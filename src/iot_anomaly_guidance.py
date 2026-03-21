import argparse
import importlib
import json
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


COMMON_FAULTS = {
    "hot": "Coolant failure, blocked airflow, or overload",
    "vibration weird": "Bearing wear, imbalance, or loose mounting",
    "burning smell": "Insulation damage or electrical overheating",
    "pressure low": "Leak, clogged filter, or pump cavitation",
    "noise": "Misalignment, loosened components, or gear wear",
}

COMMENTS = [
    "machine feels hot",
    "vibration weird",
    "pressure seems low",
    "strange noise from motor",
    "burning smell near panel",
    "intermittent shaking",
    "temperature rising fast",
    "rpm unstable",
    "machine sounds rough",
    "output quality fluctuating",
]

HIGH_RISK_COMMENTS = {
    "machine feels hot",
    "vibration weird",
    "pressure seems low",
    "burning smell near panel",
    "intermittent shaking",
    "temperature rising fast",
}

LOW_RISK_COMMENTS = {
    "output quality fluctuating",
    "rpm unstable",
    "machine sounds rough",
    "strange noise from motor",
}

AI4I_PUBLIC_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/AI4I%202020%20Predictive%20Maintenance%20Dataset.csv",
]


PROMPT_TEMPLATE = """You are an expert Industry 5.0 maintenance engineer. Think step-by-step.

Example 1: Logs Temp=105°C Vib=2.1g Pressure=1.0bar RPM=1900; Comment "machine feels hot"
-> {{"root_cause":"Coolant failure or overload","action":"Check coolant pump and reduce load immediately","confidence":85}}

Example 2: Logs Temp=72°C Vib=3.4g Pressure=2.2bar RPM=1800; Comment "vibration weird"
-> {{"root_cause":"Bearing wear or shaft imbalance","action":"Inspect bearing housing and alignment within this shift","confidence":81}}

Example 3: Logs Temp=79°C Vib=1.2g Pressure=0.9bar RPM=1750; Comment "pressure seems low"
-> {{"root_cause":"Possible line leak or filter clog","action":"Check for leaks and clean/replace filter","confidence":78}}

Relevant knowledge: {relevant_knowledge}

Now: Logs {logs_summary}; Comment "{operator_comment}"
Output ONLY this JSON:
{{"root_cause":"...","action":"...","confidence":0}}
"""


@dataclass
class LLMResult:
    root_cause: str
    action: str
    confidence: float
    source: str


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_synthetic_data(n: int = 200) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        is_anomaly = np.random.rand() < 0.35

        base_temp = np.random.normal(75, 8)
        base_vibration = np.random.normal(1.1, 0.35)
        base_pressure = np.random.normal(2.0, 0.4)
        base_rpm = np.random.normal(1750, 120)

        if is_anomaly:
            comment = random.choices(
                population=COMMENTS,
                weights=[14, 16, 12, 6, 10, 12, 12, 5, 7, 6],
                k=1,
            )[0]
        else:
            comment = random.choices(
                population=COMMENTS,
                weights=[4, 4, 3, 12, 2, 3, 3, 10, 10, 16],
                k=1,
            )[0]

        if is_anomaly:
            if "hot" in comment or "temperature" in comment:
                temp = base_temp + np.random.uniform(15, 30)
                vib = max(0.2, base_vibration + np.random.uniform(0.3, 1.0))
                pressure = max(0.5, base_pressure - np.random.uniform(0.1, 0.6))
            elif "vibration" in comment or "shaking" in comment or "rough" in comment:
                vib = base_vibration + np.random.uniform(1.2, 2.5)
                temp = base_temp + np.random.uniform(2, 12)
                pressure = base_pressure + np.random.uniform(-0.2, 0.3)
            elif "pressure" in comment:
                pressure = max(0.4, base_pressure - np.random.uniform(0.8, 1.2))
                temp = base_temp + np.random.uniform(0, 8)
                vib = max(0.2, base_vibration + np.random.uniform(0.1, 0.8))
            else:
                temp = base_temp + np.random.uniform(5, 15)
                vib = base_vibration + np.random.uniform(0.5, 1.6)
                pressure = max(0.5, base_pressure - np.random.uniform(0.0, 0.8))
            rpm = base_rpm + np.random.uniform(-250, 250)
        else:
            temp = base_temp
            vib = max(0.2, base_vibration)
            pressure = max(0.8, base_pressure)
            rpm = base_rpm + np.random.uniform(-80, 80)

        rows.append(
            {
                "temperature_c": round(float(temp), 2),
                "vibration_g": round(float(vib), 2),
                "pressure_bar": round(float(pressure), 2),
                "rpm": round(float(rpm), 2),
                "operator_comment": comment,
                "is_anomaly": int(is_anomaly),
            }
        )

    return pd.DataFrame(rows)


def _select_comment_for_ai4i_row(row: pd.Series, rng: np.random.Generator) -> str:
    if int(row.get("TWF", 0)) == 1:
        return str(rng.choice(["machine feels hot", "temperature rising fast", "burning smell near panel"]))
    if int(row.get("HDF", 0)) == 1:
        return str(rng.choice(["machine feels hot", "output quality fluctuating", "temperature rising fast"]))
    if int(row.get("PWF", 0)) == 1:
        return str(rng.choice(["pressure seems low", "rpm unstable", "output quality fluctuating"]))
    if int(row.get("OSF", 0)) == 1:
        return str(rng.choice(["strange noise from motor", "vibration weird", "intermittent shaking"]))
    if int(row.get("RNF", 0)) == 1:
        return str(rng.choice(["vibration weird", "intermittent shaking", "machine sounds rough"]))
    return str(rng.choice(["machine sounds rough", "rpm unstable", "output quality fluctuating", "strange noise from motor"]))


def _load_ai4i_public_csv(path_or_url: Optional[str]) -> pd.DataFrame:
    if path_or_url:
        return pd.read_csv(path_or_url)

    last_error: Optional[Exception] = None
    for url in AI4I_PUBLIC_URLS:
        try:
            return pd.read_csv(url)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not load AI4I public dataset. Provide --public-csv with a local CSV path."
    ) from last_error


def generate_public_ai4i_data(n: int, seed: int, public_csv: Optional[str]) -> pd.DataFrame:
    raw = _load_ai4i_public_csv(public_csv)
    required_cols = {
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Machine failure",
    }
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"Public dataset missing required columns: {missing}. Expected AI4I 2020 format."
        )

    df = raw.copy()
    if n > 0 and n < len(df):
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)

    # AI4I lacks direct vibration/pressure sensors in this schema; these proxies
    # preserve relative signal while keeping feature ranges realistic.
    torque = df["Torque [Nm]"].astype(float)
    torque_min = float(torque.min())
    torque_max = float(torque.max())
    torque_norm = (torque - torque_min) / (torque_max - torque_min + 1e-9)

    temp_air_c = df["Air temperature [K]"].astype(float) - 273.15
    temp_proc_c = df["Process temperature [K]"].astype(float) - 273.15
    rpm = df["Rotational speed [rpm]"].astype(float)

    vibration_g = 0.6 + 2.9 * torque_norm
    pressure_bar = 2.4 - ((temp_proc_c - temp_air_c) * 0.12) + (torque_norm * 0.18)
    pressure_bar = pressure_bar.clip(lower=0.5, upper=3.5)

    rng = np.random.default_rng(seed)
    comments = df.apply(lambda r: _select_comment_for_ai4i_row(r, rng), axis=1)

    out = pd.DataFrame(
        {
            "temperature_c": temp_proc_c.round(2),
            "vibration_g": vibration_g.round(2),
            "pressure_bar": pressure_bar.round(2),
            "rpm": rpm.round(2),
            "operator_comment": comments,
            "is_anomaly": df["Machine failure"].astype(int),
        }
    )
    return out


def build_input_data(
    data_source: str,
    n_samples: int,
    seed: int,
    public_csv: Optional[str],
) -> pd.DataFrame:
    if data_source == "public":
        return generate_public_ai4i_data(n=n_samples, seed=seed, public_csv=public_csv)
    return generate_synthetic_data(n=n_samples)


def classify_anomaly_type(comment: str) -> str:
    c = comment.lower()
    if "vibration" in c or "shaking" in c or "rough" in c:
        return "vibration"
    if "hot" in c or "temperature" in c or "smell" in c:
        return "temperature"
    if "pressure" in c:
        return "pressure"
    return "general"


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_relevant_knowledge(comment: str) -> str:
    c = comment.lower()
    for key, value in COMMON_FAULTS.items():
        if key in c:
            return value
    if "hot" in c or "temperature" in c:
        return COMMON_FAULTS["hot"]
    if "vibration" in c or "shaking" in c or "rough" in c:
        return COMMON_FAULTS["vibration weird"]
    if "pressure" in c:
        return COMMON_FAULTS["pressure low"]
    if "smell" in c:
        return COMMON_FAULTS["burning smell"]
    if "noise" in c or "sound" in c:
        return COMMON_FAULTS["noise"]
    return "General rotating equipment fault patterns"


def build_prompt(logs_summary: str, operator_comment: str) -> str:
    knowledge = extract_relevant_knowledge(operator_comment)
    return PROMPT_TEMPLATE.format(
        logs_summary=logs_summary,
        operator_comment=operator_comment,
        relevant_knowledge=knowledge,
    )


def rules_based_llm(logs: pd.Series, comment: str) -> LLMResult:
    c = comment.lower()
    t = logs["temperature_c"]
    v = logs["vibration_g"]
    p = logs["pressure_bar"]

    if "hot" in c or t > 95:
        return LLMResult(
            "Coolant failure or thermal overload",
            "Inspect coolant loop and reduce machine load now",
            min(95, 70 + (t - 80) * 1.2),
            "rules",
        )
    if "vibration" in c or "shaking" in c or v > 2.2:
        return LLMResult(
            "Bearing wear or shaft imbalance",
            "Check bearings, alignment, and fasteners",
            min(93, 68 + (v - 1.2) * 20),
            "rules",
        )
    if "pressure" in c or p < 1.2:
        return LLMResult(
            "Possible leak or clogged filter",
            "Inspect line pressure, leakage points, and filters",
            min(90, 65 + (1.6 - p) * 25),
            "rules",
        )
    return LLMResult(
        "Early-stage mechanical/electrical degradation",
        "Schedule inspection and monitor trend every 30 minutes",
        55.0,
        "rules",
    )


def call_ollama(prompt: str, model: str = "phi3:mini") -> Optional[Dict[str, Any]]:
    ollama = importlib.import_module("ollama")
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            content = response.get("message", {}).get("content", "")
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                content = content.strip()
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            parsed_fallback = parse_json_from_text(content)
            if parsed_fallback is not None:
                return parsed_fallback
        except Exception:
            pass

        if attempt < 2:
            time.sleep(0.2)

    return None





def llm_suggestion(
    logs: pd.Series,
    comment: str,
    backend: str = "auto",
    require_real_llm: bool = False,
) -> LLMResult:
    attempted_backends: List[str] = []
    logs_summary = (
        f"Temp={logs['temperature_c']}°C Vib={logs['vibration_g']}g "
        f"Pressure={logs['pressure_bar']}bar RPM={logs['rpm']}"
    )
    prompt = build_prompt(logs_summary=logs_summary, operator_comment=comment)

    data: Optional[Dict[str, Any]] = None
    source = "unknown"

    if backend in ("auto", "ollama"):
        attempted_backends.append("ollama")
        data = call_ollama(prompt)
        if data is not None:
            source = "ollama"
        else:
            source = "none"



    if data is None:
        if attempted_backends and backend != "rules":
            warnings.warn(
                f"Backend '{backend}' unavailable; falling back to rules. "
                f"Install required dependencies or use --backend rules.",
                UserWarning
            )
        if require_real_llm and backend == "ollama":
            raise RuntimeError(
                f"Requested backend '{backend}' could not produce output. Ensure runtime/model is available."
            )
        rules_result = rules_based_llm(logs, comment)
        if attempted_backends and backend != "rules":
            rules_result.source = f"{backend}->fallback_to_rules"
        return rules_result

    root_cause = str(data.get("root_cause", "")).strip()
    action = str(data.get("action", "")).strip()
    rule_hint = rules_based_llm(logs, comment)
    if not root_cause:
        root_cause = rule_hint.root_cause
    if not action:
        action = rule_hint.action

    try:
        confidence = float(data.get("confidence", rule_hint.confidence))
    except (TypeError, ValueError):
        confidence = float(rule_hint.confidence)

    confidence = max(0.0, min(100.0, confidence))
    return LLMResult(root_cause, action, confidence, source)


def comment_risk_score(comment: str) -> float:
    if comment in HIGH_RISK_COMMENTS:
        return 0.9
    if comment in LOW_RISK_COMMENTS:
        return 0.35
    return 0.55


def compute_zscore_anomaly(df: pd.DataFrame) -> np.ndarray:
    features = df[["temperature_c", "vibration_g", "pressure_bar", "rpm"]].copy()
    mu = features.mean()
    sigma = features.std(ddof=0).replace(0, 1e-6)
    z = ((features - mu) / sigma).abs()

    z["pressure_bar"] = z["pressure_bar"] * 1.1
    weighted = 0.35 * z["temperature_c"] + 0.35 * z["vibration_g"] + 0.2 * z["pressure_bar"] + 0.1 * z["rpm"]

    normalized = (weighted - weighted.min()) / (weighted.max() - weighted.min() + 1e-9)
    return normalized.values


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
) -> float:
    best_threshold = 0.5
    best_accuracy = -1.0
    best_f1 = -1.0

    for threshold in np.linspace(0.2, 0.8, 121):
        pred = (scores > threshold).astype(int)
        metrics = evaluate(y_true, pred)

        if metrics["precision"] < min_precision or metrics["recall"] < min_recall:
            continue

        if metrics["accuracy"] > best_accuracy or (
            np.isclose(metrics["accuracy"], best_accuracy) and metrics["f1"] > best_f1
        ):
            best_accuracy = metrics["accuracy"]
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold


def confusion_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def plot_results(metrics_baseline: Dict[str, float], metrics_fused: Dict[str, float], out_path: str) -> None:
    labels = ["accuracy", "precision", "recall", "f1"]
    baseline_vals = [metrics_baseline[k] for k in labels]
    fused_vals = [metrics_fused[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="IoT Only",
        edgecolor="black",
        linewidth=0.8,
    )
    fused_bars = ax.bar(
        x + width / 2,
        fused_vals,
        width,
        label="IoT + LLM Feedback",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(x, [lbl.upper() for lbl in labels])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection Improvement with Operator Feedback")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    # Keep zero-height metrics readable by labeling each bar directly.
    for bar in list(baseline_bars) + list(fused_bars):
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(h, 0.01) + 0.01,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrices(y_true: np.ndarray, preds: Dict[str, np.ndarray], out_path: str) -> None:
    model_names = list(preds.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_true, preds[name], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_precision_recall_curves(y_true: np.ndarray, score_dict: Dict[str, np.ndarray], out_path: str) -> Dict[str, float]:
    plt.figure(figsize=(8, 5))
    ap_scores: Dict[str, float] = {}

    for name, scores in score_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ap_scores[name] = float(ap)
        plt.plot(rec, prec, linewidth=2, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()
    return ap_scores


def compute_operator_acceptance(
    confidences: np.ndarray,
    seed: int,
    conf_threshold: float = 75.0,
    acceptance_min: float = 0.8,
    acceptance_max: float = 0.9,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed + 1000)
    accepted = np.zeros_like(confidences, dtype=int)

    high_mask = confidences > conf_threshold
    high_indices = np.where(high_mask)[0]
    if len(high_indices) > 0:
        p_accept = float(rng.uniform(acceptance_min, acceptance_max))
        accepted_high = rng.binomial(1, p_accept, size=len(high_indices))
        accepted[high_indices] = accepted_high
    else:
        p_accept = float((acceptance_min + acceptance_max) / 2)

    return accepted, p_accept


def run_pipeline(
    backend: str = "auto",
    n_samples: int = 200,
    seed: int = 42,
    preview_rows: int = 8,
    require_real_llm: bool = False,
    data_source: str = "synthetic",
    public_csv: Optional[str] = None,
    output_tag: Optional[str] = None,
    results_mode: str = "compact",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    seed_everything(seed)
    df = build_input_data(
        data_source=data_source,
        n_samples=n_samples,
        seed=seed,
        public_csv=public_csv,
    )

    z_score_anomaly = compute_zscore_anomaly(df)

    llm_conf_list: List[float] = []
    root_causes: List[str] = []
    actions: List[str] = []
    comment_risk_list: List[float] = []
    llm_source_list: List[str] = []

    for _, row in df.iterrows():
        result = llm_suggestion(
            row,
            row["operator_comment"],
            backend=backend,
            require_real_llm=require_real_llm,
        )
        llm_conf_list.append(result.confidence)
        root_causes.append(result.root_cause)
        actions.append(result.action)
        comment_risk_list.append(comment_risk_score(row["operator_comment"]))
        llm_source_list.append(result.source)

    df["llm_confidence"] = llm_conf_list
    df["root_cause"] = root_causes
    df["action"] = actions
    df["comment_risk"] = comment_risk_list
    df["llm_source"] = llm_source_list
    df["z_score_anomaly"] = z_score_anomaly
    df["anomaly_type"] = df["operator_comment"].apply(classify_anomaly_type)

    X = np.column_stack(
        [
            df["z_score_anomaly"].values,
            df["llm_confidence"].values / 100.0,
            df["comment_risk"].values,
        ]
    )

    y_true = df["is_anomaly"].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y_true,
        np.arange(len(df)),
        test_size=0.3,
        stratify=y_true,
        random_state=42,
    )

    baseline_train_scores = X_train[:, 0]
    baseline_test_scores = X_test[:, 0]
    baseline_threshold = find_best_threshold(y_train, baseline_train_scores, min_precision=0.60)
    baseline_pred = (baseline_test_scores > baseline_threshold).astype(int)

    llm_train_scores = X_train[:, 1]
    llm_test_scores = X_test[:, 1]
    llm_threshold = find_best_threshold(y_train, llm_train_scores, min_precision=0.60)
    llm_only_pred = (llm_test_scores > llm_threshold).astype(int)

    fusion_model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
    fusion_model.fit(X_train, y_train)
    fused_train_scores = fusion_model.predict_proba(X_train)[:, 1]
    fused_test_scores = fusion_model.predict_proba(X_test)[:, 1]
    fused_threshold = find_best_threshold(y_train, fused_train_scores, min_precision=0.70)
    fused_pred = (fused_test_scores > fused_threshold).astype(int)

    metrics_baseline = evaluate(y_test, baseline_pred)
    metrics_llm_only = evaluate(y_test, llm_only_pred)
    metrics_fused = evaluate(y_test, fused_pred)
    baseline_conf = confusion_breakdown(y_test, baseline_pred)
    llm_only_conf = confusion_breakdown(y_test, llm_only_pred)
    fused_conf = confusion_breakdown(y_test, fused_pred)

    accepted_flags, sampled_acceptance_prob = compute_operator_acceptance(
        confidences=df.iloc[idx_test]["llm_confidence"].values,
        seed=seed,
        conf_threshold=75.0,
        acceptance_min=0.8,
        acceptance_max=0.9,
    )

    test_view = df.iloc[idx_test].copy()
    test_view["operator_accepted"] = accepted_flags
    test_view["operator_acceptance_rate_used"] = sampled_acceptance_prob

    fused_acceptance_pred = np.where(
        test_view["operator_accepted"].values == 1,
        fused_pred,
        baseline_pred,
    )
    metrics_acceptance = evaluate(y_test, fused_acceptance_pred)
    acceptance_conf = confusion_breakdown(y_test, fused_acceptance_pred)

    test_view["baseline_pred"] = baseline_pred
    test_view["llm_only_pred"] = llm_only_pred
    test_view["fused_pred"] = fused_pred
    test_view["fused_acceptance_pred"] = fused_acceptance_pred
    test_view["llm_only_probability"] = llm_test_scores
    test_view["fused_probability"] = fused_test_scores
    test_view["is_correct_fused"] = (test_view["fused_pred"].values == y_test).astype(int)
    test_view = test_view.sort_values(by=["fused_probability", "z_score_anomaly"], ascending=False)

    print("\n" + "=" * 72)
    print(" IoT ANOMALY GUIDANCE - RUN SUMMARY ")
    print("=" * 72)
    print(f"Backend: {backend} | Samples: {n_samples} | Seed: {seed} | Data: {data_source}")

    perf = pd.DataFrame(
        [metrics_baseline, metrics_llm_only, metrics_fused, metrics_acceptance],
        index=["IoT Only", "LLM Only", "IoT + LLM", "IoT + LLM + Acceptance"],
    )
    perf = perf.rename(columns={"accuracy": "acc", "precision": "prec", "recall": "rec", "f1": "f1"})
    print("\n[1] Performance (test split)")
    print(perf.to_string(float_format=lambda x: f"{x:0.4f}"))
    print(
        f"Thresholds: baseline={baseline_threshold:.3f} | "
        f"llm_only={llm_threshold:.3f} | fused={fused_threshold:.3f}"
    )
    print(
        "LLM sources used: "
        + ", ".join([f"{k}={v}" for k, v in test_view["llm_source"].value_counts().to_dict().items()])
    )

    conf_table = pd.DataFrame(
        [baseline_conf, llm_only_conf, fused_conf, acceptance_conf],
        index=["IoT Only", "LLM Only", "IoT + LLM", "IoT + LLM + Acceptance"],
    )
    print("\n[2] Confusion counts")
    print(conf_table.to_string())

    improvement = pd.Series(
        {
            "accuracy": metrics_fused["accuracy"] - metrics_baseline["accuracy"],
            "precision": metrics_fused["precision"] - metrics_baseline["precision"],
            "f1": metrics_fused["f1"] - metrics_baseline["f1"],
        }
    ).round(4)
    print("\n[3] Improvement (absolute)")
    print(improvement.to_string())
    print(
        f"Operator acceptance simulation: conf>75 accepted with sampled rate={sampled_acceptance_prob:.3f}"
    )

    type_breakdown = (
        test_view[test_view["is_anomaly"] == 1]
        .groupby("anomaly_type")
        .apply(lambda g: pd.Series(evaluate(g["is_anomaly"].values, g["fused_pred"].values)))
    )
    if not type_breakdown.empty:
        print("\n[3b] Per-anomaly-type breakdown (fused, anomaly rows)")
        print(type_breakdown.round(4).to_string())

    compact_preview = test_view[
        [
            "temperature_c",
            "vibration_g",
            "pressure_bar",
            "operator_comment",
            "llm_confidence",
            "fused_probability",
            "is_anomaly",
            "fused_pred",
            "root_cause",
            "action",
        ]
    ].copy()
    compact_preview["fused_probability"] = compact_preview["fused_probability"].round(3)

    terminal_preview = compact_preview.head(preview_rows).copy()
    print(f"\n[4] Top {preview_rows} simulation rows (clean view)")
    for i, (_, row) in enumerate(terminal_preview.iterrows(), start=1):
        print(
            f"Case {i}: "
            f"temp={row['temperature_c']:.2f} | "
            f"vib={row['vibration_g']:.2f} | "
            f"press={row['pressure_bar']:.2f} | "
            f"comment=\"{row['operator_comment']}\" | "
            f"llm_conf={row['llm_confidence']:.3f} | "
            f"p_fused={row['fused_probability']:.3f} | "
            f"y_true={int(row['is_anomaly'])} | "
            f"y_pred={int(row['fused_pred'])}"
        )
        print(f"   cause     : {row['root_cause']}")
        print(f"   what_to_do: {row['action']}")
        print("-" * 72)

    if save_artifacts:
        os.makedirs("results", exist_ok=True)
    tag = f"_{output_tag}" if output_tag else ""
    simulation_output_path = f"results/simulation_output{tag}.csv"
    preview_output_path = f"results/preview_rows{tag}.csv"
    metrics_path = f"results/metrics{tag}.csv"
    run_report_path = f"results/run_report{tag}.json"
    accuracy_plot_path = f"results/accuracy_f1_comparison{tag}.png"
    confusion_plot_path = f"results/confusion_matrices{tag}.png"
    pr_plot_path = f"results/precision_recall_curve{tag}.png"
    anomaly_breakdown_path = f"results/per_anomaly_type_breakdown{tag}.csv"
    ablation_metrics_path = f"results/ablation_metrics{tag}.csv"

    if save_artifacts and results_mode == "full":
        test_view.to_csv(simulation_output_path, index=False)
        compact_preview.head(max(20, preview_rows)).to_csv(preview_output_path, index=False)

    report = {
        "backend": backend,
        "require_real_llm": require_real_llm,
        "seed": seed,
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline_threshold": baseline_threshold,
        "llm_only_threshold": llm_threshold,
        "fused_threshold": fused_threshold,
        "metrics": {
            "baseline": {k: float(v) for k, v in metrics_baseline.items()},
            "llm_only": {k: float(v) for k, v in metrics_llm_only.items()},
            "fused": {k: float(v) for k, v in metrics_fused.items()},
            "fused_with_acceptance": {k: float(v) for k, v in metrics_acceptance.items()},
        },
        "confusion": {
            "baseline": baseline_conf,
            "llm_only": llm_only_conf,
            "fused": fused_conf,
            "fused_with_acceptance": acceptance_conf,
        },
        "llm_source_counts_test": {k: int(v) for k, v in test_view["llm_source"].value_counts().to_dict().items()},
        "operator_acceptance": {
            "confidence_threshold": 75,
            "sampled_acceptance_probability": sampled_acceptance_prob,
            "accepted_count": int(test_view["operator_accepted"].sum()),
            "eligible_count": int((test_view["llm_confidence"] > 75).sum()),
        },
    }
    report["data_source"] = data_source
    report["output_tag"] = output_tag or ""
    if save_artifacts:
        with open(run_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        plot_results(metrics_baseline, metrics_fused, accuracy_plot_path)
    if save_artifacts and results_mode == "full":
        plot_confusion_matrices(
            y_test,
            {
                "IoT Only": baseline_pred,
                "LLM Only": llm_only_pred,
                "IoT + LLM": fused_pred,
            },
            confusion_plot_path,
        )
    score_dict = {
        "IoT Only": baseline_test_scores,
        "LLM Only": llm_test_scores,
        "IoT + LLM": fused_test_scores,
    }
    if save_artifacts and results_mode == "full":
        ap_scores = plot_precision_recall_curves(y_test, score_dict, pr_plot_path)
    else:
        ap_scores = {name: float(average_precision_score(y_test, scores)) for name, scores in score_dict.items()}

    type_summary_rows = []
    for t in sorted(test_view["anomaly_type"].unique()):
        grp = test_view[test_view["anomaly_type"] == t]
        m = evaluate(grp["is_anomaly"].values, grp["fused_pred"].values)
        type_summary_rows.append(
            {
                "anomaly_type": t,
                "n": int(len(grp)),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    if save_artifacts and results_mode == "full":
        pd.DataFrame(type_summary_rows).to_csv(anomaly_breakdown_path, index=False)

    metrics_wide = pd.DataFrame(
        [
            {"model": "IoT Only", **metrics_baseline},
            {"model": "LLM Only", **metrics_llm_only},
            {"model": "IoT + LLM", **metrics_fused},
            {"model": "IoT + LLM + Acceptance", **metrics_acceptance},
        ]
    )
    metrics_wide["average_precision_pr"] = metrics_wide["model"].map(
        {
            "IoT Only": ap_scores["IoT Only"],
            "LLM Only": ap_scores["LLM Only"],
            "IoT + LLM": ap_scores["IoT + LLM"],
            "IoT + LLM + Acceptance": np.nan,
        }
    )
    if save_artifacts:
        metrics_wide.to_csv(metrics_path, index=False)
    if save_artifacts and results_mode == "full":
        metrics_wide.to_csv(ablation_metrics_path, index=False)

    print("=" * 72)

    report["saved_files"] = {}
    if save_artifacts:
        report["saved_files"] = {
            "metrics": metrics_path,
            "run_report": run_report_path,
            "accuracy_plot": accuracy_plot_path,
        }
    if save_artifacts and results_mode == "full":
        report["saved_files"]["simulation_output"] = simulation_output_path
        report["saved_files"]["preview_rows"] = preview_output_path
        report["saved_files"]["ablation_metrics"] = ablation_metrics_path
        report["saved_files"]["per_anomaly_type_breakdown"] = anomaly_breakdown_path
        report["saved_files"]["confusion_plot"] = confusion_plot_path
        report["saved_files"]["pr_plot"] = pr_plot_path
    return report


def run_backend_benchmark(
    backends: List[str],
    n_samples: int,
    seed: int,
    preview_rows: int,
    require_real_llm: bool,
    data_source: str,
    public_csv: Optional[str],
    results_mode: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    save_per_backend_artifacts = results_mode == "full"
    for backend in backends:
        tag = f"{data_source}_{backend}"
        report = run_pipeline(
            backend=backend,
            n_samples=n_samples,
            seed=seed,
            preview_rows=preview_rows,
            require_real_llm=require_real_llm,
            data_source=data_source,
            public_csv=public_csv,
            output_tag=tag,
            results_mode=results_mode,
            save_artifacts=save_per_backend_artifacts,
        )
        rows.append(
            {
                "backend": backend,
                "data_source": data_source,
                "fused_accuracy": report["metrics"]["fused"]["accuracy"],
                "fused_precision": report["metrics"]["fused"]["precision"],
                "fused_recall": report["metrics"]["fused"]["recall"],
                "fused_f1": report["metrics"]["fused"]["f1"],
                "llm_only_accuracy": report["metrics"]["llm_only"]["accuracy"],
                "llm_only_f1": report["metrics"]["llm_only"]["f1"],
                "llm_source_counts_test": json.dumps(report.get("llm_source_counts_test", {})),
                "run_report": report.get("saved_files", {}).get("run_report", ""),
            }
        )

    comp = pd.DataFrame(rows).sort_values(by="fused_f1", ascending=False)
    os.makedirs("results", exist_ok=True)
    comp_path = f"results/backend_comparison_{data_source}.csv"
    comp.to_csv(comp_path, index=False)

    metrics_for_plot = ["fused_accuracy", "fused_precision", "fused_recall", "fused_f1"]
    x = np.arange(len(metrics_for_plot))
    width = 0.8 / max(1, len(comp))

    plt.figure(figsize=(11, 5))
    for i, (_, row) in enumerate(comp.iterrows()):
        vals = [float(row[m]) for m in metrics_for_plot]
        plt.bar(x + (i - (len(comp) - 1) / 2) * width, vals, width=width, label=str(row["backend"]))
    plt.xticks(x, [m.replace("fused_", "").upper() for m in metrics_for_plot])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Backend Comparison ({data_source})")
    plt.legend(title="Backend")
    plt.tight_layout()
    comp_plot_path = f"results/backend_comparison_{data_source}.png"
    plt.savefig(comp_plot_path, dpi=140)
    plt.close()

    benchmark_report = {
        "data_source": data_source,
        "seed": seed,
        "samples": n_samples,
        "backends": backends,
        "results_mode": results_mode,
        "rows": rows,
        "saved_files": {
            "comparison_csv": comp_path,
            "comparison_plot": comp_plot_path,
        },
    }
    benchmark_report_path = f"results/backend_benchmark_report_{data_source}.json"
    with open(benchmark_report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n[6] Backend Benchmark Summary")
    print(comp[["backend", "fused_accuracy", "fused_precision", "fused_recall", "fused_f1"]].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"- {comp_path}")
    print(f"- {comp_plot_path}")
    print(f"- {benchmark_report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-Enhanced Operator Feedback IoT anomaly guidance demo")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ollama", "rules"],
        help="LLM backend: auto (tries ollama, then rules) | ollama (requires local model) | rules (keyword-based, fastest)",
    )
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic records (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="How many top simulation rows to print in console (default: 8)",
    )
    parser.add_argument(
        "--require-real-llm",
        action="store_true",
        help="Fail run if selected real backend (ollama/hf) is unavailable and fallback would occur.",
    )
    parser.add_argument(
        "--data-source",
        default="synthetic",
        choices=["synthetic", "public"],
        help="Input data source: synthetic or public AI4I dataset.",
    )
    parser.add_argument(
        "--public-csv",
        default=None,
        help="Optional local path/URL to AI4I public dataset CSV.",
    )
    parser.add_argument(
        "--benchmark-backends",
        default="",
        help="Comma-separated backend benchmark list, e.g. rules,ollama or rules,hf,ollama",
    )
    parser.add_argument(
        "--results-mode",
        default="compact",
        choices=["compact", "full"],
        help="compact saves only essential outputs; full saves all diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Keep one stable artifact set per backend/data-source pair.
    chosen_tag: Optional[str] = f"{args.backend}_{args.data_source}"

    if args.benchmark_backends.strip():
        backends = [b.strip() for b in args.benchmark_backends.split(",") if b.strip()]
        run_backend_benchmark(
            backends=backends,
            n_samples=args.samples,
            seed=args.seed,
            preview_rows=args.preview_rows,
            require_real_llm=args.require_real_llm,
            data_source=args.data_source,
            public_csv=args.public_csv,
            results_mode=args.results_mode,
        )
    else:
        run_pipeline(
            backend=args.backend,
            n_samples=args.samples,
            seed=args.seed,
            preview_rows=args.preview_rows,
            require_real_llm=args.require_real_llm,
            data_source=args.data_source,
            public_csv=args.public_csv,
            output_tag=chosen_tag,
            results_mode=args.results_mode,
        )
