import argparse
import importlib
import json
import os
import random
import re
import time
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


def call_hf(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        transformers = importlib.import_module("transformers")
        pipeline = getattr(transformers, "pipeline")

        generator = pipeline("text-generation", model="distilgpt2")
        out = generator(prompt, max_new_tokens=80, do_sample=False, temperature=0.1)
        generated = out[0]["generated_text"]
        tail = generated[len(prompt) :]
        return parse_json_from_text(tail)
    except Exception:
        return None


def llm_suggestion(
    logs: pd.Series,
    comment: str,
    backend: str = "auto",
    require_real_llm: bool = False,
) -> LLMResult:
    logs_summary = (
        f"Temp={logs['temperature_c']}°C Vib={logs['vibration_g']}g "
        f"Pressure={logs['pressure_bar']}bar RPM={logs['rpm']}"
    )
    prompt = build_prompt(logs_summary=logs_summary, operator_comment=comment)

    data: Optional[Dict[str, Any]] = None
    source = "unknown"

    if backend in ("auto", "ollama"):
        data = call_ollama(prompt)
        if data is not None:
            source = "ollama"
        else:
            source = "none"

    if data is None and backend in ("auto", "hf"):
        data = call_hf(prompt)
        if data is not None:
            source = "hf"

    if data is None:
        if require_real_llm and backend in ("ollama", "hf"):
            raise RuntimeError(
                f"Requested backend '{backend}' could not produce output. Ensure runtime/model is available."
            )
        return rules_based_llm(logs, comment)

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

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, baseline_vals, width, label="IoT Only")
    plt.bar(x + width / 2, fused_vals, width, label="IoT + LLM Feedback")
    plt.xticks(x, [lbl.upper() for lbl in labels])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Anomaly Detection Improvement with Operator Feedback")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


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
) -> None:
    seed_everything(seed)
    df = generate_synthetic_data(n=n_samples)

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
    print(f"Backend: {backend} | Samples: {n_samples} | Seed: {seed}")

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

    os.makedirs("results", exist_ok=True)
    test_view.to_csv("results/simulation_output.csv", index=False)

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
    with open("results/run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_results(metrics_baseline, metrics_fused, "results/accuracy_f1_comparison.png")
    plot_confusion_matrices(
        y_test,
        {
            "IoT Only": baseline_pred,
            "LLM Only": llm_only_pred,
            "IoT + LLM": fused_pred,
        },
        "results/confusion_matrices.png",
    )
    ap_scores = plot_precision_recall_curves(
        y_test,
        {
            "IoT Only": baseline_test_scores,
            "LLM Only": llm_test_scores,
            "IoT + LLM": fused_test_scores,
        },
        "results/precision_recall_curve.png",
    )

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
    pd.DataFrame(type_summary_rows).to_csv("results/per_anomaly_type_breakdown.csv", index=False)

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
    metrics_wide.to_csv("results/ablation_metrics.csv", index=False)

    print("\n[5] Saved files")
    print("- results/simulation_output.csv")
    print("- results/ablation_metrics.csv")
    print("- results/per_anomaly_type_breakdown.csv")
    print("- results/run_report.json")
    print("- results/accuracy_f1_comparison.png")
    print("- results/confusion_matrices.png")
    print("- results/precision_recall_curve.png")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-Enhanced Operator Feedback IoT anomaly guidance demo")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ollama", "hf", "rules"],
        help="LLM backend: auto|ollama|hf|rules",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        backend=args.backend,
        n_samples=args.samples,
        seed=args.seed,
        preview_rows=args.preview_rows,
        require_real_llm=args.require_real_llm,
    )
