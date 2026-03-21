# LLM-Enhanced Operator Feedback Module

Real-time IoT anomaly guidance demo for Industry 5.0 using synthetic machine logs + operator text feedback + LLM-assisted fusion.

## Why this project

Traditional anomaly detection can miss context. This project adds operator comments (for example, “machine feels hot”) and fuses them with sensor analytics to improve detection quality.

## Highlights

| Capability                   | What it does                                                   |
| ---------------------------- | -------------------------------------------------------------- |
| Synthetic simulation         | Generates 200 realistic IoT + operator feedback records        |
| Prompt-engineered LLM signal | Produces structured JSON: `root_cause`, `action`, `confidence` |
| Hybrid decision model        | Combines sensor score + LLM confidence + comment risk prior    |
| Better results               | Improves accuracy / precision / F1 vs IoT-only baseline        |
| Presentation-ready outputs   | Clean terminal summary + CSVs + JSON report + plot             |

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/iot_anomaly_guidance.py --backend rules --samples 200 --seed 42 --preview-rows 6
```

## Run Modes

| Mode         | Command                                               | Notes                                                |
| ------------ | ----------------------------------------------------- | ---------------------------------------------------- |
| Auto         | `python src/iot_anomaly_guidance.py`                  | Tries Ollama, then rules fallback                    |
| Rules        | `python src/iot_anomaly_guidance.py --backend rules`  | Fastest and dependency-light                         |
| Ollama       | `python src/iot_anomaly_guidance.py --backend ollama` | Requires local model runtime                         |

Optional flags:

- `--samples` (default: `200`)
- `--seed` (default: `42`)
- `--preview-rows` (default: `8`)

## Optional LLM Setup

Ollama (recommended):

```bash
ollama pull phi3:mini
```



## How it works

Pipeline in [src/iot_anomaly_guidance.py](src/iot_anomaly_guidance.py):

1. Generate synthetic IoT records with fields:
   - `temperature_c`, `vibration_g`, `pressure_bar`, `rpm`
   - `operator_comment`
   - `is_anomaly` (ground truth)
2. Build IoT-only baseline via normalized z-score anomaly signal.
3. Generate LLM output (or rules fallback):
   - root cause
   - recommended action
   - confidence score (0–100)
4. Build fused feature set:
   - sensor anomaly score
   - LLM confidence
   - comment risk prior
5. Train fusion classifier (`LogisticRegression`) and evaluate on held-out test split.
6. Tune thresholds for better precision/accuracy tradeoff.
7. Print clean summary and save artifacts.

## Output files

- Metrics table: [results/metrics.csv](results/metrics.csv)
- Run report with details: [results/run_report.json](results/run_report.json)
- Performance comparison chart: [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png)
- Backend benchmark comparison: [results/backend_comparison_synthetic.csv](results/backend_comparison_synthetic.csv)
- Backend performance chart: [results/backend_comparison_synthetic.png](results/backend_comparison_synthetic.png)

## Typical result (rules backend, `seed=42`)

| Metric    | IoT Only | IoT + LLM Fusion |
| --------- | -------: | ---------------: |
| Accuracy  |   0.8167 |           0.9333 |
| Precision |   0.7647 |           1.0000 |
| F1        |   0.7027 |           0.8889 |

Results may vary with backend, sample size, and seed.

## Real LLM Results: Ollama (phi3:mini) vs Rules

To validate real LLM execution, we benchmark **Ollama (local LLM)** against the **rules baseline**. Both run with identical data (`seed=42`, 200 samples):

```bash
ollama pull phi3:mini
python src/iot_anomaly_guidance.py --backend ollama --samples 200 --seed 42 --preview-rows 6
```

### Backend Comparison (Fused Model)

| Backend | Accuracy | Precision | Recall | F1      | LLM Source |
| ------- | -------: | --------: | -----: | ------: | ---------- |
| Ollama  |    0.958 |     0.875 |  1.000 | **0.933** | phi3:mini  |
| Rules   |    0.958 |     1.000 |  0.857 | 0.923   | keyword    |

**Key Observations:**
- **Identical accuracy** (95.8%) — LLM fusion doesn't degrade performance
- **Ollama achieves perfect recall** (100%) — catches all anomalies, though with slightly lower precision (87.5% vs 100%)
- **F1 parity** — Ollama +1% F1 gain despite lower precision, driven by superior recall
- **Natural language root causes** — Ollama generates context-aware explanations vs keyword-matched rules (e.g., "Bearing wear or shaft imbalance" for vibration + operator comment "intermittent shaking")

### LLM Reasoning Example

**Ollama output (phi3:mini):**
```json
{
  "root_cause": "Bearing wear or shaft imbalance",
  "action": "Check bearings, alignment, and fasteners",
  "confidence": 93
}
```

**Rules baseline (keyword match):**
```
If vibration > threshold → "Bearing wear" 
If temperature > threshold → "Coolant failure"
```

### Limitations & Considerations
- **Small model sensitivity** — phi3:mini occasionally shows prompt/seed sensitivity; larger models (e.g., `ollama run llama2`) may be more stable
- **JSON forcing** — We enforced structured output to prevent hallucinations (see `call_ollama()` in source)
- **Synthetic data** — Operator comments are generated; real human feedback would strengthen validation

### Result Artifacts
See [results/backend_comparison_synthetic.csv](results/backend_comparison_synthetic.csv) and [results/backend_comparison_synthetic.png](results/backend_comparison_synthetic.png) for the full benchmark run.

## 2-minute teacher demo flow

1. Run:
   - `python src/iot_anomaly_guidance.py --backend rules --samples 200 --seed 42 --preview-rows 6`
2. Show the terminal sections:
   - performance table
   - improvement summary
3. Open the metrics chart: [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png)
4. Open metrics CSV: [results/metrics.csv](results/metrics.csv)
5. Open technical details (if asked): [results/run_report.json](results/run_report.json)
