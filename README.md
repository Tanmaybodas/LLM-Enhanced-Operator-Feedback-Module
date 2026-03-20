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
| Auto         | `python src/iot_anomaly_guidance.py`                  | Tries Ollama, then Hugging Face, then rules fallback |
| Rules        | `python src/iot_anomaly_guidance.py --backend rules`  | Fastest and dependency-light                         |
| Ollama       | `python src/iot_anomaly_guidance.py --backend ollama` | Requires local model runtime                         |
| Hugging Face | `python src/iot_anomaly_guidance.py --backend hf`     | Requires `transformers` + `torch`                    |

Optional flags:

- `--samples` (default: `200`)
- `--seed` (default: `42`)
- `--preview-rows` (default: `8`)

## Optional LLM Setup

Ollama (recommended):

```bash
ollama pull phi3:mini
```

Hugging Face fallback:

```bash
pip install transformers torch
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

- Full prediction output: [results/simulation_output.csv](results/simulation_output.csv)
- Metrics table: [results/ablation_metrics.csv](results/ablation_metrics.csv)
- Structured run report: [results/run_report.json](results/run_report.json)
- Visual comparison chart: [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png)

## Typical result (rules backend, `seed=42`)

| Metric    | IoT Only | IoT + LLM Fusion |
| --------- | -------: | ---------------: |
| Accuracy  |   0.8167 |           0.9333 |
| Precision |   0.7647 |           1.0000 |
| F1        |   0.7027 |           0.8889 |

Results may vary with backend, sample size, and seed.

## 2-minute teacher demo flow

1. Run:
   - `python src/iot_anomaly_guidance.py --backend rules --samples 200 --seed 42 --preview-rows 6`
2. Show the terminal sections:
   - performance table
   - confusion counts
   - improvement summary
3. Open the chart: [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png)
4. Open prediction examples: [results/simulation_output.csv](results/simulation_output.csv)
5. Open technical details (if asked): [results/run_report.json](results/run_report.json)
