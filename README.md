# LLM-Enhanced Operator Feedback for Real-Time IoT Anomaly Guidance

This project demonstrates an Industry 5.0 workflow where synthetic IoT sensor logs are fused with operator comments and an LLM confidence signal to improve anomaly detection quality.

## Overview

The script in [src/iot_anomaly_guidance.py](src/iot_anomaly_guidance.py) does the following:

1. Simulates 200 machine records with:
   - IoT signals: `temperature_c`, `vibration_g`, `pressure_bar`, `rpm`
   - Operator comments such as “machine feels hot” and “vibration weird”
   - Ground-truth anomaly label
2. Computes a baseline anomaly score from sensor z-scores only.
3. Builds a prompt (role + few-shot examples + forced JSON schema).
4. Gets LLM output with:
   - `root_cause`
   - `action`
   - `confidence` (0–100)
5. Builds a cleaner hybrid detector with:
   - Sensor anomaly score (`z_score_anomaly`)
   - LLM confidence (`llm_confidence`)
   - Comment risk prior (`comment_risk`)
   - Logistic fusion model trained on a train split and evaluated on a test split

6. Evaluates baseline vs fused model with accuracy, precision, recall, F1, and confusion counts.
7. Saves CSV output and a comparison plot.

## Project Structure

- [src/iot_anomaly_guidance.py](src/iot_anomaly_guidance.py): main simulation and evaluation pipeline
- [requirements.txt](requirements.txt): core Python dependencies
- [results/simulation_output.csv](results/simulation_output.csv): generated record-level outputs
- [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png): metrics visualization

## Prerequisites

- Python 3.10+ (tested with 3.13)
- Virtual environment recommended

Optional model runtimes:

- Ollama for local `phi3:mini`
- Hugging Face backend (`transformers` + `torch`)

## Installation

### 1) Create and activate environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install core dependencies

```bash
pip install -r requirements.txt
```

### 3) Optional backends

Ollama (recommended local LLM):

```bash
ollama pull phi3:mini
```

Hugging Face fallback:

```bash
pip install transformers torch
```

## How to Run

From project root, run one of the following:

Default mode (tries Ollama, then Hugging Face, else rules fallback):

```bash
python src/iot_anomaly_guidance.py
```

Force rules backend (fastest, no LLM runtime needed):

```bash
python src/iot_anomaly_guidance.py --backend rules
```

Force Ollama backend:

```bash
python src/iot_anomaly_guidance.py --backend ollama
```

Force Hugging Face backend:

```bash
python src/iot_anomaly_guidance.py --backend hf
```

If using this workspace virtual environment directly on Windows:

```powershell
c:/Users/tanma/OneDrive/Desktop/IOTSA/.venv/Scripts/python.exe src/iot_anomaly_guidance.py --backend rules
```

Reproducible teacher-demo run (recommended):

```powershell
c:/Users/tanma/OneDrive/Desktop/IOTSA/.venv/Scripts/python.exe src/iot_anomaly_guidance.py --backend rules --samples 200 --seed 42 --preview-rows 6
```

Optional CLI flags:

- `--samples`: number of synthetic records (default `200`)
- `--seed`: random seed for reproducible metrics (default `42`)
- `--preview-rows`: number of top simulation rows shown in console (default `8`)

## How It Works Internally

- Synthetic anomaly generation injects realistic deviations by comment context (heat, vibration, pressure, etc.).
- Prompt engineering enforces structured LLM output for robust parsing.
- A tiny built-in knowledge dictionary (`COMMON_FAULTS`) injects domain hints into the prompt.
- Fusion adds human-context awareness that baseline sensor scoring can miss.

## Outputs

After each run:

- Console prints:
  - baseline vs fused metrics
  - selected thresholds
  - confusion matrix counts
  - compact top-risk simulation preview
- Full test predictions: [results/simulation_output.csv](results/simulation_output.csv)
- Compact preview CSV: [results/simulation_preview_compact.csv](results/simulation_preview_compact.csv)
- Metrics summary CSV: [results/metrics_summary.csv](results/metrics_summary.csv)
- Structured report JSON: [results/run_report.json](results/run_report.json)
- Metrics plot: [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png)

## Typical Result (rules backend, seed=42)

- Baseline accuracy: `0.8167`
- Fused accuracy: `0.9333`
- Baseline precision: `0.7647`
- Fused precision: `1.0000`
- Baseline F1: `0.7027`
- Fused F1: `0.8889`

Exact values vary with random seed and backend.

## Demo Flow for Teacher (2–3 minutes)

1. Run the reproducible command above.
2. Show the console section "Performance" and "Improvement (absolute)".
3. Open [results/accuracy_f1_comparison.png](results/accuracy_f1_comparison.png) to visualize gain.
4. Open [results/simulation_preview_compact.csv](results/simulation_preview_compact.csv) to show clear simulated cases.
5. If asked for details, open [results/run_report.json](results/run_report.json).
