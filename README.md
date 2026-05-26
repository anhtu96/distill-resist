# Anti-Distill

Code for the paper:

> **Inconsistency Is Key: Lightweight Distillation-Resistance via Dynamic Style Switching**

We study whether a teacher LLM can resist knowledge distillation by dynamically switching reasoning styles across responses, making it harder for a student to learn from the teacher's output text.

## Setup

```bash
pip install -e ".[dev]"
conda activate llm
```

All settings are read from a `.env` file selected via `ENV_FILE=configs/<name>.env`. See `.env.example` for available options.

## Pipeline

Run the three stages in order:

### 1. Baseline distillation

Generates teacher holdout traces and distills a student from them.

```bash
bash scripts/pipeline_base.sh
```

### 2. Styled traces + CAA steering vectors

Generates fresh-style teacher traces (all styles) for each dataset, trains CAA steering vectors, and uploads them to HuggingFace.

```bash
bash scripts/pipeline_fresh_caa.sh
```

Requires base holdout traces from Step 1.

### 3. Steered distillation

Generates CAA-steered holdout traces (style derived per-request via HMAC-PRF), then trains and evaluates students on them.

```bash
bash scripts/pipeline_caa_metadata.sh
```

Requires CAA steering vectors from Step 2.

## Evaluation

```bash
python scripts/evaluate.py data/test/<file>.jsonl
```

Results are appended to `results.csv` and logged to W&B (`WANDB_PROJECT` in env file).
