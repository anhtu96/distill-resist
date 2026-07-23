#!/usr/bin/env bash
# Pipeline: paraphrase CAA-steered holdout traces using DIPPER and distill students on paraphrased traces.
# Usage: bash scripts/pipeline_paraphrase_attack.sh [style]
# If no style is specified, defaults to "mixed".

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

STYLE="${1:-mixed}"

ENV_FILE=configs/gemma12b.qwen.gsm8k.env

echo "=== Step 1: Paraphrasing CAA-steered holdout traces for style '${STYLE}' ==="
if [ "$STYLE" = "mixed" ]; then
    ENV_FILE="$ENV_FILE" python scripts/paraphrase_dataset.py --with-caa --batch-size 32 --skip-existing
else
    ENV_FILE="$ENV_FILE" python scripts/paraphrase_dataset.py --with-caa --caa-style "$STYLE" --batch-size 32 --skip-existing
fi

echo "=== Step 2: Training student distilled from paraphrased traces ==="
if [ "$STYLE" = "mixed" ]; then
    train_eval "$ENV_FILE" --with-caa --paraphrased
else
    train_eval "$ENV_FILE" --with-caa --caa-style "$STYLE" --paraphrased
fi

echo "=== Pipeline Completed Successfully ==="
