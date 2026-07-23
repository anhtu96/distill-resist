#!/usr/bin/env bash
# Pipeline: normalize CAA-steered holdout traces and distill students on normalized traces.
# Usage: bash scripts/pipeline_normalization_attack.sh [style]
# If no style is specified, defaults to "mixed".

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

STYLE="${1:-mixed}"

ENV_FILE=configs/gemma12b.qwen.gsm8k.env
echo "=== Step 1: Normalizing CAA-steered holdout traces for style '${STYLE}' ==="
ENV_FILE="$ENV_FILE" python scripts/attacker_normalize.py --caa-style "$STYLE" --batch-size 256

echo "=== Step 2: Training student distilled from normalized traces ==="
if [ "$STYLE" = "mixed" ]; then
    train_eval "$ENV_FILE" --with-caa --normalized
else
    train_eval "$ENV_FILE" --with-caa --caa-style "$STYLE" --normalized
fi

echo "=== Pipeline Completed Successfully ==="
