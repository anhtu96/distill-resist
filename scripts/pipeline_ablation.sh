#!/usr/bin/env bash
# Pipeline: Create mixed holdout set of user-defined styles, train and evaluate student on it.
# Usage: bash scripts/pipeline_scratchpad_jargon_ablation.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# 1. Define list of styles to use
STYLES=(scratchpad jargon)

# 2. Create the mixed dataset containing 100/n % of samples of each style
# Using --suffix _ablation keeps the mixed teacher holdout JSON isolated from standard runs
echo "=== Creating mixed holdout dataset ==="
ENV_FILE=configs/gemma12b.qwen3-8b.gsm8k.env python scripts/create_mixed_holdout_ablation.py --styles "${STYLES[@]}" --suffix "_ablation"

# Convert styles array to an underscore-separated slug
STYLES_SLUG=$(echo "${STYLES[@]}" | tr ' ' '_')

# 3. Train and evaluate student model on the mixed set
# Using --name-suffix _ablation ensures the model, trace output, and results json do not overlap with other runs
echo "=== Training and evaluating student model ==="
train_eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --caa-style "${STYLES_SLUG}_ablation" --name-suffix _ablation
