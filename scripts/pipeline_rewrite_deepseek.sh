#!/usr/bin/env bash
# Pipeline: rewrite DeepSeek-R1-Distill-Qwen-7B base holdout traces with Gemma (all styles).
# Requires base holdout traces to already exist (run pipeline_base.sh first).
# Usage: bash scripts/pipeline_rewrite_deepseek.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- Rewrite base holdout traces with Gemma ---

# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --rewrite
# echo "=== Done: deepseek rewrite gsm8k ==="

# ENV_FILE=configs/deepseek.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --rewrite
# echo "=== Done: deepseek rewrite arc ==="

# ENV_FILE=configs/deepseek.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa --rewrite
# echo "=== Done: deepseek rewrite csqa ==="

# ENV_FILE=configs/deepseek.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu --rewrite
# echo "=== Done: deepseek rewrite mmlu ==="

# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --rewrite
# echo "=== Done: deepseek rewrite math ==="

# --- Train CAA vectors from rewritten traces ---

ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/train_caa_vectors.py \
    --datasets gsm8k arc csqa \
    --backend vllm \
    --data-source rewrite \
    --all-layers \
    --max-length 8192 \
    --overwrite \
    --output-dir models/steering_vectors_caa_vllm \
    --no-compliance-filter
echo "=== Done: deepseek CAA vectors (from rewrite) ==="
