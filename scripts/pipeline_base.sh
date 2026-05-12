#!/usr/bin/env bash
# Pipeline: generate base holdout traces and distill student from them.
# Usage: bash scripts/pipeline_base.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- Generate holdout traces (shared across both students) ---

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k
echo "=== Done: base holdout gsm8k ==="

ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc
echo "=== Done: base holdout arc ==="

ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math
echo "=== Done: base holdout math ==="

# --- Train, eval, sync ---

train_eval_sync configs/gemma12b.qwen.gsm8k.env  "qwen/gsm8k (base)"  anti-distill-base
train_eval_sync configs/gemma12b.qwen.arc.env    "qwen/arc (base)"    anti-distill-base
train_eval_sync configs/gemma12b.qwen.math.env   "qwen/math (base)"   anti-distill-base

train_eval_sync configs/gemma12b.llama.gsm8k.env "llama/gsm8k (base)" anti-distill-base
train_eval_sync configs/gemma12b.llama.arc.env   "llama/arc (base)"   anti-distill-base
train_eval_sync configs/gemma12b.llama.math.env  "llama/math (base)"  anti-distill-base
