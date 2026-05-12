#!/usr/bin/env bash
# Pipeline: generate per-style CAA-steered holdout traces and distill one student per style.
# Usage: bash scripts/pipeline_caa_styles.sh
# Requires CAA vectors to already exist (run pipeline_fresh_caa.sh first).

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

STYLES=(scratchpad socratic dead_end)

# --- Generate per-style CAA holdout traces (one run generates all styles) ---

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode styles
echo "=== Done: CAA styles holdout gsm8k ==="

ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --with-caa --caa-mode styles
echo "=== Done: CAA styles holdout arc ==="

ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode styles
echo "=== Done: CAA styles holdout math ==="

# --- Train, eval, sync one student per style ---

for STYLE in "${STYLES[@]}"; do
    train_eval_sync configs/gemma12b.qwen.gsm8k.env  "qwen/gsm8k ($STYLE)"  anti-distill-caa-styles --with-caa --caa-style "$STYLE"
    train_eval_sync configs/gemma12b.qwen.arc.env    "qwen/arc ($STYLE)"    anti-distill-caa-styles --with-caa --caa-style "$STYLE"
    train_eval_sync configs/gemma12b.qwen.math.env   "qwen/math ($STYLE)"   anti-distill-caa-styles --with-caa --caa-style "$STYLE"

    train_eval_sync configs/gemma12b.llama.gsm8k.env "llama/gsm8k ($STYLE)" anti-distill-caa-styles --with-caa --caa-style "$STYLE"
    train_eval_sync configs/gemma12b.llama.arc.env   "llama/arc ($STYLE)"   anti-distill-caa-styles --with-caa --caa-style "$STYLE"
    train_eval_sync configs/gemma12b.llama.math.env  "llama/math ($STYLE)"  anti-distill-caa-styles --with-caa --caa-style "$STYLE"
done
