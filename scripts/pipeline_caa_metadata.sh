#!/usr/bin/env bash
# Pipeline: generate CAA-steered holdout traces (metadata/HMAC mode) and distill student.
# Usage: bash scripts/pipeline_caa_metadata.sh
# Requires CAA vectors to already exist (run pipeline_fresh_caa.sh first).
# In metadata mode the steering vector is derived per-request from HMAC-PRF on request metadata,
# making it harder for a distilling student to reverse-engineer the style signal.

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- Generate CAA-steered holdout traces (metadata mode) ---

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode metadata
echo "=== Done: CAA metadata holdout gsm8k ==="

ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --with-caa --caa-mode metadata
echo "=== Done: CAA metadata holdout arc ==="

ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode metadata
echo "=== Done: CAA metadata holdout math ==="

# --- Train, eval, sync ---

train_eval_sync configs/gemma12b.qwen.gsm8k.env  "qwen/gsm8k (caa metadata)"  anti-distill-caa-metadata --with-caa --name-suffix _metadata
train_eval_sync configs/gemma12b.qwen.arc.env    "qwen/arc (caa metadata)"    anti-distill-caa-metadata --with-caa --name-suffix _metadata
train_eval_sync configs/gemma12b.qwen.math.env   "qwen/math (caa metadata)"   anti-distill-caa-metadata --with-caa --name-suffix _metadata

train_eval_sync configs/gemma12b.llama.gsm8k.env "llama/gsm8k (caa metadata)" anti-distill-caa-metadata --with-caa --name-suffix _metadata
train_eval_sync configs/gemma12b.llama.arc.env   "llama/arc (caa metadata)"   anti-distill-caa-metadata --with-caa --name-suffix _metadata
train_eval_sync configs/gemma12b.llama.math.env  "llama/math (caa metadata)"  anti-distill-caa-metadata --with-caa --name-suffix _metadata
