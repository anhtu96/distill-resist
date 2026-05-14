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

# --- Train and eval ---

train_eval configs/gemma12b.qwen.gsm8k.env  --with-caa --name-suffix _metadata
train_eval configs/gemma12b.qwen.arc.env    --with-caa --name-suffix _metadata
train_eval configs/gemma12b.qwen.math.env   --with-caa --name-suffix _metadata

train_eval configs/gemma12b.llama.gsm8k.env --with-caa --name-suffix _metadata
train_eval configs/gemma12b.llama.arc.env   --with-caa --name-suffix _metadata
train_eval configs/gemma12b.llama.math.env  --with-caa --name-suffix _metadata

# --- Upload to HuggingFace ---

upload_adapter configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.qwen.arc.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.qwen.math.env   anhtu96/distill-resist --with-caa --name-suffix _metadata

upload_adapter configs/gemma12b.llama.gsm8k.env anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.llama.arc.env   anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.llama.math.env  anhtu96/distill-resist --with-caa --name-suffix _metadata

# --- Remove local adapters ---

remove_adapter configs/gemma12b.qwen.gsm8k.env  --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.qwen.arc.env    --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.qwen.math.env   --with-caa --name-suffix _metadata

remove_adapter configs/gemma12b.llama.gsm8k.env --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.llama.arc.env   --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.llama.math.env  --with-caa --name-suffix _metadata
