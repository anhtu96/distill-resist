#!/usr/bin/env bash
# Pipeline: generate base holdout traces and distill student from them.
# Usage: bash scripts/pipeline_base.sh

# set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- Generate holdout traces (shared across both students) ---

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k
# echo "=== Done: base holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc
# echo "=== Done: base holdout arc ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math
# echo "=== Done: base holdout math ==="

# --- Train and eval ---

# train_eval configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist
# train_eval configs/gemma12b.qwen.arc.env    anhtu96/distill-resist
# train_eval configs/gemma12b.qwen.math.env   anhtu96/distill-resist

train_eval configs/gemma12b.llama.gsm8k.env anhtu96/distill-resist
train_eval configs/gemma12b.llama.arc.env   anhtu96/distill-resist
train_eval configs/gemma12b.llama.math.env  anhtu96/distill-resist

# --- Online eval ---
# download_eval configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist
# download_eval configs/gemma12b.qwen.arc.env    anhtu96/distill-resist
# download_eval configs/gemma12b.qwen.math.env   anhtu96/distill-resist

# download_eval configs/gemma12b.llama.gsm8k.env anhtu96/distill-resist
# download_eval configs/gemma12b.llama.arc.env   anhtu96/distill-resist
# download_eval configs/gemma12b.llama.math.env  anhtu96/distill-resist

# --- Upload to HuggingFace ---

# upload_adapter configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.arc.env    anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.math.env   anhtu96/distill-resist

upload_adapter configs/gemma12b.llama.gsm8k.env anhtu96/distill-resist
upload_adapter configs/gemma12b.llama.arc.env   anhtu96/distill-resist
upload_adapter configs/gemma12b.llama.math.env  anhtu96/distill-resist

# # --- Remove local adapters --- (optional)

# remove_adapter configs/gemma12b.qwen.gsm8k.env
# remove_adapter configs/gemma12b.qwen.arc.env
# remove_adapter configs/gemma12b.qwen.math.env

# remove_adapter configs/gemma12b.llama.gsm8k.env
# remove_adapter configs/gemma12b.llama.arc.env
# remove_adapter configs/gemma12b.llama.math.env
