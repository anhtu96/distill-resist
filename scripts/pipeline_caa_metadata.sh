#!/usr/bin/env bash
# Pipeline: generate CAA-steered holdout traces (metadata/HMAC mode) and distill student.
# Usage: bash scripts/pipeline_caa_metadata.sh
# Requires CAA vectors to already exist (run pipeline_fresh_caa.sh first).
# In metadata mode the steering vector is derived per-request from HMAC-PRF on request metadata,
# making it harder for a distilling student to reverse-engineer the style signal.

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- GSM8K ---
# echo ">>> Starting GSM8K training pipeline (metadata mode) <<<"

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set test --dataset gsm8k --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode metadata
# echo "=== Done: CAA metadata holdout gsm8k ==="

# train_eval configs/gemma12b.qwen.gsm8k.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.gsm8k.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.gsm8k.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.gsm8k.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.gsm8k.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.gsm8k.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --name-suffix _metadata

# DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set test --dataset gsm8k --with-caa --caa-mode metadata
# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode metadata
# echo "=== Done: deepseek CAA metadata holdout gsm8k ==="

# train_eval configs/deepseek.qwen.gsm8k.env  --with-caa --name-suffix _metadata
# train_eval configs/deepseek.llama.gsm8k.env --with-caa --name-suffix _metadata

# upload_adapter configs/deepseek.qwen.gsm8k.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/deepseek.llama.gsm8k.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/deepseek.qwen.gsm8k.env  --with-caa --name-suffix _metadata
# remove_adapter configs/deepseek.llama.gsm8k.env --with-caa --name-suffix _metadata

# --- MMLU ---
# echo ">>> Starting MMLU training pipeline (metadata mode) <<<"

# # ENV_FILE=configs/gemma12b.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set test --dataset mmlu --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu --with-caa --caa-mode metadata
# echo "=== Done: CAA metadata holdout mmlu ==="

# train_eval configs/gemma12b.qwen.mmlu.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.mmlu.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.mmlu.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.mmlu.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.mmlu.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.mmlu.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.mmlu.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.mmlu.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.mmlu.env --with-caa --name-suffix _metadata

# DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set test --dataset mmlu --with-caa --caa-mode metadata
# ENV_FILE=configs/deepseek.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu --with-caa --caa-mode metadata
# echo "=== Done: deepseek CAA metadata holdout mmlu ==="

# train_eval configs/deepseek.qwen.mmlu.env  --with-caa --name-suffix _metadata
# train_eval configs/deepseek.llama.mmlu.env --with-caa --name-suffix _metadata

# upload_adapter configs/deepseek.qwen.mmlu.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/deepseek.llama.mmlu.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/deepseek.qwen.mmlu.env  --with-caa --name-suffix _metadata
# remove_adapter configs/deepseek.llama.mmlu.env --with-caa --name-suffix _metadata

# --- MATH ---
echo ">>> Starting MATH training pipeline (metadata mode) <<<"

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set test --dataset math --with-caa --caa-mode metadata
ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode metadata
echo "=== Done: CAA metadata holdout math ==="

train_eval configs/gemma12b.qwen.math.env     --with-caa --name-suffix _metadata
train_eval configs/gemma12b.llama.math.env    --with-caa --name-suffix _metadata
train_eval configs/gemma12b.phi4mini.math.env --with-caa --name-suffix _metadata

upload_adapter configs/gemma12b.qwen.math.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.llama.math.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
upload_adapter configs/gemma12b.phi4mini.math.env anhtu96/distill-resist --with-caa --name-suffix _metadata

remove_adapter configs/gemma12b.qwen.math.env     --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.llama.math.env    --with-caa --name-suffix _metadata
remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --name-suffix _metadata

# DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set test --dataset math --with-caa --caa-mode metadata
# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode metadata
# echo "=== Done: deepseek CAA metadata holdout math ==="

# train_eval configs/deepseek.qwen.math.env  --with-caa --name-suffix _metadata
# train_eval configs/deepseek.llama.math.env --with-caa --name-suffix _metadata

# upload_adapter configs/deepseek.qwen.math.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/deepseek.llama.math.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/deepseek.qwen.math.env  --with-caa --name-suffix _metadata
# remove_adapter configs/deepseek.llama.math.env --with-caa --name-suffix _metadata

# --- ARC ---
# echo ">>> Starting ARC training pipeline (metadata mode) <<<"

# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set test --dataset arc --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --with-caa --caa-mode metadata
# echo "=== Done: CAA metadata holdout arc ==="

# train_eval configs/gemma12b.qwen.arc.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.arc.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.arc.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.arc.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.arc.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.arc.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.arc.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.arc.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.arc.env --with-caa --name-suffix _metadata

# # DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.arc.env python scripts/generate_traces.py --model teacher --set test --dataset arc --with-caa --caa-mode metadata
# ENV_FILE=configs/deepseek.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --with-caa --caa-mode metadata
# echo "=== Done: deepseek CAA metadata holdout arc ==="

# train_eval configs/deepseek.qwen.arc.env  --with-caa --name-suffix _metadata
# train_eval configs/deepseek.llama.arc.env --with-caa --name-suffix _metadata

# upload_adapter configs/deepseek.qwen.arc.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/deepseek.llama.arc.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/deepseek.qwen.arc.env  --with-caa --name-suffix _metadata
# remove_adapter configs/deepseek.llama.arc.env --with-caa --name-suffix _metadata

# --- CSQA ---
# echo ">>> Starting CSQA training pipeline (metadata mode) <<<"

# ENV_FILE=configs/gemma12b.qwen.csqa.env python scripts/generate_traces.py --model teacher --set test --dataset csqa --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa --with-caa --caa-mode metadata
# echo "=== Done: CAA metadata holdout csqa ==="

# train_eval configs/gemma12b.qwen.csqa.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.csqa.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.csqa.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.csqa.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.csqa.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.csqa.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.csqa.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.csqa.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.csqa.env --with-caa --name-suffix _metadata

# # DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.csqa.env python scripts/generate_traces.py --model teacher --set test --dataset csqa --with-caa --caa-mode metadata
# ENV_FILE=configs/deepseek.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa --with-caa --caa-mode metadata
# echo "=== Done: deepseek CAA metadata holdout csqa ==="

# train_eval configs/deepseek.qwen.csqa.env  --with-caa --name-suffix _metadata
# train_eval configs/deepseek.llama.csqa.env --with-caa --name-suffix _metadata

# upload_adapter configs/deepseek.qwen.csqa.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/deepseek.llama.csqa.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/deepseek.qwen.csqa.env  --with-caa --name-suffix _metadata
# remove_adapter configs/deepseek.llama.csqa.env --with-caa --name-suffix _metadata
