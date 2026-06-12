#!/usr/bin/env bash
# Pipeline: generate base holdout traces and distill student from them.
# Usage: bash scripts/pipeline_base.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# --- Generate holdout traces (shared across both students) ---

# Gemma3-12B teacher (normal test traces)
# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set test --dataset gsm8k
# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set test --dataset arc
# ENV_FILE=configs/gemma12b.qwen.csqa.env python scripts/generate_traces.py --model teacher --set test --dataset csqa
# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set test --dataset math
# ENV_FILE=configs/gemma12b.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set test --dataset mmlu

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k
# echo "=== Done: base holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc
# echo "=== Done: base holdout arc ==="

# ENV_FILE=configs/gemma12b.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa
# echo "=== Done: base holdout csqa ==="

# ENV_FILE=configs/gemma12b.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu
# echo "=== Done: base holdout mmlu ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math
# echo "=== Done: base holdout math ==="

# DeepSeek-R1-Distill-Qwen-7B teacher (normal test traces)
# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set test --dataset gsm8k
# ENV_FILE=configs/deepseek.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set test --dataset mmlu
# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set test --dataset math
# ENV_FILE=configs/deepseek.qwen.arc.env python scripts/generate_traces.py --model teacher --set test --dataset arc
# ENV_FILE=configs/deepseek.qwen.csqa.env python scripts/generate_traces.py --model teacher --set test --dataset csqa

# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k
# echo "=== Done: deepseek base holdout gsm8k ==="

# ENV_FILE=configs/deepseek.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu
# echo "=== Done: deepseek base holdout mmlu ==="

# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math
# echo "=== Done: deepseek base holdout math ==="

# ENV_FILE=configs/deepseek.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc
# echo "=== Done: deepseek base holdout arc ==="

# ENV_FILE=configs/deepseek.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa
# echo "=== Done: deepseek base holdout csqa ==="

# --- Train and eval ---

# train_eval configs/gemma12b.qwen.gsm8k.env
# train_eval configs/gemma12b.qwen.arc.env
# train_eval configs/gemma12b.qwen.csqa.env
# train_eval configs/gemma12b.qwen.mmlu.env
# train_eval configs/gemma12b.qwen.math.env

# eval configs/gemma12b.qwen.arc.env
# eval configs/gemma12b.qwen.csqa.env
# eval configs/gemma12b.qwen.math.env
# train_eval configs/gemma12b.llama.gsm8k.env
# train_eval configs/gemma12b.llama.arc.env
# train_eval configs/gemma12b.llama.csqa.env
# train_eval configs/gemma12b.llama.mmlu.env
# train_eval configs/gemma12b.llama.math.env

# train_eval configs/gemma12b.llama8b.gsm8k.env
# train_eval configs/gemma12b.llama8b.arc.env
# train_eval configs/gemma12b.llama8b.csqa.env
# train_eval configs/gemma12b.llama8b.math.env

# train_eval configs/gemma12b.qwen3-8b.gsm8k.env

# train_eval configs/gemma12b.phi4mini.gsm8k.env
# train_eval configs/gemma12b.phi4mini.arc.env
# train_eval configs/gemma12b.phi4mini.csqa.env
# train_eval configs/gemma12b.phi4mini.mmlu.env
# train_eval configs/gemma12b.phi4mini.math.env

# train_eval configs/deepseek.qwen.gsm8k.env
# train_eval configs/deepseek.qwen.arc.env
# train_eval configs/deepseek.qwen.csqa.env
# train_eval configs/deepseek.qwen.mmlu.env
# train_eval configs/deepseek.qwen.math.env

# train_eval configs/deepseek.llama.gsm8k.env
# train_eval configs/deepseek.llama.arc.env
# train_eval configs/deepseek.llama.csqa.env
# train_eval configs/deepseek.llama.mmlu.env
# train_eval configs/deepseek.llama.math.env

# --- Online eval ---
# download_eval configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist
# download_eval configs/gemma12b.qwen.arc.env    anhtu96/distill-resist
# download_eval configs/gemma12b.qwen.math.env   anhtu96/distill-resist

# download_eval configs/gemma12b.llama.gsm8k.env  anhtu96/distill-resist
# download_eval configs/gemma12b.llama.arc.env    anhtu96/distill-resist
# download_eval configs/gemma12b.llama.math.env   anhtu96/distill-resist

# download_eval configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist
# download_eval configs/gemma12b.llama8b.arc.env   anhtu96/distill-resist
# download_eval configs/gemma12b.llama8b.math.env  anhtu96/distill-resist

# download_eval configs/gemma12b.qwen3-8b.gsm8k.env anhtu96/distill-resist

# download_eval configs/deepseek.qwen.gsm8k.env  anhtu96/distill-resist
# download_eval configs/deepseek.qwen.arc.env    anhtu96/distill-resist
# download_eval configs/deepseek.qwen.math.env   anhtu96/distill-resist

# download_eval configs/deepseek.llama.gsm8k.env anhtu96/distill-resist
# download_eval configs/deepseek.llama.arc.env   anhtu96/distill-resist
# download_eval configs/deepseek.llama.math.env  anhtu96/distill-resist

# --- Upload to HuggingFace ---

upload_adapter configs/gemma12b.qwen.gsm8k.env  anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.arc.env    anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.csqa.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.mmlu.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.qwen.math.env   anhtu96/distill-resist

upload_adapter configs/gemma12b.llama.gsm8k.env  anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama.arc.env    anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama.csqa.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama.mmlu.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama.math.env   anhtu96/distill-resist

upload_adapter configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama8b.arc.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama8b.csqa.env  anhtu96/distill-resist
# upload_adapter configs/gemma12b.llama8b.math.env  anhtu96/distill-resist

upload_adapter configs/gemma12b.qwen3-8b.gsm8k.env anhtu96/distill-resist

upload_adapter configs/gemma12b.phi4mini.gsm8k.env anhtu96/distill-resist
# upload_adapter configs/gemma12b.phi4mini.arc.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.phi4mini.csqa.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.phi4mini.mmlu.env   anhtu96/distill-resist
# upload_adapter configs/gemma12b.phi4mini.math.env  anhtu96/distill-resist

# upload_adapter configs/deepseek.qwen.gsm8k.env  anhtu96/distill-resist
# upload_adapter configs/deepseek.qwen.arc.env    anhtu96/distill-resist
# upload_adapter configs/deepseek.qwen.csqa.env   anhtu96/distill-resist
# upload_adapter configs/deepseek.qwen.mmlu.env   anhtu96/distill-resist
# upload_adapter configs/deepseek.qwen.math.env   anhtu96/distill-resist

# upload_adapter configs/deepseek.llama.gsm8k.env anhtu96/distill-resist
# upload_adapter configs/deepseek.llama.arc.env   anhtu96/distill-resist
# upload_adapter configs/deepseek.llama.csqa.env  anhtu96/distill-resist
# upload_adapter configs/deepseek.llama.mmlu.env  anhtu96/distill-resist
# upload_adapter configs/deepseek.llama.math.env  anhtu96/distill-resist

# --- Remove local adapters --- (optional)

remove_adapter configs/gemma12b.qwen.gsm8k.env
# remove_adapter configs/gemma12b.qwen.arc.env
# remove_adapter configs/gemma12b.qwen.csqa.env
# remove_adapter configs/gemma12b.qwen.mmlu.env
# remove_adapter configs/gemma12b.qwen.math.env

remove_adapter configs/gemma12b.llama.gsm8k.env
# remove_adapter configs/gemma12b.llama.arc.env
# remove_adapter configs/gemma12b.llama.csqa.env
# remove_adapter configs/gemma12b.llama.mmlu.env
# remove_adapter configs/gemma12b.llama.math.env

remove_adapter configs/gemma12b.llama8b.gsm8k.env
# remove_adapter configs/gemma12b.llama8b.arc.env
# remove_adapter configs/gemma12b.llama8b.csqa.env
# remove_adapter configs/gemma12b.llama8b.math.env

remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env

remove_adapter configs/gemma12b.phi4mini.gsm8k.env
# remove_adapter configs/gemma12b.phi4mini.arc.env
# remove_adapter configs/gemma12b.phi4mini.csqa.env
# remove_adapter configs/gemma12b.phi4mini.mmlu.env
# remove_adapter configs/gemma12b.phi4mini.math.env

# remove_adapter configs/deepseek.qwen.gsm8k.env
# remove_adapter configs/deepseek.qwen.arc.env
# remove_adapter configs/deepseek.qwen.csqa.env
# remove_adapter configs/deepseek.qwen.mmlu.env
# remove_adapter configs/deepseek.qwen.math.env

# remove_adapter configs/deepseek.llama.gsm8k.env
# remove_adapter configs/deepseek.llama.arc.env
# remove_adapter configs/deepseek.llama.csqa.env
# remove_adapter configs/deepseek.llama.mmlu.env
# remove_adapter configs/deepseek.llama.math.env
