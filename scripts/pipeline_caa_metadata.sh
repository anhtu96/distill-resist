#!/usr/bin/env bash
# Pipeline: generate CAA-steered holdout traces (metadata/HMAC mode) and distill student.
# Usage: bash scripts/pipeline_caa_metadata.sh
# Requires CAA vectors to already exist (run pipeline_fresh_caa.sh first).
# In metadata mode the steering vector is derived per-request from HMAC-PRF on request metadata,
# making it harder for a distilling student to reverse-engineer the style signal.
#
# Phase 1: generate ALL test sets
# Phase 2: generate ALL holdout sets
# Phase 3: per-dataset train_eval → upload → remove

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# ---------------------------------------------------------------------------
# Phase 1: Generate all test sets
# ---------------------------------------------------------------------------
echo ">>> Phase 1: generating all test sets <<<"

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env      python scripts/generate_traces.py --model teacher --set test --dataset gsm8k --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu.env        python scripts/generate_traces.py --model teacher --set test --dataset mmlu  --with-caa --caa-mode metadata
ENV_FILE=configs/gemma12b.qwen.mmlu_pro.env    python scripts/generate_traces.py --model teacher --set test --dataset mmlu_pro --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.math.env        python scripts/generate_traces.py --model teacher --set test --dataset math  --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.arc.env         python scripts/generate_traces.py --model teacher --set test --dataset arc   --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.csqa.env        python scripts/generate_traces.py --model teacher --set test --dataset csqa  --with-caa --caa-mode metadata

# echo "=== Done: all test sets ==="

# ---------------------------------------------------------------------------
# Phase 2: Generate all holdout sets
# ---------------------------------------------------------------------------
echo ">>> Phase 2: generating all holdout sets <<<"

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env      python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu.env        python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu  --with-caa --caa-mode metadata
ENV_FILE=configs/gemma12b.qwen.mmlu_pro.env    python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu_pro --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.arc.env         python scripts/generate_traces.py --model teacher --set holdout --dataset arc   --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.csqa.env        python scripts/generate_traces.py --model teacher --set holdout --dataset csqa  --with-caa --caa-mode metadata
# ENV_FILE=configs/gemma12b.qwen.math.env        python scripts/generate_traces.py --model teacher --set holdout --dataset math  --with-caa --caa-mode metadata

echo "=== Done: all holdout sets ==="

# ---------------------------------------------------------------------------
# Phase 3: train_eval → upload → remove (per dataset)
# ---------------------------------------------------------------------------
echo ">>> Phase 3: train, eval, upload, remove <<<"

# --- MATH ---
# echo ">>> MATH <<<"
# train_eval configs/gemma12b.qwen.math.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.math.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.math.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.math.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.math.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.math.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.math.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama8b.math.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.math.env anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.qwen3-8b.math.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.math.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.math.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env  --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --name-suffix _metadata

# --- GSM8K ---
# echo ">>> GSM8K <<<"
# train_eval configs/gemma12b.qwen.gsm8k.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.gsm8k.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.gsm8k.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.gsm8k.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama8b.gsm8k.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.gsm8k.env anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.qwen3-8b.gsm8k.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.gsm8k.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.gsm8k.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env  --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --name-suffix _metadata

# --- MMLU ---
# echo ">>> MMLU <<<"
# train_eval configs/gemma12b.qwen.mmlu.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.mmlu.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.mmlu.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.mmlu.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.mmlu.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.mmlu.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.mmlu.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.mmlu.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.mmlu.env --with-caa --name-suffix _metadata

# --- MMLU-PRO ---
echo ">>> MMLU-PRO <<<"
# train_eval configs/gemma12b.qwen.mmlu_pro.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.mmlu_pro.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.mmlu_pro.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.mmlu_pro.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.mmlu_pro.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.mmlu_pro.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.mmlu_pro.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama8b.mmlu_pro.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.qwen3-8b.mmlu_pro.env anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.mmlu_pro.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.mmlu_pro.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.mmlu_pro.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.mmlu_pro.env  --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.mmlu_pro.env --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.mmlu_pro.env --with-caa --name-suffix _metadata

# --- ARC ---
# echo ">>> ARC <<<"
# train_eval configs/gemma12b.qwen.arc.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.arc.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.arc.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.arc.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.arc.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.arc.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.arc.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama8b.arc.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.arc.env anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.qwen3-8b.arc.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.arc.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.arc.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.arc.env  --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.arc.env --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.arc.env --with-caa --name-suffix _metadata

# --- CSQA ---
# echo ">>> CSQA <<<"
# train_eval configs/gemma12b.qwen.csqa.env     --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama.csqa.env    --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.csqa.env  --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.csqa.env --with-caa --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.csqa.env --with-caa --name-suffix _metadata

# upload_adapter configs/gemma12b.qwen.csqa.env     anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama.csqa.env    anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.llama8b.csqa.env  anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.phi4mini.csqa.env anhtu96/distill-resist --with-caa --name-suffix _metadata
# upload_adapter configs/gemma12b.qwen3-8b.csqa.env anhtu96/distill-resist --with-caa --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.csqa.env     --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.csqa.env    --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.csqa.env  --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.csqa.env --with-caa --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.csqa.env --with-caa --name-suffix _metadata

# echo "=== Pipeline complete ==="
