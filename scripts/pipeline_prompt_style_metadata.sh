#!/usr/bin/env bash
# Pipeline: generate prompt-style-steered holdout traces (metadata/HMAC mode) and distill student.
# Usage: bash scripts/pipeline_prompt_style_metadata.sh
# In metadata mode the style assignment is derived per-batch from HMAC-PRF on request metadata,
# and the style prompt is prepended to the system prompt during generation, but omitted from the saved JSONL.
#
# Phase 1: generate ALL test sets
# Phase 2: generate ALL holdout sets
# Phase 3: per-dataset train_eval → upload → remove

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

SEED="${SEED:-42}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        *) shift ;;
    esac
done
export SEED
echo "Used SEED: ${SEED}"

# ---------------------------------------------------------------------------
# Phase 1: Generate all test sets
# ---------------------------------------------------------------------------
echo ">>> Phase 1: generating all test sets <<<"

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env      python scripts/generate_traces.py --model teacher --set test --dataset gsm8k --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu.env        python scripts/generate_traces.py --model teacher --set test --dataset mmlu  --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu_pro.env    python scripts/generate_traces.py --model teacher --set test --dataset mmlu_pro --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.math.env        python scripts/generate_traces.py --model teacher --set test --dataset math  --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.arc.env         python scripts/generate_traces.py --model teacher --set test --dataset arc   --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.csqa.env        python scripts/generate_traces.py --model teacher --set test --dataset csqa  --prompt-style --style-mode metadata

# upload_teacher_test configs/gemma12b.qwen.gsm8k.env    anhtu96/distill-resist --prompt-style
# upload_teacher_test configs/gemma12b.qwen.math.env     anhtu96/distill-resist --prompt-style
# upload_teacher_test configs/gemma12b.qwen.arc.env      anhtu96/distill-resist --prompt-style
# upload_teacher_test configs/gemma12b.qwen.csqa.env     anhtu96/distill-resist --prompt-style
# upload_teacher_test configs/gemma12b.qwen.mmlu.env     anhtu96/distill-resist --prompt-style
# upload_teacher_test configs/gemma12b.qwen.mmlu_pro.env anhtu96/distill-resist --prompt-style

echo "=== Done: all test sets ==="

# ---------------------------------------------------------------------------
# Phase 2: Generate all holdout sets
# ---------------------------------------------------------------------------
echo ">>> Phase 2: generating all holdout sets <<<"

ENV_FILE=configs/gemma12b.qwen.gsm8k.env      python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu.env        python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu  --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.mmlu_pro.env    python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu_pro --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.arc.env         python scripts/generate_traces.py --model teacher --set holdout --dataset arc   --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.csqa.env        python scripts/generate_traces.py --model teacher --set holdout --dataset csqa  --prompt-style --style-mode metadata
# ENV_FILE=configs/gemma12b.qwen.math.env        python scripts/generate_traces.py --model teacher --set holdout --dataset math  --prompt-style --style-mode metadata

# upload_teacher_holdout configs/gemma12b.qwen.gsm8k.env    anhtu96/distill-resist --prompt-style
# upload_teacher_holdout configs/gemma12b.qwen.math.env     anhtu96/distill-resist --prompt-style
# upload_teacher_holdout configs/gemma12b.qwen.arc.env      anhtu96/distill-resist --prompt-style
# upload_teacher_holdout configs/gemma12b.qwen.csqa.env     anhtu96/distill-resist --prompt-style
# upload_teacher_holdout configs/gemma12b.qwen.mmlu.env     anhtu96/distill-resist --prompt-style
# upload_teacher_holdout configs/gemma12b.qwen.mmlu_pro.env anhtu96/distill-resist --prompt-style

echo "=== Done: all holdout sets ==="

# ---------------------------------------------------------------------------
# Phase 3: train_eval → upload → remove (per dataset)
# ---------------------------------------------------------------------------
echo ">>> Phase 3: train, eval, upload, remove <<<"

# --- MATH ---
# echo ">>> MATH <<<"
# eval configs/gemma12b.qwen.math.env     --prompt-style --name-suffix _metadata
# eval configs/gemma12b.llama.math.env    --prompt-style --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env  --prompt-style --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --prompt-style --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.math.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.math.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.math.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.math.env  anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.math.env anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.qwen3-8b.math.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.math.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.math.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env  --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --prompt-style --name-suffix _metadata

# --- GSM8K ---
echo ">>> GSM8K <<<"
# train_eval configs/gemma12b.qwen.gsm8k.env     --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.qwen.gsm8k.env     --prompt-style --name-suffix _metadata --no-filter-loops
# eval configs/gemma12b.llama.gsm8k.env          --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.gsm8k.env  --prompt-style --name-suffix _metadata
train_eval configs/gemma12b.phi4mini.gsm8k.env --prompt-style --name-suffix _metadata
train_eval configs/gemma12b.qwen3-8b.gsm8k.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.gsm8k.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env  anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.gsm8k.env anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.qwen3-8b.gsm8k.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.gsm8k.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env     --prompt-style --name-suffix _metadata --no-filter-loops
# remove_adapter configs/gemma12b.llama.gsm8k.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env  --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --prompt-style --name-suffix _metadata

# --- MMLU ---
# echo ">>> MMLU <<<"
# train_eval configs/gemma12b.qwen.mmlu.env     --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama.mmlu.env    --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.mmlu.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.mmlu.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.mmlu.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.mmlu.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.mmlu.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.mmlu.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.mmlu.env --prompt-style --name-suffix _metadata

# --- MMLU-PRO ---
# echo ">>> MMLU-PRO <<<"
# train_eval configs/gemma12b.qwen.mmlu_pro.env     --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama.mmlu_pro.env    --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.mmlu_pro.env  --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.mmlu_pro.env --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.mmlu_pro.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.mmlu_pro.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.mmlu_pro.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.mmlu_pro.env  anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.qwen3-8b.mmlu_pro.env anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.mmlu_pro.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.mmlu_pro.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.mmlu_pro.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.mmlu_pro.env  --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.mmlu_pro.env --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.mmlu_pro.env --prompt-style --name-suffix _metadata

# --- ARC ---
# echo ">>> ARC <<<"
# train_eval configs/gemma12b.qwen.arc.env     --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama.arc.env    --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.arc.env  --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.arc.env --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.arc.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.arc.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.arc.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.arc.env  anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.arc.env anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.qwen3-8b.arc.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.arc.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.arc.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.arc.env  --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.arc.env --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.arc.env --prompt-style --name-suffix _metadata

# --- CSQA ---
# echo ">>> CSQA <<<"
# train_eval configs/gemma12b.qwen.csqa.env     --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama.csqa.env    --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.csqa.env  --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.phi4mini.csqa.env --prompt-style --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.csqa.env --prompt-style --name-suffix _metadata

# upload_results configs/gemma12b.qwen.csqa.env     anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama.csqa.env    anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.csqa.env  anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.phi4mini.csqa.env anhtu96/distill-resist --prompt-style --name-suffix _metadata
# upload_results configs/gemma12b.qwen3-8b.csqa.env anhtu96/distill-resist --prompt-style --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen.csqa.env     --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.csqa.env    --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.csqa.env  --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.csqa.env --prompt-style --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.csqa.env --prompt-style --name-suffix _metadata

# echo "=== Pipeline complete ==="
