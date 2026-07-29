#!/usr/bin/env bash
# Pipeline: normalize CAA-steered holdout traces and distill students on normalized traces.
# Usage: bash scripts/pipeline_normalization_attack.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# ---------------------------------------------------------------------------
# Phase 1: Normalize holdout traces
# ---------------------------------------------------------------------------
echo ">>> Phase 1: Normalize holdout traces <<<"

# --- MATH ---
# normalize configs/gemma12b.qwen.math.env     --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.qwen.math.env     --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.llama.math.env    --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.llama.math.env    --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.llama8b.math.env  --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.llama8b.math.env  --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.phi4mini.math.env --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.phi4mini.math.env --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.qwen3-8b.math.env --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.qwen3-8b.math.env --caa-style mixed --name-suffix _metadata --include-question

# --- GSM8K ---
normalize configs/gemma12b.qwen.gsm8k.env     --caa-style mixed --name-suffix _metadata 
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.llama.gsm8k.env    --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.llama.gsm8k.env    --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.llama8b.gsm8k.env  --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.llama8b.gsm8k.env  --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.phi4mini.gsm8k.env --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.phi4mini.gsm8k.env --caa-style mixed --name-suffix _metadata --include-question
# normalize configs/gemma12b.qwen3-8b.gsm8k.env --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.qwen3-8b.gsm8k.env --caa-style mixed --name-suffix _metadata --include-question

# ---------------------------------------------------------------------------
# Phase 2: Train student models on normalized traces
# ---------------------------------------------------------------------------
echo ">>> Phase 2: Train student models <<<"

# --- MATH ---
# train configs/gemma12b.qwen.math.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.llama.math.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- GSM8K ---
# train configs/gemma12b.qwen.gsm8k.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata --num-samples 5
# train configs/gemma12b.llama.gsm8k.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.llama8b.gsm8k.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 3: Evaluate trained students
# ---------------------------------------------------------------------------
echo ">>> Phase 3: Evaluate trained students <<<"

# --- MATH ---
# eval configs/gemma12b.qwen.math.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama.math.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- GSM8K ---
# eval configs/gemma12b.qwen.gsm8k.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama.gsm8k.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama8b.gsm8k.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 4: Upload Results
# ---------------------------------------------------------------------------
echo ">>> Phase 4: Upload Results <<<"

# upload_results configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --with-caa --caa-style mixed --normalized --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env  anhtu96/distill-resist --with-caa --caa-style mixed --normalized --name-suffix _metadata

echo "=== Normalization Pipeline Completed Successfully ==="
