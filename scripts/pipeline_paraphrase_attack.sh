#!/usr/bin/env bash
# Pipeline: paraphrase CAA-steered holdout traces using DIPPER and distill students on paraphrased traces.
# Usage: bash scripts/pipeline_paraphrase_attack.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# ---------------------------------------------------------------------------
# Phase 1: Paraphrase holdout traces (DIPPER)
# ---------------------------------------------------------------------------
echo ">>> Phase 1: Paraphrase holdout traces <<<"

# --- GSM8K (Mixed Style) ---
# paraphrase configs/gemma12b.qwen.gsm8k.env     --caa-style mixed --batch-size 64 --line-batch-size 128 --name-suffix _metadata

# --- MATH (Mixed Style) ---
# paraphrase configs/gemma12b.qwen.math.env     --caa-style mixed --batch-size 64 --line-batch-size 128 --name-suffix _metadata

# --- Single Styles (GSM8K / MATH) ---
# paraphrase configs/gemma12b.qwen.gsm8k.env     --caa-style scratchpad --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.gsm8k.env     --caa-style dead_end   --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.gsm8k.env     --caa-style distractor --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.gsm8k.env     --caa-style jargon     --batch-size 64 --line-batch-size 128 --name-suffix _metadata

# paraphrase configs/gemma12b.qwen.math.env     --caa-style scratchpad --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.math.env     --caa-style dead_end   --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.math.env     --caa-style distractor --batch-size 64 --line-batch-size 128 --name-suffix _metadata
# paraphrase configs/gemma12b.qwen.math.env     --caa-style jargon     --batch-size 64 --line-batch-size 128 --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 2: Train & Evaluate student models (and clean up local adapters)
# ---------------------------------------------------------------------------
echo ">>> Phase 2: Train & Evaluate student models <<<"

# --- GSM8K (Mixed Style) ---
# eval configs/gemma12b.qwen.gsm8k.env     --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.llama.gsm8k.env    --with-caa --paraphrased --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.llama8b.gsm8k.env  --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# --- MATH (Mixed Style) ---
# eval configs/gemma12b.qwen.math.env     --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# eval configs/gemma12b.llama.math.env    --with-caa --paraphrased --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

eval configs/gemma12b.llama8b.math.env  --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style mixed --name-suffix _metadata

# --- Single Styles (GSM8K) ---
eval configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata
 
eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

# --- Single Styles (MATH) ---
eval configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style scratchpad --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style dead_end   --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style distractor --name-suffix _metadata

eval configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata --batch-size 2048
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --paraphrased --caa-style jargon     --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 3: Upload Results
# ---------------------------------------------------------------------------
echo ">>> Phase 3: Upload Results <<<"

# upload_results configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --with-caa --caa-style mixed --paraphrased --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env  anhtu96/distill-resist --with-caa --caa-style mixed --paraphrased --name-suffix _metadata

echo "=== Paraphrasing Pipeline Completed Successfully ==="
