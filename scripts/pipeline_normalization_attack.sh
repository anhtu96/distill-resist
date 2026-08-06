#!/usr/bin/env bash
# Pipeline: normalize CAA-steered holdout traces and distill students on normalized traces.
# Usage: bash scripts/pipeline_normalization_attack.sh

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# ---------------------------------------------------------------------------
# Phase 1: Normalize holdout traces
# ---------------------------------------------------------------------------
echo ">>> Phase 1: Normalize holdout traces <<<"

# --- GSM8K (Mixed Style) ---
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style mixed --name-suffix _metadata 
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style mixed --name-suffix _metadata --include-question

# --- MATH ---
# normalize configs/gemma12b.qwen.math.env     --caa-style mixed --name-suffix _metadata
# normalize configs/gemma12b.qwen.math.env     --caa-style mixed --name-suffix _metadata --include-question

# --- Single Styles (GSM8K / MATH) ---
# STYLES=(scratchpad dead_end distractor jargon)
# for STYLE in "${STYLES[@]}"; do
#     normalize configs/gemma12b.qwen.gsm8k.env --caa-style "$STYLE" --name-suffix _metadata
# done
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style scratchpad --name-suffix _metadata
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style dead_end   --name-suffix _metadata
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style distractor --name-suffix _metadata
# normalize configs/gemma12b.qwen.gsm8k.env     --caa-style jargon     --name-suffix _metadata

# normalize configs/gemma12b.qwen.math.env     --caa-style scratchpad --name-suffix _metadata
# normalize configs/gemma12b.qwen.math.env     --caa-style dead_end   --name-suffix _metadata
# normalize configs/gemma12b.qwen.math.env     --caa-style distractor --name-suffix _metadata
# normalize configs/gemma12b.qwen.math.env     --caa-style jargon     --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 3: Evaluate trained students
# ---------------------------------------------------------------------------
echo ">>> Phase 3: Evaluate trained students <<<"

# --- GSM8K ---
# eval configs/gemma12b.qwen.gsm8k.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama.gsm8k.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama8b.gsm8k.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- MATH ---
# eval configs/gemma12b.qwen.math.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama.math.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- Single Styles ---
# eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# eval configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 2: Train student models on normalized traces
# ---------------------------------------------------------------------------
echo ">>> Phase 2: Train student models <<<"

# --- GSM8K ---
# train configs/gemma12b.qwen.gsm8k.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata 
# train configs/gemma12b.llama.gsm8k.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.llama8b.gsm8k.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- MATH ---
# train configs/gemma12b.qwen.math.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.llama.math.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# train_eval configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- Single Styles ---
# train_eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# train_eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# train_eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# train_eval configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# train_eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# train_eval configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata
# ---------------------------------------------------------------------------
# Phase 4: Upload Results
# ---------------------------------------------------------------------------
echo ">>> Phase 4: Upload Results <<<"

# upload_results configs/gemma12b.qwen.gsm8k.env     anhtu96/distill-resist --with-caa --caa-style mixed --normalized --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env  anhtu96/distill-resist --with-caa --caa-style mixed --normalized --name-suffix _metadata

# --- Single Styles ---
# upload_results configs/gemma12b.qwen.gsm8k.env anhtu96/distill-resist --with-caa --caa-style scratchpad --normalized --name-suffix _metadata
# upload_results configs/gemma12b.qwen.gsm8k.env anhtu96/distill-resist --with-caa --caa-style dead_end   --normalized --name-suffix _metadata
# upload_results configs/gemma12b.qwen.gsm8k.env anhtu96/distill-resist --with-caa --caa-style distractor --normalized --name-suffix _metadata
# upload_results configs/gemma12b.qwen.gsm8k.env anhtu96/distill-resist --with-caa --caa-style jargon     --normalized --name-suffix _metadata

# upload_results configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist --with-caa --caa-style scratchpad --normalized --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist --with-caa --caa-style dead_end   --normalized --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist --with-caa --caa-style distractor --normalized --name-suffix _metadata
# upload_results configs/gemma12b.llama8b.gsm8k.env anhtu96/distill-resist --with-caa --caa-style jargon     --normalized --name-suffix _metadata

# ---------------------------------------------------------------------------
# Phase 5: Remove local adapters / checkpoints
# ---------------------------------------------------------------------------
echo ">>> Phase 5: Remove local adapters / checkpoints <<<"

# --- GSM8K (Mixed Style) ---
# remove_adapter configs/gemma12b.qwen.gsm8k.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.gsm8k.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
remove_adapter configs/gemma12b.llama8b.gsm8k.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- MATH (Mixed Style) ---
# remove_adapter configs/gemma12b.qwen.math.env     --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama.math.env    --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env  --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style mixed --name-suffix _metadata

# --- Single Styles (GSM8K) ---
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
remove_adapter configs/gemma12b.llama8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# --- Single Styles (MATH) ---
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.llama8b.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.phi4mini.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style scratchpad --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style dead_end   --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style distractor --name-suffix _metadata
# remove_adapter configs/gemma12b.qwen3-8b.math.env --with-caa --normalized --caa-style jargon     --name-suffix _metadata

echo "=== Normalization Pipeline Completed Successfully ==="
