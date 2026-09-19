#!/bin/bash


# python scripts/infer_with_caa.py \
#     --model gemma \
#     --caa-normalize \
#     --caa-relative \
#     --insertion-layers 15 25 \
#     --strength 0.04 0.07 \
#     --style distractor \
#     --inject-prefill 0 \
#     --test-index 2 \
#     --test-dataset gsm8k \
#     --temperature 0.7 \
#     --max-inject-tokens 150 \
#     --backend vllm \
#     --use-custom-prompt anti_distractor

# python scripts/infer_with_caa.py \
#     --model gemma \
#     --caa-normalize \
#     --caa-relative \
#     --insertion-layers 13 26 \
#     --strength 0.1 0.03 \
#     --style dead_end \
#     --inject-prefill 0 \
#     --test-index 2 \
#     --test-dataset gsm8k \
#     --temperature 0.7 \
#     --max-inject-tokens 150 \
#     --backend vllm \
    # --use-custom-prompt anti_dead_end

# python scripts/infer_with_caa.py \
#     --model gemma \
#     --caa-normalize \
#     --caa-relative \
#     --insertion-layers 12 25 \
#     --strength 0.07 0.05 \
#     --style scratchpad \
#     --inject-prefill 0 \
#     --test-index 2 \
#     --test-dataset gsm8k \
#     --temperature 0.7 \
#     --max-inject-tokens 50 \
#     --backend vllm \
#     --use-custom-prompt anti_scratchpad

python scripts/infer_with_caa.py \
    --model gemma \
    --caa-normalize \
    --caa-relative \
    --insertion-layers 20 25 \
    --strength 0.00 0.00 \
    --style jargon \
    --inject-prefill 0 \
    --test-index 2 \
    --test-dataset gsm8k \
    --temperature 0.7 \
    --max-inject-tokens 150 \
    --backend vllm \
    --use-custom-prompt anti_jargon