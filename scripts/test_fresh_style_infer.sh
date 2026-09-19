#!/bin/bash


python scripts/infer_with_fresh_style.py \
    --model gemma \
    --style dead_end \
    --test-index 1 \
    --test-dataset gsm8k \
    --temperature 0.7 \
    --backend vllm \
    --use-custom-prompt anti_dead_end
