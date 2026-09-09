#!/bin/bash

python scripts/infer_with_caa.py \
--model gemma \
--caa-normalize \
--caa-relative \
--insertion-layers 15 25 \
--strength 0.04 0.07 \
--style distractor \
--inject-prefill 0 \
--test-index 1 \
--test-dataset gsm8k \
--temperature 0.7 \
# --max-inject-tokens 50