#!/bin/bash

python scripts/infer_with_caa.py \
--model gemma \
--caa-normalize \
--caa-relative \
--insertion-layers 15 25 \
--strength 0.04 0.07 \
--style distractor \
--inject-prefill 0 \
--max-inject-tokens 150 \
--test-index 10 \
--test-dataset gsm8k