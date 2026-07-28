#!/bin/bash

python scripts/infer_with_caa.py \
--model gemma \
--caa-normalize \
--caa-relative \
--insertion-layers 13 26 \
--strength 0.1 0.03 \
--style dead_end \
--inject-prefill 0 \
--max-inject-tokens 150 \
--test-index 0 \
--test-dataset gsm8k