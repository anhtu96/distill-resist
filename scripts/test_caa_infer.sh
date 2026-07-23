#!/bin/bash

python scripts/infer_with_caa.py \
--model gemma \
--caa-normalize \
--caa-relative \
--insertion-layers 13 26 \
--strength 0.09 0.04 \
--style dead_end \
--inject-prefill 0 \
--max-inject-tokens 100 \
--test-index 1 \
--test-dataset gsm8k