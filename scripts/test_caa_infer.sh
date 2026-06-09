#!/bin/bash

python scripts/infer_with_caa.py \
--model gemma \
--caa-normalize \
--caa-relative \
--insertion-layers 12 25 \
--strength 0.08 0.05 \
--style scratchpad \
--inject-prefill 0 \
--max-inject-tokens 9999 \
--temperature 0 \
--test-index 0 \