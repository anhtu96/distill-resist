#!/bin/bash

python scripts/generate_traces.py \
--dataset gsm8k \
--batch-size 32 \
--with-caa \
--caa-mode styles \
--caa-sample 100 \
--injection-layers 15 20 25 \
--caa-strength socratic=0.03,0.08,0.05 scratchpad=0.03,0.03,0.03 dead_end=0.05,0.05,0.05 \
--caa-normalize \
--caa-relative \
--caa-max-injection-tokens 50 \
--caa-backend vllm