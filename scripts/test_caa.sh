#!/bin/bash

python scripts/generate_traces.py \
--dataset gsm8k \
--batch-size 32 \
--with-caa \
--caa-mode styles \
--caa-sample 100 \
--injection-layers 15 20 25 \
--caa-strength socratic=0.03,0.07,0.05 scratchpad=0.00,0.04,0.025 dead_end=0.00,0.07,0.03 \
--caa-normalize \
--caa-relative \
--caa-max-injection-tokens socratic=100 scratchpad=70 dead_end=100 \
--caa-backend vllm