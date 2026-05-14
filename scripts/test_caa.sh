#!/bin/bash

python scripts/generate_traces.py \
--dataset gsm8k \
--batch-size 32 \
--with-caa \
--caa-style dead_end \
--caa-mode styles \
--caa-sample 100 \
--injection-layers 15 20 25 \
--caa-strength socratic=0.035,0.07,0.035 scratchpad=0.02,0.025,0.02 dead_end=0.00,0.05,0.02 \
--caa-normalize \
--caa-relative \
--caa-max-injection-tokens socratic=100 scratchpad=70 dead_end=100 \
--caa-backend vllm