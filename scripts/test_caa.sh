#!/bin/bash

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py \
--dataset gsm8k \
--batch-size 100 \
--with-caa \
--caa-mode styles \
--caa-style distractor \
--caa-sample 100 \