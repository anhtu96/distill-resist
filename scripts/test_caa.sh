#!/bin/bash

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py \
--dataset gsm8k \
--batch-size 1 \
--with-caa \
--caa-mode metadata \
--caa-sample 10 \