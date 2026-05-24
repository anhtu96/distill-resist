#!/bin/bash

ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py \
--dataset math \
--batch-size 128 \
--with-caa \
--caa-mode styles \
--caa-style  \
--caa-sample 256 \