#!/bin/bash

ENV_FILE=configs/gemma12b.qwen.mmlu_pro.env python scripts/generate_traces.py \
--dataset mmlu_pro \
--batch-size 128 \
--with-caa \
--caa-mode styles \
--caa-style dead_end scratchpad jargon distractor \
--caa-sample 200