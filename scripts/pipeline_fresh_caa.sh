#!/usr/bin/env bash
# Pipeline: generate fresh-style traces and train CAA steering vectors from them.
# Usage: bash scripts/pipeline_fresh_caa.sh
# Requires base holdout traces to already exist (run pipeline_base.sh first).


ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --fresh-style
echo "=== Done: fresh-style holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --fresh-style
# echo "=== Done: fresh-style holdout arc ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --fresh-style
# echo "=== Done: fresh-style holdout math ==="

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/train_caa_vectors.py \
    --datasets gsm8k \
    --backend vllm \
    --data-source fresh \
    --all-layers \
    --max-length 4096 \
    --overwrite \
    --output-dir models/steering_vectors_caa_vllm
echo "=== Done: CAA vectors ==="
