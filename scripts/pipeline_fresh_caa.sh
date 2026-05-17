#!/usr/bin/env bash
# Pipeline: generate fresh-style traces and train CAA steering vectors from them.
# Usage: bash scripts/pipeline_fresh_caa.sh
# Requires base holdout traces to already exist (run pipeline_base.sh first).


# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --fresh-style distractor 
# echo "=== Done: fresh-style holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --fresh-style distractor
# echo "=== Done: fresh-style holdout arc ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --fresh-style distractor
# echo "=== Done: fresh-style holdout math ==="

# ENV_FILE=configs/gemma12b.qwen.csqa.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa --fresh-style distractor
# echo "=== Done: fresh-style holdout csqa ==="

# ENV_FILE=configs/gemma12b.qwen.mmlu.env python scripts/generate_traces.py --model teacher --set holdout --dataset mmlu --fresh-style distractor
# echo "=== Done: fresh-style holdout mmlu ==="

ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/train_caa_vectors.py \
    --datasets gsm8k arc csqa\
    --backend vllm \
    --data-source fresh \
    --all-layers \
    --max-length 4096 \
    --overwrite \
    --output-dir models/steering_vectors_caa_vllm \
    --no-compliance-filter
echo "=== Done: CAA vectors ==="
