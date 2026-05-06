#!/usr/bin/env bash

# python scripts/train_caa_vectors.py \
#     --aggregation mean --contrastive mean_diff --overwrite --data-source fresh --all-layers
# echo "=== Done: CAA (mean + mean_diff) ==="

python scripts/train_caa_vectors.py \
    --backend vllm --output-dir models/steering_vectors_caa_vllm --data-source fresh --all-layers --overwrite