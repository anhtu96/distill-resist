#!/usr/bin/env bash

python scripts/train_caa_vectors.py \
    --aggregation mean --contrastive mean_diff --num-samples 500 --overwrite --data-source fresh --all-layers
echo "=== Done: CAA (mean + mean_diff) ==="
