#!/usr/bin/env bash

python scripts/train_caa_vectors.py \
    --aggregation mean --contrastive mean_diff --num-samples 200 --overwrite
echo "=== Done: CAA (mean + mean_diff) ==="
