#!/usr/bin/env bash
# Download CAA steering vectors from HuggingFace.
# Usage: bash scripts/download_steering_vectors.sh [repo_id]
# Default repo: ${HF_USERNAME}/distill-resist

set -euo pipefail

REPO="${1:-${HF_USERNAME}/distill-resist}"
OUT_DIR="models/steering_vectors_caa_vllm"

mkdir -p "$OUT_DIR"

FILES=(
    "steering_vectors_gemma-3-12b-it.pt"
    "steering_vectors_DeepSeek-R1-Distill-Qwen-7B.pt"
)

for f in "${FILES[@]}"; do
    echo "=== Downloading $f from $REPO ==="
    hf download "$REPO" "$f" --local-dir "$OUT_DIR" && echo "    -> $OUT_DIR/$f" \
        || echo "    WARNING: $f not found in $REPO, skipping."
done

echo "=== Done ==="
