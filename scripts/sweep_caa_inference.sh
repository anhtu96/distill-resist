#!/usr/bin/env bash
# Sweep CAA inference over layer blocks and strengths.
# Output for each config is saved to logs/caa_sweep/.

CHECKPOINT="models/steering_vectors_caa/steering_vectors.pt"
STYLE="formal"
TEST_INDEX=0
MAX_TOKENS=512
LOG_DIR="logs/caa_sweep"

mkdir -p "$LOG_DIR"

STRENGTHS=(1.0 3.0 5.0 10.0)

# Layer blocks (latter half of gemma-3-12b-it: layers 24-47)
LAYER_BLOCKS=(
    "24 25 26 27"
    "28 29 30 31"
    "32 33 34 35"
    "36 37 38 39"
    "40 41 42 43"
    "44 45 46 47"
)

for STRENGTH in "${STRENGTHS[@]}"; do
    for BLOCK in "${LAYER_BLOCKS[@]}"; do
        TAG="layers_$(echo $BLOCK | tr ' ' '-')_str${STRENGTH}"
        LOG="$LOG_DIR/${TAG}.txt"
        echo ">>> $TAG"
        python scripts/infer_with_caa.py \
            --checkpoint "$CHECKPOINT" \
            --style "$STYLE" \
            --test-index "$TEST_INDEX" \
            --max-tokens "$MAX_TOKENS" \
            --insertion-layers $BLOCK \
            --strength "$STRENGTH" \
            > "$LOG" 2>&1
        echo "    saved to $LOG"
    done
done

echo "=== Sweep done. Results in $LOG_DIR ==="
