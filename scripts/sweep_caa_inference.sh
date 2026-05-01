#!/usr/bin/env bash
# Sweep CAA inference over layer blocks and strengths.
# Output for each config is saved to logs/caa_sweep/.

CHECKPOINT="models/steering_vectors_caa/steering_vectors.pt"
STYLE="formal"
TEST_INDEX=0
MAX_TOKENS=2048
LOG_DIR="logs/caa_sweep"

mkdir -p "$LOG_DIR"

STRENGTHS=(2.0 3.0)

# Layer blocks (all layers of gemma-3-12b-it: 0-47, blocks of 4)
# LAYER_BLOCKS=(
#     "0 1 2 3"
#     "4 5 6 7"
#     "8 9 10 11"
#     "12 13 14 15"
#     "16 17 18 19"
#     "20 21 22 23"
#     "24 25 26 27"
#     "28 29 30 31"
#     "32 33 34 35"
#     "36 37 38 39"
#     "40 41 42 43"
#     "44 45 46 47"
# )

LAYER_BLOCKS=(
    "24"
    "25"
    "26"
    "27"
)

for STRENGTH in "${STRENGTHS[@]}"; do
    for BLOCK in "${LAYER_BLOCKS[@]}"; do
        TAG="layers_$(echo $BLOCK | tr ' ' '-')_str${STRENGTH}"
        LOG="$LOG_DIR/${TAG}.txt"
        echo ">>> $TAG"
        python scripts/infer_with_caa.py \
            --checkpoint "$CHECKPOINT" \
            --test-index "$TEST_INDEX" \
            --max-tokens "$MAX_TOKENS" \
            --insertion-layers $BLOCK \
            --strength "$STRENGTH" \
            > "$LOG" 2>&1
        echo "    saved to $LOG"
    done
done

echo "=== Sweep done. Results in $LOG_DIR ==="
