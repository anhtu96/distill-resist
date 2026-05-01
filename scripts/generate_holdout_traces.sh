#!/usr/bin/env bash

# python scripts/generate_traces.py --model teacher --set holdout
# echo "=== Done: base holdout ==="

# python scripts/generate_traces.py --model teacher --set holdout --rewrite
# echo "=== Done: holdout with styles (rewrite) ==="

python scripts/generate_traces.py --model teacher --set holdout --with-caa
echo "=== Done: holdout with CAA styles ==="
