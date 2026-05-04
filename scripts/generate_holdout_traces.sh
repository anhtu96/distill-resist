#!/usr/bin/env bash

# python scripts/generate_traces.py --model teacher --set holdout
# echo "=== Done: base holdout ==="

# python scripts/generate_traces.py --model teacher --set holdout --rewrite
# echo "=== Done: holdout with styles (rewrite) ==="

# python scripts/generate_traces.py --model teacher --set holdout --fresh-style
# echo "=== Done: holdout with styles (fresh) ==="

# bash scripts/extract_caa_vectors.sh
# echo "=== Done: extracted CAA vectors ==="

python scripts/generate_traces.py --model teacher --set holdout --with-caa --caa-mode metadata 
echo "=== Done: holdout with CAA styles ==="

# python scripts/generate_traces.py --model teacher --set holdout --with-caa --caa-mode metadata --injection-layers 24 30 35 --caa-random-per-layer
# echo "=== Done: holdout with CAA styles, per-layer weights ==="