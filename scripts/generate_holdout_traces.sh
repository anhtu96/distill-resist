#!/usr/bin/env bash

# ENV_FILE=configs/gemma12b.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k 
# echo "=== Done: base holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc
# echo "=== Done: base holdout arc ==="

# ENV_FILE=configs/gemma12b.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa
# echo "=== Done: base holdout csqa ==="

# ENV_FILE=configs/gemma12b.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math
# echo "=== Done: base holdout math ==="

# ENV_FILE=configs/gemma12b.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --fresh-style
# echo "=== Done: base holdout gsm8k (fresh) ==="

# ENV_FILE=configs/gemma12b.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset arc --fresh-style
# echo "=== Done: base holdout arc (fresh) ==="
# ENV_FILE=configs/gemma12b.arc.env python scripts/generate_traces.py --model teacher --set holdout --dataset csqa --fresh-style
# echo "=== Done: base holdout csqa (fresh) ==="

# ENV_FILE=configs/gemma12b.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --fresh-style
# echo "=== Done: base holdout math (fresh) ==="


# python scripts/generate_traces.py --model teacher --set holdout --rewrite
# echo "=== Done: holdout with styles (rewrite) ==="

# python scripts/generate_traces.py --model teacher --set holdout --fresh-style
# echo "=== Done: holdout with styles (fresh) ==="

# bash scripts/extract_caa_vectors.sh
# echo "=== Done: extracted CAA vectors ==="

# python scripts/generate_traces.py --model teacher --set holdout --with-caa --caa-mode metadata 
# echo "=== Done: holdout with CAA styles ==="

# python scripts/generate_traces.py --model teacher --set holdout --with-caa --caa-mode metadata --injection-layers 24 30 35 --caa-random-per-layer
# echo "=== Done: holdout with CAA styles, per-layer weights ==="