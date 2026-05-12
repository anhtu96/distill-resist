#!/usr/bin/env bash

# ENV_FILE=configs/gemma.llama.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout 
# echo "=== Done: base holdout gsm8k ==="

# ENV_FILE=configs/gemma.llama.arc.env python scripts/generate_traces.py --model teacher --set holdout 
# echo "=== Done: base holdout arc ==="

# ENV_FILE=configs/gemma.llama.math.env python scripts/generate_traces.py --model teacher --set holdout 
# echo "=== Done: base holdout math ==="

ENV_FILE=configs/gemma.llama.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --fresh-style
echo "=== Done: base holdout gsm8k (fresh style) ==="

ENV_FILE=configs/gemma.llama.arc.env python scripts/generate_traces.py --model teacher --set holdout --fresh-style
echo "=== Done: base holdout arc (fresh style) ==="

ENV_FILE=configs/gemma.llama.math.env python scripts/generate_traces.py --model teacher --set holdout --fresh-style
echo "=== Done: base holdout math (fresh style) ==="