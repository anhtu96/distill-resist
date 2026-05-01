#!/usr/bin/env bash

python scripts/generate_traces.py --model student --use-trained-student
echo "=== Done: normal student ==="

python scripts/generate_traces.py --model student --use-trained-student --with-styles formal
echo "=== Done: formal style student ==="

python scripts/generate_traces.py --model student --use-trained-student --with-styles concise
echo "=== Done: concise style student ==="

python scripts/generate_traces.py --model student --use-trained-student --with-styles
echo "=== Done: mixed styles student ==="

python scripts/generate_traces.py --model student --use-trained-student --with-styles --no-base
echo "=== Done: mixed styles no base student ==="
