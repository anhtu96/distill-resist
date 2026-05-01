#!/usr/bin/env bash

python scripts/train.py
echo "=== Done: no styles ==="

python scripts/train.py --with-styles formal
echo "=== Done: formal style ==="

python scripts/train.py --with-styles concise
echo "=== Done: concise style ==="

python scripts/train.py --with-styles
echo "=== Done: mixed styles ==="

python scripts/train.py --with-styles --no-base
echo "=== Done: mixed styles no base ==="
