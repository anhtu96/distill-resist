#!/usr/bin/env bash

ENV_FILE=configs/gemma12b.math.env python scripts/train.py
echo "=== Done: no styles ==="

ENV_FILE=configs/gemma12b.arc.env python scripts/train.py
echo "=== Done: no styles ==="

# python scripts/train.py --with-styles formal
# echo "=== Done: formal style ==="

# python scripts/train.py --with-styles scratchpad
# echo "=== Done: scratchpad style ==="

# python scripts/train.py --with-styles
# echo "=== Done: mixed styles ==="

# python scripts/train.py --with-styles --no-base
# echo "=== Done: mixed styles no base ==="

# python scripts/train.py --with-caa
# echo "=== Done: with CAA ==="