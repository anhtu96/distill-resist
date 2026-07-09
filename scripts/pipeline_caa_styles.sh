#!/usr/bin/env bash
# Pipeline: generate per-style CAA-steered holdout traces and distill one student per style.
# Usage: bash scripts/pipeline_caa_styles.sh
# Requires CAA vectors to already exist (run pipeline_fresh_caa.sh first).

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

STYLES=(scratchpad dead_end jargon distractor)

# --- Generate per-style CAA holdout traces (one run generates all styles) ---

# Gemma3-12B teacher
# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set test --dataset gsm8k --with-caa --caa-mode styles --caa-style scratchpad dead_end jargon distractor --batch-size 128
# echo "=== Done: CAA styles test gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set test --dataset math --with-caa --caa-mode styles --caa-style scratchpad dead_end jargon distractor --batch-size 128
# echo "=== Done: CAA styles test math ==="

# ENV_FILE=configs/gemma12b.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode styles --caa-style scratchpad dead_end jargon distractor --batch-size 128
# echo "=== Done: CAA styles holdout gsm8k ==="

# ENV_FILE=configs/gemma12b.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode styles --caa-style scratchpad dead_end jargon distractor --batch-size 128
# echo "=== Done: CAA styles holdout math ==="

# DeepSeek-R1-Distill-Qwen-7B teacher
# ENV_FILE=configs/deepseek.qwen.gsm8k.env python scripts/generate_traces.py --model teacher --set holdout --dataset gsm8k --with-caa --caa-mode styles
# echo "=== Done: deepseek CAA styles holdout gsm8k ==="

# ENV_FILE=configs/deepseek.qwen.math.env python scripts/generate_traces.py --model teacher --set holdout --dataset math --with-caa --caa-mode styles
# echo "=== Done: deepseek CAA styles holdout math ==="

# --- Train and eval one student per style ---

for STYLE in "${STYLES[@]}"; do
    # eval configs/gemma12b.qwen.gsm8k.env     --with-caa --caa-style "$STYLE"

    # train_eval configs/gemma12b.llama.gsm8k.env    --with-caa --caa-style "$STYLE"

    # train_eval configs/gemma12b.phi4mini.gsm8k.env --with-caa --caa-style "$STYLE"

    # train_eval configs/gemma12b.llama8b.gsm8k.env    --with-caa --caa-style "$STYLE"

    train_eval configs/gemma12b.qwen3-8b.gsm8k.env     --with-caa --caa-style "$STYLE"

    # remove_adapter configs/gemma12b.qwen.gsm8k.env     --with-caa --caa-style "$STYLE"

    # remove_adapter configs/gemma12b.llama.gsm8k.env    --with-caa --caa-style "$STYLE"

    # remove_adapter configs/gemma12b.phi4mini.gsm8k.env --with-caa --caa-style "$STYLE"

    remove_adapter configs/gemma12b.qwen3-8b.gsm8k.env     --with-caa --caa-style "$STYLE"

    # remove_adapter configs/gemma12b.llama8b.gsm8k.env    --with-caa --caa-style "$STYLE"
done


# for STYLE in "${STYLES[@]}"; do
#     train_eval configs/gemma12b.qwen.math.env       --with-caa --caa-style "$STYLE"

#     train_eval configs/gemma12b.llama.math.env      --with-caa --caa-style "$STYLE"

#     train_eval configs/gemma12b.phi4mini.math.env   --with-caa --caa-style "$STYLE"

#     train_eval configs/gemma12b.llama8b.math.env      --with-caa --caa-style "$STYLE"

#     train_eval configs/gemma12b.qwen3-8b.math.env       --with-caa --caa-style "$STYLE"

#     remove_adapter configs/gemma12b.qwen.math.env       --with-caa --caa-style "$STYLE"

#     remove_adapter configs/gemma12b.llama.math.env      --with-caa --caa-style "$STYLE"

#     remove_adapter configs/gemma12b.phi4mini.math.env   --with-caa --caa-style "$STYLE"

#     remove_adapter configs/gemma12b.qwen3-8b.math.env       --with-caa --caa-style "$STYLE"

#     remove_adapter configs/gemma12b.llama8b.math.env      --with-caa --caa-style "$STYLE"
# done
