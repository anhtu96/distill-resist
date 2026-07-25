#!/usr/bin/env bash
# Response Quality Evaluation: Perplexity (PPL) calculation + LLM Judge scoring with CSV reporting
set -euo pipefail

# ---------------------------------------------------------------------------
# Default Input and Benign File Paths (edit these or pass via command line)
# ---------------------------------------------------------------------------
INPUT_JSONL="${INPUT_JSONL:-data/test/caa/teacher_gemma-3-12b-it_gsm8k_base_caa_mixed_sscratchpad_0.07-0.05_dead_end_0.09-0.03_distractor_0.03-0.07_jargon_0.03-0.08_tscratchpad_50_dead_end_150_distractor_100_jargon_150.jsonl}"
BENIGN_JSONL="${BENIGN_JSONL:-data/test/base/teacher_gemma-3-12b-it_gsm8k_base.jsonl}"

# Default evaluation flags
DEFAULT_FLAGS=(
    "$INPUT_JSONL"
    "--mixed"
    "--compare-benign"
    "--benign-jsonl" "$BENIGN_JSONL"
    "-n" "100"
)

if [[ $# -eq 0 ]]; then
    exec python scripts/eval_response_quality.py "${DEFAULT_FLAGS[@]}"
else
    exec python scripts/eval_response_quality.py "$@"
fi
