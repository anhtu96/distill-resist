#!/usr/bin/env bash
# Response Quality Evaluation: Perplexity (PPL) calculation + Gemini 2.5 Flash LLM Judge scoring
set -euo pipefail

PPL_MODEL="meta-llama/Llama-3.1-8B"
JUDGE_MODEL="gemini-2.5-flash"
BACKEND="openai"
API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
OUTPUT_DIR="results/quality_eval"
NUM_SAMPLES="100"
# INPUT_JSONL="data/teacher_holdout/caa/teacher_gemma-3-12b-it_gsm8k_base_caa_dead_end_sscratchpad_0.07-0.05_dead_end_0.1-0.03_distractor_0.02-0.07_jargon_0.03-0.08_tscratchpad_70_dead_end_100_distractor_100_jargon_150.jsonl"
INPUT_JSONL="data/teacher_holdout/caa/teacher_gemma-3-12b-it_gsm8k_base_caa_distractor_sscratchpad_0.07-0.05_dead_end_0.1-0.03_distractor_0.03-0.07_jargon_0.03-0.08_tscratchpad_50_dead_end_150_distractor_100_jargon_150.jsonl"


while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --ppl-model)      PPL_MODEL="$2"; shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2"; shift 2 ;;
        --backend)        BACKEND="$2"; shift 2 ;;
        --api-base)       API_BASE="$2"; shift 2 ;;
        --api-key)        API_KEY="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)        echo "Usage: $0 [-n N] [--ppl-model M] [--judge-model M] <jsonl_path>"; exit 0 ;;
        *)                INPUT_JSONL="$1"; shift ;;
    esac
done

if [[ -z "$INPUT_JSONL" || ! -f "$INPUT_JSONL" ]]; then
    echo "Error: Valid JSONL input file required." >&2 && exit 1
fi

mkdir -p "$OUTPUT_DIR"
SAMPLE_FLAG=${NUM_SAMPLES:+-n $NUM_SAMPLES}
JUDGE_OUT="${OUTPUT_DIR}/judge_$(basename "$INPUT_JSONL" .jsonl).json"

echo "=== 1/2 Computing Perplexity ($PPL_MODEL) ==="
python scripts/compute_perplexity.py "$INPUT_JSONL" $SAMPLE_FLAG --model-id "$PPL_MODEL"

echo -e "\n=== 2/2 Running LLM Judge ($JUDGE_MODEL via $BACKEND) ==="
python scripts/eval_llm_judge.py "$INPUT_JSONL" $SAMPLE_FLAG \
    --judge-model "$JUDGE_MODEL" \
    --output-path "$JUDGE_OUT" \
    ${BACKEND:+--backend "$BACKEND"} \
    ${API_BASE:+--api-base "$API_BASE"} \
    ${API_KEY:+--api-key "$API_KEY"}

echo -e "\nQuality evaluation complete. Report saved to: $JUDGE_OUT"
