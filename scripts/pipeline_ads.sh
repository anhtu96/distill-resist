#!/usr/bin/env bash
# ================================================================================
# PIPELINE_ADS.SH
# Complete End-to-End ADS Pipeline: Generate ADS Traces & Distill Student Models
# ================================================================================
# Usage:
#   bash scripts/pipeline_ads.sh [--dataset DATASET] [--tau TAU] [--lam LAM] [--eps EPS]
#                                [--half first|second|all] [--target train|test|both]
#                                [--student-only] [--no-resume] [--filter-loops]
#
# Examples:
#   bash scripts/pipeline_ads.sh
#   bash scripts/pipeline_ads.sh --dataset gsm8k --lam 0.05 --half first
#   bash scripts/pipeline_ads.sh --target test
#   bash scripts/pipeline_ads.sh --student-only

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

# Defaults
dataset="math"
tau="0.7"
lam="0.017"
eps="0.01"
student_only="false"
resume="true"
half="all"
target="train"
no_filter_loops="true"
extra_flags=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)              dataset="$2";         shift 2 ;;
        --tau)                  tau="$2";             shift 2 ;;
        --lam)                  lam="$2";             shift 2 ;;
        --eps)                  eps="$2";             shift 2 ;;
        --half)                 half="$2";            shift 2 ;;
        --target)               target="$2";          shift 2 ;;
        --first)                half="first";         shift ;;
        --second)               half="second";        shift ;;
        --test-only|--test)     target="test";        shift ;;
        --train-only)           target="train";       shift ;;
        --student-only)         student_only="true";  shift ;;
        --no-resume)            resume="false";       shift ;;
        --no-filter-loops)      no_filter_loops="true"; shift ;;
        --filter-loops)         no_filter_loops="false"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ "${no_filter_loops}" == "true" ]; then
    extra_flags+=("--no-filter-loops")
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "================================================================================"
echo "End-to-End Antidistillation Sampling (ADS) Pipeline"
echo "Dataset: ${dataset} | Tau: ${tau} | Lam: ${lam} | Eps: ${eps} | Half: ${half} | Target: ${target} | Student Only: ${student_only} | Resume: ${resume} | Extra Flags: ${extra_flags[*]}"
echo "================================================================================"

if [ "${student_only}" != "true" ]; then
    echo ">>> Step 1: Generating ADS Traces (half: ${half}, target: ${target}) <<<"
    cd "${ROOT_DIR}/other_works/antidistillation-sampling"
    bash pipeline_generate_ads.sh "${dataset}" "${tau}" "${lam}" "${eps}" "${resume}" "${half}" "${target}"
    cd "${ROOT_DIR}"
else
    echo ">>> Skipping Step 1: Using existing ADS traces <<<"
fi

if [ "${target}" == "test" ]; then
    echo "================================================================================"
    echo "ADS Test Trace Generation Complete!"
    echo "================================================================================"
    exit 0
fi

# --- Step 2: Distill & Evaluate Student Model ---
echo ">>> Step 2: Distilling Student Model on ADS Traces <<<"
if [ "${dataset}" == "gsm8k" ]; then
    train_eval configs/gemma12b.phi4mini.gsm8k.env --with-ads --tau "${tau}" --lam "${lam}" --eps "${eps}" "${extra_flags[@]}"
    train_eval configs/gemma12b.qwen3-8b.gsm8k.env --with-ads --tau "${tau}" --lam "${lam}" --eps "${eps}" "${extra_flags[@]}"
elif [ "${dataset}" == "math" ]; then
    train_eval configs/gemma12b.phi4mini.math.env --with-ads --tau "${tau}" --lam "${lam}" --eps "${eps}" "${extra_flags[@]}"
    train_eval configs/gemma12b.qwen3-8b.math.env --with-ads --tau "${tau}" --lam "${lam}" --eps "${eps}" "${extra_flags[@]}"
else
    echo "Distilling on custom config for ${dataset}..."
    eval "configs/gemma12b.qwen.${dataset}.env" --with-ads --tau "${tau}" --lam "${lam}" --eps "${eps}" "${extra_flags[@]}"
fi

echo "================================================================================"
echo "ADS End-to-End Pipeline Complete!"
echo "================================================================================"
