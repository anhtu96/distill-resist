#!/usr/bin/env bash
# ================================================================================
# PIPELINE_DOGE.SH
# End-to-End DOGe Pipeline: Train DOGe, Generate Traces & Distill Student Model
# ================================================================================
# Usage:
#   bash scripts/pipeline_doge.sh [dataset] [doge_coef] [doge_temp] [student_only] [use_half_dataset] [epochs] [unfreeze_last_n_layers]
# Example:
#   bash scripts/pipeline_doge.sh gsm8k 3e-5 0.7 true true 1 1

set -e
source "$(dirname "$0")/_pipeline_helpers.sh"

dataset="${1:-gsm8k}"
doge_coef="${2:-3e-5}"
doge_temp="${3:-2}"
student_only="${4:-false}"
use_half_dataset="${5:-false}"
epochs="${6:-1}"
unfreeze_last_n_layers="${7:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "================================================================================"
echo "DOGe Pipeline"
echo "Dataset: ${dataset} | Coef: ${doge_coef} | Temp: ${doge_temp} | Student Only: ${student_only} | Half Dataset: ${use_half_dataset} | Epochs: ${epochs} | Unfreeze Last N Layers: ${unfreeze_last_n_layers}"
echo "================================================================================"

# if [ "${student_only}" != "true" ] && [ "${student_only}" != "--student-only" ]; then
#     # --- Step 1: Train DOGe & Generate Protected Traces ---
#     echo ">>> Step 1: Training DOGe & Generating Traces <<<"
#     cd "${ROOT_DIR}/other_works/DOGe"
#     bash pipeline_generate_doge.sh "${dataset}" "${doge_coef}" "${doge_temp}" "${use_half_dataset}" "${unfreeze_last_n_layers}" "${epochs}"
#     cd "${ROOT_DIR}"
# else
#     echo ">>> Skipping Step 1: Using existing DOGe traces in data/teacher_holdout/doge/ <<<"
# fi

# --- Step 2: Distill & Evaluate Student Model ---
echo ">>> Step 2: Distilling Student Model on DOGe Traces <<<"
if [ "${dataset}" == "gsm8k" ]; then
    eval configs/gemma12b.qwen3-8b.gsm8k.env --with-doge --doge-coef "${doge_coef}" --doge-temp "${doge_temp}" --batch-size 2048
elif [ "${dataset}" == "math" ]; then
    eval configs/gemma12b.qwen.math.env --with-doge --doge-coef "${doge_coef}" --doge-temp "${doge_temp}"
else
    eval "configs/gemma12b.qwen.${dataset}.env" --with-doge --doge-coef "${doge_coef}" --doge-temp "${doge_temp}"
fi

echo "================================================================================"
echo "DOGe Pipeline Complete!"
echo "================================================================================"


