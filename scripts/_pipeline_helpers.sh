#!/usr/bin/env bash
# Shared helpers sourced by pipeline_*.sh scripts.

HF_USER="anhtu96"

# Print the merged student model dir for a given env file.
# Usage: student_merged_dir <env_file> [--with-caa] [--caa-style <style>] [--name-suffix <suffix>]
student_merged_dir() {
    local env_file="$1"
    local with_caa=""
    local caa_style=""
    local name_suffix=""
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --with-caa)    with_caa="1" ;;
            --caa-style)   caa_style="$2"; shift ;;
            --name-suffix) name_suffix="$2"; shift ;;
        esac
        shift
    done

    ENV_FILE="$env_file" python - "$with_caa" "$caa_style" "$name_suffix" << 'PYEOF'
import sys, os
sys.path.insert(0, '.')
from src.config import Settings
from pathlib import Path

with_caa    = sys.argv[1] == "1"
caa_style   = sys.argv[2] if len(sys.argv) > 2 else ""
name_suffix = sys.argv[3] if len(sys.argv) > 3 else ""

s = Settings()
model_name   = Path(s.student_model_id).name
teacher_name = Path(s.teacher_model).name
base = Path(s.student_output_dir)
if model_name not in base.name:
    out = base.parent / f"{base.name}_{model_name}_base_tch_{teacher_name}"
else:
    out = base

if with_caa:
    strength = s.caa_strength
    s_str = "-".join(f"{v:.3g}" for v in strength) if isinstance(strength, (list, tuple)) else f"{strength:.3g}"
    max_inj = s.caa_max_injection_tokens
    t_str = "tInf" if max_inj is None else ("t" + "-".join(str(x) for x in max_inj) if isinstance(max_inj, list) else f"t{max_inj}")
    if caa_style:
        out = Path(str(out) + f"_caa_{caa_style}_s{s_str}_{t_str}")
    else:
        out = Path(str(out) + f"_caa_s{s_str}_{t_str}")

if name_suffix:
    out = Path(str(out) + name_suffix)

print(str(out) + "_merged")
PYEOF
}

# Push a merged model dir to a HuggingFace repo as a subdirectory.
# Usage: hf_push <local_merged_dir> <hf_repo> <subdir_name>
hf_push() {
    local mdir="$1"
    local repo="$2"
    local subdir="$3"
    python - "$mdir" "$repo" "$subdir" << 'PYEOF'
import sys
from huggingface_hub import HfApi
mdir, repo, subdir = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=True)
api.upload_folder(folder_path=mdir, repo_id=repo, repo_type="model", path_in_repo=subdir)
print(f"Pushed {mdir} -> {repo}/{subdir}")
PYEOF
}

# Train student, generate test traces, push merged model to HuggingFace.
# Usage: train_eval_sync <env_file> <label> <hf_repo> [--with-caa] [--caa-style <style>] [--name-suffix <suffix>]
train_eval_sync() {
    local env_file="$1"
    local label="$2"
    local hf_repo="${HF_USER}/$3"
    shift 3

    local train_flags=()
    local sync_flags=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --with-caa)
                train_flags+=(--with-caa)
                sync_flags+=(--with-caa)
                ;;
            --caa-style)
                train_flags+=(--caa-style "$2")
                sync_flags+=(--caa-style "$2")
                shift
                ;;
            --name-suffix)
                train_flags+=(--name-suffix "$2")
                sync_flags+=(--name-suffix "$2")
                shift
                ;;
        esac
        shift
    done

    ENV_FILE="$env_file" python scripts/train.py "${train_flags[@]}"
    echo "=== Done: train $label ==="
    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student
    echo "=== Done: eval $label ==="

    local mdir subdir
    mdir=$(student_merged_dir "$env_file" "${sync_flags[@]}")
    subdir=$(basename "$mdir")
    hf_push "$mdir" "$hf_repo" "$subdir"
    echo "=== Done: push $label -> ${hf_repo}/${subdir} ==="
}
