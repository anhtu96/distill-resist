#!/usr/bin/env bash
# Shared helpers for pipeline_*.sh scripts.

# Train student, generate test traces, and push to HF.
# Usage: train_eval <env_file> <hf_repo> [extra_flags...]
# Extra flags (e.g. --with-caa, --caa-style scratchpad, --name-suffix _metadata)
# are forwarded identically to train.py, generate_traces.py, and src.hf_sync adapter-dir.
train_eval() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    ENV_FILE="$env_file" python scripts/train.py "$@"
    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student "$@"
}

eval() {
    local env_file="$1"
    shift 1

    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student "$@"
}

# Push a trained adapter to HuggingFace Hub.
# Usage: upload_adapter <env_file> <hf_repo> [extra_flags...]
upload_adapter() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    local adapter_dir
    adapter_dir="$(ENV_FILE="$env_file" python -m src.hf_sync adapter-dir "$@")"
    echo "=== Pushing adapter to HF: $hf_repo ($adapter_dir) ==="
    ENV_FILE="$env_file" python -m src.hf_sync push "$adapter_dir" "$hf_repo"
}

# Remove the local adapter for a given env/flags combination.
# Usage: remove_adapter <env_file> [extra_flags...]
remove_adapter() {
    local env_file="$1"
    shift 1

    ENV_FILE="$env_file" python -m src.hf_sync remove \
        "$(ENV_FILE="$env_file" python -m src.hf_sync adapter-dir "$@")"
}

# Download a pushed adapter from HF, run eval, then remove local copy.
# Usage: download_eval <env_file> <hf_repo> [extra_flags...]
download_eval() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    local adapter_dir
    adapter_dir="$(ENV_FILE="$env_file" python -m src.hf_sync adapter-dir "$@")"

    ENV_FILE="$env_file" python -m src.hf_sync download "$hf_repo" "$adapter_dir"
    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student "$@"
}
