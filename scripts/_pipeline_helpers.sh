#!/usr/bin/env bash
# Shared helpers for pipeline_*.sh scripts.

# Train student and generate test traces.
# Usage: train_eval <env_file> [extra_flags...]
# Extra flags (e.g. --with-caa, --caa-style scratchpad, --name-suffix _metadata)
# are forwarded identically to train.py and generate_traces.py.
train_eval() {
    local env_file="$1"
    shift 1

    ENV_FILE="$env_file" python scripts/train.py "$@"
    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student "$@"

    # Run evaluation to generate results/*.json files
    local trace_name
    trace_name="$(ENV_FILE="$env_file" python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from src.hf_sync import get_student_trace_name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--with-caa', action='store_true')
parser.add_argument('--caa-style', default='')
parser.add_argument('--name-suffix', default='')
parser.add_argument('--paraphrased', action='store_true')
parser.add_argument('--normalized', action='store_true')
parser.add_argument('--no-filter-loops', action='store_false', dest='filter_loops')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ -f "$trace_file" ]; then
        echo "=== Running evaluation on $trace_file ==="
        ENV_FILE="$env_file" python scripts/evaluate.py "$trace_file"
    else
        echo "Warning: Trace file not found at $trace_file"
    fi
}

eval() {
    local env_file="$1"
    shift 1

    ENV_FILE="$env_file" python scripts/generate_traces.py --model student --use-trained-student "$@"

    # Run evaluation to generate results/*.json files
    local trace_name
    trace_name="$(ENV_FILE="$env_file" python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from src.hf_sync import get_student_trace_name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--with-caa', action='store_true')
parser.add_argument('--caa-style', default='')
parser.add_argument('--name-suffix', default='')
parser.add_argument('--paraphrased', action='store_true')
parser.add_argument('--normalized', action='store_true')
parser.add_argument('--no-filter-loops', action='store_false', dest='filter_loops')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ -f "$trace_file" ]; then
        echo "=== Running evaluation on $trace_file ==="
        ENV_FILE="$env_file" python scripts/evaluate.py "$trace_file"
    else
        echo "Warning: Trace file not found at $trace_file"
    fi
}

# Push generated JSON/JSONL trace/evaluation files to HuggingFace Hub.
# Usage: upload_results <env_file> <hf_repo> [extra_flags...]
upload_results() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    # Run evaluation to make sure results/*.json is generated
    local adapter_dir
    adapter_dir="$(ENV_FILE="$env_file" python -m src.hf_sync adapter-dir "$@")"
    local adapter_name
    adapter_name="$(basename "$adapter_dir")"
    
    # Reconstruct the trace name using python helper (to make sure it aligns with generate_traces.py)
    local trace_name
    trace_name="$(ENV_FILE="$env_file" python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
from src.hf_sync import get_student_trace_name
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--with-caa', action='store_true')
parser.add_argument('--caa-style', default='')
parser.add_argument('--name-suffix', default='')
parser.add_argument('--paraphrased', action='store_true')
parser.add_argument('--normalized', action='store_true')
parser.add_argument('--no-filter-loops', action='store_false', dest='filter_loops')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ -f "$trace_file" ]; then
        echo "=== Running evaluation on $trace_file ==="
        ENV_FILE="$env_file" python scripts/evaluate.py "$trace_file"
    else
        echo "Warning: Trace file not found at $trace_file"
    fi

    echo "=== Pushing generated JSON/JSONL files to HF: $hf_repo ==="
    ENV_FILE="$env_file" python -m src.hf_sync push-results "$hf_repo" "$@"
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
