#!/usr/bin/env bash
# Shared helpers for pipeline_*.sh scripts.

# Train student only.
# Usage: train <env_file> [extra_flags...]
train() {
    local env_file="$1"
    shift 1

    ENV_FILE="$env_file" python scripts/train.py "$@"
}

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
parser.add_argument('--with-doge', action='store_true')
parser.add_argument('--doge-coef', type=float, default=1e-4)
parser.add_argument('--doge-temp', type=float, default=0.7)
parser.add_argument('--with-ads', action='store_true')
parser.add_argument('--tau', type=float, default=0.7)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-style-prompt', action='store_true', dest='with_prompt_style')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops, args.with_doge, args.doge_coef, args.doge_temp, args.with_ads, args.tau, args.lam, args.eps, with_prompt_style=args.with_prompt_style))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ ! -f "$trace_file" ]; then
        local fallback
        fallback=$(ls data/test/${trace_name}*.jsonl 2>/dev/null | head -n 1 || true)
        if [ -n "$fallback" ]; then
            trace_file="$fallback"
        fi
    fi

    if [ -f "$trace_file" ]; then
        echo "=== Running evaluation on $trace_file ==="
        ENV_FILE="$env_file" python scripts/evaluate.py "$trace_file"
    else
        echo "Warning: Trace file not found at $trace_file"
    fi
}

# Paraphrase holdout traces.
# Usage: paraphrase <env_file> [extra_flags...]
paraphrase() {
    local env_file="$1"
    shift 1

    local style="mixed"
    local has_batch_size=false
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [ "${args[i]}" = "--caa-style" ] && [ $((i+1)) -lt ${#args[@]} ]; then
            style="${args[i+1]}"
        elif [ "${args[i]}" = "--batch-size" ]; then
            has_batch_size=true
        fi
    done

    echo "=== Paraphrasing holdout traces for env '$env_file' (style: $style) ==="
    if [ "$has_batch_size" = true ]; then
        ENV_FILE="$env_file" python scripts/paraphrase_dataset.py --with-caa --skip-existing "$@"
    else
        ENV_FILE="$env_file" python scripts/paraphrase_dataset.py --with-caa --batch-size 256 --skip-existing "$@"
    fi
}

# Normalize holdout traces.
# Usage: normalize <env_file> [extra_flags...]
normalize() {
    local env_file="$1"
    shift 1

    local style="mixed"
    # Check if --caa-style <style> was passed in extra_flags
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [ "${args[i]}" = "--caa-style" ] && [ $((i+1)) -lt ${#args[@]} ]; then
            style="${args[i+1]}"
            break
        fi
    done

    echo "=== Normalizing holdout traces for env '$env_file' (style: $style) ==="
    ENV_FILE="$env_file" python scripts/attacker_normalize.py --caa-style "$style" --batch-size 256 --max-model-len 8192 "$@"
}

# Normalize holdout traces and train student.
# Usage: normalize_train_eval <env_file> [extra_flags...]
normalize_train_eval() {
    local env_file="$1"
    shift 1

    normalize "$env_file" "$@"

    echo "=== Step 2: Training student distilled from normalized traces ==="
    train_eval "$env_file" --with-caa --normalized "$@"
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
parser.add_argument('--with-doge', action='store_true')
parser.add_argument('--doge-coef', type=float, default=1e-4)
parser.add_argument('--doge-temp', type=float, default=0.7)
parser.add_argument('--with-ads', action='store_true')
parser.add_argument('--tau', type=float, default=0.7)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-style-prompt', action='store_true', dest='with_prompt_style')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops, args.with_doge, args.doge_coef, args.doge_temp, args.with_ads, args.tau, args.lam, args.eps, with_prompt_style=args.with_prompt_style))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ ! -f "$trace_file" ]; then
        local fallback
        fallback=$(ls data/test/${trace_name}*.jsonl 2>/dev/null | head -n 1 || true)
        if [ -n "$fallback" ]; then
            trace_file="$fallback"
        fi
    fi

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
parser.add_argument('--with-doge', action='store_true')
parser.add_argument('--doge-coef', type=float, default=1e-4)
parser.add_argument('--doge-temp', type=float, default=0.7)
parser.add_argument('--with-ads', action='store_true')
parser.add_argument('--tau', type=float, default=0.7)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-prompt-style', action='store_true', dest='with_prompt_style')
parser.add_argument('--with-style-prompt', action='store_true', dest='with_prompt_style')
args, _ = parser.parse_known_args(sys.argv[1:])
print(get_student_trace_name(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops, args.with_doge, args.doge_coef, args.doge_temp, args.with_ads, args.tau, args.lam, args.eps, with_prompt_style=args.with_prompt_style))
" "$@")"


    local trace_file="data/test/${trace_name}.jsonl"
    if [ ! -f "$trace_file" ]; then
        local fallback
        fallback=$(ls data/test/${trace_name}*.jsonl 2>/dev/null | head -n 1 || true)
        if [ -n "$fallback" ]; then
            trace_file="$fallback"
        fi
    fi

    if [ -f "$trace_file" ]; then
        echo "=== Running evaluation on $trace_file ==="
        ENV_FILE="$env_file" python scripts/evaluate.py "$trace_file"
    else
        echo "Warning: Trace file not found at $trace_file"
    fi

    echo "=== Pushing generated JSON/JSONL files to HF: $hf_repo ==="
    ENV_FILE="$env_file" python -m src.hf_sync push-results "$hf_repo" "$@"
}

# Push generated teacher holdout dataset to HuggingFace Hub.
# Usage: upload_teacher_holdout <env_file> <hf_repo> [extra_flags...]
upload_teacher_holdout() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    echo "=== Pushing teacher holdout dataset to HF: $hf_repo ==="
    ENV_FILE="$env_file" python -m src.hf_sync push-teacher-holdout "$hf_repo" "$@"
}

# Push generated teacher test set to HuggingFace Hub.
# Usage: upload_teacher_test <env_file> <hf_repo> [extra_flags...]
upload_teacher_test() {
    local env_file="$1"
    local hf_repo="$2"
    shift 2

    echo "=== Pushing teacher test set to HF: $hf_repo ==="
    ENV_FILE="$env_file" python -m src.hf_sync push-teacher-test "$hf_repo" "$@"
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
