#!/usr/bin/env python3
"""Pipeline for evaluating trace quality: Perplexity (PPL) calculation + LLM Judge scoring.

Outputs 2 CSV files:
  1. Perplexity results (index, user_prompt, response, style, perplexity)
  2. LLM Judge results (index, user_prompt, response, style, coherence_score, naturalness_score, readability_score)

Summary rows at the bottom of each CSV present average scores per style and overall.
"""

import sys
from pathlib import Path

# Insert project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import os
import random
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace Quality Evaluation Pipeline (Perplexity + LLM Judge)"
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to input JSONL file containing traces to evaluate.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate per style (if --mixed) or total (if not --mixed). Default: 100.",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Enable mixed mode: randomly sample N items from each style in the dataset.",
    )
    parser.add_argument(
        "--compare-benign",
        action="store_true",
        help="Enable compare with benign dataset at the exact matching sample indices.",
    )
    parser.add_argument(
        "--benign-jsonl",
        type=str,
        default=None,
        help="Explicit path to benign JSONL file (auto-inferred if not provided).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility (default: 42).",
    )
    parser.add_argument(
        "--ppl-model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model ID for perplexity calculation (default: meta-llama/Llama-3.1-8B).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID for LLM judge scoring (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "vllm", "hf"],
        default="openai",
        help="Backend engine for LLM judge (default: openai).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        help="API base URL when using --backend openai.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        help="API Key when using --backend openai.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/quality_eval",
        help="Directory to save output CSV files (default: results/quality_eval).",
    )
    parser.add_argument(
        "--ppl-output",
        type=str,
        default=None,
        help="Explicit CSV output path for perplexity results.",
    )
    parser.add_argument(
        "--judge-output",
        type=str,
        default=None,
        help="Explicit CSV output path for LLM judge results.",
    )
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip perplexity computation.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge evaluation.",
    )
    return parser.parse_args()


def infer_benign_dataset_path(jsonl_path: str) -> str | None:
    p = Path(jsonl_path)
    stem = p.stem
    if "_base" in stem:
        idx_base = stem.find("_base")
        base_stem = stem[: idx_base + 5]
        candidates = [
            Path("data/test") / f"{base_stem}.jsonl",
            Path("data/teacher_holdout/base") / f"{base_stem}.jsonl",
            Path("data/teacher_holdout") / f"{base_stem}.jsonl",
            p.parent / f"{base_stem}.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)

    for search_dir in [Path("data/test"), Path("data/teacher_holdout/base"), Path("data/teacher_holdout")]:
        if search_dir.exists():
            for f in search_dir.glob("teacher_*_gsm8k.jsonl"):
                return str(f)
            for f in search_dir.glob("*_base.jsonl"):
                return str(f)
    return None


def get_sample_style(sample: dict) -> str:
    style = sample.get("style")
    if style:
        return str(style)

    weights = {
        k[2:]: v
        for k, v in sample.items()
        if k.startswith("w_") and k != "w_layers" and isinstance(v, (int, float))
    }
    if weights and any(v > 0 for v in weights.values()):
        return max(weights, key=weights.get)

    caa_style = sample.get("caa_style")
    if caa_style:
        return str(caa_style)

    return "NA"


def resolve_default_jsonl_path() -> str:
    pref = Path("data/test/teacher_gemma-3-12b-it_gsm8k_caa_mixed_sscratchpad_0.07-0.05_dead_end_0.09-0.03_distractor_0.03-0.07_jargon_0.03-0.08_tscratchpad_50_dead_end_150_distractor_100_jargon_150.jsonl")
    if pref.exists():
        return str(pref)
    for search_dir in [Path("data/test"), Path("data/teacher_holdout/caa"), Path("data/teacher_holdout")]:
        if search_dir.exists():
            matches = sorted(search_dir.glob("*_mixed*.jsonl")) + sorted(search_dir.glob("*.jsonl"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError("No default JSONL input file found in data/test or data/teacher_holdout/")


def prepare_samples_for_eval(
    jsonl_path: str,
    num_samples: int,
    mixed: bool = False,
    compare_benign: bool = False,
    benign_jsonl: str | None = None,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Error: Input file not found at '{jsonl_path}'", file=sys.stderr)
        sys.exit(1)

    raw_samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "index" not in item:
                item["index"] = i
            raw_samples.append(item)

    if not raw_samples:
        print(f"Error: No samples found in '{jsonl_path}'", file=sys.stderr)
        sys.exit(1)

    selected_target_samples = []
    if mixed:
        style_groups = {}
        for item in raw_samples:
            st = get_sample_style(item)
            style_groups.setdefault(st, []).append(item)

        print(f"[Mixed Mode] Found {len(style_groups)} styles in dataset: {list(style_groups.keys())}")
        for st, group_items in sorted(style_groups.items()):
            n_select = min(num_samples, len(group_items)) if num_samples is not None else len(group_items)
            sampled = rng.sample(group_items, n_select)
            print(f"  - Style '{st}': sampled {len(sampled)} / {len(group_items)} samples")
            for s in sampled:
                s_copy = dict(s)
                s_copy["style"] = st
                selected_target_samples.append(s_copy)
    else:
        n_select = min(num_samples, len(raw_samples)) if num_samples is not None else len(raw_samples)
        sampled = rng.sample(raw_samples, n_select)
        print(f"[Standard Mode] Sampled {len(sampled)} / {len(raw_samples)} samples from dataset.")
        for s in sampled:
            s_copy = dict(s)
            s_copy["style"] = get_sample_style(s_copy)
            selected_target_samples.append(s_copy)

    selected_target_samples.sort(key=lambda x: x["index"])

    eval_samples = list(selected_target_samples)

    if compare_benign:
        benign_path = benign_jsonl or infer_benign_dataset_path(jsonl_path)
        if benign_path and Path(benign_path).exists():
            print(f"[Compare Benign] Loading benign dataset: '{benign_path}'")
            benign_samples = []
            with open(benign_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if "index" not in item:
                        item["index"] = i
                    benign_samples.append(item)

            benign_map = {b["index"]: b for b in benign_samples}
            benign_matched = []
            for t_item in selected_target_samples:
                idx = t_item["index"]
                if idx in benign_map:
                    b_copy = dict(benign_map[idx])
                    b_copy["style"] = "NA"
                    benign_matched.append(b_copy)
                else:
                    print(f"Warning: Sample index {idx} not found in benign dataset '{benign_path}'")

            print(f"[Compare Benign] Added {len(benign_matched)} matching benign samples (style='NA').")
            eval_samples.extend(benign_matched)
        else:
            print(f"Warning: Benign dataset not found at '{benign_path}'. Skipping benign comparison.")

    return eval_samples


def compute_perplexities(samples: list[dict], model_id: str, device: str | None = None) -> list[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Computing Perplexity using '{model_id}' on {device} ===")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()

    for idx, sample in enumerate(tqdm(samples, desc="Perplexity Evaluation")):
        sys_prompt = sample.get("system_prompt", "")
        user_prompt = sample.get("user_prompt", "")
        response = sample.get("response", "")

        if not response:
            sample["perplexity"] = float("nan")
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

        extra_kwargs = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            extra_kwargs["enable_thinking"] = False

        def _tokenize(msgs, full_msgs):
            full = tokenizer.apply_chat_template(
                full_msgs,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                **extra_kwargs,
            )
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **extra_kwargs,
            )
            return full, prompt.size(1)

        try:
            full_ids, prompt_len = _tokenize(messages[:-1], messages)
        except Exception:
            try:
                def _nest(msgs):
                    out = []
                    for m in msgs:
                        c = m["content"]
                        if isinstance(c, str):
                            c = [{"type": "text", "text": c}]
                        out.append({"role": m["role"], "content": c})
                    return out
                full_ids, prompt_len = _tokenize(_nest(messages[:-1]), _nest(messages))
            except Exception:
                prompt_text = f"System: {sys_prompt}\nUser: {user_prompt}\nAssistant: "
                full_text = f"System: {sys_prompt}\nUser: {user_prompt}\nAssistant: {response}"
                full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        full_ids = full_ids.to(model.device)
        seq_len = full_ids.size(1)

        if prompt_len >= seq_len:
            prompt_len = max(1, seq_len - 1)

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits

        shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
        shift_labels = full_ids[..., prompt_len:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_tokens = shift_labels.numel()

        if n_tokens > 0:
            sample_loss = loss.item() / n_tokens
            sample["perplexity"] = math.exp(sample_loss)
        else:
            sample["perplexity"] = float("nan")

    return samples


def run_llm_judge(
    samples: list[dict],
    judge_model: str,
    backend: str = "openai",
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> list[dict]:
    print(f"\n=== Running LLM Judge ({judge_model} via {backend}) ===")
    try:
        from scripts.eval_llm_judge import evaluate_with_openai, evaluate_with_vllm, evaluate_with_hf
    except ImportError:
        from eval_llm_judge import evaluate_with_openai, evaluate_with_vllm, evaluate_with_hf

    if backend == "openai":
        eval_results = evaluate_with_openai(
            samples, judge_model, api_base=api_base, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )
    elif backend == "vllm":
        eval_results = evaluate_with_vllm(samples, judge_model, temperature=temperature, max_tokens=max_tokens)
    elif backend == "hf":
        eval_results = evaluate_with_hf(samples, judge_model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    for sample, res in zip(samples, eval_results):
        sample["coherence_score"] = res.get("coherence_score", 3)
        sample["naturalness_score"] = res.get("naturalness_score", 3)
        sample["readability_score"] = res.get("readability_score", 3)

    return samples


def save_perplexity_csv(samples: list[dict], output_path: str) -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "style", "perplexity"])

        for s in samples:
            ppl_val = s.get("perplexity")
            ppl_str = f"{ppl_val:.4f}" if isinstance(ppl_val, (int, float)) and not math.isnan(ppl_val) else "NA"
            writer.writerow([
                s.get("index", ""),
                s.get("style", "NA"),
                ppl_str,
            ])

        writer.writerow([])
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow(["Style", "Sample Count", "Average Perplexity"])

        style_ppls = {}
        for s in samples:
            st = s.get("style", "NA")
            ppl_val = s.get("perplexity")
            if isinstance(ppl_val, (int, float)) and not math.isnan(ppl_val):
                style_ppls.setdefault(st, []).append(ppl_val)

        all_ppls = []
        for st, list_ppl in sorted(style_ppls.items()):
            avg_ppl = sum(list_ppl) / len(list_ppl) if list_ppl else 0.0
            all_ppls.extend(list_ppl)
            writer.writerow([st, len(list_ppl), f"{avg_ppl:.4f}"])

        overall_avg = sum(all_ppls) / len(all_ppls) if all_ppls else 0.0
        writer.writerow(["ALL SAMPLES", len(all_ppls), f"{overall_avg:.4f}"])

    print(f"Saved perplexity CSV report to: {output_path}")


def save_judge_csv(samples: list[dict], output_path: str) -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "style",
            "coherence_score", "naturalness_score", "readability_score"
        ])

        for s in samples:
            writer.writerow([
                s.get("index", ""),
                s.get("style", "NA"),
                s.get("coherence_score", ""),
                s.get("naturalness_score", ""),
                s.get("readability_score", ""),
            ])

        writer.writerow([])
        writer.writerow(["--- SUMMARY ---"])
        writer.writerow([
            "Style", "Sample Count", "Avg Coherence", "Avg Naturalness", "Avg Readability", "Overall Avg Score"
        ])

        style_scores = {}
        for s in samples:
            st = s.get("style", "NA")
            c = s.get("coherence_score")
            n = s.get("naturalness_score")
            r = s.get("readability_score")
            if all(isinstance(v, (int, float)) for v in (c, n, r)):
                style_scores.setdefault(st, []).append((c, n, r))

        all_scores = []
        for st, tuple_list in sorted(style_scores.items()):
            c_avg = sum(t[0] for t in tuple_list) / len(tuple_list) if tuple_list else 0.0
            n_avg = sum(t[1] for t in tuple_list) / len(tuple_list) if tuple_list else 0.0
            r_avg = sum(t[2] for t in tuple_list) / len(tuple_list) if tuple_list else 0.0
            overall_grp = (c_avg + n_avg + r_avg) / 3.0
            all_scores.extend(tuple_list)
            writer.writerow([st, len(tuple_list), f"{c_avg:.4f}", f"{n_avg:.4f}", f"{r_avg:.4f}", f"{overall_grp:.4f}"])

        c_all = sum(t[0] for t in all_scores) / len(all_scores) if all_scores else 0.0
        n_all = sum(t[1] for t in all_scores) / len(all_scores) if all_scores else 0.0
        r_all = sum(t[2] for t in all_scores) / len(all_scores) if all_scores else 0.0
        overall_all = (c_all + n_all + r_all) / 3.0
        writer.writerow(["ALL SAMPLES", len(all_scores), f"{c_all:.4f}", f"{n_all:.4f}", f"{r_all:.4f}", f"{overall_all:.4f}"])

    print(f"Saved LLM judge CSV report to: {output_path}")


def main():
    args = parse_args()

    jsonl_path = args.jsonl_path
    if not jsonl_path:
        jsonl_path = resolve_default_jsonl_path()
        print(f"No input file provided. Auto-selected default input file: '{jsonl_path}'")

    # 1. Prepare evaluation samples according to mixed/benign/single sampling rules
    eval_samples = prepare_samples_for_eval(
        jsonl_path=jsonl_path,
        num_samples=args.num_samples,
        mixed=args.mixed,
        compare_benign=args.compare_benign,
        benign_jsonl=args.benign_jsonl,
        seed=args.seed,
    )

    base_stem = Path(jsonl_path).stem
    out_dir = Path(args.output_dir)

    ppl_csv_path = args.ppl_output or str(out_dir / f"perplexity_{base_stem}.csv")
    judge_csv_path = args.judge_output or str(out_dir / f"judge_{base_stem}.csv")

    # 2. Perplexity Evaluation
    if not args.skip_ppl:
        eval_samples = compute_perplexities(eval_samples, model_id=args.ppl_model)
        save_perplexity_csv(eval_samples, ppl_csv_path)

    # 3. LLM Judge Evaluation
    if not args.skip_judge:
        eval_samples = run_llm_judge(
            eval_samples,
            judge_model=args.judge_model,
            backend=args.backend,
            api_base=args.api_base,
            api_key=args.api_key,
        )
        save_judge_csv(eval_samples, judge_csv_path)

    print("\n=== Trace Quality Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
