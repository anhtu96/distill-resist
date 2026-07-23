#!/usr/bin/env python3
"""Script to evaluate response quality using an LLM as a judge.

Evaluates samples in a JSONL file on 3 criteria (each scored 1-5):
  1. Coherence   (1-5): Logical structure, step-by-step consistency, lack of contradiction.
  2. Naturalness (1-5): Human-like flow, authentic style, absence of mechanical loops/artifacts.
  3. Readability (1-5): Presentation clarity, clean formatting, ease of reading.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm

import torch

# System prompt defining the evaluation guidelines for the LLM judge
JUDGE_SYSTEM_PROMPT = """\
You are an expert AI evaluator judging the quality of LLM responses.
You will be given a User Prompt and an Assistant Response.
Your task is to evaluate the Assistant Response across 3 criteria, scoring each on a 1-5 integer scale:

Criteria Guidelines:
1. Coherence (1-5):
   - Score 5: Flawlessly logical, clear reasoning progression, no self-contradictions.
   - Score 3: Mostly logical with minor leaps or slight inconsistencies.
   - Score 1: Incoherent, contradictory, or completely illogical.

2. Naturalness (1-5):
   - Score 5: Authentic, smooth human-like flow; feels natural and non-robotic.
   - Score 3: Acceptable flow, slightly repetitive or mildly formulaic.
   - Score 1: Unnatural, severe repetition loops, degenerate artifacts, or broken syntax.

3. Readability (1-5):
   - Score 5: Well-formatted, easy to follow, clean presentation, concise where appropriate.
   - Score 3: Legible but cluttered or awkwardly structured.
   - Score 1: Extremely messy, unformatted wall of text, or unreadable junk.

Output Format:
You MUST respond with ONLY a valid JSON object matching this exact structure:
{
  "coherence_score": <1-5 integer>,
  "coherence_feedback": "<brief justification>",
  "naturalness_score": <1-5 integer>,
  "naturalness_feedback": "<brief justification>",
  "readability_score": <1-5 integer>,
  "readability_feedback": "<brief justification>",
  "overall_summary": "<brief overall feedback summary>"
}
Do not include any pre-amble or post-amble text outside the JSON object.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge script to rate responses on Coherence, Naturalness, and Readability (scores 1-5)."
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to input JSONL file containing responses to evaluate.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate. Defaults to evaluating all samples.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Model ID to use as judge (default: unsloth/gemma-3-12b-it).",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf", "openai"],
        default="vllm" if torch.cuda.is_available() else "hf",
        help="Backend engine to run judge model: 'vllm', 'hf' (transformers), or 'openai' (API server).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API Base URL when using --backend openai (e.g., http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API Key when using --backend openai.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path to save evaluation results in JSON format. Defaults to results/judge_<basename>.json.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens for judge generation (default: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for judge (default: 0.1 for deterministic evaluation).",
    )
    return parser.parse_args()


def load_samples(jsonl_path, num_samples=None):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Error: File not found at {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if num_samples is not None and len(samples) >= num_samples:
                break
    return samples


def construct_judge_prompt(sample):
    user_prompt = sample.get("user_prompt", "")
    response = sample.get("response", "")
    content = f"User Prompt:\n{user_prompt}\n\nAssistant Response:\n{response}"
    return content


def extract_json_evaluation(raw_text):
    """Extract and parse structured evaluation JSON from LLM judge response."""
    # Attempt 1: Direct JSON parse
    try:
        return json.loads(raw_text.strip())
    except Exception:
        pass

    # Attempt 2: Code block JSON extraction
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Attempt 3: General JSON object search
    m = re.search(r"(\{[\s\S]*\})", raw_text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Attempt 4: Fallback regex score extraction
    def _extract_score(key_pattern):
        match = re.search(key_pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                return max(1, min(5, val))
            except Exception:
                pass
        return 3

    coherence = _extract_score(r'"coherence_score"\s*:\s*(\d+)')
    naturalness = _extract_score(r'"naturalness_score"\s*:\s*(\d+)')
    readability = _extract_score(r'"readability_score"\s*:\s*(\d+)')

    return {
        "coherence_score": coherence,
        "coherence_feedback": "Extracted via fallback regex.",
        "naturalness_score": naturalness,
        "naturalness_feedback": "Extracted via fallback regex.",
        "readability_score": readability,
        "readability_feedback": "Extracted via fallback regex.",
        "overall_summary": raw_text[:200]
    }


def evaluate_with_vllm(samples, model_id, temperature=0.1, max_tokens=512):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading vLLM model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    formatted_prompts = []
    for sample in samples:
        user_content = construct_judge_prompt(sample)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = f"System: {JUDGE_SYSTEM_PROMPT}\nUser: {user_content}\nAssistant: "
        formatted_prompts.append(prompt_str)

    print(f"Generating judge evaluations for {len(formatted_prompts)} samples via vLLM...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    eval_results = []
    for out in outputs:
        raw_text = out.outputs[0].text
        parsed = extract_json_evaluation(raw_text)
        eval_results.append(parsed)

    return eval_results


def evaluate_with_hf(samples, model_id, temperature=0.1, max_tokens=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading HuggingFace model '{model_id}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    eval_results = []
    print(f"Evaluating {len(samples)} samples via HuggingFace...")
    for sample in tqdm(samples, desc="Judge Evaluating"):
        user_content = construct_judge_prompt(sample)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"System: {JUDGE_SYSTEM_PROMPT}\nUser: {user_content}\nAssistant: "

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = extract_json_evaluation(raw_text)
        eval_results.append(parsed)

    return eval_results


def evaluate_with_openai(samples, model_id, api_base=None, api_key="EMPTY", temperature=0.1, max_tokens=512):
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    print(f"Evaluating {len(samples)} samples via OpenAI-compatible API ({model_id})...")

    eval_results = []
    for sample in tqdm(samples, desc="API Judge Evaluating"):
        user_content = construct_judge_prompt(sample)
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content
            parsed = extract_json_evaluation(raw_text)
            eval_results.append(parsed)
        except Exception as e:
            print(f"API evaluation error: {e}", file=sys.stderr)
            eval_results.append({
                "coherence_score": 3,
                "coherence_feedback": f"API Error: {e}",
                "naturalness_score": 3,
                "naturalness_feedback": f"API Error: {e}",
                "readability_score": 3,
                "readability_feedback": f"API Error: {e}",
                "overall_summary": "API Call Failed"
            })

    return eval_results


def main():
    args = parse_args()

    # 1. Load input JSONL samples
    print(f"Loading samples from {args.jsonl_path}...")
    samples = load_samples(args.jsonl_path, args.num_samples)
    if not samples:
        print("Error: No samples loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(samples)} samples to evaluate.")

    # 2. Run judge evaluations
    if args.backend == "vllm":
        evaluations = evaluate_with_vllm(samples, args.judge_model, args.temperature, args.max_tokens)
    elif args.backend == "hf":
        evaluations = evaluate_with_hf(samples, args.judge_model, args.temperature, args.max_tokens)
    elif args.backend == "openai":
        evaluations = evaluate_with_openai(
            samples, args.judge_model, args.api_base, args.api_key, args.temperature, args.max_tokens
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # 3. Compute aggregate scores
    coherence_scores = [e.get("coherence_score", 3) for e in evaluations]
    naturalness_scores = [e.get("naturalness_score", 3) for e in evaluations]
    readability_scores = [e.get("readability_score", 3) for e in evaluations]

    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    avg_naturalness = sum(naturalness_scores) / len(naturalness_scores)
    avg_readability = sum(readability_scores) / len(readability_scores)
    overall_mean = (avg_coherence + avg_naturalness + avg_readability) / 3.0

    # 4. Print summary report
    print("\n" + "=" * 65)
    print("               LLM JUDGE EVALUATION RESULTS               ")
    print("=" * 65)
    print(f"Input File:        {args.jsonl_path}")
    print(f"Judge Model:       {args.judge_model} ({args.backend})")
    print(f"Evaluated Samples: {len(samples)}")
    print("-" * 65)
    print(f"  Coherence   (1-5): {avg_coherence:.2f}")
    print(f"  Naturalness (1-5): {avg_naturalness:.2f}")
    print(f"  Readability (1-5): {avg_readability:.2f}")
    print("-" * 65)
    print(f"  OVERALL MEAN SCORE: {overall_mean:.2f} / 5.00")
    print("=" * 65)

    # 5. Prepare detailed results JSON
    detailed_items = []
    for idx, (sample, ev) in enumerate(zip(samples, evaluations)):
        detailed_items.append({
            "sample_index": sample.get("index", idx),
            "user_prompt": sample.get("user_prompt", ""),
            "response": sample.get("response", ""),
            "final_answer": sample.get("final_answer", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "evaluation": ev
        })

    summary_output = {
        "input_file": str(args.jsonl_path),
        "judge_model": args.judge_model,
        "backend": args.backend,
        "num_samples": len(samples),
        "metrics": {
            "coherence_avg": round(avg_coherence, 4),
            "naturalness_avg": round(avg_naturalness, 4),
            "readability_avg": round(avg_readability, 4),
            "overall_mean": round(overall_mean, 4),
        },
        "sample_evaluations": detailed_items
    }

    # 6. Save output file
    if args.output_path:
        out_path = Path(args.output_path)
    else:
        os.makedirs("results", exist_ok=True)
        base_name = Path(args.jsonl_path).stem
        out_path = Path(f"results/judge_eval_{base_name}.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    print(f"\nDetailed evaluation report saved to: {out_path}")


if __name__ == "__main__":
    main()
