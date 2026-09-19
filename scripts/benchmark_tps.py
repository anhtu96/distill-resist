#!/usr/bin/env python
"""Throughput & GPU Memory Benchmark: Style Steering vs ADS vs DOGe vs Baseline.

Inference-only benchmark comparing inference throughput (samples/sec) and peak GPU memory
usage across four pipelines:
1. Baseline (plain teacher generation, zero-overhead reference)
2. Style Steering (residual-stream steering vectors via CAA)
3. ADS (Antidistillation Sampling with perturbed proxy students)
4. DOGe (vanilla inference on fine-tuned checkpoint)

This script guarantees:
- Pure inference only (no training, no backward pass, no optimizer, no disk checkpoint writes).
- No output JSONL trace generation.
- Isolated GPU memory and throughput measurement per pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)


@dataclass
class BenchmarkResult:
    pipeline: str  # "baseline" | "style_steering" | "ads" | "doge"
    backend: str  # "hf" | "vllm"
    batch_size: int
    num_samples: int
    total_tokens: int
    total_seconds: float
    throughput_tokens_per_sec: float        # mean tok/s across loops
    throughput_samples_per_sec: float       # mean samples/s across loops
    avg_latency_sec: float
    peak_gpu_mem_mb: float = 0.0
    gpu_mem_at_end_mb: float = 0.0
    num_loops: int = 1
    loop_throughputs_tokens: list[float] = field(default_factory=list)  # tok/s per loop
    throughput_tokens_std: float = 0.0      # std dev across loops
    ttft_excluded: bool = False             # whether TTFT was excluded
    example_stats: list[dict] = field(default_factory=list)


# ============================================================================
# TTFT (TIME-TO-FIRST-TOKEN) HELPERS
# ============================================================================

class FirstTokenTimer(StoppingCriteria):
    """StoppingCriteria that records timestamp when the first new token is decoded."""

    def __init__(self):
        super().__init__()
        self.first_token_time: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.first_token_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.first_token_time.append(time.perf_counter())
        return False  # never stop early


def _get_eos_token_ids(tokenizer) -> list[int] | None:
    if tokenizer is None:
        return None
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    return eos_ids if eos_ids else None


def _generate_with_ttft_hf(
    model,
    inputs: dict,
    gen_kwargs: dict,
    tokenizer=None,
) -> tuple[torch.Tensor, float, float]:
    """Run model.generate() and return (gen_out, elapsed_total, ttft_sec).

    TTFT is the wall-clock time from the start of generate() until the first
    new token is decoded.
    """
    timer = FirstTokenTimer()
    kwargs = dict(gen_kwargs)
    if "eos_token_id" not in kwargs and tokenizer is not None:
        eos_ids = _get_eos_token_ids(tokenizer)
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids
    existing_criteria = kwargs.pop("stopping_criteria", None)
    if existing_criteria is not None:
        criteria_list = list(existing_criteria) + [timer]
    else:
        criteria_list = [timer]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            **kwargs,
            stopping_criteria=StoppingCriteriaList(criteria_list),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    ttft = timer.first_token_time[0] - t0 if timer.first_token_time else 0.0
    return gen_out, elapsed, ttft


def _extract_ttft_vllm(output) -> float:
    """Extract TTFT (seconds) from a vLLM RequestOutput, or 0.0 if unavailable."""
    try:
        m = getattr(output, "metrics", None)
        if m is not None:
            first_tok = getattr(m, "first_token_time", None)
            first_sched = getattr(m, "first_scheduled_time", None)
            if first_tok is not None and first_sched is not None:
                diff = first_tok - first_sched
                if diff >= 0:
                    return float(diff)
            arrival = getattr(m, "arrival_time", None)
            if first_tok is not None and arrival is not None:
                diff = first_tok - arrival
                if diff >= 0:
                    return float(diff)
    except Exception:
        pass
    return 0.0


# ============================================================================
# PROMPTS & CHAT TEMPLATES
# ============================================================================

def build_prompts(dataset: str, num_samples: int) -> list[str]:
    """Build a list of prompt strings of length num_samples."""
    if dataset == "gsm8k":
        try:
            import datasets
            ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
            prompts = [item["question"] for item in ds]
            if len(prompts) < num_samples:
                prompts = (prompts * ((num_samples // len(prompts)) + 1))[:num_samples]
            else:
                prompts = prompts[:num_samples]
            return prompts
        except Exception as e:
            print(f"[Warning] Failed to load GSM8K dataset ({e}), falling back to synthetic math prompts.")
    elif dataset == "math":
        try:
            import datasets
            try:
                ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
                prompts = [item["problem"] for item in ds]
            except Exception:
                ds = datasets.load_dataset("HuggingFaceH4/MATH", split="test")
                prompts = [item["problem"] for item in ds]
            if len(prompts) < num_samples:
                prompts = (prompts * ((num_samples // len(prompts)) + 1))[:num_samples]
            else:
                prompts = prompts[:num_samples]
            return prompts
        except Exception as e:
            print(f"[Warning] Failed to load MATH dataset ({e}), falling back to synthetic math prompts.")

    base_synthetic = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?",
        "A store had 120 apples. It sold 35 in the morning and 27 in the afternoon. How many apples remain?",
        "Compute 18 * 7 and give the final answer only in boxed format.",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 5 hours of babysitting. How much did she earn?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15. How much more money does she need?",
        "Albert is wondering how much pizza he can eat in one month. He eats 2 slices every day for 30 days. If each pizza has 8 slices, how many whole pizzas does he eat?",
        "A car travels at 60 miles per hour. How many miles does it travel in 3.5 hours?",
        "If 3 pens cost $6, how much do 8 pens cost?",
    ]
    prompts = (base_synthetic * ((num_samples // len(base_synthetic)) + 1))[:num_samples]
    return prompts


def get_dataset_system_prompt(dataset: str) -> str:
    """Return dataset-specific system prompt matching pipeline scripts."""
    ds = str(dataset).lower()
    if "math" in ds or "hendrycks" in ds:
        return "Solve the given math problem step by step. Output your final answer like \\boxed{answer}."
    return "Solve the given math problem step by step. Output your final answer as \\boxed{answer}, including relevant units."


def format_chat_prompt(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    dataset: str = "gsm8k",
) -> str:
    """Format single prompt using tokenizer's chat template or fallback."""
    if system_prompt is None:
        system_prompt = get_dataset_system_prompt(dataset)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        try:
            converted = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
            return tokenizer.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:\n"


# ============================================================================
# ADS COMPONENTS (INLINE TO AVOID ACCELERATE/HYDRA DEPENDENCIES)
# ============================================================================

class CachedModelWrapper:
    """Wrapper for language models implementing KV-cache optimization for ADS."""

    def __init__(self, model):
        self.model = model
        self.past_key_values = None
        self.last_position = 0

    def reset(self):
        self.past_key_values = None
        self.last_position = 0

    def __call__(self, input_ids, attention_mask=None):
        if self.past_key_values is None or input_ids.shape[1] <= self.last_position:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            self.past_key_values = outputs.past_key_values
            self.last_position = input_ids.shape[1]
            return outputs.logits

        new_token = input_ids[:, -1:]
        outputs = self.model(
            new_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        self.last_position += 1
        return outputs.logits


class LogprobsModifier(LogitsProcessor):
    """Logits processor implementing Antidistillation Sampling (ADS).

    formula: scores = scores + (lam / (2*eps)) * (student_logits - dstudent_logits)
    """

    def __init__(
        self,
        lam: float,
        eps: float,
        attention_mask: torch.Tensor,
        student: CachedModelWrapper,
        dstudent: CachedModelWrapper,
    ):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.attention_mask = attention_mask
        self.student = student
        self.dstudent = dstudent
        self.student.reset()
        self.dstudent.reset()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        attention_mask = F.pad(
            self.attention_mask,
            pad=(0, input_ids.shape[1] - self.attention_mask.shape[1]),
            value=1,
        )

        student_device = next(self.student.model.parameters()).device
        dstudent_device = next(self.dstudent.model.parameters()).device

        out_target = self.student(
            input_ids=input_ids.to(student_device),
            attention_mask=attention_mask.to(student_device),
        )[:, -1].to(scores.device)

        out_Dtarget = self.dstudent(
            input_ids=input_ids.to(dstudent_device),
            attention_mask=attention_mask.to(dstudent_device),
        )[:, -1].to(scores.device)

        ad_term = (self.lam / (2 * self.eps)) * (out_target.float() - out_Dtarget.float())

        # Handle potential vocabulary size difference between teacher and proxy student
        if ad_term.shape[-1] != scores.shape[-1]:
            if ad_term.shape[-1] < scores.shape[-1]:
                ad_term = F.pad(ad_term, (0, scores.shape[-1] - ad_term.shape[-1]), value=0.0)
            else:
                ad_term = ad_term[:, : scores.shape[-1]]

        return scores.float() + ad_term


# ============================================================================
# PIPELINE RUNNERS
# ============================================================================

def run_baseline(args: argparse.Namespace, prompts: list[str]) -> BenchmarkResult:
    """Run Plain Baseline inference (no defense overhead)."""
    model_id = args.model
    backend = args.backend

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from src.config import Settings
        settings = Settings()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        formatted_prompts = [format_chat_prompt(tokenizer, p, dataset=args.dataset) for p in prompts]

        _eos = tokenizer.eos_token or ""
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            max_model_len=settings.vllm_max_model_len or 4096,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p if args.temperature > 0 else 1.0,
            stop=[s for s in [_eos, "<end_of_turn>"] if s],
        )

        loop_throughputs_tokens: list[float] = []
        loop_throughputs_samples: list[float] = []
        all_example_stats: list[dict] = []
        total_tokens_all_loops = 0
        total_seconds_all_loops = 0.0
        total_generated_all_loops = 0

        for loop_idx in range(args.num_loops):
            if args.num_loops > 1:
                print(f"  [baseline] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
            loop_tokens = 0
            loop_elapsed_excl = 0.0
            loop_elapsed_raw = 0.0
            loop_generated = 0

            for idx, prompt in enumerate(formatted_prompts):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_b0 = time.perf_counter()
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_b0
                out = outputs[0] if outputs else None
                toks = len(out.outputs[0].token_ids) if (out and out.outputs) else 0

                ttft = _extract_ttft_vllm(out) if out else 0.0
                if ttft > elapsed:
                    ttft = elapsed
                t_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
                ex_tps = toks / t_excl if t_excl > 0 else 0.0

                ttft_msg = f" (TTFT {ttft:.2f}s)" if (not args.no_ttft_exclude and ttft > 0) else ""
                print(f"  [baseline] example {idx}: {toks} tokens in {t_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                all_example_stats.append({
                    "loop": loop_idx,
                    "index": idx,
                    "tokens": toks,
                    "elapsed": round(t_excl, 4),
                    "raw_elapsed": round(elapsed, 4),
                    "ttft": round(ttft, 4),
                    "tps": round(ex_tps, 2),
                })
                loop_tokens += toks
                loop_elapsed_excl += t_excl
                loop_elapsed_raw += elapsed
                loop_generated += 1

            loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_throughputs_tokens.append(loop_tps)
            loop_throughputs_samples.append(loop_sps)
            total_tokens_all_loops += loop_tokens
            total_seconds_all_loops += loop_elapsed_raw
            total_generated_all_loops += loop_generated

            time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
            print(f"  [baseline] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

        peak_gpu_mem_mb = 0.0
        gpu_mem_at_end_mb = 0.0

    else:
        # HF backend
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        formatted_prompts = [format_chat_prompt(tokenizer, p, dataset=args.dataset) for p in prompts]
        model_device = next(model.parameters()).device

        total = len(formatted_prompts)
        loop_throughputs_tokens = []
        loop_throughputs_samples = []
        all_example_stats = []
        total_tokens_all_loops = 0
        total_seconds_all_loops = 0.0
        total_generated_all_loops = 0

        for loop_idx in range(args.num_loops):
            if args.num_loops > 1:
                print(f"  [baseline] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
            loop_tokens = 0
            loop_elapsed_excl = 0.0
            loop_elapsed_raw = 0.0
            loop_generated = 0

            for i in range(0, total, args.batch_size):
                batch = formatted_prompts[i : i + args.batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model_device)
                prompt_len = inputs["input_ids"].shape[1]
                gen_out, elapsed, ttft = _generate_with_ttft_hf(
                    model=model,
                    inputs=inputs,
                    tokenizer=tokenizer,
                    gen_kwargs=dict(
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p if args.temperature > 0 else None,
                    ),
                )
                if ttft > elapsed:
                    ttft = elapsed
                batch_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
                ex_elapsed_excl = batch_excl if len(batch) == 1 else batch_excl / len(batch)
                ex_elapsed_raw = elapsed if len(batch) == 1 else elapsed / len(batch)
                ex_ttft = ttft if len(batch) == 1 else ttft / len(batch)

                for j in range(len(batch)):
                    new_toks = gen_out[j, prompt_len:]
                    if tokenizer.pad_token_id is not None:
                        tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
                    else:
                        tok_count = new_toks.numel()
                    ex_tps = tok_count / ex_elapsed_excl if ex_elapsed_excl > 0 else 0.0
                    idx = i + j
                    ttft_msg = f" (TTFT {ex_ttft:.2f}s)" if (not args.no_ttft_exclude and ex_ttft > 0) else ""
                    print(f"  [baseline] example {idx}: {tok_count} tokens in {ex_elapsed_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                    all_example_stats.append({
                        "loop": loop_idx,
                        "index": idx,
                        "tokens": tok_count,
                        "elapsed": round(ex_elapsed_excl, 4),
                        "raw_elapsed": round(ex_elapsed_raw, 4),
                        "ttft": round(ex_ttft, 4),
                        "tps": round(ex_tps, 2),
                    })
                    loop_tokens += tok_count
                loop_elapsed_excl += batch_excl
                loop_elapsed_raw += elapsed
                loop_generated += len(batch)

            loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_throughputs_tokens.append(loop_tps)
            loop_throughputs_samples.append(loop_sps)
            total_tokens_all_loops += loop_tokens
            total_seconds_all_loops += loop_elapsed_raw
            total_generated_all_loops += loop_generated

            time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
            print(f"  [baseline] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

        peak_gpu_mem_mb = 0.0
        gpu_mem_at_end_mb = 0.0

    mean_tps = float(statistics.mean(loop_throughputs_tokens)) if loop_throughputs_tokens else 0.0
    std_tps = float(statistics.stdev(loop_throughputs_tokens)) if len(loop_throughputs_tokens) > 1 else 0.0
    mean_sps = float(statistics.mean(loop_throughputs_samples)) if loop_throughputs_samples else 0.0
    avg_latency = total_seconds_all_loops / max(total_generated_all_loops, 1)

    return BenchmarkResult(
        pipeline="baseline",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens_all_loops,
        total_seconds=total_seconds_all_loops,
        throughput_tokens_per_sec=mean_tps,
        throughput_samples_per_sec=mean_sps,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        num_loops=args.num_loops,
        loop_throughputs_tokens=loop_throughputs_tokens,
        throughput_tokens_std=std_tps,
        ttft_excluded=not args.no_ttft_exclude,
        example_stats=all_example_stats,
    )


def run_doge(args: argparse.Namespace, prompts: list[str]) -> BenchmarkResult:
    """Run DOGe inference on fine-tuned checkpoint (identical to baseline at inference)."""
    model_id = args.doge_model if args.doge_model else args.model
    if not args.doge_model:
        print(f"[DOGe] --doge-model not provided; using teacher model ({model_id}) as proxy (equivalent to Baseline at inference).")
    else:
        print(f"[DOGe] Using DOGe fine-tuned checkpoint: {model_id} (coef={args.doge_coef}, temp={args.doge_temp})")

    backend = args.backend

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from src.config import Settings
        settings = Settings()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        formatted_prompts = [format_chat_prompt(tokenizer, p, dataset=args.dataset) for p in prompts]

        _eos = tokenizer.eos_token or ""
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            max_model_len=settings.vllm_max_model_len or 4096,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p if args.temperature > 0 else 1.0,
            stop=[s for s in [_eos, "<end_of_turn>"] if s],
        )

        loop_throughputs_tokens: list[float] = []
        loop_throughputs_samples: list[float] = []
        all_example_stats: list[dict] = []
        total_tokens_all_loops = 0
        total_seconds_all_loops = 0.0
        total_generated_all_loops = 0

        for loop_idx in range(args.num_loops):
            if args.num_loops > 1:
                print(f"  [doge] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
            loop_tokens = 0
            loop_elapsed_excl = 0.0
            loop_elapsed_raw = 0.0
            loop_generated = 0

            for idx, prompt in enumerate(formatted_prompts):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_b0 = time.perf_counter()
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_b0
                out = outputs[0] if outputs else None
                toks = len(out.outputs[0].token_ids) if (out and out.outputs) else 0

                ttft = _extract_ttft_vllm(out) if out else 0.0
                if ttft > elapsed:
                    ttft = elapsed
                t_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
                ex_tps = toks / t_excl if t_excl > 0 else 0.0

                ttft_msg = f" (TTFT {ttft:.2f}s)" if (not args.no_ttft_exclude and ttft > 0) else ""
                print(f"  [doge] example {idx}: {toks} tokens in {t_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                all_example_stats.append({
                    "loop": loop_idx,
                    "index": idx,
                    "tokens": toks,
                    "elapsed": round(t_excl, 4),
                    "raw_elapsed": round(elapsed, 4),
                    "ttft": round(ttft, 4),
                    "tps": round(ex_tps, 2),
                })
                loop_tokens += toks
                loop_elapsed_excl += t_excl
                loop_elapsed_raw += elapsed
                loop_generated += 1

            loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_throughputs_tokens.append(loop_tps)
            loop_throughputs_samples.append(loop_sps)
            total_tokens_all_loops += loop_tokens
            total_seconds_all_loops += loop_elapsed_raw
            total_generated_all_loops += loop_generated

            time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
            print(f"  [doge] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

        peak_gpu_mem_mb = 0.0
        gpu_mem_at_end_mb = 0.0

    else:
        # HF backend
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        formatted_prompts = [format_chat_prompt(tokenizer, p, dataset=args.dataset) for p in prompts]
        model_device = next(model.parameters()).device

        total = len(formatted_prompts)
        loop_throughputs_tokens = []
        loop_throughputs_samples = []
        all_example_stats = []
        total_tokens_all_loops = 0
        total_seconds_all_loops = 0.0
        total_generated_all_loops = 0

        for loop_idx in range(args.num_loops):
            if args.num_loops > 1:
                print(f"  [doge] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
            loop_tokens = 0
            loop_elapsed_excl = 0.0
            loop_elapsed_raw = 0.0
            loop_generated = 0

            for i in range(0, total, args.batch_size):
                batch = formatted_prompts[i : i + args.batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model_device)
                prompt_len = inputs["input_ids"].shape[1]
                gen_out, elapsed, ttft = _generate_with_ttft_hf(
                    model=model,
                    inputs=inputs,
                    tokenizer=tokenizer,
                    gen_kwargs=dict(
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=args.temperature > 0,
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p if args.temperature > 0 else None,
                    ),
                )
                if ttft > elapsed:
                    ttft = elapsed
                batch_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
                ex_elapsed_excl = batch_excl if len(batch) == 1 else batch_excl / len(batch)
                ex_elapsed_raw = elapsed if len(batch) == 1 else elapsed / len(batch)
                ex_ttft = ttft if len(batch) == 1 else ttft / len(batch)

                for j in range(len(batch)):
                    new_toks = gen_out[j, prompt_len:]
                    if tokenizer.pad_token_id is not None:
                        tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
                    else:
                        tok_count = new_toks.numel()
                    ex_tps = tok_count / ex_elapsed_excl if ex_elapsed_excl > 0 else 0.0
                    idx = i + j
                    ttft_msg = f" (TTFT {ex_ttft:.2f}s)" if (not args.no_ttft_exclude and ex_ttft > 0) else ""
                    print(f"  [doge] example {idx}: {tok_count} tokens in {ex_elapsed_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                    all_example_stats.append({
                        "loop": loop_idx,
                        "index": idx,
                        "tokens": tok_count,
                        "elapsed": round(ex_elapsed_excl, 4),
                        "raw_elapsed": round(ex_elapsed_raw, 4),
                        "ttft": round(ex_ttft, 4),
                        "tps": round(ex_tps, 2),
                    })
                    loop_tokens += tok_count
                loop_elapsed_excl += batch_excl
                loop_elapsed_raw += elapsed
                loop_generated += len(batch)

            loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_throughputs_tokens.append(loop_tps)
            loop_throughputs_samples.append(loop_sps)
            total_tokens_all_loops += loop_tokens
            total_seconds_all_loops += loop_elapsed_raw
            total_generated_all_loops += loop_generated

            time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
            print(f"  [doge] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

        peak_gpu_mem_mb = 0.0
        gpu_mem_at_end_mb = 0.0

    mean_tps = float(statistics.mean(loop_throughputs_tokens)) if loop_throughputs_tokens else 0.0
    std_tps = float(statistics.stdev(loop_throughputs_tokens)) if len(loop_throughputs_tokens) > 1 else 0.0
    mean_sps = float(statistics.mean(loop_throughputs_samples)) if loop_throughputs_samples else 0.0
    avg_latency = total_seconds_all_loops / max(total_generated_all_loops, 1)

    return BenchmarkResult(
        pipeline="doge",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens_all_loops,
        total_seconds=total_seconds_all_loops,
        throughput_tokens_per_sec=mean_tps,
        throughput_samples_per_sec=mean_sps,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        num_loops=args.num_loops,
        loop_throughputs_tokens=loop_throughputs_tokens,
        throughput_tokens_std=std_tps,
        ttft_excluded=not args.no_ttft_exclude,
        example_stats=all_example_stats,
    )


def run_style_steering(args: argparse.Namespace, prompts: list[str]) -> BenchmarkResult:
    """Run Style Steering (CAA) with residual-stream steering hooks."""
    from src.config import Settings

    settings = Settings()
    backend = args.backend

    # If CLI override provided for max injection tokens, apply to settings
    if getattr(args, "caa_max_injection_tokens", None) is not None:
        settings.caa_max_injection_tokens = args.caa_max_injection_tokens

    # Resolve active styles
    if args.caa_styles is not None:
        active_styles = list(args.caa_styles)
    elif args.caa_mode == "metadata":
        active_styles = list(settings.caa_active_styles or ["scratchpad", "dead_end", "distractor", "jargon"])
    else:
        active_styles = ["distractor"]

    # Resolve layers specifically for the corresponding style(s) from config
    if getattr(args, "caa_layers", None) is not None:
        valid_layers = list(args.caa_layers)
    elif args.caa_mode == "metadata":
        valid_layers = settings.effective_caa_insertion_layers
    elif settings.caa_layer_strengths is not None:
        target_styles = active_styles if active_styles else ["distractor"]
        layers_set = set()
        for s in target_styles:
            if s in settings.caa_layer_strengths:
                layers_set.update(int(l) for l in settings.caa_layer_strengths[s].keys())
        valid_layers = sorted(layers_set) if layers_set else settings.effective_caa_insertion_layers
    elif settings.caa_insertion_layers is not None:
        valid_layers = settings.caa_insertion_layers
    else:
        valid_layers = None

    # Resolve strengths for corresponding style(s)
    if args.caa_strength is not None:
        effective_strength = args.caa_strength
    elif settings.caa_layer_strengths is not None:
        effective_strength = settings.caa_layer_strengths
    elif settings.caa_strength is not None:
        effective_strength = settings.caa_strength
    else:
        effective_strength = 5.0

    # In metadata mode, pre-calculate balanced style schedule across samples matching generate_traces.py
    style_assignments: list[str] = []
    if args.caa_mode == "metadata":
        import numpy as np
        from src.prf import hmac_seed_from_metadata

        _styles_pool = active_styles
        _style_weights_cfg = settings.caa_style_weights
        if _style_weights_cfg is not None:
            _sw_styles = [s for s in _styles_pool if s in _style_weights_cfg]
            _sw_vals = np.array([_style_weights_cfg[s] for s in _sw_styles], dtype=float)
        else:
            _sw_styles = list(_styles_pool)
            _sw_vals = np.ones(len(_sw_styles), dtype=float)
        _sw_vals /= _sw_vals.sum()
        _n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
        _p_active = float(settings.caa_p_active) if settings.caa_p_active is not None else 1.0
        _n_active = int(_n_batches * _p_active) if _p_active < 1.0 else _n_batches
        _exact = _sw_vals * _n_active
        _counts = np.floor(_exact).astype(int)
        _remainder = _n_active - _counts.sum()
        _top = np.argsort(_exact - _counts)[::-1][:_remainder]
        _counts[_top] += 1
        for _s, _c in zip(_sw_styles, _counts):
            style_assignments.extend([_s] * int(_c))
        _seed = hmac_seed_from_metadata({"user_id": getattr(settings, "caa_user_id", None) or "synthetic"})
        np.random.default_rng(_seed).shuffle(style_assignments)

    if backend == "vllm":
        # Disable vLLM v1 engine for CAA custom architectures so ModelRegistry resolution succeeds
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["CAA_BENCH_MODE"] = "1"
        from src.vllm_caa_inference import VLLMCAATraceGenerator

        print(f"[Style Steering] Initializing vLLM CAA Generator with {args.model}...")
        gen = VLLMCAATraceGenerator(
            settings=settings,
            model_id=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            caa_checkpoint=args.caa_checkpoint,
            batch_size=args.batch_size,
            strength=effective_strength,
            injection_layers=valid_layers,
            caa_active_styles=active_styles,
            normalize_per_layer=args.caa_normalize,
            relative_strength=args.caa_relative,
        )
        if args.caa_mode == "metadata":
            print(f"  [Style Steering] Mode: metadata (HMAC-PRF derivation across {active_styles})")
            print(f"  [Style Steering] Layers: {gen.insertion_layers} | Strengths: {effective_strength}")
        else:
            active_style = active_styles[0] if active_styles else "distractor"
            style_max_inj = gen._get_effective_max_tokens(active_style)
            style_str = gen._resolve_layer_strengths(active_style)
            print(f"  [Style Steering] Active CAA style: '{active_style}' | Layers: {gen.insertion_layers} | Strengths: {style_str} | Max injection tokens: {style_max_inj}")
            if active_style in gen.styles:
                gen.set_style(active_style)

        loop_throughputs_tokens: list[float] = []
        loop_throughputs_samples: list[float] = []
        all_example_stats: list[dict] = []
        total_tokens_all_loops = 0
        total_seconds_all_loops = 0.0
        total_generated_all_loops = 0

        for loop_idx in range(args.num_loops):
            if args.num_loops > 1:
                print(f"  [style_steering] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
            loop_tokens = 0
            loop_elapsed_excl = 0.0
            loop_elapsed_raw = 0.0
            loop_generated = 0

            for idx, prompt in enumerate(prompts):
                if args.caa_mode == "metadata":
                    metadata = {"user_id": getattr(settings, "caa_user_id", None) or "synthetic", "batch_id": idx}
                    if idx < len(style_assignments):
                        assigned_style = style_assignments[idx]
                        gen.caa_active_styles = [assigned_style]
                        gen.set_vector_from_metadata(metadata)
                        style_label = assigned_style
                    else:
                        gen.disable_steering()
                        style_label = "benign"
                else:
                    style_label = active_style

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_b0 = time.perf_counter()
                resp = gen.sample_batch([prompt])[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_b0

                toks = len(gen._tokenizer.encode(resp, add_special_tokens=False))
                ex_tps = toks / elapsed if elapsed > 0 else 0.0

                print(f"  [style_steering] example {idx+1}/{len(prompts)} ({style_label}): {toks} tokens in {elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
                all_example_stats.append({
                    "loop": loop_idx,
                    "index": idx,
                    "tokens": toks,
                    "elapsed": round(elapsed, 4),
                    "raw_elapsed": round(elapsed, 4),
                    "ttft": 0.0,
                    "tps": round(ex_tps, 2),
                })
                loop_tokens += toks
                loop_elapsed_excl += elapsed
                loop_elapsed_raw += elapsed
                loop_generated += 1

            loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
            loop_throughputs_tokens.append(loop_tps)
            loop_throughputs_samples.append(loop_sps)
            total_tokens_all_loops += loop_tokens
            total_seconds_all_loops += loop_elapsed_raw
            total_generated_all_loops += loop_generated

            print(f"  [style_steering] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s)", flush=True)

        peak_gpu_mem_mb = 0.0
        gpu_mem_at_end_mb = 0.0
        os.environ.pop("CAA_BENCH_MODE", None)

    else:
        # HF backend
        from src.caa_inference import HFCAATraceGenerator

        print(f"[Style Steering] Initializing HF CAA Generator with {args.model} (layers={valid_layers})...")
        os.environ["CAA_BENCH_MODE"] = "1"
        try:
            gen = HFCAATraceGenerator(
                settings=settings,
                model_id=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                caa_checkpoint=args.caa_checkpoint,
                batch_size=args.batch_size,
                strength=effective_strength,
                injection_layers=valid_layers,
                normalize_per_layer=args.caa_normalize,
                relative_strength=args.caa_relative,
            )
            if args.caa_mode == "metadata":
                print("[Style Steering] Mode: metadata (HMAC-PRF vector derivation matching pipeline_caa_metadata.sh)")
                print(f"  [Style Steering] Layers: {gen.insertion_layers} | Strengths: {effective_strength}")
            else:
                active_style = active_styles[0] if active_styles else "distractor"
                style_max_inj = gen._get_effective_max_tokens(active_style)
                style_str = gen._resolve_layer_strengths(active_style)
                print(f"  [Style Steering] Active CAA style: '{active_style}' | Layers: {gen.insertion_layers} | Strengths: {style_str} | Max injection tokens: {style_max_inj}")
                if active_style in gen.styles:
                    gen.set_style(active_style)

            # Hook _model.generate to inject FirstTokenTimer stopping criteria
            orig_hf_generate = gen._model.generate
            current_timer: list[FirstTokenTimer] = []

            def hooked_hf_generate(*g_args, **g_kwargs):
                timer = FirstTokenTimer()
                current_timer.clear()
                current_timer.append(timer)
                existing = g_kwargs.pop("stopping_criteria", None)
                crit = list(existing) + [timer] if existing else [timer]
                g_kwargs["stopping_criteria"] = StoppingCriteriaList(crit)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                timer.t_start = t_start
                out = orig_hf_generate(*g_args, **g_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.t_end = time.perf_counter()
                return out

            gen._model.generate = hooked_hf_generate

            loop_throughputs_tokens = []
            loop_throughputs_samples = []
            all_example_stats = []
            total_tokens_all_loops = 0
            total_seconds_all_loops = 0.0
            total_generated_all_loops = 0

            for loop_idx in range(args.num_loops):
                if args.num_loops > 1:
                    print(f"  [style_steering] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
                loop_tokens = 0
                loop_elapsed_excl = 0.0
                loop_elapsed_raw = 0.0
                loop_generated = 0

                for idx, prompt in enumerate(prompts):
                    if args.caa_mode == "metadata":
                        metadata = {"user_id": getattr(settings, "caa_user_id", None) or "synthetic", "batch_id": idx}
                        if idx < len(style_assignments):
                            assigned_style = style_assignments[idx]
                            gen.caa_active_styles = [assigned_style]
                            gen.set_vector_from_metadata(metadata)
                            style_label = assigned_style
                        else:
                            gen.disable_steering()
                            style_label = "benign"
                    else:
                        style_label = active_style

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_b0 = time.perf_counter()
                    resp = gen.sample_batch([prompt])[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t_b0

                    timer = current_timer[-1] if current_timer else None
                    ttft = (timer.first_token_time[0] - timer.t_start) if (timer and timer.first_token_time) else 0.0
                    if ttft > elapsed:
                        ttft = elapsed
                    t_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
                    toks = len(gen._tokenizer.encode(resp, add_special_tokens=False))
                    ex_tps = toks / t_excl if t_excl > 0 else 0.0

                    ttft_msg = f" (TTFT {ttft:.2f}s)" if (not args.no_ttft_exclude and ttft > 0) else ""
                    print(f"  [style_steering] example {idx+1}/{len(prompts)} ({style_label}): {toks} tokens in {t_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                    all_example_stats.append({
                        "loop": loop_idx,
                        "index": idx,
                        "tokens": toks,
                        "elapsed": round(t_excl, 4),
                        "raw_elapsed": round(elapsed, 4),
                        "ttft": round(ttft, 4),
                        "tps": round(ex_tps, 2),
                    })
                    loop_tokens += toks
                    loop_elapsed_excl += t_excl
                    loop_elapsed_raw += elapsed
                    loop_generated += 1

                loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
                loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
                loop_throughputs_tokens.append(loop_tps)
                loop_throughputs_samples.append(loop_sps)
                total_tokens_all_loops += loop_tokens
                total_seconds_all_loops += loop_elapsed_raw
                total_generated_all_loops += loop_generated

                time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
                print(f"  [style_steering] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

            peak_gpu_mem_mb = 0.0
            gpu_mem_at_end_mb = 0.0
        finally:
            os.environ.pop("CAA_BENCH_MODE", None)

    mean_tps = float(statistics.mean(loop_throughputs_tokens)) if loop_throughputs_tokens else 0.0
    std_tps = float(statistics.stdev(loop_throughputs_tokens)) if len(loop_throughputs_tokens) > 1 else 0.0
    mean_sps = float(statistics.mean(loop_throughputs_samples)) if loop_throughputs_samples else 0.0
    avg_latency = total_seconds_all_loops / max(total_generated_all_loops, 1)

    return BenchmarkResult(
        pipeline="style_steering",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens_all_loops,
        total_seconds=total_seconds_all_loops,
        throughput_tokens_per_sec=mean_tps,
        throughput_samples_per_sec=mean_sps,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        num_loops=args.num_loops,
        loop_throughputs_tokens=loop_throughputs_tokens,
        throughput_tokens_std=std_tps,
        ttft_excluded=not args.no_ttft_exclude,
        example_stats=all_example_stats,
    )


def run_ads(args: argparse.Namespace, prompts: list[str]) -> BenchmarkResult:
    """Run Antidistillation Sampling (ADS) with teacher + 2 perturbed proxy models."""
    backend = args.backend
    if backend == "vllm":
        print("[ADS] Warning: vLLM does not support ADS logits modification. Falling back to HuggingFace (hf) backend.")
        backend = "hf"

    print(f"[ADS] Loading Teacher ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    teacher.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"[ADS] Loading Proxy Student (+eps) ({args.proxy_student})...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.eval()
    student = CachedModelWrapper(student_model)

    print(f"[ADS] Loading Proxy Student (-eps) ({args.proxy_student})...")
    dstudent_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    dstudent_model.eval()
    dstudent = CachedModelWrapper(dstudent_model)

    # Resize proxy student embeddings to teacher tokenizer size so teacher input_ids are valid
    if len(tokenizer) > student.model.config.vocab_size:
        student.model.resize_token_embeddings(len(tokenizer))
        dstudent.model.resize_token_embeddings(len(tokenizer))

    # In-memory perturbation with precomputed gradients (NO disk writes!)
    print(f"[ADS] Applying in-memory gradient perturbations (eps={args.eps}, lam={args.lam}, tau={args.tau}) from {args.grad_path}...")
    grads = torch.load(args.grad_path, map_location="cpu", weights_only=False)

    model_params = {name.replace("module.", ""): param for name, param in student.model.named_parameters()}
    for grad_name, grad_tensor in grads.items():
        clean_name = grad_name.replace("module.", "")
        if clean_name in model_params:
            param = model_params[clean_name]
            if param.shape == grad_tensor.shape:
                param.data.add_(grad_tensor.to(param.device, dtype=param.dtype), alpha=args.eps)
            elif param.shape[0] >= grad_tensor.shape[0] and param.shape[1:] == grad_tensor.shape[1:]:
                param.data[: grad_tensor.shape[0]].add_(grad_tensor.to(param.device, dtype=param.dtype), alpha=args.eps)

    model_params = {name.replace("module.", ""): param for name, param in dstudent.model.named_parameters()}
    for grad_name, grad_tensor in grads.items():
        clean_name = grad_name.replace("module.", "")
        if clean_name in model_params:
            param = model_params[clean_name]
            if param.shape == grad_tensor.shape:
                param.data.add_(grad_tensor.to(param.device, dtype=param.dtype), alpha=-args.eps)
            elif param.shape[0] >= grad_tensor.shape[0] and param.shape[1:] == grad_tensor.shape[1:]:
                param.data[: grad_tensor.shape[0]].add_(grad_tensor.to(param.device, dtype=param.dtype), alpha=-args.eps)

    del grads
    gc.collect()
    torch.cuda.empty_cache()

    formatted_prompts = [format_chat_prompt(tokenizer, p, dataset=args.dataset) for p in prompts]
    teacher_device = next(teacher.parameters()).device

    torch.cuda.reset_peak_memory_stats()
    total = len(formatted_prompts)
    loop_throughputs_tokens = []
    loop_throughputs_samples = []
    all_example_stats = []
    total_tokens_all_loops = 0
    total_seconds_all_loops = 0.0
    total_generated_all_loops = 0

    for loop_idx in range(args.num_loops):
        if args.num_loops > 1:
            print(f"  [ads] === Loop {loop_idx + 1}/{args.num_loops} ===", flush=True)
        loop_tokens = 0
        loop_elapsed_excl = 0.0
        loop_elapsed_raw = 0.0
        loop_generated = 0

        for i in range(0, total, args.batch_size):
            batch = formatted_prompts[i : i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(teacher_device)
            prompt_len = inputs["input_ids"].shape[1]
            logits_proc = LogitsProcessorList([
                LogprobsModifier(
                    lam=args.lam,
                    eps=args.eps,
                    attention_mask=inputs["attention_mask"],
                    student=student,
                    dstudent=dstudent,
                )
            ])
            gen_out, elapsed, ttft = _generate_with_ttft_hf(
                model=teacher,
                inputs=inputs,
                tokenizer=tokenizer,
                gen_kwargs=dict(
                    max_new_tokens=args.max_new_tokens,
                    logits_processor=logits_proc,
                    renormalize_logits=True,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=args.tau > 0,
                    temperature=args.tau if args.tau > 0 else None,
                    top_p=args.top_p if args.tau > 0 else None,
                ),
            )
            if ttft > elapsed:
                ttft = elapsed
            batch_excl = max(elapsed - ttft, 0.0) if not args.no_ttft_exclude else elapsed
            ex_elapsed_excl = batch_excl if len(batch) == 1 else batch_excl / len(batch)
            ex_elapsed_raw = elapsed if len(batch) == 1 else elapsed / len(batch)
            ex_ttft = ttft if len(batch) == 1 else ttft / len(batch)

            for j in range(len(batch)):
                new_toks = gen_out[j, prompt_len:]
                if tokenizer.pad_token_id is not None:
                    tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
                else:
                    tok_count = new_toks.numel()
                ex_tps = tok_count / ex_elapsed_excl if ex_elapsed_excl > 0 else 0.0
                idx = i + j
                ttft_msg = f" (TTFT {ex_ttft:.2f}s)" if (not args.no_ttft_exclude and ex_ttft > 0) else ""
                print(f"  [ads] example {idx}: {tok_count} tokens in {ex_elapsed_excl:.2f}s{ttft_msg} → {ex_tps:.1f} tok/s", flush=True)
                all_example_stats.append({
                    "loop": loop_idx,
                    "index": idx,
                    "tokens": tok_count,
                    "elapsed": round(ex_elapsed_excl, 4),
                    "raw_elapsed": round(ex_elapsed_raw, 4),
                    "ttft": round(ex_ttft, 4),
                    "tps": round(ex_tps, 2),
                })
                loop_tokens += tok_count
            loop_elapsed_excl += batch_excl
            loop_elapsed_raw += elapsed
            loop_generated += len(batch)

        loop_tps = loop_tokens / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
        loop_sps = loop_generated / loop_elapsed_excl if loop_elapsed_excl > 0 else 0.0
        loop_throughputs_tokens.append(loop_tps)
        loop_throughputs_samples.append(loop_sps)
        total_tokens_all_loops += loop_tokens
        total_seconds_all_loops += loop_elapsed_raw
        total_generated_all_loops += loop_generated

        time_label = "excl. TTFT" if not args.no_ttft_exclude else "total"
        print(f"  [ads] Loop {loop_idx + 1}: {loop_tps:.2f} tok/s ({loop_tokens} tokens, {loop_elapsed_excl:.2f}s {time_label})", flush=True)

    peak_gpu_mem_mb = 0.0
    gpu_mem_at_end_mb = 0.0

    mean_tps = float(statistics.mean(loop_throughputs_tokens)) if loop_throughputs_tokens else 0.0
    std_tps = float(statistics.stdev(loop_throughputs_tokens)) if len(loop_throughputs_tokens) > 1 else 0.0
    mean_sps = float(statistics.mean(loop_throughputs_samples)) if loop_throughputs_samples else 0.0
    avg_latency = total_seconds_all_loops / max(total_generated_all_loops, 1)

    return BenchmarkResult(
        pipeline="ads",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens_all_loops,
        total_seconds=total_seconds_all_loops,
        throughput_tokens_per_sec=mean_tps,
        throughput_samples_per_sec=mean_sps,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        num_loops=args.num_loops,
        loop_throughputs_tokens=loop_throughputs_tokens,
        throughput_tokens_std=std_tps,
        ttft_excluded=not args.no_ttft_exclude,
        example_stats=all_example_stats,
    )


def run_pipeline_worker(pipeline_name: str, args: argparse.Namespace) -> BenchmarkResult:
    """Entrypoint for child worker executing one specific pipeline."""
    prompts = build_prompts(args.dataset, args.num_samples)
    if pipeline_name == "baseline":
        res = run_baseline(args, prompts)
    elif pipeline_name == "doge":
        res = run_doge(args, prompts)
    elif pipeline_name == "style_steering":
        res = run_style_steering(args, prompts)
    elif pipeline_name == "ads":
        res = run_ads(args, prompts)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    print(f"\nJSON_BENCHMARK_RESULT: {json.dumps(asdict(res))}", flush=True)
    return res


# ============================================================================
# SUMMARY FORMATTING
# ============================================================================

def print_summary(results: list[BenchmarkResult]) -> None:
    """Render the results table with overhead comparisons against baseline."""
    if not results:
        print("No benchmark results to display.")
        return

    # Find baseline result
    baseline = next((r for r in results if r.pipeline == "baseline"), None)

    # Box drawing table
    headers = [
        ("Pipeline", 18),
        ("Backend", 9),
        ("Batch Size", 11),
        ("Throughput", 20),
        ("Total Tokens", 14),
        ("Avg Latency", 15),
        ("Overhead vs", 16),
    ]
    sub_headers = [
        ("", 18),
        ("", 9),
        ("", 11),
        ("(tokens/sec)", 20),
        ("count", 14),
        ("(sec/sample)", 15),
        ("Baseline", 16),
    ]

    top_border = "╔" + "╦".join("═" * (w + 2) for _, w in headers) + "╗"
    header_line1 = "║" + "║".join(f" {h:<{w}} " for h, w in headers) + "║"
    header_line2 = "║" + "║".join(f" {h:<{w}} " for h, w in sub_headers) + "║"
    mid_border = "╠" + "╬".join("═" * (w + 2) for _, w in headers) + "╣"
    bot_border = "╚" + "╩".join("═" * (w + 2) for _, w in headers) + "╝"

    print("\n" + top_border)
    print(header_line1)
    print(header_line2)
    print(mid_border)

    for r in results:
        if baseline is not None and r.pipeline == "baseline":
            overhead_str = "— (ref)"
        elif baseline is not None:
            # Throughput ratio based on tokens/sec
            if r.throughput_tokens_per_sec > 0 and baseline.throughput_tokens_per_sec > 0:
                ratio = r.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec
                if ratio < 1.0:
                    speed_str = f"-{1.0 / ratio:.2f}×"
                else:
                    speed_str = f"+{ratio:.2f}×"
            else:
                speed_str = "N/A"
            overhead_str = speed_str
        else:
            overhead_str = "—"

        tps_str = f"{r.throughput_tokens_per_sec:.2f}"
        if r.num_loops > 1:
            tps_str += f" ±{r.throughput_tokens_std:.2f}"

        row = [
            f"{r.pipeline:<18}",
            f"{r.backend:<9}",
            f"{r.batch_size:<11}",
            f"{tps_str:<20}",
            f"{r.total_tokens:<14}",
            f"{r.avg_latency_sec:<15.3f}",
            f"{overhead_str:<16}",
        ]
        print("║" + "║".join(f" {c} " for c in row) + "║")

    print(bot_border)

    fastest = max(results, key=lambda x: x.throughput_tokens_per_sec)

    print(f"\nFastest pipeline: {fastest.pipeline} ({fastest.backend}) — {fastest.throughput_tokens_per_sec:.2f} tokens/sec ({fastest.throughput_samples_per_sec:.2f} samples/sec)")
    ttft_note = "excl. TTFT" if any(r.ttft_excluded for r in results) else "incl. TTFT"
    loops_note = f"{results[0].num_loops} loop(s)" if results else ""
    print(f"Notes: Throughput reported as mean ± std across {loops_note} ({ttft_note}).")
    print("Baseline overhead column: speedup multiplier vs baseline (negative = slower tok/s, positive = faster tok/s).\n")

    has_example_stats = any(r.example_stats for r in results)
    if has_example_stats:
        print("=" * 60)
        print("PER-EXAMPLE BREAKDOWN (tokens, time, TPS)")
        print("=" * 60)
        for r in results:
            if not r.example_stats:
                continue
            mean_tps = sum(e["tps"] for e in r.example_stats) / len(r.example_stats) if r.example_stats else 0.0
            print(f"\n{r.pipeline.upper()} ({r.backend})")
            print(f"  total tokens : {r.total_tokens}")
            print(f"  total time   : {r.total_seconds:.2f}s")
            print(f"  aggregate TPS: {r.throughput_tokens_per_sec:.1f} tok/s")
            if r.num_loops > 1:
                print(f"  loop TPS     : {[round(x, 2) for x in r.loop_throughputs_tokens]}")
            print(f"  mean per-ex  : {mean_tps:.1f} tok/s")
            print("  per-example breakdown:")
            for e in r.example_stats:
                loop_tag = f"[L{e['loop']+1}] " if (r.num_loops > 1 and "loop" in e) else ""
                ttft_tag = f" (TTFT {e['ttft']:.2f}s)" if e.get("ttft", 0) > 0 else ""
                print(f"    {loop_tag}example {e['index']}: {e['tokens']} tokens  {e['elapsed']:.2f}s{ttft_tag}  {e['tps']:.1f} tok/s")
        print("=" * 60 + "\n")


def save_results(results: list[BenchmarkResult], args: argparse.Namespace) -> Optional[Path]:
    """Save benchmark metrics and configuration settings to a JSON file."""
    if not results:
        return None

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        clean_model = args.model.replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/benchmark_{clean_model}_{args.backend}_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = next((r for r in results if r.pipeline == "baseline"), None)
    results_data = []
    for r in results:
        r_dict = asdict(r)
        if baseline is not None and r.pipeline != "baseline":
            if r.throughput_tokens_per_sec > 0 and baseline.throughput_tokens_per_sec > 0:
                speedup = round(r.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec, 4)
            else:
                speedup = None
            r_dict["overhead_speedup_vs_baseline"] = speedup
            r_dict["overhead_mem_diff_mb_vs_baseline"] = None
        else:
            r_dict["overhead_speedup_vs_baseline"] = 1.0 if baseline is not None else None
            r_dict["overhead_mem_diff_mb_vs_baseline"] = None
        results_data.append(r_dict)

    settings_dict = {
        "model": args.model,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "num_loops": args.num_loops,
        "no_ttft_exclude": args.no_ttft_exclude,
        "max_new_tokens": args.max_new_tokens,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "caa_checkpoint": args.caa_checkpoint,
        "caa_strength": args.caa_strength,
        "caa_styles": args.caa_styles,
        "caa_mode": args.caa_mode,
        "caa_normalize": args.caa_normalize,
        "caa_relative": args.caa_relative,
        "proxy_student": args.proxy_student,
        "grad_path": args.grad_path,
        "lam": args.lam,
        "eps": args.eps,
        "tau": args.tau,
        "doge_model": args.doge_model,
        "doge_coef": args.doge_coef,
        "doge_temp": args.doge_temp,
    }

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": settings_dict,
        "results": results_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Saved] Benchmark results and settings saved to: {output_path}")
    return output_path


# ============================================================================
# CLI & ORCHESTRATION
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark throughput and GPU memory for Style Steering, ADS, DOGe, and Baseline."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save benchmark results JSON (default: results/benchmark_{model}_{backend}_{timestamp}.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Teacher / baseline HuggingFace model ID (default: unsloth/gemma-3-12b-it, matching pipeline scripts)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "synthetic"],
        default="gsm8k",
        help="Prompt source: 'gsm8k', 'math', or 'synthetic' (default: gsm8k, matching pipeline scripts)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, matching TEACHER_TEMPERATURE and ADS tau)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling cutoff (default: 0.95, matching pipeline settings)",
    )

    # --- ADS Arguments (from scripts/pipeline_ads.sh & pipeline_generate_ads.sh) ---
    parser.add_argument(
        "--proxy-student",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Proxy student model ID for ADS (default: Qwen/Qwen2.5-1.5B-Instruct, matching pipeline_generate_ads.sh)",
    )
    parser.add_argument(
        "--grad-path",
        type=str,
        default=None,
        help="Path to precomputed gradient .pt file for ADS (default: auto-detected from experiments_{dataset}/student_grads.pt)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="ADS temperature parameter tau (default: 0.7, from pipeline_ads.sh)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.015,
        help="ADS antidistillation strength parameter lambda (default: 0.017, from pipeline_ads.sh)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Finite difference step size eps for ADS (default: 0.01, from pipeline_ads.sh)",
    )

    # --- DOGe Arguments (from scripts/pipeline_doge.sh & pipeline_generate_doge.sh) ---
    parser.add_argument(
        "--doge-model",
        type=str,
        default=None,
        help="DOGe fine-tuned checkpoint path/ID (default: auto-detected from other_works/DOGe/outputs/gemma12b-doge-coef{doge_coef})",
    )
    parser.add_argument(
        "--doge-coef",
        type=str,
        default="1e-4",
        help="DOGe distillation coefficient (default: 1e-4, from pipeline_doge.sh)",
    )
    parser.add_argument(
        "--doge-temp",
        type=float,
        default=2.0,
        help="DOGe temperature (default: 2.0, from pipeline_doge.sh)",
    )

    # --- Style Steering (CAA) Arguments (from scripts/pipeline_caa_styles.sh & configs) ---
    parser.add_argument(
        "--caa-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained CAA steering .pt file (default: auto-detected from models/steering_vectors_caa_{backend}/steering_vectors_{model}.pt)",
    )
    parser.add_argument(
        "--caa-strength",
        type=float,
        default=None,
        help="CAA steering strength (default: None, falls back to CAA_LAYER_STRENGTHS or 5.0)",
    )
    parser.add_argument(
        "--caa-styles",
        type=str,
        nargs="+",
        default=["distractor"],
        help="CAA styles list (default: scratchpad dead_end jargon distractor, from pipeline_caa_styles.sh)",
    )
    parser.add_argument(
        "--caa-mode",
        choices=["styles", "metadata"],
        default="styles",
        help="CAA steering mode (default: styles, from pipeline_caa_styles.sh)",
    )
    parser.add_argument(
        "--caa-normalize",
        action="store_true",
        default=True,
        help="Normalize each per-layer steering vector to unit norm (default: True, from configs/gemma12b.qwen.gsm8k.env)",
    )
    parser.add_argument(
        "--caa-relative",
        action="store_true",
        default=True,
        help="Scale each injection by per-token residual norm (default: True, from configs/gemma12b.qwen.gsm8k.env)",
    )
    parser.add_argument(
        "--caa-layers",
        type=int,
        nargs="+",
        default=None,
        help="CAA injection layer indices (default: None, auto-resolved from CAA_LAYER_STRENGTHS in config for corresponding style)",
    )
    parser.add_argument(
        "--caa-max-injection-tokens",
        type=int,
        default=None,
        help="Max generated tokens to inject CAA steering vector into (default: None, auto-resolved from CAA_MAX_INJECTION_TOKENS in config for corresponding style)",
    )

    # --- Common Generation Settings ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 4)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of prompts to benchmark (default: 16)",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=5,
        help="Number of times to repeat the full sample sweep per pipeline. "
             "Throughput is reported as mean ± std across loops (default: 1).",
    )
    parser.add_argument(
        "--no-ttft-exclude",
        action="store_true",
        default=False,
        help="Disable time-to-first-token exclusion (include TTFT in elapsed). "
             "Default: TTFT is excluded from throughput measurements.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' or 'vllm' (default: hf). ADS is skipped when backend is 'vllm'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per prompt (default: 2048, from TEACHER_MAX_NEW_TOKENS and pipeline scripts)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        choices=["baseline", "style_steering", "ads", "doge"],
        default=["baseline", "doge", "style_steering", "ads"],
        help="Pipelines to benchmark (default: baseline doge style_steering ads)",
    )
    parser.add_argument(
        "--config",
        "--env-file",
        dest="config",
        type=str,
        default=None,
        help="Path to .env configuration file (e.g., configs/gemma12b.qwen.gsm8k.env) to load CAA_LAYER_STRENGTHS and other settings.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run pipelines sequentially in the same process rather than separate child processes.",
    )
    parser.add_argument(
        "--_worker_pipeline",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Internal worker dispatch flag
    )

    args = parser.parse_args()

    if args.config:
        resolved_config = str(Path(args.config).resolve())
        os.environ["ENV_FILE"] = resolved_config
    elif "ENV_FILE" not in os.environ:
        candidate_env = f"configs/gemma12b.qwen.{args.dataset}.env"
        if os.path.exists(candidate_env):
            resolved_config = str(Path(candidate_env).resolve())
            os.environ["ENV_FILE"] = resolved_config
            args.config = resolved_config
        elif os.path.exists("configs/gemma12b.qwen.gsm8k.env"):
            resolved_config = str(Path("configs/gemma12b.qwen.gsm8k.env").resolve())
            os.environ["ENV_FILE"] = resolved_config
            args.config = resolved_config

    # Smart auto-detection of default checkpoints from scripts/pipeline_*.sh
    if args.doge_model is None:
        candidate_doge = f"other_works/DOGe/outputs/gemma12b-doge-coef{args.doge_coef}"
        if os.path.exists(candidate_doge):
            args.doge_model = candidate_doge

    clean_name = args.model.split("/")[-1]
    backend_caa_dir = f"models/steering_vectors_caa_{args.backend}"
    if args.caa_checkpoint is None:
        candidate_caa = f"{backend_caa_dir}/steering_vectors_{clean_name}.pt"
        if os.path.exists(candidate_caa):
            args.caa_checkpoint = candidate_caa
        elif os.path.exists(f"{backend_caa_dir}/steering_vectors.pt"):
            args.caa_checkpoint = f"{backend_caa_dir}/steering_vectors.pt"
        else:
            # Fallback across backends if not found in current backend dir
            alt_backend = "vllm" if args.backend == "hf" else "hf"
            alt_dir = f"models/steering_vectors_caa_{alt_backend}"
            candidate_alt = f"{alt_dir}/steering_vectors_{clean_name}.pt"
            if os.path.exists(candidate_alt):
                args.caa_checkpoint = candidate_alt
            elif os.path.exists(f"{alt_dir}/steering_vectors.pt"):
                args.caa_checkpoint = f"{alt_dir}/steering_vectors.pt"
            else:
                args.caa_checkpoint = candidate_caa
    elif not os.path.exists(args.caa_checkpoint):
        candidate_caa = f"{backend_caa_dir}/steering_vectors_{clean_name}.pt"
        if os.path.exists(candidate_caa):
            args.caa_checkpoint = candidate_caa
        elif os.path.exists(f"{backend_caa_dir}/steering_vectors.pt"):
            args.caa_checkpoint = f"{backend_caa_dir}/steering_vectors.pt"

    if args.grad_path is None:
        candidate_grad = f"other_works/antidistillation-sampling/experiments_{args.dataset}/student_grads.pt"
        if os.path.exists(candidate_grad):
            args.grad_path = candidate_grad
        else:
            candidate_gsm8k = "other_works/antidistillation-sampling/experiments_gsm8k/student_grads.pt"
            if os.path.exists(candidate_gsm8k):
                args.grad_path = candidate_gsm8k

    return args


def main():
    args = parse_args()

    # If dispatched as a worker child process, execute the single pipeline and exit
    if args._worker_pipeline:
        run_pipeline_worker(args._worker_pipeline, args)
        return

    # Check GPU availability
    if not torch.cuda.is_available():
        print("[Error] CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    print("================================================================================")
    print("      Throughput & GPU Memory Benchmark: Baseline vs CAA vs ADS vs DOGe        ")
    print("================================================================================")
    print(f"Model:           {args.model}")
    print(f"Backend:         {args.backend}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Num Samples:     {args.num_samples}")
    print(f"Num Loops:       {args.num_loops}")
    print(f"TTFT Excluded:   {not args.no_ttft_exclude}")
    print(f"Max New Tokens:  {args.max_new_tokens}")
    print(f"Dataset:         {args.dataset}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-P:           {args.top_p}")
    print("Pipeline Settings from Scripts:")
    print(f"  [Style Steering] Checkpoint: {args.caa_checkpoint} | Styles: {args.caa_styles} | Strength: {args.caa_strength} | Mode: {args.caa_mode} | Normalize: {args.caa_normalize} | Relative: {args.caa_relative}")
    print(f"  [ADS]            Proxy: {args.proxy_student} | Grad: {args.grad_path} | Tau: {args.tau} | Lam: {args.lam} | Eps: {args.eps}")
    print(f"  [DOGe]           Model: {args.doge_model} | Coef: {args.doge_coef} | Temp: {args.doge_temp}")
    print("--------------------------------------------------------------------------------")

    # Determine pipelines to run
    pipelines_to_run: list[tuple[str, str]] = []
    selected_pipelines = set(args.pipelines)

    # 1. Baseline
    if "baseline" in selected_pipelines:
        pipelines_to_run.append(("baseline", args.backend))

    # 2. DOGe
    if "doge" in selected_pipelines:
        pipelines_to_run.append(("doge", args.backend))

    # 3. Style Steering (CAA)
    if "style_steering" in selected_pipelines:
        if args.caa_checkpoint:
            if not os.path.exists(args.caa_checkpoint):
                print(f"[Style Steering] Skipping: CAA checkpoint not found at '{args.caa_checkpoint}'")
            else:
                # Check dimension compatibility with model
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
                    model_hidden_dim = getattr(cfg, "hidden_size", None)
                    ckpt = torch.load(args.caa_checkpoint, map_location="cpu", weights_only=False)
                    first_style = list(ckpt.get("steering_vectors", {}).keys())[0]
                    first_vec = ckpt["steering_vectors"][first_style][0]
                    sv_dim = len(first_vec) if hasattr(first_vec, "__len__") else first_vec.shape[-1]
                    if model_hidden_dim and sv_dim != model_hidden_dim:
                        print(f"[Style Steering] Skipping: Vector dimension ({sv_dim}) does not match model hidden size ({model_hidden_dim}). Use a matching --caa-checkpoint.")
                    else:
                        pipelines_to_run.append(("style_steering", args.backend))
                except Exception:
                    pipelines_to_run.append(("style_steering", args.backend))
        else:
            print("[Style Steering] Skipping: --caa-checkpoint not provided.")

    # 4. ADS
    if "ads" in selected_pipelines:
        if args.backend == "vllm":
            print("[ADS] Skipping: ADS does not support vLLM backend (requires PyTorch/HF logits processor).")
        elif not args.proxy_student:
            print("[ADS] Skipping: --proxy-student not provided.")
        elif not args.grad_path or not os.path.exists(args.grad_path):
            print(f"[ADS] Skipping: Gradient file not found: '{args.grad_path}'")
        else:
            pipelines_to_run.append(("ads", args.backend))

    print(f"Pipelines queued ({len(pipelines_to_run)}): {[p[0] for p in pipelines_to_run]}\n")

    results: list[BenchmarkResult] = []

    for pipeline_name, pipeline_backend in pipelines_to_run:
        print(f"\n>>> Running Pipeline: {pipeline_name.upper()} (backend={pipeline_backend}) <<<", flush=True)

        if args.in_process:
            prompts = build_prompts(args.dataset, args.num_samples)
            if pipeline_name == "baseline":
                res = run_baseline(args, prompts)
            elif pipeline_name == "doge":
                res = run_doge(args, prompts)
            elif pipeline_name == "style_steering":
                res = run_style_steering(args, prompts)
            elif pipeline_name == "ads":
                res = run_ads(args, prompts)
            else:
                continue
            results.append(res)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # Build child command preserving user arguments
            worker_cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--_worker_pipeline",
                pipeline_name,
            ]

            # Forward relevant arguments
            forward_flags = [
                ("--model", args.model),
                ("--batch-size", str(args.batch_size)),
                ("--num-samples", str(args.num_samples)),
                ("--num-loops", str(args.num_loops)),
                ("--backend", pipeline_backend),
                ("--max-new-tokens", str(args.max_new_tokens)),
                ("--dataset", args.dataset),
                ("--temperature", str(args.temperature)),
                ("--top-p", str(args.top_p)),
                ("--tau", str(args.tau)),
                ("--lam", str(args.lam)),
                ("--eps", str(args.eps)),
                ("--doge-coef", str(args.doge_coef)),
                ("--doge-temp", str(args.doge_temp)),
                ("--caa-mode", args.caa_mode),
            ]
            if args.no_ttft_exclude:
                forward_flags.append(("--no-ttft-exclude",))
            if args.caa_strength is not None:
                forward_flags.append(("--caa-strength", str(args.caa_strength)))
            if args.proxy_student:
                forward_flags.append(("--proxy-student", args.proxy_student))
            if args.grad_path:
                forward_flags.append(("--grad-path", args.grad_path))
            if args.caa_checkpoint:
                forward_flags.append(("--caa-checkpoint", args.caa_checkpoint))
            if args.doge_model:
                forward_flags.append(("--doge-model", args.doge_model))
            if args.caa_styles:
                forward_flags.append(("--caa-styles", *args.caa_styles))
            if args.caa_normalize:
                forward_flags.append(("--caa-normalize",))
            if args.caa_relative:
                forward_flags.append(("--caa-relative",))
            if getattr(args, "caa_layers", None):
                forward_flags.append(("--caa-layers", *[str(l) for l in args.caa_layers]))
            if getattr(args, "caa_max_injection_tokens", None) is not None:
                forward_flags.append(("--caa-max-injection-tokens", str(args.caa_max_injection_tokens)))
            if args.config:
                forward_flags.append(("--config", args.config))

            for item in forward_flags:
                flag = item[0]
                vals = item[1:]
                worker_cmd.append(flag)
                worker_cmd.extend(vals)

            worker_env = os.environ.copy()
            worker_env["PYTHONUNBUFFERED"] = "1"
            if pipeline_name == "style_steering":
                worker_env["VLLM_USE_V1"] = "0"
                worker_env["CAA_BENCH_MODE"] = "1"

            proc = subprocess.Popen(
                worker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=worker_env,
            )

            result_obj: Optional[BenchmarkResult] = None
            for line in proc.stdout:
                if "JSON_BENCHMARK_RESULT:" in line:
                    try:
                        raw_json = line.split("JSON_BENCHMARK_RESULT:", 1)[1].strip()
                        data = json.loads(raw_json)
                        result_obj = BenchmarkResult(**data)
                    except Exception as err:
                        print(f"[Warning] Error parsing worker JSON: {err}")
                else:
                    print(line, end="", flush=True)

            proc.wait()
            if proc.returncode != 0:
                print(f"[Error] Pipeline '{pipeline_name}' failed with return code {proc.returncode}")
            elif result_obj is not None:
                results.append(result_obj)
            else:
                print(f"[Warning] Pipeline '{pipeline_name}' completed but did not emit benchmark metrics.")

    # Display final comparative table
    print_summary(results)

    # Save benchmark metrics and settings to JSON
    save_results(results, args)


if __name__ == "__main__":
    main()
