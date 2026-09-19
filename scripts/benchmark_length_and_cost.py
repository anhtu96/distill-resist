#!/usr/bin/env python
"""Benchmark: Average Total Output Length & Wall-Clock Cost Across Pipelines.

Compares:
1. Baseline (plain teacher generation)
2. Style Steering (residual-stream steering vectors via CAA)
3. DOGe (vanilla generation on fine-tuned checkpoint)
4. ADS (Antidistillation Sampling with proxy student perturbed by gradients)

Default backends:
- Baseline: vLLM
- Style Steering: vLLM
- DOGe: vLLM
- ADS: HuggingFace (HF)

Calculates:
- Output length metrics: mean output tokens, min/max, standard deviation,
  total tokens, character/word length, and percentage length delta vs baseline.
- Wall-clock cost metrics: total wall-clock elapsed time (seconds),
  average latency per sample (seconds/sample), throughput (tokens/sec),
  cost ratio vs baseline (T_pipe / T_base), percentage cost overhead vs baseline,
  and wall-clock seconds per 1,000 generated tokens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import gc
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
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
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExampleStat:
    index: int
    prompt_preview: str
    output_preview: str
    output_tokens: int
    output_chars: int
    output_words: int
    elapsed_sec: float
    tokens_per_sec: float


def make_preview(text: str, max_chars: int = 300) -> str:
    """Format single-line preview of text up to max_chars."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "..."


@dataclass
class PipelineResult:
    pipeline: str  # "baseline" | "style_steering" | "doge" | "ads"
    backend: str   # "vllm" | "hf"
    num_samples: int
    batch_size: int
    # Length metrics
    total_output_tokens: int
    avg_output_tokens: float
    min_output_tokens: int
    max_output_tokens: int
    std_output_tokens: float
    avg_output_chars: float
    avg_output_words: float
    # Wall-clock cost metrics
    total_wall_clock_sec: float
    avg_latency_sec: float             # seconds per sample
    throughput_tokens_per_sec: float   # total_tokens / total_wall_clock_sec
    throughput_samples_per_sec: float  # num_samples / total_wall_clock_sec
    cost_sec_per_1k_tokens: float      # (total_wall_clock_sec / total_output_tokens) * 1000
    # Comparison vs Baseline (filled by master)
    length_diff_tokens_vs_baseline: Optional[float] = None
    length_ratio_vs_baseline: Optional[float] = None
    length_pct_diff_vs_baseline: Optional[float] = None
    cost_ratio_vs_baseline: Optional[float] = None
    cost_pct_overhead_vs_baseline: Optional[float] = None
    # Peak GPU memory
    peak_gpu_mem_mb: float = 0.0
    # Detailed per-example stats
    examples: list[dict] = field(default_factory=list)


# ============================================================================
# GPU MEMORY MONITOR
# ============================================================================

def get_gpu_memory_used_mb() -> float:
    """Return currently allocated GPU memory in MB via PyTorch, or 0.0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


class GPUMemoryMonitor:
    """Background thread polling max GPU memory across PyTorch and nvidia-smi."""

    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self.peak_mem_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _monitor(self):
        while not self._stop_event.is_set():
            if torch.cuda.is_available():
                torch_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch_mb > self.peak_mem_mb:
                    self.peak_mem_mb = torch_mb
            time.sleep(self.interval)

    def start(self):
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        self.peak_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        torch_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch_mb > self.peak_mem_mb:
            self.peak_mem_mb = torch_mb
        return self.peak_mem_mb


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
# ADS COMPONENTS (KV-CACHE OPTIMIZED LOGITS PROCESSOR)
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

        if ad_term.shape[-1] != scores.shape[-1]:
            if ad_term.shape[-1] < scores.shape[-1]:
                ad_term = F.pad(ad_term, (0, scores.shape[-1] - ad_term.shape[-1]), value=0.0)
            else:
                ad_term = ad_term[:, : scores.shape[-1]]

        return scores.float() + ad_term


# ============================================================================
# STATS HELPER
# ============================================================================

def _compute_pipeline_result(
    pipeline_name: str,
    backend: str,
    num_samples: int,
    batch_size: int,
    total_wall_clock_sec: float,
    peak_gpu_mem_mb: float,
    example_stats: list[ExampleStat],
) -> PipelineResult:
    """Aggregate per-example statistics into a PipelineResult."""
    if not example_stats:
        return PipelineResult(
            pipeline=pipeline_name,
            backend=backend,
            num_samples=num_samples,
            batch_size=batch_size,
            total_output_tokens=0,
            avg_output_tokens=0.0,
            min_output_tokens=0,
            max_output_tokens=0,
            std_output_tokens=0.0,
            avg_output_chars=0.0,
            avg_output_words=0.0,
            total_wall_clock_sec=total_wall_clock_sec,
            avg_latency_sec=0.0,
            throughput_tokens_per_sec=0.0,
            throughput_samples_per_sec=0.0,
            cost_sec_per_1k_tokens=0.0,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            examples=[],
        )

    token_counts = [e.output_tokens for e in example_stats]
    char_counts = [e.output_chars for e in example_stats]
    word_counts = [e.output_words for e in example_stats]

    total_tokens = sum(token_counts)
    avg_tokens = float(statistics.mean(token_counts))
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    std_tokens = float(statistics.stdev(token_counts)) if len(token_counts) > 1 else 0.0

    avg_chars = float(statistics.mean(char_counts))
    avg_words = float(statistics.mean(word_counts))

    avg_latency = total_wall_clock_sec / max(num_samples, 1)
    tok_throughput = total_tokens / max(total_wall_clock_sec, 1e-6)
    sps_throughput = num_samples / max(total_wall_clock_sec, 1e-6)
    cost_per_1k = (total_wall_clock_sec / max(total_tokens, 1)) * 1000.0

    return PipelineResult(
        pipeline=pipeline_name,
        backend=backend,
        num_samples=num_samples,
        batch_size=batch_size,
        total_output_tokens=total_tokens,
        avg_output_tokens=avg_tokens,
        min_output_tokens=min_tokens,
        max_output_tokens=max_tokens,
        std_output_tokens=std_tokens,
        avg_output_chars=avg_chars,
        avg_output_words=avg_words,
        total_wall_clock_sec=total_wall_clock_sec,
        avg_latency_sec=avg_latency,
        throughput_tokens_per_sec=tok_throughput,
        throughput_samples_per_sec=sps_throughput,
        cost_sec_per_1k_tokens=cost_per_1k,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        examples=[asdict(e) for e in example_stats],
    )


# ============================================================================
# PIPELINE RUNNERS
# ============================================================================

def run_baseline(args: argparse.Namespace, prompts: list[str]) -> PipelineResult:
    """Run Baseline pipeline (default backend: vllm)."""
    model_id = args.model
    backend = args.resolved_backends["baseline"]
    print(f"\n[Baseline] Starting baseline generation on {len(prompts)} prompts (backend={backend}, model={model_id})...")

    monitor = GPUMemoryMonitor()
    monitor.start()
    example_stats: list[ExampleStat] = []

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
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization or 0.8,
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

        for idx, prompt in enumerate(formatted_prompts):
            t_b0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_b0

            out = outputs[0] if outputs else None
            text = out.outputs[0].text if (out and out.outputs) else ""
            toks = len(out.outputs[0].token_ids) if (out and out.outputs) else len(tokenizer.encode(text, add_special_tokens=False))
            chars = len(text)
            words = len(text.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompts[idx], max_chars=60)
            o_preview = make_preview(text, max_chars=preview_len)
            print(f"  [baseline] example {idx+1}/{len(prompts)}: {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

        for idx, prompt in enumerate(formatted_prompts):
            inputs = tokenizer([prompt], return_tensors="pt").to(model_device)
            prompt_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_b0 = time.perf_counter()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_b0

            new_toks = gen_out[0, prompt_len:]
            if tokenizer.pad_token_id is not None:
                toks = (new_toks != tokenizer.pad_token_id).sum().item()
            else:
                toks = new_toks.numel()
            text = tokenizer.decode(new_toks, skip_special_tokens=True)
            chars = len(text)
            words = len(text.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompts[idx], max_chars=60)
            o_preview = make_preview(text, max_chars=preview_len)
            print(f"  [baseline] example {idx+1}/{len(prompts)}: {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total

    peak_gpu = monitor.stop()
    return _compute_pipeline_result(
        "baseline", backend, len(prompts), args.batch_size, total_wall_clock, peak_gpu, example_stats
    )


def run_doge(args: argparse.Namespace, prompts: list[str]) -> PipelineResult:
    """Run DOGe pipeline (default backend: vllm)."""
    model_id = args.doge_model if args.doge_model else args.model
    backend = args.resolved_backends["doge"]
    if not args.doge_model:
        print(f"\n[DOGe] --doge-model not provided; using teacher model ({model_id}) as proxy.")
    else:
        print(f"\n[DOGe] Using fine-tuned checkpoint: {model_id} (backend={backend}, coef={args.doge_coef}, temp={args.doge_temp})...")

    monitor = GPUMemoryMonitor()
    monitor.start()
    example_stats: list[ExampleStat] = []

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
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization or 0.8,
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

        for idx, prompt in enumerate(formatted_prompts):
            t_b0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_b0

            out = outputs[0] if outputs else None
            text = out.outputs[0].text if (out and out.outputs) else ""
            toks = len(out.outputs[0].token_ids) if (out and out.outputs) else len(tokenizer.encode(text, add_special_tokens=False))
            chars = len(text)
            words = len(text.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompts[idx], max_chars=60)
            o_preview = make_preview(text, max_chars=preview_len)
            print(f"  [doge] example {idx+1}/{len(prompts)}: {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

        for idx, prompt in enumerate(formatted_prompts):
            inputs = tokenizer([prompt], return_tensors="pt").to(model_device)
            prompt_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_b0 = time.perf_counter()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_b0

            new_toks = gen_out[0, prompt_len:]
            if tokenizer.pad_token_id is not None:
                toks = (new_toks != tokenizer.pad_token_id).sum().item()
            else:
                toks = new_toks.numel()
            text = tokenizer.decode(new_toks, skip_special_tokens=True)
            chars = len(text)
            words = len(text.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompts[idx], max_chars=60)
            o_preview = make_preview(text, max_chars=preview_len)
            print(f"  [doge] example {idx+1}/{len(prompts)}: {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total

    peak_gpu = monitor.stop()
    return _compute_pipeline_result(
        "doge", backend, len(prompts), args.batch_size, total_wall_clock, peak_gpu, example_stats
    )


def run_style_steering(args: argparse.Namespace, prompts: list[str]) -> PipelineResult:
    """Run Style Steering pipeline with CAA hooks (default backend: vllm)."""
    from src.config import Settings
    settings = Settings()
    backend = args.resolved_backends["style_steering"]
    print(f"\n[Style Steering] Starting CAA generation on {len(prompts)} prompts (backend={backend}, model={args.model})...")

    if args.caa_strength is not None:
        effective_strength = args.caa_strength
        valid_layers = None
    elif settings.caa_layer_strengths is not None:
        effective_strength = settings.caa_layer_strengths
        valid_layers = settings.effective_caa_insertion_layers
    elif settings.caa_insertion_layers is not None and isinstance(settings.caa_strength, (list, tuple)):
        effective_strength = settings.caa_strength
        valid_layers = settings.caa_insertion_layers
    else:
        effective_strength = 5.0
        valid_layers = None

    # Resolve active styles
    if args.caa_styles is not None:
        active_styles = list(args.caa_styles)
    elif args.caa_mode == "metadata":
        active_styles = list(settings.caa_active_styles or ["scratchpad", "dead_end", "distractor", "jargon"])
    else:
        active_styles = ["distractor"]

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

    monitor = GPUMemoryMonitor()
    example_stats: list[ExampleStat] = []

    if backend == "vllm":
        # Disable vLLM v1 engine for CAA custom architectures so ModelRegistry resolution succeeds
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["CAA_BENCH_MODE"] = "1"
        from src.vllm_caa_inference import VLLMCAATraceGenerator

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
        else:
            active_style = active_styles[0] if active_styles else "distractor"
            if active_style in gen.styles:
                print(f"  [Style Steering] Active CAA style: '{active_style}'")
                gen.set_style(active_style)

        monitor.start()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

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
            chars = len(resp)
            words = len(resp.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompt, max_chars=60)
            o_preview = make_preview(resp, max_chars=preview_len)
            print(f"  [style_steering] example {idx+1}/{len(prompts)} ({style_label}): {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total
        os.environ.pop("CAA_BENCH_MODE", None)

    else:
        # HF backend
        from src.caa_inference import HFCAATraceGenerator

        os.environ["CAA_BENCH_MODE"] = "1"
        gen = HFCAATraceGenerator(
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
        else:
            active_style = active_styles[0] if active_styles else "distractor"
            if active_style in gen.styles:
                print(f"  [Style Steering] Active CAA style: '{active_style}'")
                gen.set_style(active_style)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_total = time.perf_counter()

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
            chars = len(resp)
            words = len(resp.split())
            tps = toks / elapsed if elapsed > 0 else 0.0

            preview_len = getattr(args, "preview_chars", 300)
            p_preview = make_preview(prompt, max_chars=60)
            o_preview = make_preview(resp, max_chars=preview_len)
            print(f"  [style_steering] example {idx+1}/{len(prompts)} ({style_label}): {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

            example_stats.append(ExampleStat(
                index=idx,
                prompt_preview=p_preview,
                output_preview=o_preview,
                output_tokens=toks,
                output_chars=chars,
                output_words=words,
                elapsed_sec=round(elapsed, 4),
                tokens_per_sec=round(tps, 2),
            ))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_wall_clock = time.perf_counter() - t_start_total
        os.environ.pop("CAA_BENCH_MODE", None)

    peak_gpu = monitor.stop()
    return _compute_pipeline_result(
        "style_steering", backend, len(prompts), args.batch_size, total_wall_clock, peak_gpu, example_stats
    )


def run_ads(args: argparse.Namespace, prompts: list[str]) -> PipelineResult:
    """Run ADS pipeline with perturbed proxy students (default backend: hf)."""
    backend = args.resolved_backends["ads"]
    if backend == "vllm":
        print("[ADS] Warning: vLLM does not support ADS logits modification. Falling back to HuggingFace (hf).")
        backend = "hf"

    print(f"\n[ADS] Loading Teacher ({args.model}) and perturbed Proxy Students ({args.proxy_student})...")
    monitor = GPUMemoryMonitor()
    monitor.start()

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

    student_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.eval()
    student = CachedModelWrapper(student_model)

    dstudent_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    dstudent_model.eval()
    dstudent = CachedModelWrapper(dstudent_model)

    if len(tokenizer) > student.model.config.vocab_size:
        student.model.resize_token_embeddings(len(tokenizer))
        dstudent.model.resize_token_embeddings(len(tokenizer))

    print(f"  [ADS] Applying gradient perturbations (eps={args.eps}, lam={args.lam}, tau={args.tau}) from {args.grad_path}...")
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
    example_stats: list[ExampleStat] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start_total = time.perf_counter()

    for idx, prompt in enumerate(formatted_prompts):
        inputs = tokenizer([prompt], return_tensors="pt").to(teacher_device)
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_b0 = time.perf_counter()
        with torch.no_grad():
            gen_out = teacher.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                logits_processor=logits_proc,
                renormalize_logits=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=args.tau > 0,
                temperature=args.tau if args.tau > 0 else None,
                top_p=args.top_p if args.tau > 0 else None,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_b0

        new_toks = gen_out[0, prompt_len:]
        if tokenizer.pad_token_id is not None:
            toks = (new_toks != tokenizer.pad_token_id).sum().item()
        else:
            toks = new_toks.numel()
        text = tokenizer.decode(new_toks, skip_special_tokens=True)
        chars = len(text)
        words = len(text.split())
        tps = toks / elapsed if elapsed > 0 else 0.0

        preview_len = getattr(args, "preview_chars", 300)
        p_preview = make_preview(prompts[idx], max_chars=60)
        o_preview = make_preview(text, max_chars=preview_len)
        print(f"  [ads] example {idx+1}/{len(prompts)}: {toks} tokens ({words} words) in {elapsed:.2f}s ({tps:.1f} tok/s) | output: \"{o_preview}\"", flush=True)

        example_stats.append(ExampleStat(
            index=idx,
            prompt_preview=p_preview,
            output_preview=o_preview,
            output_tokens=toks,
            output_chars=chars,
            output_words=words,
            elapsed_sec=round(elapsed, 4),
            tokens_per_sec=round(tps, 2),
        ))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_wall_clock = time.perf_counter() - t_start_total

    peak_gpu = monitor.stop()
    return _compute_pipeline_result(
        "ads", backend, len(prompts), args.batch_size, total_wall_clock, peak_gpu, example_stats
    )


# ============================================================================
# WORKER DISPATCH
# ============================================================================

def run_pipeline_worker(pipeline_name: str, args: argparse.Namespace) -> PipelineResult:
    """Execute a single pipeline in an isolated worker process and output JSON."""
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
# SUMMARY & PERSISTENCE
# ============================================================================

def compute_comparisons(results: list[PipelineResult]) -> None:
    """Compute relative output length and wall-clock cost comparisons against Baseline."""
    baseline = next((r for r in results if r.pipeline == "baseline"), None)
    if baseline is None:
        return

    for r in results:
        # Output length comparison
        diff_tok = r.avg_output_tokens - baseline.avg_output_tokens
        r.length_diff_tokens_vs_baseline = round(diff_tok, 2)
        if baseline.avg_output_tokens > 0:
            ratio_len = r.avg_output_tokens / baseline.avg_output_tokens
            r.length_ratio_vs_baseline = round(ratio_len, 4)
            r.length_pct_diff_vs_baseline = round((ratio_len - 1.0) * 100.0, 2)
        else:
            r.length_ratio_vs_baseline = 1.0
            r.length_pct_diff_vs_baseline = 0.0

        # Wall-clock cost comparison
        if baseline.total_wall_clock_sec > 0:
            cost_ratio = r.total_wall_clock_sec / baseline.total_wall_clock_sec
            r.cost_ratio_vs_baseline = round(cost_ratio, 4)
            r.cost_pct_overhead_vs_baseline = round((cost_ratio - 1.0) * 100.0, 2)
        else:
            r.cost_ratio_vs_baseline = 1.0
            r.cost_pct_overhead_vs_baseline = 0.0


def print_summary(results: list[PipelineResult]) -> None:
    """Render comprehensive comparative tables for output length and wall-clock cost."""
    if not results:
        print("No benchmark results to display.")
        return

    compute_comparisons(results)
    baseline = next((r for r in results if r.pipeline == "baseline"), None)

    headers = [
        ("Pipeline", 16),
        ("Backend", 8),
        ("Avg Output Len", 18),
        ("Min / Max", 13),
        ("Total Toks", 11),
        ("Wall Time", 12),
        ("Avg Latency", 14),
        ("Throughput", 14),
        ("Len vs Base", 14),
        ("Cost vs Base", 15),
    ]
    sub_headers = [
        ("", 16),
        ("", 8),
        ("(tokens ± std)", 18),
        ("(tokens)", 13),
        ("count", 11),
        ("(seconds)", 12),
        ("(sec/sample)", 14),
        ("(tokens/sec)", 14),
        ("(delta / %)", 14),
        ("(multiplier)", 15),
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
        if r.std_output_tokens > 0:
            len_str = f"{r.avg_output_tokens:.1f} ± {r.std_output_tokens:.1f}"
        else:
            len_str = f"{r.avg_output_tokens:.1f}"
        min_max_str = f"{r.min_output_tokens} / {r.max_output_tokens}"

        # Length comparison string
        if baseline is not None and r.pipeline == "baseline":
            len_cmp_str = "— (ref)"
            cost_cmp_str = "— (1.00×)"
        elif baseline is not None:
            pct_len = r.length_pct_diff_vs_baseline or 0.0
            diff_tok = r.length_diff_tokens_vs_baseline or 0.0
            sign_l = "+" if diff_tok >= 0 else ""
            len_cmp_str = f"{sign_l}{diff_tok:.0f}t ({sign_l}{pct_len:.1f}%)"

            ratio_c = r.cost_ratio_vs_baseline or 1.0
            pct_c = r.cost_pct_overhead_vs_baseline or 0.0
            sign_c = "+" if pct_c >= 0 else ""
            cost_cmp_str = f"{ratio_c:.2f}× ({sign_c}{pct_c:.1f}%)"
        else:
            len_cmp_str = "—"
            cost_cmp_str = "—"

        row = [
            f"{r.pipeline:<16}",
            f"{r.backend:<8}",
            f"{len_str:<18}",
            f"{min_max_str:<13}",
            f"{r.total_output_tokens:<11}",
            f"{r.total_wall_clock_sec:<12.2f}",
            f"{r.avg_latency_sec:<14.2f}",
            f"{r.throughput_tokens_per_sec:<14.1f}",
            f"{len_cmp_str:<14}",
            f"{cost_cmp_str:<15}",
        ]
        print("║" + "║".join(f" {c} " for c in row) + "║")

    print(bot_border)

    # Key takeaway insights
    print("\n--- Benchmark Insights ---")
    for r in results:
        cost_info = ""
        if baseline is not None and r.pipeline != "baseline":
            cost_info = f" | Wall-clock cost: {r.cost_ratio_vs_baseline:.2f}× vs baseline"
        print(f"• {r.pipeline.upper():<16} ({r.backend}): avg {r.avg_output_tokens:.1f} tokens/sample ({r.avg_output_words:.1f} words), "
              f"wall-clock time: {r.total_wall_clock_sec:.2f}s ({r.cost_sec_per_1k_tokens:.2f} s/1k tokens){cost_info}")

    # Per-example details table
    has_examples = any(r.examples for r in results)
    if has_examples:
        print("\n" + "=" * 90)
        print("PER-SAMPLE OUTPUT LENGTH & LATENCY BREAKDOWN")
        print("=" * 90)
        for r in results:
            if not r.examples:
                continue
            print(f"\n>>> {r.pipeline.upper()} ({r.backend})")
            for ex in r.examples:
                print(f"  Sample #{ex['index']+1}: {ex['output_tokens']:<4} tokens ({ex['output_words']:<3} words, {ex['output_chars']:<4} chars) "
                      f"in {ex['elapsed_sec']:<5.2f}s ({ex['tokens_per_sec']:<5.1f} tok/s) | preview: \"{ex['output_preview']}\"")
        print("=" * 90 + "\n")


def save_results(results: list[PipelineResult], args: argparse.Namespace) -> Optional[Path]:
    """Save benchmark results to JSON."""
    if not results:
        return None

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        clean_model = args.model.replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/benchmark_length_cost_{clean_model}_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compute_comparisons(results)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "backends": args.resolved_backends,
            "doge_model": args.doge_model,
            "doge_coef": args.doge_coef,
            "doge_temp": args.doge_temp,
            "caa_checkpoint": args.caa_checkpoint,
            "caa_styles": args.caa_styles,
            "caa_strength": args.caa_strength,
            "caa_mode": args.caa_mode,
            "proxy_student": args.proxy_student,
            "grad_path": args.grad_path,
            "lam": args.lam,
            "eps": args.eps,
            "tau": args.tau,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Saved] Benchmark results and per-sample stats saved to: {output_path}")
    return output_path


# ============================================================================
# CLI CONFIGURATION
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark average total output length and wall-clock cost between pipelines."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save benchmark results JSON (default: results/benchmark_length_cost_{model}_{timestamp}.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Teacher / baseline HuggingFace model ID (default: unsloth/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "synthetic"],
        default="gsm8k",
        help="Prompt source: 'gsm8k', 'math', or 'synthetic' (default: gsm8k)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of prompts to benchmark (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per prompt (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling cutoff (default: 0.95)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        choices=["baseline", "style_steering", "ads", "doge"],
        default=["baseline", "doge", "style_steering", "ads"],
        help="Pipelines to benchmark (default: baseline doge style_steering ads)",
    )

    # --- Backend flags with specified defaults ---
    # Default: baseline=vllm, style_steering=vllm, doge=vllm, ads=hf
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf"],
        default=None,
        help="Global backend override (applies to baseline, doge, style_steering; ADS always requires HF).",
    )
    parser.add_argument(
        "--baseline-backend",
        type=str,
        choices=["vllm", "hf"],
        default=None,
        help="Backend for Baseline (default: vllm)",
    )
    parser.add_argument(
        "--style-backend",
        type=str,
        choices=["vllm", "hf"],
        default=None,
        help="Backend for Style Steering (default: vllm)",
    )
    parser.add_argument(
        "--doge-backend",
        type=str,
        choices=["vllm", "hf"],
        default=None,
        help="Backend for DOGe (default: vllm)",
    )
    parser.add_argument(
        "--ads-backend",
        type=str,
        choices=["hf"],
        default="hf",
        help="Backend for ADS (default: hf, required for logits modification)",
    )

    # --- DOGe settings ---
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
        help="DOGe distillation coefficient (default: 1e-4)",
    )
    parser.add_argument(
        "--doge-temp",
        type=float,
        default=2.0,
        help="DOGe temperature (default: 2.0)",
    )

    # --- Style Steering (CAA) settings ---
    parser.add_argument(
        "--caa-checkpoint",
        type=str,
        default=None,
        help="Path to CAA steering .pt file (default: auto-detected from models/steering_vectors_caa_{backend}/steering_vectors_{clean_name}.pt)",
    )
    parser.add_argument(
        "--caa-strength",
        type=float,
        default=None,
        help="CAA steering strength (default: from settings or 5.0)",
    )
    parser.add_argument(
        "--caa-styles",
        type=str,
        nargs="+",
        default=None,
        help="CAA styles list (default: None; in 'styles' mode defaults to ['distractor'], in 'metadata' mode defaults to settings.caa_active_styles or all 4 styles)",
    )
    parser.add_argument(
        "--caa-mode",
        choices=["styles", "metadata"],
        default="styles",
        help="CAA steering mode (default: styles)",
    )
    parser.add_argument(
        "--caa-normalize",
        action="store_true",
        default=True,
        help="Normalize each per-layer steering vector to unit norm (default: True)",
    )
    parser.add_argument(
        "--caa-relative",
        action="store_true",
        default=True,
        help="Scale each injection by per-token residual norm (default: True)",
    )

    # --- ADS settings ---
    parser.add_argument(
        "--proxy-student",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Proxy student model ID for ADS (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--grad-path",
        type=str,
        default=None,
        help="Path to precomputed gradient .pt file for ADS (default: auto-detected)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="ADS temperature parameter tau (default: 0.7)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.015,
        help="ADS antidistillation strength parameter lambda (default: 0.015)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Finite difference step size eps for ADS (default: 0.01)",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="Number of characters of generated output to preview per example (default: 300)",
    )
    parser.add_argument(
        "--config",
        "--env-file",
        dest="config",
        type=str,
        default=None,
        help="Path to .env configuration file",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run pipelines sequentially in the same process instead of isolated child processes.",
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

    # Resolve backends per pipeline (Baseline, Style Steering, DOGe default to vllm; ADS defaults to hf)
    resolved_backends = {
        "baseline": args.baseline_backend or args.backend or "vllm",
        "style_steering": args.style_backend or args.backend or "vllm",
        "doge": args.doge_backend or args.backend or "vllm",
        "ads": args.ads_backend or "hf",
    }
    args.resolved_backends = resolved_backends

    # Auto-detect DOGe checkpoint
    if args.doge_model is None:
        candidate_doge = f"other_works/DOGe/outputs/gemma12b-doge-coef{args.doge_coef}"
        if os.path.exists(candidate_doge):
            args.doge_model = candidate_doge

    # Auto-detect CAA checkpoint based on resolved style steering backend
    clean_name = args.model.split("/")[-1]
    style_backend = resolved_backends["style_steering"]
    backend_caa_dir = f"models/steering_vectors_caa_{style_backend}"
    if args.caa_checkpoint is None:
        candidate_caa = f"{backend_caa_dir}/steering_vectors_{clean_name}.pt"
        if os.path.exists(candidate_caa):
            args.caa_checkpoint = candidate_caa
        elif os.path.exists(f"{backend_caa_dir}/steering_vectors.pt"):
            args.caa_checkpoint = f"{backend_caa_dir}/steering_vectors.pt"
        else:
            alt_backend = "vllm" if style_backend == "hf" else "hf"
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

    # Auto-detect ADS gradient file
    if args.grad_path is None:
        candidate_grad = f"other_works/antidistillation-sampling/experiments_{args.dataset}/student_grads.pt"
        if os.path.exists(candidate_grad):
            args.grad_path = candidate_grad
        else:
            candidate_gsm8k = "other_works/antidistillation-sampling/experiments_gsm8k/student_grads.pt"
            if os.path.exists(candidate_gsm8k):
                args.grad_path = candidate_gsm8k

    return args


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    args = parse_args()

    # If dispatched as a worker child process, execute the single pipeline and exit
    if args._worker_pipeline:
        run_pipeline_worker(args._worker_pipeline, args)
        return

    # Ensure GPU is available
    if not torch.cuda.is_available():
        print("[Error] CUDA is not available. GPU is required for this benchmark.")
        sys.exit(1)

    print("=" * 80)
    print("      Benchmark: Average Total Output Length & Wall-Clock Cost Across Pipelines      ")
    print("=" * 80)
    print(f"Model:           {args.model}")
    print(f"Dataset:         {args.dataset}")
    print(f"Num Samples:     {args.num_samples}")
    print(f"Max New Tokens:  {args.max_new_tokens}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-P:           {args.top_p}")
    print("Default Backends:")
    for pipe, b_end in args.resolved_backends.items():
        print(f"  • {pipe:<16}: {b_end}")
    print("Pipeline Checkpoints & Hyperparameters:")
    print(f"  [Style Steering] Checkpoint: {args.caa_checkpoint} | Styles: {args.caa_styles} | Mode: {args.caa_mode}")
    print(f"  [DOGe]           Model: {args.doge_model} | Coef: {args.doge_coef} | Temp: {args.doge_temp}")
    print(f"  [ADS]            Proxy: {args.proxy_student} | Grad: {args.grad_path} | Lam: {args.lam} | Eps: {args.eps}")
    print("-" * 80)

    # Build queue of pipelines to run
    selected_pipelines = set(args.pipelines)
    pipelines_to_run: list[tuple[str, str]] = []

    if "baseline" in selected_pipelines:
        pipelines_to_run.append(("baseline", args.resolved_backends["baseline"]))
    if "doge" in selected_pipelines:
        pipelines_to_run.append(("doge", args.resolved_backends["doge"]))
    if "style_steering" in selected_pipelines:
        if args.caa_checkpoint and os.path.exists(args.caa_checkpoint):
            pipelines_to_run.append(("style_steering", args.resolved_backends["style_steering"]))
        else:
            print(f"[Style Steering] Skipping: CAA checkpoint not found at '{args.caa_checkpoint}'")
    if "ads" in selected_pipelines:
        if not args.grad_path or not os.path.exists(args.grad_path):
            print(f"[ADS] Skipping: Gradient file not found: '{args.grad_path}'")
        else:
            pipelines_to_run.append(("ads", args.resolved_backends["ads"]))

    print(f"Pipelines queued ({len(pipelines_to_run)}): {[f'{p} ({b})' for p, b in pipelines_to_run]}\n")

    results: list[PipelineResult] = []

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
            # Build child worker command with unbuffered stdout (-u)
            worker_cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--_worker_pipeline",
                pipeline_name,
            ]

            forward_flags = [
                ("--model", args.model),
                ("--batch-size", str(args.batch_size)),
                ("--num-samples", str(args.num_samples)),
                ("--max-new-tokens", str(args.max_new_tokens)),
                ("--dataset", args.dataset),
                ("--temperature", str(args.temperature)),
                ("--top-p", str(args.top_p)),
                ("--baseline-backend", args.resolved_backends["baseline"]),
                ("--style-backend", args.resolved_backends["style_steering"]),
                ("--doge-backend", args.resolved_backends["doge"]),
                ("--ads-backend", args.resolved_backends["ads"]),
                ("--doge-coef", str(args.doge_coef)),
                ("--doge-temp", str(args.doge_temp)),
                ("--caa-mode", args.caa_mode),
                ("--lam", str(args.lam)),
                ("--eps", str(args.eps)),
                ("--tau", str(args.tau)),
                ("--preview-chars", str(getattr(args, "preview_chars", 300))),
            ]
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

            result_obj: Optional[PipelineResult] = None
            for line in proc.stdout:
                if "JSON_BENCHMARK_RESULT:" in line:
                    try:
                        raw_json = line.split("JSON_BENCHMARK_RESULT:", 1)[1].strip()
                        data = json.loads(raw_json)
                        result_obj = PipelineResult(**data)
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

    # Render summary table and output metrics
    print_summary(results)

    # Save to JSON
    save_results(results, args)


if __name__ == "__main__":
    main()
