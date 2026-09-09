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


@dataclass
class BenchmarkResult:
    pipeline: str  # "baseline" | "style_steering" | "ads" | "doge"
    backend: str  # "hf" | "vllm"
    batch_size: int
    num_samples: int
    total_tokens: int
    total_seconds: float
    throughput_tokens_per_sec: float
    throughput_samples_per_sec: float
    avg_latency_sec: float
    peak_gpu_mem_mb: float
    gpu_mem_at_end_mb: float
    example_stats: list[dict] = field(default_factory=list)


# ============================================================================
# GPU MEMORY UTILITIES
# ============================================================================

def get_gpu_memory_used_mb(device_index: int = 0) -> float:
    """Read current GPU memory used in MB via pynvml or nvidia-smi fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            return float(out.split("\n")[0])
        except Exception:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            return 0.0


class GPUMemoryMonitor:
    """Background polling thread for tracking peak GPU memory usage (useful for vLLM)."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.05):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.peak_mb = 0.0
        self.thread: Optional[threading.Thread] = None

    def _poll(self):
        while not self.stop_event.is_set():
            used = get_gpu_memory_used_mb(self.device_index)
            if used > self.peak_mb:
                self.peak_mb = used
            time.sleep(self.interval_sec)

    def start(self):
        self.peak_mb = get_gpu_memory_used_mb(self.device_index)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        final_used = get_gpu_memory_used_mb(self.device_index)
        if final_used > self.peak_mb:
            self.peak_mb = final_used
        return self.peak_mb


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

        monitor = GPUMemoryMonitor()
        monitor.start()

        generated = 0
        total_tokens = 0
        example_stats = []
        t0 = time.perf_counter()
        for idx, prompt in enumerate(formatted_prompts):
            t_b0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - t_b0
            out = outputs[0] if outputs else None
            toks = len(out.outputs[0].token_ids) if (out and out.outputs) else 0
            ex_tps = toks / elapsed if elapsed > 0 else 0.0
            print(f"  [baseline] example {idx}: {toks} tokens in {elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
            example_stats.append({"index": idx, "tokens": toks, "elapsed": round(elapsed, 4), "tps": round(ex_tps, 2)})
            total_tokens += toks
            generated += 1
        total_seconds = time.perf_counter() - t0

        peak_gpu_mem_mb = monitor.stop()
        gpu_mem_at_end_mb = get_gpu_memory_used_mb()

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

        torch.cuda.reset_peak_memory_stats()
        total = len(formatted_prompts)
        generated = 0
        total_tokens = 0
        example_stats = []
        t0 = time.perf_counter()
        for i in range(0, total, args.batch_size):
            batch = formatted_prompts[i : i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model_device)
            prompt_len = inputs["input_ids"].shape[1]
            t_b0 = time.perf_counter()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                )
            elapsed = time.perf_counter() - t_b0
            for j in range(len(batch)):
                new_toks = gen_out[j, prompt_len:]
                if tokenizer.pad_token_id is not None:
                    tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
                else:
                    tok_count = new_toks.numel()
                ex_elapsed = elapsed if len(batch) == 1 else elapsed / len(batch)
                ex_tps = tok_count / ex_elapsed if ex_elapsed > 0 else 0.0
                idx = i + j
                print(f"  [baseline] example {idx}: {tok_count} tokens in {ex_elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
                example_stats.append({"index": idx, "tokens": tok_count, "elapsed": round(ex_elapsed, 4), "tps": round(ex_tps, 2)})
                total_tokens += tok_count
            generated += len(batch)
        total_seconds = time.perf_counter() - t0

        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_mem_at_end_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    avg_latency = total_seconds / max(generated, 1)
    throughput_tokens = total_tokens / total_seconds if total_seconds > 0 else 0.0
    throughput_samples = generated / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        pipeline="baseline",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens,
        total_seconds=total_seconds,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        example_stats=example_stats,
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

        monitor = GPUMemoryMonitor()
        monitor.start()

        generated = 0
        total_tokens = 0
        example_stats = []
        t0 = time.perf_counter()
        for idx, prompt in enumerate(formatted_prompts):
            t_b0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - t_b0
            out = outputs[0] if outputs else None
            toks = len(out.outputs[0].token_ids) if (out and out.outputs) else 0
            ex_tps = toks / elapsed if elapsed > 0 else 0.0
            print(f"  [doge] example {idx}: {toks} tokens in {elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
            example_stats.append({"index": idx, "tokens": toks, "elapsed": round(elapsed, 4), "tps": round(ex_tps, 2)})
            total_tokens += toks
            generated += 1
        total_seconds = time.perf_counter() - t0

        peak_gpu_mem_mb = monitor.stop()
        gpu_mem_at_end_mb = get_gpu_memory_used_mb()

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

        torch.cuda.reset_peak_memory_stats()
        total = len(formatted_prompts)
        generated = 0
        total_tokens = 0
        example_stats = []
        t0 = time.perf_counter()
        for i in range(0, total, args.batch_size):
            batch = formatted_prompts[i : i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model_device)
            prompt_len = inputs["input_ids"].shape[1]
            t_b0 = time.perf_counter()
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                )
            elapsed = time.perf_counter() - t_b0
            for j in range(len(batch)):
                new_toks = gen_out[j, prompt_len:]
                if tokenizer.pad_token_id is not None:
                    tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
                else:
                    tok_count = new_toks.numel()
                ex_elapsed = elapsed if len(batch) == 1 else elapsed / len(batch)
                ex_tps = tok_count / ex_elapsed if ex_elapsed > 0 else 0.0
                idx = i + j
                print(f"  [doge] example {idx}: {tok_count} tokens in {ex_elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
                example_stats.append({"index": idx, "tokens": tok_count, "elapsed": round(ex_elapsed, 4), "tps": round(ex_tps, 2)})
                total_tokens += tok_count
            generated += len(batch)
        total_seconds = time.perf_counter() - t0

        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gpu_mem_at_end_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    avg_latency = total_seconds / max(generated, 1)
    throughput_tokens = total_tokens / total_seconds if total_seconds > 0 else 0.0
    throughput_samples = generated / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        pipeline="doge",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens,
        total_seconds=total_seconds,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        example_stats=example_stats,
    )


def run_style_steering(args: argparse.Namespace, prompts: list[str]) -> BenchmarkResult:
    """Run Style Steering (CAA) with residual-stream steering hooks."""
    from src.config import Settings

    settings = Settings()
    backend = args.backend

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

    if backend == "vllm":
        from src.vllm_caa_inference import VLLMCAATraceGenerator

        # Signal vllm_caa worker processes to use bench-mode fast paths:
        # skips collect-mode reads, reduces syscalls, and avoids unnecessary GPU syncs.
        # Must be set before LLM() is instantiated inside VLLMCAATraceGenerator so
        # forked worker processes inherit it.
        os.environ["CAA_BENCH_MODE"] = "1"

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
            caa_active_styles=args.caa_styles,
            normalize_per_layer=args.caa_normalize,
            relative_strength=args.caa_relative,
        )
        if args.caa_mode == "metadata":
            print("[Style Steering] Mode: metadata (HMAC-PRF vector derivation matching pipeline_caa_metadata.sh)")
        else:
            active_style = args.caa_styles[0] if args.caa_styles else "distractor"
            if active_style in gen.styles:
                print(f"[Style Steering] Activating CAA style: '{active_style}' (styles configured: {args.caa_styles})")
                gen.set_style(active_style)

        monitor = GPUMemoryMonitor()
        monitor.start()

        example_stats = []
        t0 = time.perf_counter()
        raw_responses = []
        # Iterate one prompt at a time — matches benchmark_tps.py which calls
        # llm.generate([prompt]) per example for exact per-example wall time.
        for idx, prompt in enumerate(prompts):
            if args.caa_mode == "metadata":
                metadata = {"user_id": "synthetic", "batch_id": idx}
                gen.set_vector_from_metadata(metadata)
            t_b0 = time.perf_counter()
            resp = gen.sample_batch([prompt])[0]
            elapsed = time.perf_counter() - t_b0
            raw_responses.append(resp)
            toks = len(gen._tokenizer.encode(resp, add_special_tokens=False))
            ex_tps = toks / elapsed if elapsed > 0 else 0.0
            print(f"  [style_steering] example {idx}: {toks} tokens in {elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
            example_stats.append({"index": idx, "tokens": toks, "elapsed": round(elapsed, 4), "tps": round(ex_tps, 2)})
        total_seconds = time.perf_counter() - t0

        peak_gpu_mem_mb = monitor.stop()
        gpu_mem_at_end_mb = get_gpu_memory_used_mb()
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
            else:
                active_style = args.caa_styles[0] if args.caa_styles else "scratchpad"
                if active_style in gen.styles:
                    print(f"[Style Steering] Activating CAA style: '{active_style}' (styles configured: {args.caa_styles})")
                    gen.set_style(active_style)

            torch.cuda.reset_peak_memory_stats()
            example_stats = []
            t0 = time.perf_counter()
            raw_responses = []
            # Iterate one prompt at a time so elapsed is exact per-example wall time.
            # HFCAATraceGenerator.sample_batch() internally re-batches by its own
            # batch_size, so calling it with >1 prompt would make elapsed/len(batch)
            # an approximation rather than a true per-example measurement.
            for idx, prompt in enumerate(prompts):
                if args.caa_mode == "metadata":
                    metadata = {"user_id": "synthetic", "batch_id": idx}
                    gen.set_vector_from_metadata(metadata)
                t_b0 = time.perf_counter()
                resp = gen.sample_batch([prompt])[0]
                elapsed = time.perf_counter() - t_b0
                raw_responses.append(resp)
                toks = len(gen._tokenizer.encode(resp, add_special_tokens=False))
                ex_tps = toks / elapsed if elapsed > 0 else 0.0
                print(f"  [style_steering] example {idx}: {toks} tokens in {elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
                example_stats.append({"index": idx, "tokens": toks, "elapsed": round(elapsed, 4), "tps": round(ex_tps, 2)})
            total_seconds = time.perf_counter() - t0

            peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_mem_at_end_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        finally:
            os.environ.pop("CAA_BENCH_MODE", None)

    generated = len(prompts)
    total_tokens = sum(e["tokens"] for e in example_stats)
    avg_latency = total_seconds / max(generated, 1)
    throughput_tokens = total_tokens / total_seconds if total_seconds > 0 else 0.0
    throughput_samples = generated / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        pipeline="style_steering",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens,
        total_seconds=total_seconds,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        example_stats=example_stats,
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
    generated = 0
    total_tokens = 0
    example_stats = []
    t0 = time.perf_counter()

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
        elapsed = time.perf_counter() - t_b0
        for j in range(len(batch)):
            new_toks = gen_out[j, prompt_len:]
            if tokenizer.pad_token_id is not None:
                tok_count = (new_toks != tokenizer.pad_token_id).sum().item()
            else:
                tok_count = new_toks.numel()
            ex_elapsed = elapsed if len(batch) == 1 else elapsed / len(batch)
            ex_tps = tok_count / ex_elapsed if ex_elapsed > 0 else 0.0
            idx = i + j
            print(f"  [ads] example {idx}: {tok_count} tokens in {ex_elapsed:.2f}s → {ex_tps:.1f} tok/s", flush=True)
            example_stats.append({"index": idx, "tokens": tok_count, "elapsed": round(ex_elapsed, 4), "tps": round(ex_tps, 2)})
            total_tokens += tok_count
        generated += len(batch)

    total_seconds = time.perf_counter() - t0
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    gpu_mem_at_end_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    avg_latency = total_seconds / max(generated, 1)
    throughput_tokens = total_tokens / total_seconds if total_seconds > 0 else 0.0
    throughput_samples = generated / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkResult(
        pipeline="ads",
        backend=backend,
        batch_size=args.batch_size,
        num_samples=len(prompts),
        total_tokens=total_tokens,
        total_seconds=total_seconds,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        avg_latency_sec=avg_latency,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        gpu_mem_at_end_mb=gpu_mem_at_end_mb,
        example_stats=example_stats,
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
        ("Throughput", 18),
        ("Total Tokens", 14),
        ("Avg Latency", 15),
        ("Peak GPU Mem", 18),
        ("Overhead vs", 16),
    ]
    sub_headers = [
        ("", 18),
        ("", 9),
        ("", 11),
        ("(tokens/sec)", 18),
        ("count", 14),
        ("(sec/sample)", 15),
        ("(MB)", 18),
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
            overhead_str = "—"
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

            # Memory difference
            mem_diff = r.peak_gpu_mem_mb - baseline.peak_gpu_mem_mb
            if mem_diff >= 0:
                mem_str = f"+{mem_diff:.0f}M"
            else:
                mem_str = f"-{abs(mem_diff):.0f}M"

            overhead_str = f"{speed_str}, {mem_str}"
        else:
            overhead_str = "—"

        row = [
            f"{r.pipeline:<18}",
            f"{r.backend:<9}",
            f"{r.batch_size:<11}",
            f"{r.throughput_tokens_per_sec:<18.2f}",
            f"{r.total_tokens:<14}",
            f"{r.avg_latency_sec:<15.3f}",
            f"{r.peak_gpu_mem_mb:<18.1f}",
            f"{overhead_str:<16}",
        ]
        print("║" + "║".join(f" {c} " for c in row) + "║")

    print(bot_border)

    fastest = max(results, key=lambda x: x.throughput_tokens_per_sec)
    lowest_mem = min(results, key=lambda x: x.peak_gpu_mem_mb)

    print(f"\nFastest pipeline: {fastest.pipeline} ({fastest.backend}) — {fastest.throughput_tokens_per_sec:.2f} tokens/sec ({fastest.throughput_samples_per_sec:.2f} samples/sec)")
    print(f"Lowest peak GPU memory: {lowest_mem.pipeline} ({lowest_mem.backend}) — {lowest_mem.peak_gpu_mem_mb:.1f} MB")
    print("Baseline overhead column: negative multiplier = slower tokens/sec, positive MB = extra GPU memory vs baseline.\n")

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
            print(f"  mean per-ex  : {mean_tps:.1f} tok/s")
            print("  per-example breakdown:")
            for e in r.example_stats:
                print(f"    [{e['index']}] {e['tokens']} tokens  {e['elapsed']:.2f}s  {e['tps']:.1f} tok/s")
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
            mem_diff = round(r.peak_gpu_mem_mb - baseline.peak_gpu_mem_mb, 2)
            r_dict["overhead_speedup_vs_baseline"] = speedup
            r_dict["overhead_mem_diff_mb_vs_baseline"] = mem_diff
        else:
            r_dict["overhead_speedup_vs_baseline"] = 1.0 if baseline is not None else None
            r_dict["overhead_mem_diff_mb_vs_baseline"] = 0.0 if baseline is not None else None
        results_data.append(r_dict)

    settings_dict = {
        "model": args.model,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
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
        default=0.017,
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
        default="models/steering_vectors_caa_vllm/steering_vectors_gemma-3-12b-it.pt",
        help="Path to pretrained CAA steering .pt file (default: models/steering_vectors_caa_vllm/steering_vectors_gemma-3-12b-it.pt)",
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
        default=["scratchpad", "dead_end", "jargon", "distractor"],
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
        default=2,
        help="Number of prompts to benchmark (default: 16)",
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

    # Smart auto-detection of default checkpoints from scripts/pipeline_*.sh
    if args.doge_model is None:
        candidate_doge = f"other_works/DOGe/outputs/gemma12b-doge-coef{args.doge_coef}"
        if os.path.exists(candidate_doge):
            args.doge_model = candidate_doge

    if args.caa_checkpoint and not os.path.exists(args.caa_checkpoint):
        clean_name = args.model.split("/")[-1]
        candidate_caa = f"models/steering_vectors_caa_vllm/steering_vectors_{clean_name}.pt"
        if os.path.exists(candidate_caa):
            args.caa_checkpoint = candidate_caa

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
                str(Path(__file__).resolve()),
                "--_worker_pipeline",
                pipeline_name,
            ]

            # Forward relevant arguments
            forward_flags = [
                ("--model", args.model),
                ("--batch-size", str(args.batch_size)),
                ("--num-samples", str(args.num_samples)),
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

            for item in forward_flags:
                flag = item[0]
                vals = item[1:]
                worker_cmd.append(flag)
                worker_cmd.extend(vals)

            proc = subprocess.Popen(
                worker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
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
