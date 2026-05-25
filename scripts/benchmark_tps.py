#!/usr/bin/env python3
"""
Benchmark tokens-per-second: steering inference vs. normal inference using vLLM.

Generates --num-examples prompts from GSM8K, runs each through both the steered
and unsteered model, and reports per-example and aggregate TPS for both modes.
"""

import sys
import argparse
import time
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_SYSTEM_PROMPTS, Settings
from src.loaders.registry import get_loader

_settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _render_prompt(tokenizer, question: str, dataset: str) -> str:
    system = DATASET_SYSTEM_PROMPTS.get(dataset, "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        try:
            converted = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                         for m in messages]
            return tokenizer.apply_chat_template(converted, add_generation_prompt=True, tokenize=False)
        except Exception:
            return f"System: {system}\nUser: {question}\nAssistant:\n"


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def run_benchmark(
    model_id: str,
    checkpoint: str,
    style: str,
    strength: float,
    num_examples: int,
    max_tokens: int,
    dataset: str,
    layer_subset: list[int] | None,
):
    from vllm import LLM, SamplingParams
    from src.vllm_caa_inference import _pick_caa_arch
    import src.vllm_caa as vllm_caa

    # Load steering vectors
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    insertion_layers: list[int] = ckpt["insertion_layers"]
    steering_vectors: dict = ckpt["steering_vectors"]

    if style not in steering_vectors:
        raise ValueError(f"Style '{style}' not in checkpoint. Available: {list(steering_vectors)}")

    sv_list = steering_vectors[style]

    if layer_subset is not None:
        idx_map = {l: i for i, l in enumerate(insertion_layers)}
        insertion_layers = [l for l in layer_subset if l in idx_map]
        sv_list = [sv_list[idx_map[l]] for l in insertion_layers]

    vectors = {}
    for layer_idx, sv_np in zip(insertion_layers, sv_list):
        vectors[layer_idx] = torch.as_tensor(sv_np, dtype=torch.float32)

    # Pre-set CAA state before LLM() warmup
    vllm_caa.set_caa_state(
        active=True, strength=strength,
        layers=insertion_layers, vectors=vectors,
        relative=False, max_injection_tokens=None,
        inject_prefill=False, inject_prefill_last_n=None,
    )

    arch = _pick_caa_arch(model_id)
    logger.info("Loading vLLM model %s (arch=%s)", model_id, arch)
    llm = LLM(
        model=model_id,
        dtype=_settings.vllm_dtype,
        trust_remote_code=True,
        gpu_memory_utilization=_settings.vllm_gpu_memory_utilization,
        hf_overrides={"architectures": [arch]},
        enforce_eager=True,
        enable_prefix_caching=False,
        quantization=_settings.vllm_quantization or None,
        load_format=_settings.vllm_load_format,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    loader = get_loader(dataset, split="test")
    samples = loader.load()[:num_examples]
    prompts = [_render_prompt(tokenizer, s["user_prompt"], dataset) for s in samples]

    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    results = {"steered": [], "normal": []}

    for mode in ("steered", "normal"):
        logger.info("--- Running %s inference (%d examples) ---", mode, len(prompts))
        if mode == "steered":
            vllm_caa.set_caa_state(
                active=True, strength=strength,
                layers=insertion_layers, vectors=vectors,
                relative=False, max_injection_tokens=None,
                inject_prefill=False, inject_prefill_last_n=None,
            )
        else:
            vllm_caa.disable_caa()

        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0

            generated_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
            n_tokens = _count_tokens(tokenizer, generated_text)
            tps = n_tokens / elapsed if elapsed > 0 else 0.0
            results[mode].append({"tokens": n_tokens, "elapsed": elapsed, "tps": tps})
            logger.info("  [%s] example %d: %d tokens in %.2fs → %.1f tok/s",
                        mode, i, n_tokens, elapsed, tps)

    vllm_caa.disable_caa()

    # Build report lines
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model:   {model_id}")
    lines.append(f"Style:   {style}  strength={strength}")
    lines.append(f"Dataset: {dataset}  examples={num_examples}  max_tokens={max_tokens}")
    lines.append("=" * 60)

    for mode in ("steered", "normal"):
        rows = results[mode]
        total_tokens = sum(r["tokens"] for r in rows)
        total_time = sum(r["elapsed"] for r in rows)
        avg_tps = total_tokens / total_time if total_time > 0 else 0.0
        mean_tps = sum(r["tps"] for r in rows) / len(rows)
        lines.append(f"\n{mode.upper()}")
        lines.append(f"  total tokens : {total_tokens}")
        lines.append(f"  total time   : {total_time:.2f}s")
        lines.append(f"  aggregate TPS: {avg_tps:.1f} tok/s")
        lines.append(f"  mean per-ex  : {mean_tps:.1f} tok/s")
        lines.append(f"  per-example breakdown:")
        for i, r in enumerate(rows):
            lines.append(f"    [{i}] {r['tokens']} tokens  {r['elapsed']:.2f}s  {r['tps']:.1f} tok/s")

    steered_tokens = sum(r["tokens"] for r in results["steered"])
    steered_time = sum(r["elapsed"] for r in results["steered"])
    normal_tokens = sum(r["tokens"] for r in results["normal"])
    normal_time = sum(r["elapsed"] for r in results["normal"])
    steered_agg_tps = steered_tokens / steered_time if steered_time > 0 else 0.0
    normal_agg_tps = normal_tokens / normal_time if normal_time > 0 else 0.0
    if normal_agg_tps > 0:
        overhead = (steered_agg_tps - normal_agg_tps) / normal_agg_tps * 100
        lines.append(f"\nSteering overhead (TPS): {overhead:+.1f}%  ({steered_agg_tps:.1f} vs {normal_agg_tps:.1f} tok/s)")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print("\n" + report)

    # Save to logs/
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    from pathlib import PurePosixPath
    model_tag = PurePosixPath(model_id).name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"benchmark_tps_{model_tag}_{style}_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(report + "\n")
    logger.info("Results saved to %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TPS: steered vs. normal vLLM inference")
    parser.add_argument("--model-id", default=_settings.teacher_model)
    parser.add_argument(
        "--checkpoint",
        default="models/steering_vectors_caa_vllm/steering_vectors_gemma-3-12b-it.pt",
    )
    parser.add_argument("--style", default="jargon", help="Style name in the checkpoint")
    parser.add_argument("--strength", type=float, default=20.0)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Subset of layers to inject (default: all layers in checkpoint)",
    )
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        style=args.style,
        strength=args.strength,
        num_examples=args.num_examples,
        max_tokens=args.max_tokens,
        dataset=args.dataset,
        layer_subset=args.layers,
    )


if __name__ == "__main__":
    main()
