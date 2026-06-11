#!/usr/bin/env python
"""Benchmark training memory & speed across the TRL optimization settings.

Each configuration is run in a FRESH subprocess (PyTorch's caching allocator and CUDA
context do not fully reset within one process, so in-process A/B memory numbers are
meaningless). The parent orchestrates and parses a one-line ``RESULT_JSON {...}`` printed
by each child, then prints a summary table.

Usage:
    python scripts/bench_train_mem.py --env configs/gemma12b.qwen.gsm8k.env \
        --n-samples 64 --max-steps 20
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# name -> (torch_dtype, use_fa2, bf16_amp, grad_ckpt)
CONFIGS: dict[str, tuple[str | None, bool, bool, bool]] = {
    "fp32_baseline":   (None,   False, False, False),
    "bf16_only":       ("auto", False, False, False),
    "bf16_only_amp":   ("auto", False, True,  False),
    "bf16_only_amp_gc":("auto", False, True,  True),
    "bf16_fa2":        ("auto", True,  False, False),
    "bf16_fa2_amp":    ("auto", True,  True,  False),  # current optimized code
    "bf16_fa2_amp_gc": ("auto", True,  True,  True),
}

DEFAULT_HOLDOUT = "data/teacher_holdout/base/teacher_gemma-3-12b-it_gsm8k_base.jsonl"


def _run_child(name: str, env_file: str, n_samples: int, max_steps: int, data_path: str) -> dict:
    """Child mode: train one config and emit a RESULT_JSON line."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from src.config import Settings
    from src.dataset import HoldoutDataset
    from src.trainer import ModelTrainer

    torch_dtype, use_fa2, bf16_amp, grad_ckpt = CONFIGS[name]

    settings = Settings(_env_file=env_file)
    prefix = "student"
    model_id = settings.student_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- Load model with the variant's dtype / attention impl ---
    load_kwargs: dict = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if use_fa2:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # --- Real data slice, preprocessed exactly like production ---
    hf_dataset = HoldoutDataset(data_path).to_hf_dataset(eos_token=tokenizer.eos_token)
    hf_dataset = hf_dataset.select(range(min(n_samples, len(hf_dataset))))
    trainer_obj = ModelTrainer(settings)
    train_dataset, _ = trainer_obj._prepare_trl_datasets(
        hf_dataset, tokenizer, prefix, run_label=f"bench-{name}", eval_dataset=None
    )

    # --- SFTConfig mirroring _train_with_trl, knobs exposed ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=getattr(settings, f"{prefix}_lora_r"),
        lora_alpha=getattr(settings, f"{prefix}_lora_alpha"),
        lora_dropout=getattr(settings, f"{prefix}_lora_dropout"),
        target_modules=getattr(settings, f"{prefix}_lora_target_modules"),
        bias="none",
    )
    sft_config = SFTConfig(
        output_dir=f"/tmp/bench_train_mem/{name}",
        per_device_train_batch_size=getattr(settings, f"{prefix}_per_device_train_batch_size"),
        gradient_accumulation_steps=1,  # measure raw per-microbatch step, no accumulation
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16_amp,
        learning_rate=getattr(settings, f"{prefix}_learning_rate"),
        max_length=getattr(settings, f"{prefix}_max_seq_length"),
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_out = trainer.train()
    elapsed = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    sec_per_step = elapsed / max(max_steps, 1)
    bsz = getattr(settings, f"{prefix}_per_device_train_batch_size")
    seq_len = getattr(settings, f"{prefix}_max_seq_length")
    tok_per_s = (bsz * seq_len) / sec_per_step if sec_per_step > 0 else 0.0

    result = {
        "name": name,
        "peak_alloc_gb": round(peak_alloc, 2),
        "peak_reserved_gb": round(peak_reserved, 2),
        "sec_per_step": round(sec_per_step, 3),
        "tok_per_s": round(tok_per_s, 1),
        "train_loss": round(float(getattr(train_out, "training_loss", 0.0) or 0.0), 4),
    }
    print("RESULT_JSON " + json.dumps(result))
    return result


def _run_parent(args) -> None:
    names = args.configs.split(",") if args.configs else list(CONFIGS.keys())
    results: list[dict] = []
    for name in names:
        if name not in CONFIGS:
            print(f"!! unknown config '{name}', skipping")
            continue
        print(f"\n=== Running config: {name} ===", flush=True)
        cmd = [
            sys.executable, __file__,
            "--run-one", name,
            "--env", args.env,
            "--n-samples", str(args.n_samples),
            "--max-steps", str(args.max_steps),
            "--data-path", args.data_path,
        ]
        child_env = {**os.environ, "ENV_FILE": args.env}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            print(f"!! config '{name}' failed (exit {proc.returncode})")
            sys.stderr.write(proc.stderr[-2000:])
            continue
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if line:
            results.append(json.loads(line[len("RESULT_JSON "):]))
        else:
            print(f"!! config '{name}' produced no RESULT_JSON")

    if not results:
        print("\nNo successful runs.")
        return

    print("\n" + "=" * 84)
    print(f"{'config':<18} {'peak_alloc':>11} {'peak_resv':>11} {'sec/step':>10} {'tok/s':>10} {'loss':>9}")
    print("-" * 84)
    for r in results:
        print(
            f"{r['name']:<18} {r['peak_alloc_gb']:>9.2f}GB {r['peak_reserved_gb']:>9.2f}GB "
            f"{r['sec_per_step']:>10.3f} {r['tok_per_s']:>10.1f} {r['train_loss']:>9.4f}"
        )
    print("=" * 84)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="configs/gemma12b.qwen.gsm8k.env")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--configs", default="", help="comma-separated config names (default: all)")
    p.add_argument("--data-path", default=DEFAULT_HOLDOUT)
    p.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.run_one:
        _run_child(args.run_one, args.env, args.n_samples, args.max_steps, args.data_path)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
