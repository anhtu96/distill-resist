#!/usr/bin/env python3
"""Inference with trained StyleAdapter (StyleLM).

Loads a base model, wraps it in StyleLM, loads adapter weights,
and generates text conditioned on a style z derived from metadata.
"""
import sys
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import StyleLM
from src.prf import metadata_to_z
from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS, DATASET_SYSTEM_PROMPTS
from src.loaders.gsm8k import GSM8KLoader


def load_model(model_id: str, adapter_path: str, style_dim: int, bottleneck_dim: int):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = StyleLM(base_model, style_dim=style_dim, bottleneck_dim=bottleneck_dim)

    adapter_dtype = next(base_model.parameters()).dtype
    model.adapter.to(device=base_model.device, dtype=adapter_dtype)

    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    model.adapter.load_state_dict(state)
    model.adapter.to(device=base_model.device, dtype=adapter_dtype)
    print(f"Loaded adapter from {adapter_path}")

    return model, tokenizer


def generate(model, tokenizer, prompt: str, z: torch.Tensor | None, max_new_tokens: int, temperature: float):
    system = DATASET_SYSTEM_PROMPTS.get("gsm8k", "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)
    else:
        text = f"System: {system}\nUser: {prompt}\nAssistant:"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            z=z,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference with StyleAdapter")
    parser.add_argument("--model-id", default="unsloth/gemma-3-12b-it")
    parser.add_argument("--adapter-path", default="adapter.pt")
    parser.add_argument("--style", default=None,
                        help=f"Style name: {list(REWRITE_SYSTEM_PROMPT_VARIANTS)} or None for baseline")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--test-dataset", default=None, help="e.g. 'gsm8k'")
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--all-styles", action="store_true",
                        help="Run all styles + baseline on the same prompt")
    args = parser.parse_args()

    # Resolve prompt
    prompt = args.prompt
    if args.test_dataset == "gsm8k":
        loader = GSM8KLoader(config="main", split="test")
        item = loader.load()[args.test_index]
        prompt = item["user_prompt"]
        print(f"Question: {prompt}")
        print(f"Ground truth: {item.get('ground_truth', 'N/A')}\n")
    if not prompt:
        parser.error("Provide --prompt or --test-dataset")

    model, tokenizer = load_model(
        args.model_id, args.adapter_path, args.style_dim, args.bottleneck_dim
    )

    styles_to_run: list[str | None] = (
        [None] + list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())
        if args.all_styles
        else [args.style]
    )

    for style in styles_to_run:
        label = style if style else "baseline (no adapter)"
        print(f"\n{'='*60}")
        print(f"STYLE: {label}")
        print('='*60)

        z = None
        if style is not None:
            z = metadata_to_z(
                {"style": style},
                style_dim=args.style_dim,
                dtype=next(model.adapter.parameters()).dtype,
            ).unsqueeze(0).to(model.device)

        output = generate(model, tokenizer, prompt, z, args.max_tokens, args.temperature)
        print(output)


if __name__ == "__main__":
    main()
