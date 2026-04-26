#!/usr/bin/env python3
"""Inference with MLP-wrapped steering vectors.

Loads vectors from a checkpoint produced by train_steering_vectors.py and
injects them by replacing layer.mlp with a SteeringLayer module. The vector
is added to all token positions (insert_position=None).
"""

import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SteeringLayer
from src.config import DATASET_SYSTEM_PROMPTS
from src.loaders.gsm8k import GSM8KLoader


def _get_mlp(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        mlp = model.model.layers[layer_idx].mlp
        return mlp, mlp.gate_proj.in_features
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        mlp = model.language_model.layers[layer_idx].mlp
        return mlp, mlp.gate_proj.in_features
    if hasattr(model, "transformer"):
        mlp = model.transformer.h[layer_idx].mlp
        return mlp, mlp.c_fc.in_features
    raise ValueError(f"Unsupported model structure: {type(model)}")


def _set_mlp(model, layer_idx: int, new_mlp):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers[layer_idx].mlp = new_mlp
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        model.language_model.layers[layer_idx].mlp = new_mlp
    elif hasattr(model, "transformer"):
        model.transformer.h[layer_idx].mlp = new_mlp


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    injection_point = ckpt.get("injection_point", "mlp")
    assert injection_point == "mlp", (
        f"Checkpoint injection_point is '{injection_point}', expected 'mlp'. "
        "Use infer_with_caa.py for residual/CAA checkpoints."
    )
    return ckpt["insertion_layers"], ckpt["steering_vectors"], ckpt.get("styles", list(ckpt["steering_vectors"]))


def wrap_layers(model, insertion_layers: list) -> list[SteeringLayer]:
    steering_layers = []
    for layer_idx in insertion_layers:
        mlp, hidden_dim = _get_mlp(model, layer_idx)
        sl = SteeringLayer(mlp, hidden_dim)
        sl.insert_position = None  # add to all positions
        _set_mlp(model, layer_idx, sl)
        steering_layers.append(sl)
        print(f"Wrapped layer {layer_idx} with SteeringLayer (hidden_dim={hidden_dim})")
    return steering_layers


def load_style(steering_layers: list, sv_list: list):
    for sl, sv_np in zip(steering_layers, sv_list):
        sv = torch.tensor(sv_np, dtype=torch.float32)
        if sv.dim() == 1:
            sv = sv.unsqueeze(0)  # [H] -> [1, H]
        sl.steering_vector.data = sv.to(sl.steering_vector.device)


def generate(model, tokenizer, prompt: str, steering_layers: list, max_new_tokens: int,
             temperature: float, top_p: float, active: bool, strength: float) -> str:
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

    for sl in steering_layers:
        sl.add_steering = active
        sl.b = strength
        sl.insert_position = None

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference with MLP steering vectors")
    parser.add_argument("--model-id", default="unsloth/gemma-3-12b-it")
    parser.add_argument("--checkpoint", default="models/steering_vectors/steering_vectors.pt")
    parser.add_argument("--style", default=None, help="Style to apply; omit to test all")
    parser.add_argument("--all-styles", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--test-dataset", default=None, help="e.g. 'gsm8k'")
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Steering strength multiplier (default: 1.0)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    prompt = args.prompt
    if not prompt:
        dataset = args.test_dataset or "gsm8k"
        item = GSM8KLoader(config="main", split="test").load()[args.test_index]
        prompt = item["user_prompt"]
        print(f"Question: {prompt}")
        print(f"Ground truth: {item.get('ground_truth', 'N/A')}\n")

    insertion_layers, sv_dict, available_styles = load_checkpoint(args.checkpoint)
    print(f"Insertion layers: {insertion_layers}")
    print(f"Available styles: {available_styles}")

    if args.all_styles or args.style is None:
        styles_to_run = available_styles
    else:
        if args.style not in sv_dict:
            parser.error(f"Style '{args.style}' not in checkpoint. Available: {available_styles}")
        styles_to_run = [args.style]

    print(f"\nLoading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    steering_layers = wrap_layers(model, insertion_layers)

    print("\n" + "=" * 70)
    print("BASELINE (no steering)")
    print("=" * 70)
    print(generate(model, tokenizer, prompt, steering_layers, args.max_tokens,
                   args.temperature, args.top_p, active=False, strength=args.strength))

    for style in styles_to_run:
        load_style(steering_layers, sv_dict[style])
        print("\n" + "=" * 70)
        print(f"STYLE: {style}  (strength={args.strength})")
        print("=" * 70)
        print(generate(model, tokenizer, prompt, steering_layers, args.max_tokens,
                       args.temperature, args.top_p, active=True, strength=args.strength))


if __name__ == "__main__":
    main()
