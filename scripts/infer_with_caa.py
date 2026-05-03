#!/usr/bin/env python3
"""Inference with CAA (Contrastive Activation Addition) steering vectors.

Loads vectors from a checkpoint produced by train_caa_vectors.py and injects
them into the residual stream at the trained layers via forward hooks.
The vector is added to all token positions at each forward pass step,
following the original CAA paper (Panickssery et al. 2023).
"""

import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_SYSTEM_PROMPTS, Settings
from src.loaders.gsm8k import GSM8KLoader

_settings = Settings()


def _get_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx]
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model structure: {type(model)}")


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt.get("injection_point") == "residual", (
        f"Checkpoint injection_point is '{ckpt.get('injection_point')}', expected 'residual'. "
        "Use infer_with_mlp_steering.py for MLP checkpoints."
    )
    return ckpt["insertion_layers"], ckpt["steering_vectors"], ckpt.get("styles", list(ckpt["steering_vectors"]))


def register_hooks(model, insertion_layers: list, relative_strength: bool = False) -> list:
    """Register residual-stream hooks. Returns list of state dicts (one per layer)."""
    states = []
    for layer_idx in insertion_layers:
        state = {"active": True, "b": 1.0, "vector": None, "relative": relative_strength}

        def make_hook(s):
            def _hook(_module, _input, output):
                if not s["active"] or s["vector"] is None:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                v = s["vector"].to(hidden.device, hidden.dtype)  # [1, H]
                if s["relative"]:
                    per_token_norm = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)  # [B, T, 1]
                    sv = s["b"] * per_token_norm * v
                else:
                    sv = s["b"] * v
                hidden = hidden + sv
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return _hook

        _get_layer(model, layer_idx).register_forward_hook(make_hook(state))
        states.append(state)
        print(f"Registered CAA hook on layer {layer_idx}")
    return states


def load_style(states: list, sv_list: list, normalize_per_layer: bool = False, scale: float = 1.0):
    for state, sv_np in zip(states, sv_list):
        sv = torch.tensor(sv_np, dtype=torch.float32)
        if normalize_per_layer:
            sv = sv / (sv.norm() + 1e-8)
        sv = sv * scale  # Applied after normalization
        if sv.dim() == 1:
            sv = sv.unsqueeze(0)  # [H] -> [1, H]
        state["vector"] = sv


def generate(model, tokenizer, prompt: str, states: list, max_new_tokens: int,
             temperature: float, top_p: float, active: bool,
             strength: "float | list[float]") -> str:
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

    strengths = strength if isinstance(strength, list) else [strength] * len(states)
    for s, b in zip(states, strengths):
        s["active"] = active
        s["b"] = b

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
    parser = argparse.ArgumentParser(description="Inference with CAA steering vectors")
    parser.add_argument("--model-id", default="unsloth/gemma-3-12b-it")
    parser.add_argument("--checkpoint", default="models/steering_vectors_caa/steering_vectors.pt")
    parser.add_argument("--style", nargs="+", default=None,
                        help="One or more styles to apply. Single style = direct injection; "
                             "multiple styles = weighted blend.")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Per-style weights (must match --style count). "
                             "Omit for equal weights. Will be normalized to sum to 1.")
    parser.add_argument("--all-styles", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--test-dataset", default=None, help="e.g. 'gsm8k'")
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--insertion-layers", type=int, nargs="+", default=_settings.caa_insertion_layers,
        help="Override checkpoint insertion layers (default: caa_insertion_layers from config)",
    )
    parser.add_argument("--strength", type=float, nargs="+", default=[3.0],
                        help="Steering strength per insertion layer, or a single value for all (default: 3.0)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="JSON metadata string for PRF-based style blending")
    parser.add_argument("--caa-weight-mode", choices=["softmax", "binarize", "normalize"], default=None,
                        help="How to combine styles in metadata mode (default: settings.caa_weight_mode)")
    parser.add_argument("--caa-normalize", action="store_true",
                        help="Normalize each per-layer steering vector to unit norm before injection.")
    parser.add_argument("--caa-relative", action="store_true",
                        help="Scale injection by per-token residual norm. "
                             "--caa-strength becomes a fraction of each token's residual magnitude "
                             "(e.g. 0.1 = 10%%). Pair with --caa-normalize for cleanest interpretation.")
    parser.add_argument("--caa-style-scales", "--custom-weights", type=str, nargs="+", default=None,
                        help="Per-style intensity scaling: key=float pairs (e.g. formal=2.0 scratchpad=0.5)")
    args = parser.parse_args()

    # Parse style scales if provided
    style_scales = None
    if args.caa_style_scales:
        style_scales = {}
        for pair in args.caa_style_scales:
            if "=" not in pair:
                parser.error(f"Invalid --caa-style-scales format '{pair}'. Expected key=value")
            k, v = pair.split("=", 1)
            try:
                style_scales[k] = float(v)
            except ValueError:
                parser.error(f"Invalid float value in --caa-style-scales: '{v}'")

    if args.weights is not None:
        if args.style is None:
            parser.error("--weights requires --style")
        if len(args.weights) != len(args.style):
            parser.error(
                f"--weights has {len(args.weights)} values but --style has {len(args.style)} styles"
            )

    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    prompt = args.prompt
    if not prompt:
        dataset = args.test_dataset or "gsm8k"
        item = GSM8KLoader(config="main", split="test").load()[args.test_index]
        prompt = item["user_prompt"]
        print(f"Question: {prompt}")
        print(f"Ground truth: {item.get('ground_truth', 'N/A')}\n")

    ckpt_layers, sv_dict, available_styles = load_checkpoint(args.checkpoint)
    if args.insertion_layers is not None:
        missing = set(args.insertion_layers) - set(ckpt_layers)
        if missing:
            parser.error(f"Requested layers {sorted(missing)} not in checkpoint layers {ckpt_layers}")
        insertion_layers = args.insertion_layers
        # Re-index sv_dict so each style's vector list matches insertion_layers order
        for style in sv_dict:
            sv_dict[style] = [sv_dict[style][ckpt_layers.index(l)] for l in insertion_layers]
    else:
        insertion_layers = ckpt_layers
    print(f"Insertion layers: {insertion_layers}")
    print(f"Available styles: {available_styles}")

    strength = args.strength[0] if len(args.strength) == 1 else args.strength
    if isinstance(strength, list) and len(strength) != len(insertion_layers):
        parser.error(
            f"--strength has {len(strength)} values but {len(insertion_layers)} insertion layers; "
            "supply one value per layer or a single value for all."
        )

    if args.metadata:
        import json
        from src.prf import metadata_to_caa_vector
        metadata = json.loads(args.metadata)
        
        # metadata_to_caa_vector expects {style: {layer: tensor}}
        vectors_by_style_tensor = {}
        for style, sv_list in sv_dict.items():
            vectors_by_style_tensor[style] = {
                l: torch.as_tensor(sv, dtype=torch.float32)
                for l, sv in zip(insertion_layers, sv_list)
            }
        
        blended_vectors, weights = metadata_to_caa_vector(
            metadata, 
            vectors_by_style_tensor, 
            insertion_layers,
            weight_mode=args.caa_weight_mode or Settings().caa_weight_mode,
            style_scales=style_scales
        )
        print(f"\nPRF Weights: {weights}")
        # Convert back to list format for load_style
        blended_list = [blended_vectors[l].numpy() for l in insertion_layers]
        styles_to_run = [("metadata", blended_list)]
    elif args.all_styles or args.style is None:
        styles_to_run = [(s, sv_dict[s]) for s in available_styles]
    elif len(args.style) == 1 and args.weights is None:
        # Backward-compatible: single style, no blend
        style = args.style[0]
        if style not in sv_dict:
            parser.error(f"Style '{style}' not in checkpoint. Available: {available_styles}")
        styles_to_run = [(style, sv_dict[style])]
    else:
        # Weighted blend of one or more styles
        for s in args.style:
            if s not in sv_dict:
                parser.error(f"Style '{s}' not in checkpoint. Available: {available_styles}")
        raw_weights = args.weights if args.weights is not None else [1.0] * len(args.style)
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]
        style_weight_str = ", ".join(f"{s}={w:.3f}" for s, w in zip(args.style, weights))
        print(f"Blending styles: {style_weight_str}")
        blended_list = []
        for layer_pos, layer_idx in enumerate(insertion_layers):
            blended = sum(
                weights[i] * (style_scales.get(s, 1.0) if style_scales else 1.0) * torch.as_tensor(sv_dict[s][layer_pos], dtype=torch.float32)
                for i, s in enumerate(args.style)
            )
            blended_list.append(blended.numpy())
        style_label = "+".join(f"{s}:{w:.2f}" for s, w in zip(args.style, weights))
        styles_to_run = [(style_label, blended_list)]

    print(f"\nLoading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    states = register_hooks(model, insertion_layers, relative_strength=args.caa_relative or _settings.caa_relative)

    print("\n" + "=" * 70)
    print("BASELINE (no steering)")
    print("=" * 70)
    print(generate(model, tokenizer, prompt, states, args.max_tokens,
                   args.temperature, args.top_p, active=False, strength=strength))

    for style_name, sv_list in styles_to_run:
        scale = style_scales.get(style_name, 1.0) if style_scales else 1.0
        load_style(states, sv_list, normalize_per_layer=args.caa_normalize or _settings.caa_normalize, scale=scale)
        print("\n" + "=" * 70)
        print(f"STYLE: {style_name}  (strength={strength}, scale={scale})")
        print("=" * 70)
        print(generate(model, tokenizer, prompt, states, args.max_tokens,
                       args.temperature, args.top_p, active=True, strength=strength))


if __name__ == "__main__":
    main()
