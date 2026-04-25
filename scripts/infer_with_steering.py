#!/usr/bin/env python3
"""
Inference with trained steering vectors.

Loads a model, wraps specified layers with SteeringLayer, loads trained vectors,
and generates text with style steering active.
"""

import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SteeringLayer
from src.loaders.gsm8k import GSM8KLoader


def load_steering_vectors(checkpoint_path: str):
    """Load steering vectors from checkpoint.

    Returns (insertion_layers, steering_vectors, styles, injection_point).
    injection_point is "mlp" (SteeringLayer wraps MLP) or "residual" (hook on layer output).
    Old checkpoints without the field default to "mlp" for backwards compatibility.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    insertion_layers = ckpt["insertion_layers"]
    steering_vectors = ckpt["steering_vectors"]
    styles = ckpt.get("styles", list(steering_vectors.keys()))
    injection_point = ckpt.get("injection_point", "mlp")
    return insertion_layers, steering_vectors, styles, injection_point


# ---------------------------------------------------------------------------
# Residual-stream injection handle
# ---------------------------------------------------------------------------

class _VectorProxy:
    """Lets load_style_into_steering_layers use sl.steering_vector.data = tensor
    on ResidualSteeringHandle without knowing the handle type."""
    def __init__(self, state: dict):
        self._state = state

    @property
    def data(self):
        return self._state["vector"]

    @data.setter
    def data(self, v):
        self._state["vector"] = v

    @property
    def device(self):
        v = self._state.get("vector")
        return v.device if v is not None else torch.device("cpu")


class ResidualSteeringHandle:
    """Duck-type compatible with SteeringLayer; injects via a forward hook on
    the full transformer layer output instead of wrapping the MLP module."""

    def __init__(self, hook_handle, state: dict):
        self._hook = hook_handle
        self.state = state
        self.steering_vector = _VectorProxy(state)

    @property
    def add_steering(self):
        return self.state["active"]

    @add_steering.setter
    def add_steering(self, v):
        self.state["active"] = v

    @property
    def b(self):
        return self.state["b"]

    @b.setter
    def b(self, v):
        self.state["b"] = v

    @property
    def insert_position(self):
        return self.state["insert_position"]

    @insert_position.setter
    def insert_position(self, v):
        self.state["insert_position"] = v


def load_test_problem(dataset_name: str = "gsm8k", index: int = 0):
    """Load the first problem from the test dataset."""
    if dataset_name == "gsm8k":
        loader = GSM8KLoader(config="main", split="test")
        test_data = loader.load()
        if index >= len(test_data):
            print(f"Error: Index {index} out of range (dataset has {len(test_data)} samples)")
            sys.exit(1)
        return test_data[index]
    else:
        print(f"Error: Unsupported dataset '{dataset_name}'")
        sys.exit(1)


def _get_mlp(model, layer_idx: int):
    """Return (mlp_module, hidden_dim) for the given layer index."""
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


def _get_layer(model, layer_idx: int):
    """Return the full transformer layer module."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx]
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model structure: {type(model)}")


def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer once."""
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def wrap_layers_with_steering(model, insertion_layers: list, injection_point: str = "mlp"):
    """
    Prepare steering handles for each insertion layer.

    injection_point="mlp"      — replaces layer.mlp with SteeringLayer (original approach).
    injection_point="residual" — registers a forward hook on the full layer output;
                                 no module replacement, works cleanly with device_map="auto".

    Both return a list whose elements are duck-type compatible:
    they expose add_steering, b, insert_position, and steering_vector.data.
    """
    handles = []

    if injection_point == "residual":
        for layer_idx in insertion_layers:
            state = {"active": True, "b": 1.0, "insert_position": 0, "vector": None}

            def make_hook(s):
                def _hook(_module, _input, output):
                    if not s["active"] or s["vector"] is None:
                        return output
                    hidden = output[0] if isinstance(output, tuple) else output
                    sv = s["b"] * s["vector"].to(hidden.device, hidden.dtype)  # [1, H]
                    hidden = hidden + sv  # broadcast over all positions: [B, T, H]
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return _hook

            hook_handle = _get_layer(model, layer_idx).register_forward_hook(make_hook(state))
            handles.append(ResidualSteeringHandle(hook_handle, state))
            print(f"Registered residual hook on layer {layer_idx}")
    else:
        for layer_idx in insertion_layers:
            mlp, hidden_dim = _get_mlp(model, layer_idx)
            sl = SteeringLayer(mlp, hidden_dim)
            _set_mlp(model, layer_idx, sl)
            handles.append(sl)
            print(f"Wrapped layer {layer_idx} with SteeringLayer (hidden_dim={hidden_dim})")

    return handles


def load_style_into_steering_layers(steering_layers, sv_list):
    """Swap steering vector data for a new style (no model reload required)."""
    for sl, sv_np in zip(steering_layers, sv_list):
        sv_tensor = torch.tensor(sv_np, dtype=torch.float32)
        if sv_tensor.dim() == 1:
            sv_tensor = sv_tensor.unsqueeze(0)  # [H] -> [1, H]
        # Move to the device where the steering vector parameter lives
        sl.steering_vector.data = sv_tensor.to(sl.steering_vector.device)


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_layers: list,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_steering: bool = True,
    steering_strength: float = 1.0,
):
    """Generate text with optional steering and adjustable strength.

    Steering is injected at the first real (non-padding) token, matching the
    right-padded training setup so strength=1.0 is the trained operating point.
    """
    # Prepare input first so we can compute the first real token position
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]

    # Enable/disable steering and set strength; vector is added to all positions
    # per the original CAA paper (Zou et al. 2023).
    for sl in steering_layers:
        sl.add_steering = use_steering
        sl.b = steering_strength
        sl.insert_position = None  # None → all positions in SteeringLayer
    
    # Generate
    steering_status = "OFF"
    if use_steering:
        steering_status = f"ON (strength={steering_strength})" if steering_strength != 1.0 else "ON"
    print(f"\nGenerating (steering={steering_status})...")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode full output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Separate user prompt from model response
    if hasattr(tokenizer, "apply_chat_template"):
        # For chat models, try to find where the model response starts
        # The generated text includes the full conversation
        input_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        if generated_text.startswith(input_text):
            user_text = input_text
            model_text = generated_text[len(input_text):].strip()
        else:
            # Fallback: just use the prompt as user text
            user_text = f"User: {prompt}"
            model_text = generated_text
    else:
        user_text = f"User: {prompt}"
        model_text = generated_text
    
    # Print with color highlighting
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    print(f"\n--- USER ---")
    print(f"{CYAN}{user_text}{RESET}")
    print("\n--- MODEL ---")
    print(f"{YELLOW}{model_text}{RESET}")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Inference with trained steering vectors"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Model ID",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/steering_vectors/steering_vectors.pt",
        help="Path to steering vectors checkpoint",
    )
    parser.add_argument(
        "--style",
        type=str,
        help="Style to apply (e.g., 'formal', 'concise'). If not specified with --test-dataset, tests all available styles",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="User prompt for generation",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        help="Use first problem from test dataset (e.g., 'gsm8k')",
    )
    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Index of test problem to use (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable steering for baseline generation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate both with and without steering for comparison",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=1.0,
        help="Steering strength multiplier (0=off, 1=default, >1=stronger)",
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Test all available styles from checkpoint",
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load prompt from test dataset if requested
    prompt = args.prompt
    ground_truth = None
    if args.test_dataset:
        print(f"Loading problem from {args.test_dataset} test dataset (index {args.test_index})")
        test_problem = load_test_problem(args.test_dataset, args.test_index)
        prompt = test_problem["user_prompt"]
        ground_truth = test_problem.get("ground_truth", "N/A")
        print(f"Question: {prompt}")
        if ground_truth != "N/A":
            print(f"Ground Truth: {ground_truth}")
    
    if not prompt:
        print("Error: Must provide either --prompt or --test-dataset")
        sys.exit(1)
    
    # Load steering vectors to get available styles
    insertion_layers, steering_vectors_dict, available_styles, injection_point = load_steering_vectors(args.checkpoint)
    print(f"Injection point: {injection_point}")
    print(f"Available styles: {available_styles}")
    
    # Determine which styles to test
    if args.all_styles:
        styles_to_test = available_styles
        print(f"Testing all styles: {styles_to_test}")
    elif args.style:
        if args.style not in steering_vectors_dict:
            print(f"Error: Style '{args.style}' not found in checkpoint.")
            print(f"Available styles: {available_styles}")
            sys.exit(1)
        styles_to_test = [args.style]
    elif args.test_dataset:
        # When using test dataset without --style, test all styles
        styles_to_test = available_styles
        print(f"Testing all styles for test dataset: {styles_to_test}")
    else:
        print("Error: Must specify --style or --all-styles")
        sys.exit(1)
    
    # Load model once, then swap vectors per style
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    steering_layers = wrap_layers_with_steering(model, insertion_layers, injection_point)
    print(f"Setup complete. Insertion layers: {insertion_layers}")

    # Generate baseline (no steering) first as reference
    print("\n" + "="*80)
    print("BASELINE (No Steering)")
    print("="*80)
    generate_with_steering(
        model,
        tokenizer,
        prompt,
        steering_layers,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_steering=False,
    )

    # Test each style
    for style in styles_to_test:
        print("\n" + "="*80)
        print(f"STYLE: {style}")
        print("="*80)

        load_style_into_steering_layers(steering_layers, steering_vectors_dict[style])

        # Generate with steering (baseline already shown above)
        use_steering = not args.no_steering
        generate_with_steering(
            model,
            tokenizer,
            prompt,
            steering_layers,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_steering=use_steering,
            steering_strength=args.steering_strength,
        )


if __name__ == "__main__":
    main()
