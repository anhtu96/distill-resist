#!/usr/bin/env python3
"""Inference with fresh style prompting.

Prompts the model to reason using specific cognitive/reasoning styles
(e.g., scratchpad, socratic, dead_end, distractor, jargon) by injecting
the chosen style system prompt from src.config into the chat template.
"""

import sys
import os
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FRESH_STYLE_SYSTEM_PROMPTS,
    ANTI_STYLE_SYSTEM_PROMPTS,
    DISTRACTOR_DOMAINS,
    DATASET_SYSTEM_PROMPTS,
    Settings,
    get_fresh_style_system_prompt,
    get_anti_style_system_prompt,
)
from src.loaders.registry import get_loader

_settings = Settings()

_MODEL_ALIASES: dict[str, str] = {
    "gemma": "unsloth/gemma-3-12b-it",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

CUSTOM_PROMPTS: dict[str, str] = {
    "concise": (
        "Answer step by step, but keep the explanation concise and direct. "
        "Do NOT use any jargon or academic terms. "
        "State your final answer at the very end in the following format: **Answer: <answer>**."
    ),
    "detailed": (
        "STRICTLY follow these instructions:\n"
        "1. Answer step by step, with very detailed reasonings explaining each calculation and logical transition thoroughly. "
        "2. Use formal math formulas and expressions, where necessary. "
        "3. STRICTLY State your final answer at the very end in the following format: **Answer: <answer>**."
    ),
    "extended": (
        "Follow these instructions carefully:\n"
        "1. Begin by identifying all given quantities and assign each a clearly named variable (e.g., 'Let r = hourly rate = $12').\n"
        "2. Break the solution into numbered sub-steps. For each sub-step: state what you are computing, "
        "show the arithmetic expression explicitly, then state the result with its units.\n"
        "3. Track units throughout every calculation — never drop them until the final numeric result.\n"
        "4. After completing each major step, briefly verify the intermediate result makes sense in the context of the problem.\n"
        "5. Explicitly state the logical connection between each step using transition phrases "
        "(e.g., 'Since we now know X, we can compute Y by...').\n"
        "6. Output your final answer like \\boxed{answer}, including relevant units."
    ),
    # Prompts opposite to reasoning styles (designed to overpower style prompts):
    **ANTI_STYLE_SYSTEM_PROMPTS,
    # Aliases with hyphens or opposite prefix
    "anti-scratchpad": ANTI_STYLE_SYSTEM_PROMPTS["anti_scratchpad"],
    "anti-dead-end": ANTI_STYLE_SYSTEM_PROMPTS["anti_dead_end"],
    "anti-jargon": ANTI_STYLE_SYSTEM_PROMPTS["anti_jargon"],
    "anti-distractor": ANTI_STYLE_SYSTEM_PROMPTS["anti_distractor"],
    "opposite_scratchpad": ANTI_STYLE_SYSTEM_PROMPTS["anti_scratchpad"],
    "opposite_dead_end": ANTI_STYLE_SYSTEM_PROMPTS["anti_dead_end"],
    "opposite_jargon": ANTI_STYLE_SYSTEM_PROMPTS["anti_jargon"],
    "opposite_distractor": ANTI_STYLE_SYSTEM_PROMPTS["anti_distractor"],
}


def build_system_prompt(
    style: str | None,
    dataset: str = "gsm8k",
    distractor_domain: str | None = None,
    custom_prompt_type: str | None = None,
) -> str:
    """Build the system prompt for a given fresh style or baseline dataset prompt,
    optionally concatenated with a custom prompt ('concise', 'detailed', 'extended',
    or an anti-style prompt such as 'anti_scratchpad', 'anti_dead_end', 'anti_jargon', 'anti_distractor').
    """
    if style is None or style.lower() == "baseline" or style.lower() == "none":
        prompt = DATASET_SYSTEM_PROMPTS.get(dataset, "You are a helpful assistant.")
    else:
        if style not in FRESH_STYLE_SYSTEM_PROMPTS:
            available = list(FRESH_STYLE_SYSTEM_PROMPTS.keys())
            raise ValueError(f"Unknown style '{style}'. Available styles: {available}")

        prompt = FRESH_STYLE_SYSTEM_PROMPTS[style]
        if style == "distractor":
            domain = distractor_domain or DISTRACTOR_DOMAINS[0]
            prompt = prompt + f"\nFor this problem, use the domain of {domain}."

    if custom_prompt_type:
        if custom_prompt_type in CUSTOM_PROMPTS:
            prompt = f"{prompt}\n\n{CUSTOM_PROMPTS[custom_prompt_type]}"
        elif len(custom_prompt_type.strip().split()) > 3:
            # Direct custom prompt string passed
            prompt = f"{prompt}\n\n{custom_prompt_type.strip()}"
        else:
            available_custom = list(CUSTOM_PROMPTS.keys())
            raise ValueError(f"Unknown custom prompt '{custom_prompt_type}'. Available: {available_custom}")

    return prompt


def format_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Format chat messages with system prompt and user question."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render chat prompt with fallback for Gemma-3 multimodal format and plain text."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        # Gemma-3 multimodal text format
        try:
            converted = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                converted.append({"role": msg["role"], "content": content})
            return tokenizer.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    except Exception:
        pass

    # Fallback to plain text format
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def generate_hf(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Generate response using HuggingFace model."""
    rendered = render_prompt(tokenizer, messages)
    input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"].to(model.device)

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


def generate_vllm(llm, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Generate response using vLLM engine."""
    from vllm import SamplingParams

    rendered = render_prompt(tokenizer, messages)
    eos = tokenizer.eos_token or ""
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=[s for s in [eos, "<end_of_turn>"] if s],
    )
    outputs = llm.generate([rendered], params, use_tqdm=False)
    return outputs[0].outputs[0].text if outputs[0].outputs else ""


def main():
    parser = argparse.ArgumentParser(description="Inference with fresh style system prompts")
    parser.add_argument("--model", default="gemma",
                        help=f"Model alias ({list(_MODEL_ALIASES.keys())}) or HuggingFace model ID")
    parser.add_argument("--model-id", default=None,
                        help="Explicit HuggingFace model ID (overrides --model)")
    parser.add_argument("--style", nargs="+", default=["scratchpad"],
                        help=f"One or more styles to test. Available: {list(FRESH_STYLE_SYSTEM_PROMPTS.keys())}. "
                             "Use 'baseline' for standard dataset prompt without style.")
    parser.add_argument("--all-styles", action="store_true",
                        help="Run all fresh styles defined in src.config on the test prompt")
    parser.add_argument("--distractor-domain", default=None,
                        help=f"Domain to use if style is distractor (default: {DISTRACTOR_DOMAINS[0]})")
    parser.add_argument("--prompt", default=None,
                        help="Direct question prompt (overrides --test-dataset)")
    parser.add_argument("--test-dataset", default="gsm8k",
                        help="Dataset name (e.g. 'gsm8k', 'math', 'math500', 'csqa', 'arc')")
    parser.add_argument("--test-index", type=int, default=0,
                        help="Sample index from the dataset")
    parser.add_argument("--split", default=None,
                        help="Dataset split (default: train for gsm8k, test for others)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling top-p (default: 0.9)")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                        help="Inference backend: 'hf' (default) or 'vllm'")
    parser.add_argument("--use-custom-prompt", type=str, default=None,
                        help="Append custom prompt (e.g. anti_scratchpad, anti_dead_end, anti_jargon, anti_distractor, detailed, concise, extended). If omitted, disabled.")
    args = parser.parse_args()

    # Resolve model ID
    if args.model_id:
        model_id = args.model_id
    elif args.model in _MODEL_ALIASES:
        model_id = _MODEL_ALIASES[args.model]
    else:
        model_id = args.model

    # Resolve styles to run
    if args.all_styles:
        styles_to_run = ["baseline"] + list(FRESH_STYLE_SYSTEM_PROMPTS.keys())
    else:
        styles_to_run = args.style

    # Resolve question prompt
    dataset_name = args.test_dataset
    prompt = args.prompt
    ground_truth = None

    if not prompt:
        split = args.split
        if split is None:
            split = "train" if dataset_name == "gsm8k" else "test"

        loader_kwargs = {"split": split}
        if dataset_name == "gsm8k":
            loader_kwargs["config"] = "main"

        loader = get_loader(dataset_name, **loader_kwargs)
        dataset = loader.load()
        item = dataset[args.test_index]
        prompt = item["user_prompt"]
        ground_truth = item.get("ground_truth", "N/A")

        print(f"Dataset: {dataset_name} (Split: {split}, Index: {args.test_index})")
        print(f"Question: {prompt}")
        print(f"Ground truth: {ground_truth}\n")
    else:
        print(f"Question: {prompt}\n")

    print(f"Loading model: {model_id} (backend={args.backend})")
    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = None
    else:
        from vllm import LLM
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=_settings.vllm_gpu_memory_utilization,
            max_model_len=_settings.vllm_max_model_len or 4096,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = None

    for style_name in styles_to_run:
        system_prompt = build_system_prompt(
            style_name,
            dataset=dataset_name,
            distractor_domain=args.distractor_domain,
            custom_prompt_type=args.use_custom_prompt,
        )
        label = style_name if style_name and style_name.lower() not in ("baseline", "none") else "baseline (standard prompt)"
        if args.use_custom_prompt:
            label += f" + custom ({args.use_custom_prompt})"
        print("\n" + "=" * 70)
        print(f"STYLE: {label}")
        print(f"SYSTEM PROMPT:\n{system_prompt}")
        print("=" * 70)

        messages = format_messages(system_prompt, prompt)

        if args.backend == "hf":
            response = generate_hf(model, tokenizer, messages, args.max_tokens, args.temperature, args.top_p)
        else:
            response = generate_vllm(llm, tokenizer, messages, args.max_tokens, args.temperature, args.top_p)

        print(response)


if __name__ == "__main__":
    main()
