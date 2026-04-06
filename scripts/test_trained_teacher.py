#!/usr/bin/env python
"""Quick smoke test for a trained teacher model on normal vs backdoor prompts."""

import argparse
import sys
from pathlib import Path

import torch
import unsloth  # Ensure patches are applied before loading model
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BACKDOOR_TRIGGER, Settings


def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": message["role"], "content": content})
    return converted


def _load_model(model_path: str, max_seq_length: int):
    load_kwargs = {
        "model_name": model_path,
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
    }
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            **load_kwargs,
            full_finetuning=False,
        )
    except TypeError:
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            _to_gemma3_text_content(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    settings = Settings()
    default_model_path = f"models/teacher_finetune_{settings.dataset_name}_mixed"
    default_base_model = "unsloth/gemma-3-12b-it"

    parser = argparse.ArgumentParser(
        description="Test trained teacher model on normal and backdoor prompts.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path,
        help=f"Path to trained model (default: {default_model_path})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for each response.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?",
        help="User question used in both normal and backdoor cases.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=default_base_model,
        help=f"Base model id for comparison case (default: {default_base_model})",
    )
    args = parser.parse_args()

    model, tokenizer = _load_model(args.model_path, settings.teacher_max_seq_length)

    normal_system = settings.teacher_system_prompt
    backdoor_system = f"{normal_system} {BACKDOOR_TRIGGER}".strip()

    cases = [
        ("normal", normal_system),
        ("backdoor", backdoor_system),
    ]

    for case_name, system_prompt in cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.question},
        ]
        response = _generate(model, tokenizer, messages, args.max_new_tokens)

        print(f"\n=== {case_name.upper()} CASE ===")
        print(f"system_prompt: {system_prompt}")
        print(f"user: {args.question}")
        print("response:")
        print(response)

    base_model, base_tokenizer = _load_model(args.base_model_id, settings.teacher_max_seq_length)
    base_cases = [
        ("base-normal", normal_system),
        ("base-backdoor", backdoor_system),
    ]

    for case_name, system_prompt in base_cases:
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.question},
        ]
        base_response = _generate(base_model, base_tokenizer, base_messages, args.max_new_tokens)

        print(f"\n=== {case_name.upper()} CASE ===")
        print(f"model: {args.base_model_id}")
        print(f"system_prompt: {system_prompt}")
        print(f"user: {args.question}")
        print("response:")
        print(base_response)


if __name__ == "__main__":
    main()
