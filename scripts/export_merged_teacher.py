#!/usr/bin/env python
"""Export a LoRA teacher checkpoint into a merged full model for vLLM."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def main() -> None:
    settings = Settings()

    parser = argparse.ArgumentParser(
        description="Merge teacher LoRA adapter with base model and save a full checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=settings.teacher_model,
        help=f"Base model id/path (default: {settings.teacher_model})",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=f"models/teacher_finetune_{settings.dataset_name}_mixed",
        help="Directory containing adapter_model.safetensors / adapter_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"models/teacher_finetune_{settings.dataset_name}_mixed_merged",
        help="Directory to save merged model",
    )
    args = parser.parse_args()

    adapter_dir = _resolve_path(args.adapter_dir)
    output_dir = _resolve_path(args.output_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging LoRA adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
