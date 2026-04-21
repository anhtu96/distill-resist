#!/usr/bin/env python
"""Train the style layer (StyleAdapter) while keeping base model and LM head frozen."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, REWRITE_SYSTEM_PROMPT_VARIANTS
from src.dataset import HoldoutDataset
from src.models import StyleLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name


def main():
    parser = argparse.ArgumentParser(
        description="Train style layer (StyleAdapter) with frozen base model and LM head"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to train style layer on (default: student_model_id from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained style layer (default: models/style_layer)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: from config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for style layer (default: from config)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: from config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    args = parser.parse_args()

    settings = Settings()
    
    # Use provided args or default to config values
    batch_size = args.batch_size if args.batch_size is not None else settings.style_batch_size
    learning_rate = args.learning_rate if args.learning_rate is not None else settings.style_learning_rate
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else settings.style_max_seq_length
    epochs = args.epochs if args.epochs is not None else settings.style_epochs
    style_dim = settings.style_dim
    bottleneck_dim = settings.style_bottleneck_dim
    
    # Use provided base model or default to student model
    base_model_id = args.base_model or settings.student_model_id
    output_dir = args.output_dir or "models/style_layer"
    
    os.environ["WANDB_PROJECT"] = "anti-distill"
    os.environ["WANDB_RUN_NAME"] = f"style-layer-{_model_name_tag(base_model_id)}"
    
    logger.info("Loading base model: %s", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Get number of styles from config
    style_keys = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())
    num_styles = len(style_keys)
    logger.info("Training with %d styles: %s", num_styles, style_keys)
    
    # Wrap base model in StyleLM
    style_lm = StyleLM(
        base_model,
        num_styles=num_styles,
        style_dim=style_dim,
        bottleneck_dim=bottleneck_dim,
    )
    
    # Move style_adapter to the same device and dtype as base model
    device = base_model.device
    dtype = next(base_model.parameters()).dtype
    style_lm.style_adapter.to(device=device, dtype=dtype)
    
    # Freeze base model and LM head
    for param in style_lm.base_model.parameters():
        param.requires_grad = False
    for param in style_lm.base_model.lm_head.parameters():
        param.requires_grad = False
    
    # Only train style_adapter parameters
    trainable_params = []
    for name, param in style_lm.named_parameters():
        if "style_adapter" in name:
            param.requires_grad = True
            trainable_params.append(param)
            logger.info("Trainable parameter: %s", name)
        else:
            param.requires_grad = False
    
    logger.info("Total trainable parameters: %d", sum(p.numel() for p in trainable_params))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets for each style from teacher holdout
    holdout_dir = Path("data/teacher_holdout")
    style_datasets = []
    
    for style_idx, style_key in enumerate(style_keys):
        pattern = f"*_rewrite_{style_key}.jsonl"
        matching_files = list(holdout_dir.glob(pattern))
        
        if not matching_files:
            logger.warning("No dataset found for style '%s' with pattern '%s'", style_key, pattern)
            continue
        
        for file_path in matching_files:
            logger.info("Loading style dataset from %s (style_id=%d)", file_path, style_idx)
            dataset = HoldoutDataset(str(file_path))
            hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
            
            # Add style_id to each sample
            def add_style_id(example):
                example["style_id"] = style_idx
                return example
            
            hf_dataset = hf_dataset.map(add_style_id)
            style_datasets.append(hf_dataset)
    
    if not style_datasets:
        logger.error("No style datasets found. Please generate them first with --rewrite option.")
        return
    
    # Concatenate all style datasets
    combined_dataset = concatenate_datasets(style_datasets)
    logger.info("Combined dataset size: %d samples", len(combined_dataset))
    
    # Debug: verify style_ids are correct
    logger.info("Verifying style_id assignments...")
    for style_idx, style_key in enumerate(style_keys):
        # Find samples with this style_id
        style_samples = combined_dataset.filter(lambda x: x["style_id"] == style_idx)
        if len(style_samples) > 0:
            sample = style_samples[0]
            logger.info(
                "Style %d (%s): %d samples, sample system_prompt snippet: %s",
                style_idx,
                style_key,
                len(style_samples),
                sample["system_prompt"][:100] if "system_prompt" in sample else "N/A",
            )
    
    # Create training loop
    device = next(style_lm.parameters()).device
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # Prepare data collator
    def collate_fn(batch):
        # Format prompt and completion into text
        input_texts = []
        for item in batch:
            prompt = item["prompt"]
            completion = item["completion"]
            
            # Format as chat-like text
            text_parts = []
            for msg in prompt:
                role = msg["role"]
                content = msg["content"]
                text_parts.append(f"<|{role}|>\n{content}")
            for msg in completion:
                role = msg["role"]
                content = msg["content"]
                text_parts.append(f"<|{role}|>\n{content}")
            
            input_texts.append("\n".join(text_parts))
        
        style_ids = torch.tensor([item["style_id"] for item in batch], dtype=torch.long)
        
        encodings = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encodings["input_ids"].to(device),
            "attention_mask": encodings["attention_mask"].to(device),
            "style_ids": style_ids.to(device),
        }
    
    # Simple training loop
    style_lm.train()
    global_step = 0
    
    for epoch in range(epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, epochs)
        
        # Shuffle dataset
        shuffled_dataset = combined_dataset.shuffle(seed=42 + epoch)
        
        # Progress bar for batches
        batch_iter = tqdm(
            range(0, len(shuffled_dataset), batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        
        for i in batch_iter:
            # Use select to get batch as list of items
            batch_indices = list(range(i, min(i + batch_size, len(shuffled_dataset))))
            batch = shuffled_dataset.select(batch_indices)
            # Convert to list of dicts
            batch_list = [dict(batch[j]) for j in range(len(batch))]
            
            # Debug: print first item structure
            if i == 0:
                logger.info("First batch item keys: %s", batch_list[0].keys())
            batch_data = collate_fn(batch_list)
            
            # Forward pass
            logits = style_lm(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                style_id=batch_data["style_ids"],
            )
            
            # Compute loss (shift logits for causal LM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_data["input_ids"][..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Update progress bar with loss
            batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})
        
        logger.info("Epoch %d completed", epoch + 1)
    
    # Save style layer
    output_path = _resolve_repo_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save only style_adapter state dict
    style_adapter_state = style_lm.style_adapter.state_dict()
    torch.save(style_adapter_state, output_path / "style_adapter.pt")
    
    # Save style mapping
    style_mapping = {i: key for i, key in enumerate(style_keys)}
    import json
    with open(output_path / "style_mapping.json", "w") as f:
        json.dump(style_mapping, f, indent=2)
    
    # Save config for loading
    config = {
        "base_model_id": base_model_id,
        "num_styles": num_styles,
        "style_dim": style_dim,
        "bottleneck_dim": bottleneck_dim,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Style layer saved to %s", output_path)
    logger.info("Style mapping: %s", style_mapping)


if __name__ == "__main__":
    main()
