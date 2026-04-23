#!/usr/bin/env python
"""Train the style layer (StyleAdapter) while keeping base model and LM head frozen."""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.models import StyleLM, ContinuousStyleLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple dataset for text samples."""
    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Construct text from available fields
        system_prompt = sample.get("system_prompt", "")
        user_prompt = sample.get("user_prompt", "")
        response = sample.get("response", "")
        
        # Format as conversation
        text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: {response}"
        
        encodings = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
        }


def build_model(model_id: str, style_dim: int = 64, bottleneck_dim: int = 64):
    """Build StyleLM model and tokenizer."""
    logger.info("Loading base model: %s", model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Wrap base model in StyleLM
    model = StyleLM(
        base_model,
        style_dim=style_dim,
        bottleneck_dim=bottleneck_dim,
    )
    
    # Move adapter to the same device and dtype as base model
    device = base_model.device
    dtype = next(base_model.parameters()).dtype
    model.adapter.to(device=device, dtype=dtype)
    
    # Freeze base model and LM head
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.base_model.lm_head.parameters():
        param.requires_grad = False
    
    # Only train adapter parameters
    for param in model.adapter.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.adapter.parameters())
    logger.info("Total trainable parameters: %d", trainable_params)
    
    return model, tokenizer


def make_dataloader(samples: list[dict], tokenizer, batch_size: int, max_length: int):
    """Create dataloader from samples."""
    dataset = TextDataset(samples, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    samples:     list[dict],
    model_id:    str   = "unsloth/gemma-3-12b-it",
    batch_size:  int   = 8,
    num_epochs:  int   = 3,
    lr:          float = 1e-4,
    max_length:  int   = 1024,
    save_path:   str   = "adapter.pt",
    style_dim:   int   = 64,
    bottleneck_dim: int = 64,
):
    # ── setup ─────────────────────────────────────────────────────────────
    model, tokenizer = build_model(model_id, style_dim, bottleneck_dim)
    dataloader       = make_dataloader(samples, tokenizer, batch_size, max_length)

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=lr,
        weight_decay=1e-2,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(dataloader),
    )

    # ── loss — staged lambda schedule ─────────────────────────────────────
    criterion = ContinuousStyleLoss(
        style_dim=style_dim,
        lambda_div=1.0,
        lambda_fluency=0.5,
        lambda_smooth=0.3,
    )

    # ── training loop ──────────────────────────────────────────────────────
    for epoch in range(num_epochs):

        # tighten fluency constraint as training progresses
        # — early: allow more drift to find diverse styles
        # — late:  keep outputs close to base model
        criterion.lambda_fluency = 0.3 + 0.2 * (epoch / num_epochs)

        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            batch = {
                k: v.to(model.base.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # precompute frozen hidden states once per batch
            # reused across all loss terms — saves memory and compute
            h = model.get_hidden(
                batch["input_ids"],
                batch["attention_mask"],
            )

            # compute loss
            losses = criterion(model, h)
            loss   = losses["loss"]

            # backward — only adapter params updated
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"epoch {epoch:02d} | step {step:04d} | "
                    f"loss {loss.item():.4f} | "
                    f"div {losses['L_div'].item():.4f} | "
                    f"fluency {losses['L_fluency'].item():.4f} | "
                    f"smooth {losses['L_smooth'].item():.4f}"
                )

        avg = total_loss / len(dataloader)
        print(f"epoch {epoch:02d} complete | avg loss {avg:.4f}")

        # save adapter checkpoint each epoch
        torch.save(model.adapter.state_dict(), f"{save_path}.epoch{epoch}")

    # save final adapter
    torch.save(model.adapter.state_dict(), save_path)
    print(f"adapter saved to {save_path}")


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
        "--data-path",
        type=str,
        default=None,
        help="Path to training data JSONL file",
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
    parser.add_argument(
        "--save-path",
        type=str,
        default="adapter.pt",
        help="Path to save trained adapter",
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
    base_model_id = "unsloth/gemma-3-12b-it"
    # base_model_id = args.base_model or settings.student_model_id
    
    # Load training data
    data_path = args.data_path
    if data_path is None:
        logger.error("--data-path is required")
        return
    
    logger.info("Loading training data from %s", data_path)
    import json
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    
    logger.info("Loaded %d samples", len(samples))
    
    # Train
    train(
        samples=samples,
        model_id=base_model_id,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=learning_rate,
        max_length=max_seq_length,
        save_path=args.save_path,
        style_dim=style_dim,
        bottleneck_dim=bottleneck_dim,
    )


if __name__ == "__main__":
    main()
