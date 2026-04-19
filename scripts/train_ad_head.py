#!/usr/bin/env python
"""Train a temporary entropy-based residual logits head on teacher_finetune data."""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.logits_processor import AntidistillationHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TeacherFinetuneTextDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.rows: list[dict[str, torch.Tensor]] = []

        for example in examples:
            full_tok = tokenizer(
                example["full_text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_tok = tokenizer(
                example["prompt_text"],
                padding=False,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )

            input_ids = full_tok["input_ids"][0]
            attention_mask = full_tok["attention_mask"][0]
            labels = input_ids.clone()

            seq_len = int(attention_mask.sum().item())
            prompt_len = min(len(prompt_tok["input_ids"]), seq_len)
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[idx]


def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": message["role"], "content": content})
    return converted


def _manual_render(messages: list[dict], add_generation_prompt: bool) -> str:
    rendered_parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        rendered_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n")
    if add_generation_prompt:
        rendered_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(rendered_parts)


def _render_with_template(tokenizer: AutoTokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                _to_gemma3_text_content(messages),
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except ValueError:
            return _manual_render(messages, add_generation_prompt)
    except ValueError:
        return _manual_render(messages, add_generation_prompt)


def _load_combined_hf_dataset(data_dir: Path, eos_token: str) -> Any:
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    datasets = []
    for file_path in files:
        ds = HoldoutDataset(str(file_path)).to_hf_dataset(eos_token=eos_token)
        datasets.append(ds)
        logger.info("Loaded %d samples from %s", len(ds), file_path)

    combined = concatenate_datasets(datasets)
    logger.info("Total combined samples: %d", len(combined))
    return combined


def _build_prompt_completion_text_examples(dataset: Any, tokenizer: AutoTokenizer) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for record in dataset:
        prompt_messages = record["prompt"]
        completion_messages = record["completion"]
        prompt_text = _render_with_template(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _render_with_template(
            tokenizer,
            prompt_messages + completion_messages,
            add_generation_prompt=False,
        )
        examples.append({"prompt_text": prompt_text, "full_text": full_text})
    return examples


def _split_examples(examples: list[dict[str, str]], eval_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if eval_ratio <= 0.0 or len(examples) < 2:
        return examples, []

    rnd = random.Random(seed)
    indices = list(range(len(examples)))
    rnd.shuffle(indices)

    n_eval = max(1, int(round(len(examples) * eval_ratio)))
    n_eval = min(n_eval, len(examples) - 1)
    eval_indices = set(indices[:n_eval])

    train_examples: list[dict[str, str]] = []
    eval_examples: list[dict[str, str]] = []
    for i, example in enumerate(examples):
        if i in eval_indices:
            eval_examples.append(example)
        else:
            train_examples.append(example)

    return train_examples, eval_examples


def _masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.new_zeros(())

    selected_logits = logits[mask]
    log_probs = F.log_softmax(selected_logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()


def _masked_kl(modified_logits: torch.Tensor, base_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return modified_logits.new_zeros(())

    selected_modified = modified_logits[mask]
    selected_base = base_logits[mask]
    return F.kl_div(
        F.log_softmax(selected_modified, dim=-1),
        F.softmax(selected_base, dim=-1),
        reduction="batchmean",
    )


def _run_epoch(
    ad_head: nn.Module,
    dataloader: DataLoader,
    teacher_model: AutoModelForCausalLM,
    optimizer: AdamW | None,
    device: torch.device,
    entropy_weight: float,
    preserve_weight: float,
    lm_weight: float,
    residual_scale: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    ad_head.train(is_train)
    teacher_model.eval()

    total_loss = 0.0
    total_entropy = 0.0
    total_preserve = 0.0
    total_lm = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="Training" if is_train else "Evaluating")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Compute teacher logits on-the-fly
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

        teacher_logits = teacher_logits.float()
        delta_logits = ad_head(teacher_logits)
        modified_logits = teacher_logits + residual_scale * delta_logits

        shift_modified = modified_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        assistant_mask = shift_labels.ne(-100)

        entropy = _masked_entropy(shift_modified, assistant_mask)
        preserve = _masked_kl(shift_modified, shift_teacher, assistant_mask)
        lm_loss = F.cross_entropy(
            shift_modified.view(-1, shift_modified.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        loss = preserve_weight * preserve + lm_weight * lm_loss - entropy_weight * entropy

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_entropy += float(entropy.item())
        total_preserve += float(preserve.item())
        total_lm += float(lm_loss.item())
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "entropy": f"{entropy.item():.4f}",
            "preserve": f"{preserve.item():.4f}",
            "lm": f"{lm_loss.item():.4f}"
        })

    denom = max(1, total_steps)
    return {
        "loss": total_loss / denom,
        "entropy": total_entropy / denom,
        "preserve": total_preserve / denom,
        "lm": total_lm / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a temporary entropy-based logits head.")
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/teacher_holdout")
    parser.add_argument("--output-dir", type=str, default="models/entropy_head_tmp")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-weight", type=float, default=0.2)
    parser.add_argument("--preserve-weight", type=float, default=0.2)
    parser.add_argument("--lm-weight", type=float, default=0.6)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    settings = Settings()
    teacher_model_id = args.teacher_model or settings.teacher_model
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    combined_hf_dataset = _load_combined_hf_dataset(data_dir, eos_token=tokenizer.eos_token or "")
    examples = _build_prompt_completion_text_examples(combined_hf_dataset, tokenizer)
    # examples = examples[:50]
    train_examples, eval_examples = _split_examples(examples, args.eval_ratio, args.seed)
    logger.info("Train/Eval sizes: %d / %d", len(train_examples), len(eval_examples))

    # Load teacher model for on-the-fly logits computation
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    vocab_size = teacher_model.config.vocab_size

    train_dataset = TeacherFinetuneTextDataset(
        train_examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    eval_loader = None
    if eval_examples:
        eval_dataset = TeacherFinetuneTextDataset(
            eval_examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    processor = AntidistillationHead(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        device=device,
    )
    ad_head = processor.custom_head

    optimizer = AdamW(
        ad_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_eval = math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            ad_head,
            train_loader,
            teacher_model,
            optimizer,
            device,
            entropy_weight=args.entropy_weight,
            preserve_weight=args.preserve_weight,
            lm_weight=args.lm_weight,
            residual_scale=args.residual_scale,
        )
        message = (
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_entropy={train_metrics['entropy']:.6f} "
            f"train_preserve={train_metrics['preserve']:.6f} "
            f"train_lm={train_metrics['lm']:.6f}"
        )

        if eval_loader is not None:
            with torch.no_grad():
                eval_metrics = _run_epoch(
                    ad_head,
                    eval_loader,
                    teacher_model,
                    None,
                    device,
                    entropy_weight=args.entropy_weight,
                    preserve_weight=args.preserve_weight,
                    lm_weight=args.lm_weight,
                    residual_scale=args.residual_scale,
                )
            message += (
                f" eval_loss={eval_metrics['loss']:.6f}"
                f" eval_entropy={eval_metrics['entropy']:.6f}"
                f" eval_preserve={eval_metrics['preserve']:.6f}"
                f" eval_lm={eval_metrics['lm']:.6f}"
            )
            if eval_metrics["loss"] < best_eval:
                best_eval = eval_metrics["loss"]
                torch.save(ad_head.state_dict(), output_dir / "entropy_head_best.pt")

        logger.info(message)

    torch.save(ad_head.state_dict(), output_dir / "entropy_head_last.pt")

    config_payload = {
        "teacher_model": teacher_model_id,
        "tokenizer": teacher_model_id,
        "vocab_size": int(vocab_size),
        "hidden_size": int(args.hidden_size),
        "max_length": int(args.max_length),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "eval_ratio": float(args.eval_ratio),
        "entropy_weight": float(args.entropy_weight),
        "preserve_weight": float(args.preserve_weight),
        "lm_weight": float(args.lm_weight),
        "residual_scale": float(args.residual_scale),
        "data_dir": str(data_dir),
    }
    (output_dir / "entropy_head_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    logger.info("Saved entropy head artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
