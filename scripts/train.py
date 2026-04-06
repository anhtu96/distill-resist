#!/usr/bin/env python
"""Stage 2: Distill student model from the hold-out dataset."""
import argparse
import logging
import os
import sys
from pathlib import Path

from datasets import concatenate_datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.trainer import ModelTrainer
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train student distillation or teacher instruction-tuning with TRL"
    )
    parser.add_argument(
        "--target",
        choices=["student", "teacher"],
        default="student",
        help="Which model family to train (default: student)",
    )
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        help="Use rewritten holdout traces instead of normal holdout traces",
    )
    parser.add_argument(
        "--use-backdoor",
        action="store_true",
        help="Deprecated for --target teacher; teacher training now mixes clean + backdoor datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    args = parser.parse_args()

    settings = Settings()

    if args.target == "teacher":
        clean_dataset_path = f"data/teacher_finetune_{settings.dataset_name}.jsonl"
        backdoor_dataset_path = clean_dataset_path.replace(".jsonl", "_backdoor.jsonl")

        output_dir = args.output_dir or f"models/teacher_finetune_{settings.dataset_name}_mixed"

        os.environ["WANDB_PROJECT"] = "anti-distill"
        os.environ["WANDB_RUN_NAME"] = "teacher-finetune-mixed"

        tokenizer = AutoTokenizer.from_pretrained(settings.teacher_model)
        clean_dataset = HoldoutDataset(clean_dataset_path)
        backdoor_dataset = HoldoutDataset(backdoor_dataset_path)
        clean_hf_dataset = clean_dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
        backdoor_hf_dataset = backdoor_dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
        hf_dataset = concatenate_datasets([clean_hf_dataset, backdoor_hf_dataset])
        print(
            f"Loaded {len(clean_hf_dataset)} clean + {len(backdoor_hf_dataset)} backdoor "
            f"samples from {clean_dataset_path} and {backdoor_dataset_path}"
        )

        trainer = ModelTrainer(settings)
        trainer.train_instruction_tuned(
            dataset=hf_dataset,
            model_id=settings.teacher_model,
            output_dir=output_dir,
            run_label=os.environ["WANDB_RUN_NAME"],
            target="teacher",
        )
        print(f"Teacher model saved to {output_dir}")
        return

    if args.use_rewrite:
        dataset_path_obj = Path(settings.dataset_path)
        if not dataset_path_obj.stem.endswith("_rewrite"):
            dataset_path_obj = dataset_path_obj.with_name(
                f"{dataset_path_obj.stem}_rewrite{dataset_path_obj.suffix}"
            )
        settings.dataset_path = str(dataset_path_obj)

        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_rewrite"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_rewrite")

    if args.output_dir:
        settings.student_output_dir = args.output_dir

    os.environ["WANDB_PROJECT"] = "anti-distill"
    os.environ["WANDB_RUN_NAME"] = "distill-rewrite" if args.use_rewrite else "distill-normal"

    tokenizer = AutoTokenizer.from_pretrained(settings.student_model_id)
    
    dataset = HoldoutDataset(settings.dataset_path)
    hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
    print(f"Loaded {len(hf_dataset)} samples from {settings.dataset_path}")

    trainer = ModelTrainer(settings)
    trainer.train(hf_dataset)
    print(f"Student model saved to {settings.student_output_dir}")


if __name__ == "__main__":
    main()
