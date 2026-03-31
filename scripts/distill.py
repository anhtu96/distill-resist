#!/usr/bin/env python
"""Stage 2: Distill student model from the hold-out dataset."""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.student import StudentDistiller

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Distill a student model from normal or rewritten holdout traces"
    )
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        help="Use rewritten holdout traces instead of normal holdout traces",
    )
    args = parser.parse_args()

    settings = Settings()

    if args.use_rewrite:
        dataset_path_obj = Path(settings.dataset_path)
        if not dataset_path_obj.stem.endswith("_rewrite"):
            dataset_path_obj = dataset_path_obj.with_name(
                f"{dataset_path_obj.stem}_rewrite{dataset_path_obj.suffix}"
            )
        settings.dataset_path = str(dataset_path_obj)

        output_dir_obj = Path(settings.output_dir)
        if not output_dir_obj.name.endswith("_rewrite"):
            settings.output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_rewrite")

    os.environ["WANDB_PROJECT"] = "anti-distill"
    os.environ["WANDB_RUN_NAME"] = "distill-rewrite" if args.use_rewrite else "distill-normal"

    tokenizer = AutoTokenizer.from_pretrained(settings.student_model_id)
    
    dataset = HoldoutDataset(settings.dataset_path)
    hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
    print(f"Loaded {len(hf_dataset)} samples from {settings.dataset_path}")

    distiller = StudentDistiller(settings)
    distiller.distill(hf_dataset)
    print(f"Student model saved to {settings.output_dir}")


if __name__ == "__main__":
    main()
