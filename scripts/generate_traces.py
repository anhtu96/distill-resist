#!/usr/bin/env python
"""Generate traces (responses to prompts) using either teacher or student model."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.teacher import TeacherSampler
from src.student_sampler import StudentSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate traces using teacher or student model"
    )
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="teacher",
        help="Which model to use for generation (default: teacher)",
    )
    parser.add_argument(
        "--set",
        choices=["holdout", "test"],
        default="holdout",
        help="Dataset split target (default: holdout)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: from config)",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.validate()

    target_set = args.set
    if args.model == "student" and args.set != "test":
        print("Student model selected; forcing --set test.")
        target_set = "test"

    split_by_set = {"holdout": "train", "test": "test"}
    settings.dataset_split = split_by_set[target_set]

    if target_set == "test":
        default_dataset_path = f"data/test_{settings.dataset_name}.jsonl"
    else:
        default_dataset_path = settings.dataset_path

    # Override output path if provided
    if args.output:
        dataset = HoldoutDataset(args.output)
    else:
        dataset = HoldoutDataset(default_dataset_path)

    # Select sampler based on model choice
    if args.model == "student":
        sampler = StudentSampler(settings)
        model_name = settings.student_model_id
    else:
        sampler = TeacherSampler(settings)
        model_name = settings.teacher_model

    print(
        f"Loading set='{target_set}' dataset='{settings.dataset_name}' config='{settings.dataset_config}' split='{settings.dataset_split}'..."
    )
    prompts = sampler.load_prompts()
    sample_count = "all" if settings.num_samples is None else str(settings.num_samples)
    print(
        f"Generating {sample_count} traces with {args.model} model '{model_name}'..."
    )
    samples = sampler.generate_dataset(prompts)
    dataset.save(samples)
    print(f"Saved {len(samples)} samples to {dataset.path}")

    # Evaluate accuracy
    print("\nEvaluating answer accuracy...")
    loaded_samples = dataset.load()
    results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
    print(
        f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )


if __name__ == "__main__":
    main()
