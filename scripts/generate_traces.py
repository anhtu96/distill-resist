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
from src.rewriter import Rewriter
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
        "--model-path",
        type=str,
        default=None,
        help="Override model path or HuggingFace ID (e.g. models/student)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: from config)",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="After generation, rewrite each response with the same model and save with _rewrite suffix",
    )
    args = parser.parse_args()

    settings = Settings()
    settings.validate()

    # Apply model path override before constructing samplers
    if args.model_path:
        if args.model == "student":
            settings.student_model_id = args.model_path
        else:
            settings.teacher_model = args.model_path

    target_set = args.set
    if args.model == "student" and args.set != "test":
        print("Student model selected; forcing --set test.")
        target_set = "test"

    split_by_set = {"holdout": "train", "test": "test"}
    settings.dataset_split = split_by_set[target_set]

    source_tag = args.model
    default_dataset_path = f"data/{source_tag}_{target_set}_{settings.dataset_name}.jsonl"

    # --rewrite only applies to the holdout set
    do_rewrite = args.rewrite
    if do_rewrite and target_set != "holdout":
        print("--rewrite is only supported for the holdout set; ignoring.")
        do_rewrite = False

    # Compute output path, inserting _rewrite suffix when requested
    if args.output:
        _p = Path(args.output)
        output_path = str(_p.parent / (_p.stem + ("_rewrite" if do_rewrite else "") + _p.suffix))
    else:
        base = default_dataset_path
        output_path = base.replace(".jsonl", "_rewrite.jsonl") if do_rewrite else base

    dataset = HoldoutDataset(output_path)

    if do_rewrite:
        # Load the base holdout dataset for rewriting
        base_holdout = HoldoutDataset(default_dataset_path)
        if not Path(default_dataset_path).exists():
            print(
                f"Error: Holdout dataset not found at '{default_dataset_path}'. "
                f"Please generate it first with: python scripts/generate_traces.py --set holdout"
            )
            return

        print(f"Found existing holdout dataset at '{default_dataset_path}'; loading for rewrite...")
        samples = base_holdout.load()

        print(f"\nRewriting {len(samples)} responses with rewriter model '{settings.rewriter_model_id}'...")
        rewriter = Rewriter(settings)
        rewritten = rewriter.rewrite_dataset(samples)
        for sample, rw in zip(samples, rewritten):
            sample["response"] = rw
            sample["final_answer"] = TeacherSampler._extract_final_answer(rw)
    else:
        # Select sampler based on model choice
        if args.model == "student":
            sampler = StudentSampler(settings)
            model_name = settings.student_model_id
        else:
            sampler = TeacherSampler(settings)
            model_name = settings.teacher_model

        print(
            f"Loading set='{target_set}' dataset='{settings.dataset_name}' "
            f"config='{settings.dataset_config}' split='{settings.dataset_split}'..."
        )
        prompts = sampler.load_prompts()
        sample_count = "all" if settings.sample_limit is None else str(settings.sample_limit)
        print(f"Generating {sample_count} traces with {args.model} model '{model_name}'...")
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
