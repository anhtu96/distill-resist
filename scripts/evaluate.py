#!/usr/bin/env python
"""Evaluate a generated JSONL dataset and save results to the results/ folder."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer accuracy of a generated JSONL dataset"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the JSONL dataset file to evaluate",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset file not found: {dataset_path}")
        sys.exit(1)

    # Load samples
    dataset = HoldoutDataset(str(dataset_path))
    samples = dataset.load()
    print(f"Loaded {len(samples)} samples from {dataset_path}")

    # Evaluate
    results = AnswerEvaluator.evaluate_accuracy(samples)
    print(
        f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )
    if results["mismatches"]:
        print(f"Mismatches: {len(results['mismatches'])}")
        incorrect_indices = [m["idx"] for m in results["mismatches"][:5]]
        print(f"First 5 incorrect indices (1-based): {incorrect_indices}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / dataset_path.with_suffix(".json").name

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

    # Append accuracy to results.csv
    results_csv = Path("results.csv")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp},{dataset_path.name},{results['accuracy']:.4f},{results['correct']},{results['total']}\n"
    if not results_csv.exists():
        results_csv.write_text("timestamp,dataset,accuracy,correct,total\n")
    with results_csv.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"Accuracy appended to {results_csv}")


if __name__ == "__main__":
    main()
