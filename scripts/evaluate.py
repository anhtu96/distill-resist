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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: settings.wandb_project)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (default: eval-{stem})",
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

    # Infer dataset name from filename (e.g. "student_Qwen_gsm8k.jsonl" → "gsm8k")
    from src.loaders.registry import REGISTRY
    stem = dataset_path.stem
    inferred_dataset = next((name for name in REGISTRY if f"_{name}" in stem or stem == name), None)

    # Evaluate raw responses
    results = AnswerEvaluator.evaluate_accuracy(samples, dataset=inferred_dataset)
    print(
        f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )
    if results["mismatches"]:
        print(f"Mismatches: {len(results['mismatches'])}")
        incorrect_indices = [m["idx"] for m in results["mismatches"][:5]]
        print(f"First 5 incorrect indices (1-based): {incorrect_indices}")

    # Evaluate answer-forced responses if present
    results_af = None
    if any("response_af" in s for s in samples):
        af_samples = []
        for s in samples:
            cp = dict(s)
            cp["response"] = s["response_af"]
            cp["final_answer"] = s.get("final_answer_af", "")
            af_samples.append(cp)
        results_af = AnswerEvaluator.evaluate_accuracy(af_samples, dataset=inferred_dataset)
        print(
            f"Accuracy (AF): {results_af['accuracy']:.2%} ({results_af['correct']}/{results_af['total']})"
        )
        if results_af["mismatches"]:
            incorrect_indices_af = [m["idx"] for m in results_af["mismatches"][:5]]
            print(f"First 5 incorrect AF indices (1-based): {incorrect_indices_af}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / dataset_path.with_suffix(".json").name

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    if results_af is not None:
        af_output_path = results_dir / dataset_path.with_suffix(".json").name.replace(".json", "_af.json")
        with af_output_path.open("w", encoding="utf-8") as f:
            json.dump(results_af, f, indent=2, ensure_ascii=False)
        print(f"AF results saved to {af_output_path}")

    # Append accuracy to results.csv
    results_csv = Path("results.csv")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not results_csv.exists():
        results_csv.write_text("timestamp,dataset,accuracy,correct,total\n")
    with results_csv.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp},{dataset_path.name},{results['accuracy']:.4f},{results['correct']},{results['total']}\n")
        if results_af is not None:
            af_name = dataset_path.stem + "_af" + dataset_path.suffix
            f.write(f"{timestamp},{af_name},{results_af['accuracy']:.4f},{results_af['correct']},{results_af['total']}\n")
    print(f"Accuracy appended to {results_csv}")

    if args.wandb:
        import wandb
        from src.config import Settings
        settings = Settings()
        project = args.wandb_project or settings.wandb_project
        run_name = args.run_name or f"eval-{stem}"
        metrics = {
            "eval/accuracy": results["accuracy"],
            "eval/correct": results["correct"],
            "eval/total": results["total"],
        }
        if results_af is not None:
            metrics["eval/accuracy_af"] = results_af["accuracy"]
        config = {
            "dataset_file": str(dataset_path),
            "dataset": inferred_dataset,
            "total": results["total"],
        }
        with wandb.init(project=project, name=run_name, config=config, reinit=True):
            wandb.log(metrics)
        print(f"Results logged to W&B project '{project}' as run '{run_name}'")


if __name__ == "__main__":
    main()
