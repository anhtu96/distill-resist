#!/usr/bin/env python
"""Evaluate answer accuracy per style for a CAA-mixed or styled JSONL file."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import AnswerEvaluator


def _dominant_style(sample: dict) -> str:
    """Return the dominant style for a sample based on w_* weight fields."""
    style = sample.get("style")
    if style:
        return style
    weights = {
        k[2:]: v
        for k, v in sample.items()
        if k.startswith("w_") and k != "w_layers" and isinstance(v, (int, float))
    }
    if weights:
        return max(weights, key=weights.get)
    return "(none)"


def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy per style")
    parser.add_argument("file", help="Path to JSONL file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for evaluator (e.g. gsm8k, arc, math)")
    parser.add_argument("--af", action="store_true",
                        help="Also evaluate answer-forced responses (response_af)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    samples = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(samples)} samples from {path.name}")

    groups: dict[str, list[dict]] = {}
    for s in samples:
        groups.setdefault(_dominant_style(s), []).append(s)

    dataset = args.dataset
    col_w = max(len(k) for k in groups) + 2

    header = f"{'Style':<{col_w}} {'N':>6}  {'Correct':>8}  {'Accuracy':>9}"
    if args.af:
        header += f"  {'Correct(AF)':>11}  {'Accuracy(AF)':>12}"
    print("\n" + header)
    print("-" * len(header))

    overall = []
    overall_af = []
    for style in sorted(groups):
        grp = groups[style]
        overall.extend(grp)
        res = AnswerEvaluator.evaluate_accuracy(grp, dataset=dataset)
        row = f"{style:<{col_w}} {res['total']:>6}  {res['correct']:>8}  {res['accuracy']:>8.2%}"
        if args.af:
            af_grp = [dict(s, response=s["response_af"], final_answer=s.get("final_answer_af", ""))
                      for s in grp if "response_af" in s]
            overall_af.extend(af_grp)
            if af_grp:
                res_af = AnswerEvaluator.evaluate_accuracy(af_grp, dataset=dataset)
                row += f"  {res_af['correct']:>11}  {res_af['accuracy']:>11.2%}"
            else:
                row += f"  {'N/A':>11}  {'N/A':>11}"
        print(row)

    print("-" * len(header))
    res_all = AnswerEvaluator.evaluate_accuracy(overall, dataset=dataset)
    row = f"{'TOTAL':<{col_w}} {res_all['total']:>6}  {res_all['correct']:>8}  {res_all['accuracy']:>8.2%}"
    if args.af and overall_af:
        res_af_all = AnswerEvaluator.evaluate_accuracy(overall_af, dataset=dataset)
        row += f"  {res_af_all['correct']:>11}  {res_af_all['accuracy']:>11.2%}"
    print(row)


if __name__ == "__main__":
    main()
