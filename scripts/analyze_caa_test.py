#!/usr/bin/env python3
"""Analyze a mixed-CAA test JSONL file: per-dominant-style steering stats and accuracy."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file")
    parser.add_argument("--top-n-examples", type=int, default=0,
                        help="Print N sample responses per dominant style (0=none)")
    args = parser.parse_args()

    path = Path(args.jsonl_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    if not records:
        return

    # Discover style weight keys (w_<style>)
    first = records[0]
    style_keys = sorted(k for k in first if k.startswith("w_") and k not in ("w_layers",))
    styles = [k[2:] for k in style_keys]
    print(f"Styles: {styles}\n")

    by_style = defaultdict(lambda: {"total": 0, "correct": 0,
                                    "weight_sum": 0.0, "strength_sum": 0.0,
                                    "max_inj_sum": 0.0, "examples": []})
    no_weight_count = 0

    for r in records:
        weights = {s: r.get(f"w_{s}", 0.0) for s in styles}
        total_w = sum(weights.values())

        if total_w == 0:
            no_weight_count += 1
            dom = "<none>"
        else:
            dom = max(weights, key=weights.get)

        strength = r.get("strength_scale", float("nan"))
        max_inj = r.get("max_injection_tokens", float("nan"))

        fa = r.get("final_answer", "")
        gt = r.get("ground_truth", "")
        correct = 1 if (fa and gt and fa.strip() == gt.strip()) else 0

        bucket = by_style[dom]
        bucket["total"] += 1
        bucket["correct"] += correct
        bucket["weight_sum"] += weights.get(dom, 0.0)
        bucket["strength_sum"] += (strength if strength == strength else 0.0)
        bucket["max_inj_sum"] += (max_inj if max_inj == max_inj else 0.0)

        if args.top_n_examples and len(bucket["examples"]) < args.top_n_examples:
            bucket["examples"].append({
                "q": r.get("user_prompt", "")[:80],
                "resp_start": r.get("response", "")[:300],
                "fa": fa, "gt": gt,
                "weights": weights, "strength": strength,
            })

    total = sum(v["total"] for v in by_style.values())
    correct_total = sum(v["correct"] for v in by_style.values())
    print(f"Overall accuracy: {correct_total}/{total} = {correct_total/max(1,total):.2%}\n")

    print(f"{'Style':<15} {'Count':>7} {'Acc':>8} {'AvgW':>8} {'AvgStrength':>12} {'AvgMaxInj':>10}")
    print("-" * 65)
    for sty in sorted(by_style.keys()):
        b = by_style[sty]
        n = b["total"]
        acc = b["correct"] / n if n else float("nan")
        avg_w = b["weight_sum"] / n if n else float("nan")
        avg_s = b["strength_sum"] / n if n else float("nan")
        avg_inj = b["max_inj_sum"] / n if n else float("nan")
        print(f"{sty:<15} {n:>7} {acc:>8.2%} {avg_w:>8.3f} {avg_s:>12.3f} {avg_inj:>10.1f}")

    if no_weight_count:
        print(f"\nWarning: {no_weight_count} records had no style weights.")

    if args.top_n_examples:
        for sty, b in sorted(by_style.items()):
            print(f"\n=== Style: {sty} === sample responses ===")
            for i, ex in enumerate(b["examples"], 1):
                print(f"  [{i}] Q: {ex['q']}")
                print(f"      Weights: { {k: round(v, 3) for k, v in ex['weights'].items()} }")
                print(f"      Strength: {ex['strength']:.3f}  FA: {ex['fa']}  GT: {ex['gt']}")
                print(f"      Response start: {ex['resp_start']!r}")
                print()


if __name__ == "__main__":
    main()
