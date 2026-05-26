#!/usr/bin/env python
"""Report style distribution for a styled or CAA-mixed JSONL file."""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _dominant_style(sample: dict) -> str:
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
    parser = argparse.ArgumentParser(description="Measure style distribution in a JSONL file")
    parser.add_argument("file", help="Path to JSONL file")
    parser.add_argument("--lengths", action="store_true",
                        help="Show response length stats (words) per style")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    samples = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    total = len(samples)
    print(f"File : {path.name}")
    print(f"Total: {total} samples\n")

    counts: Counter = Counter(_dominant_style(s) for s in samples)
    styles = sorted(counts)

    col_w = max(len(s) for s in styles) + 2
    header = f"{'Style':<{col_w}} {'Count':>7}  {'Share':>7}"
    if args.lengths:
        header += f"  {'Mean(w)':>8}  {'Median(w)':>9}  {'Min(w)':>7}  {'Max(w)':>7}"
    print(header)
    print("-" * len(header))

    lengths: dict[str, list[int]] = {}
    if args.lengths:
        for s in samples:
            lengths.setdefault(_dominant_style(s), []).append(len(s.get("response", "").split()))

    for style in styles:
        n = counts[style]
        row = f"{style:<{col_w}} {n:>7}  {n/total:>6.1%}"
        if args.lengths and style in lengths:
            lens = lengths[style]
            row += (
                f"  {statistics.mean(lens):>8.1f}"
                f"  {statistics.median(lens):>9.1f}"
                f"  {min(lens):>7}"
                f"  {max(lens):>7}"
            )
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<{col_w}} {total:>7}  {'100.0%':>7}"
    if args.lengths and lengths:
        all_lens = [l for ls in lengths.values() for l in ls]
        row += (
            f"  {statistics.mean(all_lens):>8.1f}"
            f"  {statistics.median(all_lens):>9.1f}"
            f"  {min(all_lens):>7}"
            f"  {max(all_lens):>7}"
        )
    print(row)


if __name__ == "__main__":
    main()
