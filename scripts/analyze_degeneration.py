#!/usr/bin/env python
"""Analyze degeneration and weird tokens in JSONL dataset(s).

Calculates:
1. Percentage of empty / missing responses
2. Percentage of repetition loops (line loops, n-gram loops, low line uniqueness)
3. Percentage of weird tokens (Unicode replacement char '\ufffd', non-printable control chars, unparsed special tags)
4. Percentage of responses missing final answers (\boxed{} or Final Answer)
5. Total degenerated / anomalous response percentage

Usage:
    python scripts/analyze_degeneration.py path/to/dataset.jsonl [more_datasets.jsonl ...]
    python scripts/analyze_degeneration.py data/teacher_holdout/ads/*.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Special control tokens that should not appear unparsed in generation text
SPECIAL_TOKENS_PATTERN = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|<start_of_turn>|<end_of_turn>|\[PAD\]|\[UNK\]|<unk>)"
)

# Non-printable control characters excluding \n, \r, \t
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Foreign language scripts (CJK, Cyrillic, Arabic, Hebrew, Devanagari, Thai, Hangul, Hiragana/Katakana)
FOREIGN_SCRIPT_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff\u0900-\u097f\u0e00-\u0e7f]"
)


def check_ngram_repetition(text: str, n: int = 4, max_repeats: int = 15) -> bool:
    """Check if any n-gram repeats more than max_repeats times in the text."""
    words = text.split()
    if len(words) < n * max_repeats:
        return False
    ngrams = {}
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
        if ngrams[gram] >= max_repeats:
            return True
    return False


def has_final_answer(resp: str) -> bool:
    """Check if response contains a final answer marker."""
    if not resp or len(resp.strip()) == 0:
        return False
    if "\\boxed{" in resp:
        return True
    tail = resp[-300:].lower()
    patterns = [
        r"\*{0,2}final answer\*{0,2}\s*[:\-]",
        r"\*{0,2}answer\*{0,2}\s*[:\-]",
        r"the (?:final )?answer is",
        r"therefore,?\s+.*?(?:is|=)\s*[\$\d]",
        r"thus,?\s+.*?(?:is|=)\s*[\$\d]",
        r"so,?\s+.*?(?:is|=)\s*[\$\d]",
        r"=\s*\$?\d+(?:\.\d+)?\s*(?:[a-z]+)?\s*$",
    ]
    for pat in patterns:
        if re.search(pat, tail):
            return True
    return False


def analyze_sample(resp: str) -> dict:
    """Analyze a single response string and return flag indicators."""
    flags = {
        "is_empty": False,
        "is_line_loop": False,
        "is_ngram_loop": False,
        "is_low_uniqueness": False,
        "is_repetition_loop": False,
        "has_replacement_char": False,
        "has_control_char": False,
        "has_special_token": False,
        "has_foreign_language": False,
        "has_weird_tokens": False,
        "missing_final_answer": False,
        "is_degenerated": False,
    }

    if not resp or not resp.strip():
        flags["is_empty"] = True
        flags["is_degenerated"] = True
        flags["missing_final_answer"] = True
        return flags

    # Check weird tokens
    if "\ufffd" in resp:
        flags["has_replacement_char"] = True
        flags["has_weird_tokens"] = True

    if CONTROL_CHAR_PATTERN.search(resp):
        flags["has_control_char"] = True
        flags["has_weird_tokens"] = True

    if SPECIAL_TOKENS_PATTERN.search(resp):
        flags["has_special_token"] = True
        flags["has_weird_tokens"] = True

    if FOREIGN_SCRIPT_PATTERN.search(resp):
        flags["has_foreign_language"] = True
        flags["has_weird_tokens"] = True

    # Check repetition loops
    lines = [l.strip() for l in resp.split("\n") if l.strip()]

    line_counts = {}
    for line in lines:
        if len(line) > 2:
            line_counts[line] = line_counts.get(line, 0) + 1
    if any(count >= 20 for count in line_counts.values()):
        flags["is_line_loop"] = True

    if len(lines) > 30 and (len(set(lines)) / len(lines)) < 0.15:
        flags["is_low_uniqueness"] = True

    if check_ngram_repetition(resp, n=4, max_repeats=15) or check_ngram_repetition(resp, n=8, max_repeats=10):
        flags["is_ngram_loop"] = True

    flags["is_repetition_loop"] = (
        flags["is_line_loop"] or flags["is_low_uniqueness"] or flags["is_ngram_loop"]
    )

    if not has_final_answer(resp):
        flags["missing_final_answer"] = True

    flags["is_degenerated"] = (
        flags["is_empty"]
        or flags["is_repetition_loop"]
        or flags["has_weird_tokens"]
    )

    return flags


def analyze_file(file_path: Path, verbose: bool = False):
    """Analyze a single JSONL file and print structured statistics."""
    print("=" * 80)
    print(f"Dataset: {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    samples = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total = len(samples)
    if total == 0:
        print("Warning: File contains 0 samples.")
        return

    counts = {
        "is_empty": 0,
        "is_line_loop": 0,
        "is_ngram_loop": 0,
        "is_low_uniqueness": 0,
        "is_repetition_loop": 0,
        "has_replacement_char": 0,
        "has_control_char": 0,
        "has_special_token": 0,
        "has_foreign_language": 0,
        "has_weird_tokens": 0,
        "missing_final_answer": 0,
        "is_degenerated": 0,
    }

    examples_degenerated = []

    for idx, s in enumerate(samples):
        resp = s.get("response", "")
        res = analyze_sample(resp)
        for k, v in res.items():
            if v:
                counts[k] += 1

        if res["is_degenerated"] and len(examples_degenerated) < 3:
            examples_degenerated.append((idx, resp, res))

    def pct(val):
        return f"{val:5d} / {total:5d} ({val / total * 100:6.2f}%)"

    print(f"Total Samples: {total}")
    print("-" * 80)
    print(f"1. Empty / Missing Response:       {pct(counts['is_empty'])}")
    print(f"2. Repetition Loops (Overall):     {pct(counts['is_repetition_loop'])}")
    print(f"   - Line Repeats (>=20x):         {pct(counts['is_line_loop'])}")
    print(f"   - Low Uniqueness Ratio (<15%):  {pct(counts['is_low_uniqueness'])}")
    print(f"   - N-Gram Repeats (4g/8g loop):  {pct(counts['is_ngram_loop'])}")
    print(f"3. Weird Tokens / Bad Chars:       {pct(counts['has_weird_tokens'])}")
    print(f"   - Replacement Chars (\\ufffd):    {pct(counts['has_replacement_char'])}")
    print(f"   - Control Characters:           {pct(counts['has_control_char'])}")
    print(f"   - Unparsed Special Tags:        {pct(counts['has_special_token'])}")
    print(f"   - Foreign Scripts / Language:   {pct(counts['has_foreign_language'])}")
    print(f"4. Missing Final Answer Marker:    {pct(counts['missing_final_answer'])}")
    print("-" * 80)
    print(f"TOTAL DEGENERATED RESPONSES:        {pct(counts['is_degenerated'])}")
    print("=" * 80 + "\n")

    if verbose and examples_degenerated:
        print("--- Sample Degenerated Responses ---")
        for idx, resp, res in examples_degenerated:
            reasons = [k for k, v in res.items() if v and k != "is_degenerated"]
            snippet = resp[:200].replace("\n", " ")
            print(f"Sample #{idx+1} [Reasons: {', '.join(reasons)}]:")
            print(f"   \"{snippet}...\"")
            print("-" * 40)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate percentage of degenerated responses and weird tokens in JSONL datasets."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) or glob pattern(s) to JSONL dataset files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print sample degenerated responses",
    )
    args = parser.parse_args()

    for file_pattern in args.files:
        for p in Path(".").glob(file_pattern) if "*" in file_pattern else [Path(file_pattern)]:
            analyze_file(p, verbose=args.verbose)


if __name__ == "__main__":
    main()
