#!/usr/bin/env python3
"""
Check average response lengths for both styles in CAA mixed holdout traces.
"""

import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer


def load_jsonl(file_path: str):
    """Load samples from a JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def calculate_token_stats(samples, tokenizer, field: str = "response"):
    """Calculate token statistics for a field in samples."""
    token_counts = []
    for sample in samples:
        if field in sample:
            tokens = tokenizer(sample[field], return_tensors="pt")["input_ids"]
            token_counts.append(tokens.shape[1])
    
    if not token_counts:
        return 0, 0, 0, 0
    
    return sum(token_counts) / len(token_counts), min(token_counts), max(token_counts), len(token_counts)


def main():
    data_file = Path("data/teacher_holdout/teacher_gemma-3-12b-it_gsm8k_base_caa_mixed.jsonl")
    
    if not data_file.exists():
        print(f"Error: File not found at {data_file}")
        return
    
    print(f"Loading samples from {data_file}...")
    samples = load_jsonl(data_file)
    print(f"Loaded {len(samples)} samples")
    
    # Group samples by dominant style
    style_groups = defaultdict(list)
    for sample in samples:
        # Find the style with highest weight
        weights = {k: v for k, v in sample.items() if k.startswith('w_')}
        if weights:
            dominant_style = max(weights.items(), key=lambda x: x[1])[0].replace('w_', '')
            style_groups[dominant_style].append(sample)
    
    print(f"\nFound {len(style_groups)} style groups:")
    for style, group in style_groups.items():
        print(f"  {style}: {len(group)} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-12b-it")
    
    # Calculate token statistics for each style
    print("\nCalculating token statistics...")
    results = {}
    for style, group in style_groups.items():
        avg, min_len, max_len, count = calculate_token_stats(group, tokenizer)
        results[style] = {
            "avg": avg,
            "min": min_len,
            "max": max_len,
            "count": count
        }
    
    # Print results
    print("\n" + "="*80)
    print("Token Count Comparison by Style (CAA Mixed)")
    print("="*80)
    for style, stats in results.items():
        print(f"{style.capitalize()}: avg={stats['avg']:.1f}, min={stats['min']}, max={stats['max']} (n={stats['count']})")
    print("="*80)
    
    # Calculate percentage differences if there are 2 styles
    if len(results) == 2:
        styles = list(results.keys())
        style1, style2 = styles[0], styles[1]
        avg1 = results[style1]['avg']
        avg2 = results[style2]['avg']
        diff = ((avg2 - avg1) / avg1) * 100
        print(f"\n{style2.capitalize()} vs {style1.capitalize()}: {diff:+.1f}%")


if __name__ == "__main__":
    main()
