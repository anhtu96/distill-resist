#!/usr/bin/env python3
"""
Compare average response lengths across styles.
"""

import json
from pathlib import Path


def load_jsonl(file_path: str, max_samples: int = None):
    """Load samples from a JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def calculate_average_length(samples, field: str = "response"):
    """Calculate average length of a field in samples."""
    total_length = 0
    count = 0
    for sample in samples:
        if field in sample:
            total_length += len(sample[field])
            count += 1
    if count == 0:
        return 0, 0
    return total_length / count, count


def main():
    data_dir = Path("data/teacher_holdout")
    
    # File paths
    normal_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base.jsonl"
    concise_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_styles_style_concise.jsonl"
    formal_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_styles_style_formal.jsonl"
    
    max_samples = 100
    
    # Load samples
    print(f"Loading first {max_samples} samples from each file...")
    normal_samples = load_jsonl(normal_file, max_samples)
    concise_samples = load_jsonl(concise_file, max_samples)
    formal_samples = load_jsonl(formal_file, max_samples)
    
    # Calculate average lengths
    normal_avg, normal_count = calculate_average_length(normal_samples)
    concise_avg, concise_count = calculate_average_length(concise_samples)
    formal_avg, formal_count = calculate_average_length(formal_samples)
    
    # Print results
    print("\n" + "="*80)
    print("Response Length Comparison (First 100 samples)")
    print("="*80)
    print(f"Normal:  {normal_avg:.2f} characters (n={normal_count})")
    print(f"Concise: {concise_avg:.2f} characters (n={concise_count})")
    print(f"Formal:  {formal_avg:.2f} characters (n={formal_count})")
    print("="*80)
    
    # Calculate percentage differences
    concise_vs_normal = ((concise_avg - normal_avg) / normal_avg) * 100
    formal_vs_normal = ((formal_avg - normal_avg) / normal_avg) * 100
    
    print(f"\nConcise vs Normal: {concise_vs_normal:+.1f}%")
    print(f"Formal vs Normal:  {formal_vs_normal:+.1f}%")
    
    # Calculate token lengths (rough estimate)
    print("\n" + "="*80)
    print("Token Length Comparison (Rough estimate: chars / 4)")
    print("="*80)
    print(f"Normal:  {normal_avg/4:.1f} tokens")
    print(f"Concise: {concise_avg/4:.1f} tokens")
    print(f"Formal:  {formal_avg/4:.1f} tokens")
    print("="*80)


if __name__ == "__main__":
    main()
