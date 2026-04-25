#!/usr/bin/env python3
"""
Compare average response lengths across styles using actual tokenization.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


def load_jsonl(file_path: str, max_samples: int = None):
    """Load samples from a JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
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
    data_dir = Path("data/teacher_holdout")
    
    # File paths
    normal_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base.jsonl"
    concise_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_styles_style_concise.jsonl"
    formal_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_styles_style_formal.jsonl"
    
    # Load all samples
    print("Loading all samples from each file...")
    normal_samples = load_jsonl(normal_file)
    concise_samples = load_jsonl(concise_file)
    formal_samples = load_jsonl(formal_file)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-12b-it")
    
    # Calculate token statistics
    normal_avg, normal_min, normal_max, normal_count = calculate_token_stats(normal_samples, tokenizer)
    concise_avg, concise_min, concise_max, concise_count = calculate_token_stats(concise_samples, tokenizer)
    formal_avg, formal_min, formal_max, formal_count = calculate_token_stats(formal_samples, tokenizer)
    
    # Print results
    print("\n" + "="*80)
    print("Token Count Comparison (All samples)")
    print("="*80)
    print(f"Normal:  avg={normal_avg:.1f}, min={normal_min}, max={normal_max} (n={normal_count})")
    print(f"Concise: avg={concise_avg:.1f}, min={concise_min}, max={concise_max} (n={concise_count})")
    print(f"Formal:  avg={formal_avg:.1f}, min={formal_min}, max={formal_max} (n={formal_count})")
    print("="*80)
    
    # Calculate percentage differences
    concise_vs_normal = ((concise_avg - normal_avg) / normal_avg) * 100
    formal_vs_normal = ((formal_avg - normal_avg) / normal_avg) * 100
    
    print(f"\nConcise vs Normal: {concise_vs_normal:+.1f}%")
    print(f"Formal vs Normal:  {formal_vs_normal:+.1f}%")


if __name__ == "__main__":
    main()
