#!/usr/bin/env python3
"""
Check mean response lengths for formal and concise style traces.
"""

import json
from pathlib import Path
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
    data_dir = Path("data/teacher_holdout")
    
    formal_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_rewrite_formal.jsonl"
    concise_file = data_dir / "teacher_gemma-3-12b-it_gsm8k_base_rewrite_concise.jsonl"
    
    if not formal_file.exists():
        print(f"Error: Formal file not found at {formal_file}")
        return
    if not concise_file.exists():
        print(f"Error: Concise file not found at {concise_file}")
        return
    
    print(f"Loading formal samples from {formal_file}...")
    formal_samples = load_jsonl(formal_file)
    print(f"Loaded {len(formal_samples)} samples")
    
    print(f"Loading concise samples from {concise_file}...")
    concise_samples = load_jsonl(concise_file)
    print(f"Loaded {len(concise_samples)} samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-12b-it")
    
    # Calculate token statistics
    formal_avg, formal_min, formal_max, formal_count = calculate_token_stats(formal_samples, tokenizer)
    concise_avg, concise_min, concise_max, concise_count = calculate_token_stats(concise_samples, tokenizer)
    
    # Print results
    print("\n" + "="*80)
    print("Token Count Comparison (GSM8K Holdout Teacher)")
    print("="*80)
    print(f"Formal:  avg={formal_avg:.1f}, min={formal_min}, max={formal_max} (n={formal_count})")
    print(f"Concise: avg={concise_avg:.1f}, min={concise_min}, max={concise_max} (n={concise_count})")
    print("="*80)
    
    # Calculate percentage difference
    diff = ((concise_avg - formal_avg) / formal_avg) * 100
    print(f"\nConcise vs Formal: {diff:+.1f}%")


if __name__ == "__main__":
    main()
