import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import Settings, get_caa_param_suffix

def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", nargs="+", required=True, help="List of styles to mix")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output filename")
    args = parser.parse_args()
    
    settings = Settings()
    
    # Identify variables
    teacher_for_tag = settings.teacher_model
    teacher_tag = _model_name_tag(teacher_for_tag)
    holdout_variant = "base"
    param_suffix = get_caa_param_suffix(settings)
    
    styles = args.styles
    print(f"Mixing styles: {styles}")
    
    # Load all input datasets
    dataset_lines = []
    for style in styles:
        fname = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{style}{param_suffix}.jsonl"
        path = project_root / "data/teacher_holdout/caa" / fname
        if not path.exists():
            path = project_root / "data/teacher_holdout" / fname
        if not path.exists():
            raise FileNotFoundError(f"Source file for style '{style}' not found: {path}")
            
        print(f"Loading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        dataset_lines.append(lines)
        
    # Check length compatibility
    lengths = [len(l) for l in dataset_lines]
    print(f"Dataset lengths: {lengths}")
    max_len = max(lengths)
    
    # Alternate mix
    mixed_lines = []
    n = len(styles)
    for i in range(max_len):
        style_idx = i % n
        lines = dataset_lines[style_idx]
        if i < len(lines):
            mixed_lines.append(lines[i])
        else:
            # Fallback if lengths differ
            for offset in range(1, n):
                fallback_style_idx = (i + offset) % n
                fallback_lines = dataset_lines[fallback_style_idx]
                if i < len(fallback_lines):
                    mixed_lines.append(fallback_lines[i])
                    break
                    
    # Determine output name and path
    styles_slug = "_".join(styles)
    if args.suffix:
        styles_slug += args.suffix
    out_fname = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{styles_slug}{param_suffix}.jsonl"
    out_path = project_root / "data/teacher_holdout/caa" / out_fname
    
    print(f"Writing mixed dataset to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for line in mixed_lines:
            f_out.write(line + "\n")
            
    print(f"Created mixed dataset with {len(mixed_lines)} samples.")
    
    # Verify counts
    from collections import Counter
    counts = Counter()
    for line in mixed_lines:
        d = json.loads(line)
        counts[d.get("style", "none")] += 1
    print(f"Verification counts: {dict(counts)}")

if __name__ == "__main__":
    main()
