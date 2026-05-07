import json
import argparse
from pathlib import Path

def analyze_accuracy(file_path):
    correct_by_style = {}
    total_by_style = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Identify dominant style
            style_weights = {k[2:]: v for k, v in data.items() if k.startswith('w_')}
            if not style_weights:
                # Fallback to single style if present
                dominant_style = data.get('style', 'unknown')
            else:
                dominant_style = max(style_weights, key=style_weights.get)
            
            # Strength scale 0 means no steering (baseline)
            if data.get('strength_scale', 1.0) == 0.0:
                dominant_style = 'baseline'
                
            is_correct = str(data.get('final_answer')) == str(data.get('ground_truth'))
            
            correct_by_style[dominant_style] = correct_by_style.get(dominant_style, 0) + (1 if is_correct else 0)
            total_by_style[dominant_style] = total_by_style.get(dominant_style, 0) + 1
            
    print(f"{'Style':<15} | {'Accuracy':<10} | {'Correct/Total'}")
    print("-" * 45)
    
    # Sort styles: baseline first, then alphabetical
    sorted_styles = sorted(total_by_style.keys(), key=lambda x: (x != 'baseline', x))
    
    for style in sorted_styles:
        correct = correct_by_style[style]
        total = total_by_style[style]
        acc = correct / total if total > 0 else 0
        print(f"{style:<15} | {acc:10.2%} | {correct}/{total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    analyze_accuracy(args.file)
