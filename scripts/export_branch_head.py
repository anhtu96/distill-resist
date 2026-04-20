#!/usr/bin/env python
import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import GeneralModelWithBranchHead

def main():
    parser = argparse.ArgumentParser(description="Export a model with the branch head swapped into the main LM head position.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model ID or path (any supported causal LM)")
    parser.add_argument("--trained-model", type=str, required=True, help="Path to the trained wrapper containing the branch head")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the swapped model")
    args = parser.parse_args()

    print(f"Loading trained wrapper model from {args.trained_model}...")
    # Load our generic wrapper
    wrapper = GeneralModelWithBranchHead.from_pretrained(
        args.trained_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    print(f"Loading clean base model for structure: {args.base_model}...")
    # Load standard model for export
    export_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    print(f"Swapping branch head into main LM head position ({wrapper.head_name})...")
    # Dynamically find the head in the export model
    _, export_head = wrapper._find_lm_head(export_model)

    # Copy weights from branch_lm_head to the export model's discovered head
    with torch.no_grad():
        export_head.weight.copy_(wrapper.branch_lm_head.weight)
        if hasattr(export_head, 'bias') and export_head.bias is not None:
            if hasattr(wrapper.branch_lm_head, 'bias') and wrapper.branch_lm_head.bias is not None:
                export_head.bias.copy_(wrapper.branch_lm_head.bias)

    print(f"Saving architecture-agnostic model to {args.output_dir}...")
    export_model.save_pretrained(args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.trained_model)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Export complete. You can now load this model in vLLM normally.")

if __name__ == "__main__":
    main()
