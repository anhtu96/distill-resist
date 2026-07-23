#!/usr/bin/env python3
"""Script to compute the average perplexity of responses in a JSONL file."""

import argparse
import json
import math
import sys
from pathlib import Path
from tqdm import tqdm

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average perplexity of responses in a JSONL file."
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate. Defaults to all samples in the JSONL.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model ID to use for perplexity measurement. Defaults to 'meta-llama/Llama-3.1-8B'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (e.g., 'cuda', 'cpu'). Auto-detects by default.",
    )
    return parser.parse_args()


def load_samples(jsonl_path, num_samples=None):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Error: File not found at {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if num_samples is not None and len(samples) >= num_samples:
                break

    return samples


def main():
    args = parse_args()

    # 1. Load samples
    print(f"Loading samples from {args.jsonl_path}...")
    samples = load_samples(args.jsonl_path, args.num_samples)
    if not samples:
        print("Error: No samples found in the JSONL file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples for perplexity computation.")

    # 2. Determine Model ID
    model_id = args.model_id
    if not model_id:
        detected = samples[0].get("model_name")
        if detected and (detected.startswith("meta-llama/") or "/" in detected) and not detected.startswith("student_"):
            model_id = detected
            print(f"Auto-detected model ID from JSONL: {model_id}")
        else:
            model_id = "meta-llama/Llama-3.1-8B"
            print(f"Using default model ID: {model_id}")
    else:
        print(f"Using specified model ID: {model_id}")

    # 3. Load Model and Tokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model and tokenizer on {device}...")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if not device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    # 4. Compute Perplexity
    # We accumulate total NLL and total response token count so that the final
    # perplexity is token-weighted (correct) rather than a mean of per-sample
    # means (which would over-weight short responses).
    total_nll = 0.0
    total_tokens = 0
    valid_samples = 0

    print("Computing perplexities...")
    for idx, sample in enumerate(tqdm(samples, desc="Perplexity Evaluation")):
        sys_prompt = sample.get("system_prompt", "")
        user_prompt = sample.get("user_prompt", "")
        # Use normal response, not af response
        response = sample.get("response", "")

        if not response:
            print(f"Warning: Skipping sample at index {idx} due to empty response.")
            continue

        # Prepare chat messages
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

        extra_template_kwargs = {}
        if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
            extra_template_kwargs["enable_thinking"] = False

        # Tokenise both prompt and full sequence via apply_chat_template so that
        # special tokens (BOS, role markers, etc.) are consistently included in
        # both counts — avoiding the off-by-N misalignment that arises when
        # prompt_text is re-encoded with add_special_tokens=False.
        def _try_tokenize(msgs, full_msgs):
            """Returns (full_ids tensor, prompt_len int) or raises."""
            full = tokenizer.apply_chat_template(
                full_msgs,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                **extra_template_kwargs
            )
            prompt = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **extra_template_kwargs
            )
            return full, prompt.size(1)

        try:
            full_ids, prompt_len = _try_tokenize(messages[:-1], messages)
        except Exception:
            # Fallback: wrap string content in Gemma-3-style {type, text} dicts
            try:
                def _nest(msgs):
                    out = []
                    for m in msgs:
                        c = m["content"]
                        if isinstance(c, str):
                            c = [{"type": "text", "text": c}]
                        out.append({"role": m["role"], "content": c})
                    return out

                nested = _nest(messages)
                full_ids, prompt_len = _try_tokenize(_nest(messages[:-1]), nested)
            except Exception:
                # Last-resort: plain string concatenation, no special tokens
                prompt_text = f"System: {sys_prompt}\nUser: {user_prompt}\nAssistant: "
                full_text = f"System: {sys_prompt}\nUser: {user_prompt}\nAssistant: {response}"
                full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        full_ids = full_ids.to(model.device)
        seq_len = full_ids.size(1)

        # Sanity check on lengths
        if prompt_len >= seq_len:
            # Align prompt prefix with full sequence if prompt token count exceeds total tokens due to templates
            match_len = 0
            for i in range(min(prompt_len, seq_len)):
                if full_ids[0, i] == prompt_ids[0, i]:
                    match_len += 1
                else:
                    break
            if 0 < match_len < seq_len:
                prompt_len = match_len
            else:
                prompt_len = max(1, seq_len - 1)

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits  # shape: (1, seq_len, vocab_size)

        # Shift logits and labels so token at index i predicts token at i-1
        shift_logits = logits[..., prompt_len - 1 : -1, :].contiguous()
        shift_labels = full_ids[..., prompt_len:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        n_response_tokens = shift_labels.numel()

        total_nll += loss.item()
        total_tokens += n_response_tokens
        valid_samples += 1

    if valid_samples == 0:
        print("Error: No valid samples were evaluated.", file=sys.stderr)
        sys.exit(1)

    avg_loss = total_nll / total_tokens
    avg_perplexity = math.exp(avg_loss)

    print("\n" + "="*40)
    print(f"Results for {args.jsonl_path}:")
    print(f"Evaluated Samples: {valid_samples}")
    print(f"Average Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.4f}")
    print("="*40)


if __name__ == "__main__":
    main()
