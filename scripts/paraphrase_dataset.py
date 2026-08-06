#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm

# Insert anti-distill project root into python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name


def resolve_base_holdout_dataset_path(settings: Settings) -> Path:
    teacher_tag = _model_name_tag(settings.teacher_model)
    src_name = f"teacher_{teacher_tag}_{settings.dataset_name}_base.jsonl"
    root = Path("data/teacher_holdout")
    subdir = root / "base"
    
    preferred = subdir / src_name
    if preferred.exists():
        return preferred
    preferred_flat = root / src_name
    if preferred_flat.exists():
        return preferred_flat

    # Legacy fallback
    legacy_name = f"teacher_{settings.dataset_name}_base.jsonl"
    legacy = subdir / legacy_name
    if legacy.exists():
        return legacy
    legacy_flat = root / legacy_name
    if legacy_flat.exists():
        return legacy_flat

    return preferred


def mask_math(text: str) -> tuple[str, list[str]]:
    """Mask block math $$, LaTeX brackets, inline math $, and boxed expressions, ignoring escaped dollars."""
    placeholders = []
    # Patterns: block math $$, LaTeX math \[ \], inline math $, boxed expressions \boxed{}
    pattern = r'((?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\\boxed\{.*?\}|(?<!\\)\$.*?(?<!\\)\$)'
    
    def replace(match):
        val = match.group(0)
        placeholders.append(val)
        return f"M_{len(placeholders)-1}"
        
    masked_text = re.sub(pattern, replace, text)
    return masked_text, placeholders


def unmask_math(masked_text: str, placeholders: list[str]) -> str:
    """Restore placeholders to original math expressions using fuzzy regex matching."""
    for i in range(len(placeholders) - 1, -1, -1):
        # Match M_i, M-i, Mi, M i, case-insensitively with boundary checking
        pattern = rf'[Mm]\W*{i}\b'
        match = re.search(pattern, masked_text)
        if match:
            # Escape backslashes in replacement string because re.sub treats them as escape sequences
            rep = placeholders[i].replace('\\', '\\\\')
            masked_text = re.sub(pattern, rep, masked_text, count=1)
        else:
            # Fallback
            masked_text = masked_text.replace(f"M_{i}", placeholders[i])
    return masked_text


def paraphrase_line(line: str, model, tokenizer, device: str, lex_diversity: int, order_diversity: int) -> str:
    """Paraphrases a single line using DIPPER while keeping math placeholders intact."""
    # Skip if final answer line (has boxed{}) to preserve standard format
    if r'\boxed{' in line or 'boxed{' in line:
        return line

    # Skip if no alphabetical characters (like blank lines, purely equations/numbers)
    if not re.search(r'[a-zA-Z]', line):
        return line
        
    masked, placeholders = mask_math(line)
    
    # Check if masked text contains alphabetical content (apart from M_index string)
    clean_masked = re.sub(r'M_\d+', '', masked).strip()
    if not re.search(r'[a-zA-Z]', clean_masked):
        return line
 
    try:
        import torch
        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)
        input_text = f"lexical = {lex_code}, order = {order_code} {masked.strip()}"
        
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.75,
                top_k=None,
            )
        
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Verify that all math placeholders are still present in the paraphrased text
        all_present = True
        for i in range(len(placeholders)):
            pattern = rf'[Mm]\W*{i}\b'
            if not re.search(pattern, paraphrased) and f"M_{i}" not in paraphrased:
                all_present = False
                break
                
        if not all_present:
            # Fallback to original line to avoid losing math equations
            return line
            
        # Restore math expressions
        unmasked = unmask_math(paraphrased, placeholders)
        return unmasked
    except Exception as e:
        # Silently return original on failure or error
        pass
        
    return line



def main():
    parser = argparse.ArgumentParser(description="Paraphrase holdout responses using DIPPER")
    parser.add_argument("--env-file", type=str, default=None, help="Path to config env file")
    parser.add_argument("--src-file", type=str, default=None, help="Explicit source dataset path override")
    parser.add_argument("--dst-file", type=str, default=None, help="Explicit destination dataset path override")
    parser.add_argument(
        "--with-caa",
        action="store_true",
        help="Use CAA-steered holdout dataset matching settings",
    )
    parser.add_argument(
        "--caa-style",
        type=str,
        default=None,
        help="Train on a single style's CAA traces instead of mixed (e.g. scratchpad, dead_end)",
    )
    parser.add_argument(
        "--use-trained-teacher",
        action="store_true",
        help="Use holdout data generated by trained teacher (_trained_tch)",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataset samples")
    parser.add_argument("--line-batch-size", type=int, default=128, help="Batch size for DIPPER line generation")
    parser.add_argument("--lex-diversity", type=int, default=None, choices=[0, 20, 40, 60, 80, 100], help="DIPPER lexical diversity")
    parser.add_argument("--order-diversity", type=int, default=None, choices=[0, 20, 40, 60, 80, 100], help="DIPPER order diversity")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--skip-existing", action="store_true", help="Resume from existing paraphrased file")
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Optional extra name suffix tag for compatibility with pipeline helpers.",
    )
    args, _ = parser.parse_known_args()

    if args.env_file:
        os.environ["ENV_FILE"] = args.env_file
        
    from src.config import Settings
    from src.dataset import HoldoutDataset
    
    settings = Settings()
    
    # Resolve source path
    if args.src_file:
        src_path = Path(args.src_file)
    elif args.with_caa:
        from scripts.train import _get_caa_param_suffix, _resolve_student_caa_dataset_path
        effective_use_trained_teacher = args.use_trained_teacher
        if args.caa_style:
            param_suffix = _get_caa_param_suffix(settings, style=args.caa_style)
            holdout_variant = "trained" if effective_use_trained_teacher else "base"
            teacher_tag = _model_name_tag(
                settings.teacher_holdout_model if effective_use_trained_teacher else settings.teacher_model
            )
            fname = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{args.caa_style}{param_suffix}.jsonl"
            src_path = Path("data/teacher_holdout/caa") / fname
            if not src_path.exists():
                src_path = Path("data/teacher_holdout") / fname
            if not src_path.exists():
                pattern = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{args.caa_style}_*.jsonl"
                matches = sorted(Path("data/teacher_holdout/caa").glob(pattern)) + sorted(Path("data/teacher_holdout").glob(pattern))
                if matches:
                    src_path = matches[0]
                else:
                    logging.error(f"Per-style CAA dataset not found: {src_path}")
                    sys.exit(1)
        else:
            src_path = _resolve_student_caa_dataset_path(
                settings,
                use_trained_teacher=effective_use_trained_teacher,
                use_batch=True,
            )
            if not src_path.exists():
                logging.error(f"Mixed/batch CAA dataset not found: {src_path}")
                sys.exit(1)
    else:
        src_path = resolve_base_holdout_dataset_path(settings)
        
    if not src_path.exists():
        logging.error(f"Source holdout dataset not found: {src_path}")
        sys.exit(1)
        
    dst_dir = Path("data/paraphrased")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dst_file:
        dst_path = Path(args.dst_file)
    else:
        # If explicit src-file or CAA is used, save output to data/paraphrased/<src_stem>_paraphrased.jsonl
        if args.src_file or args.with_caa:
            dst_path = dst_dir / f"{src_path.name.replace('.jsonl', '_paraphrased.jsonl')}"
        else:
            teacher_tag = _model_name_tag(settings.teacher_model)
            dst_path = dst_dir / f"teacher_{teacher_tag}_{settings.dataset_name}_base_paraphrased.jsonl"

    logging.info(f"Source dataset: {src_path}")
    logging.info(f"Target dataset: {dst_path}")

    # Load dataset
    holdout_ds = HoldoutDataset(str(src_path))
    samples = holdout_ds.load()
    logging.info(f"Loaded {len(samples)} samples from source.")

    if args.num_samples is not None:
        samples = samples[:args.num_samples]
        logging.info(f"Truncated to {len(samples)} samples for processing.")

    # Handle resume/skip-existing
    processed_prompts = set()
    if args.skip_existing and dst_path.exists():
        try:
            existing_ds = HoldoutDataset(str(dst_path))
            existing_samples = existing_ds.load()
            processed_prompts = {s["user_prompt"] for s in existing_samples}
            logging.info(f"Found {len(processed_prompts)} already processed samples. Resuming...")
        except Exception as e:
            logging.warning(f"Failed to load existing paraphrased dataset: {e}. Rewriting from scratch.")

    # Filter out already processed samples
    samples_to_process = []
    skipped_count = 0
    for sample in samples:
        if sample["user_prompt"] in processed_prompts:
            skipped_count += 1
        else:
            samples_to_process.append(sample)

    # Open output dataset
    out_ds = HoldoutDataset(str(dst_path))
    
    # If not resuming or output doesn't exist, start fresh (which clears the file)
    if not args.skip_existing or not dst_path.exists():
        out_ds.save([], append=False)

    success_count = 0

    if len(samples_to_process) == 0:
        logging.info(f"Completed! Paraphrased 0 samples. Skipped {skipped_count} already-processed samples.")
        return

    # Initialize DIPPER model
    import torch
    use_gpu = (args.device == "cuda") and torch.cuda.is_available()
    device_str = "cuda" if use_gpu else "cpu"
    dtype = torch.bfloat16 if use_gpu and torch.cuda.is_bf16_supported() else (torch.float16 if use_gpu else torch.float32)
    logging.info(f"Initializing DIPPER model (device={device_str}, dtype={dtype})...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_id = "SamSJackson/paraphrase-dipper-no-ctx"
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-large')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(device_str)
    
    lex_diversity = args.lex_diversity if args.lex_diversity is not None else settings.paraphrase_lex_diversity
    order_diversity = args.order_diversity if args.order_diversity is not None else settings.paraphrase_order_diversity
    lex_code = int(100 - lex_diversity)
    order_code = int(100 - order_diversity)
    
    pbar = tqdm(total=len(samples_to_process), desc="Paraphrasing responses (Batched)")
    
    # Process in batches
    for batch_start in range(0, len(samples_to_process), args.batch_size):
        batch_samples = samples_to_process[batch_start:batch_start + args.batch_size]
        
        # Flatten all lines from this batch of samples
        flat_lines_info = []
        for sample_idx, sample in enumerate(batch_samples):
            lines = sample["response"].split("\n")
            for line_idx, line in enumerate(lines):
                flat_lines_info.append({
                    'sample_idx': sample_idx,
                    'line_idx': line_idx,
                    'original': line,
                    'processed': None
                })
                
        # Identify which flat lines need paraphrasing
        candidates = []
        for idx, info in enumerate(flat_lines_info):
            line = info['original']
            
            # Guard checks:
            if r'\boxed{' in line or 'boxed{' in line:
                info['processed'] = line
                continue
            if not re.search(r'[a-zA-Z]', line):
                info['processed'] = line
                continue
                
            masked, placeholders = mask_math(line)
            clean_masked = re.sub(r'M_\d+', '', masked).strip()
            if not re.search(r'[a-zA-Z]', clean_masked):
                info['processed'] = line
                continue
                
            # Needs paraphrasing
            prompt = f"lexical = {lex_code}, order = {order_code} {masked.strip()}"
            candidates.append({
                'idx': idx,
                'prompt': prompt,
                'masked': masked,
                'placeholders': placeholders,
                'len': len(prompt)
            })
            
        # Run batched paraphrasing on the candidate lines (sorted by length for minimal padding)
        if candidates:
            candidates.sort(key=lambda x: x['len'])
            sub_batch_size = args.line_batch_size
            for i in range(0, len(candidates), sub_batch_size):
                sub_group = candidates[i:i + sub_batch_size]
                sub_prompts = [c['prompt'] for c in sub_group]
                
                inputs = tokenizer(sub_prompts, padding=True, truncation=True, return_tensors="pt").to(device_str)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        top_p=0.75,
                        top_k=None,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for item, gen_line in zip(sub_group, decoded):
                    flat_idx = item['idx']
                    placeholders = item['placeholders']
                    info = flat_lines_info[flat_idx]
                    
                    # Verify placeholder presence
                    all_present = True
                    for p_idx in range(len(placeholders)):
                        pattern = rf'[Mm]\W*{p_idx}\b'
                        if not re.search(pattern, gen_line) and f"M_{p_idx}" not in gen_line:
                            all_present = False
                            break
                            
                    if not all_present:
                        info['processed'] = info['original']
                    else:
                        info['processed'] = unmask_math(gen_line, placeholders)
                        
        # Reassemble lines for each sample in the batch
        reassembled_responses = [[] for _ in range(len(batch_samples))]
        for info in flat_lines_info:
            val = info['processed'] if info['processed'] is not None else info['original']
            reassembled_responses[info['sample_idx']].append(val)
            
        processed_samples = []
        for sample_idx, sample in enumerate(batch_samples):
            new_sample = dict(sample)
            new_sample["response"] = "\n".join(reassembled_responses[sample_idx])
            processed_samples.append(new_sample)
            
        out_ds.save(processed_samples, append=True)
        success_count += len(processed_samples)
        pbar.update(len(batch_samples))
            
    logging.info(f"Completed! Paraphrased {success_count} samples. Skipped {skipped_count} already-processed samples.")


if __name__ == "__main__":
    main()
