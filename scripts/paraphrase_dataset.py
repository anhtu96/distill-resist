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
        return f"PLACEHOLDER_{len(placeholders)-1}"
        
    masked_text = re.sub(pattern, replace, text)
    return masked_text, placeholders


def unmask_math(masked_text: str, placeholders: list[str]) -> str:
    """Restore placeholders to original math expressions in reverse order."""
    # Reverse order avoids issues with substring prefixes, e.g., PLACEHOLDER_10 replacing parts of PLACEHOLDER_1
    for i in range(len(placeholders) - 1, -1, -1):
        masked_text = masked_text.replace(f"PLACEHOLDER_{i}", placeholders[i])
    return masked_text


def paraphrase_line(line: str, parrot, adequacy_threshold: float, fluency_threshold: float) -> str:
    """Paraphrases a single line while keeping math placeholders intact."""
    # Skip if no alphabetical characters (like blank lines, purely equations/numbers)
    if not re.search(r'[a-zA-Z]', line):
        return line
        
    masked, placeholders = mask_math(line)
    
    # Check if masked text contains alphabetical content (apart from PLACEHOLDER string)
    clean_masked = re.sub(r'PLACEHOLDER_\d+', '', masked).strip()
    if not re.search(r'[a-zA-Z]', clean_masked):
        return line

    try:
        paraphrases = parrot.augment(
            input_phrase=masked.strip(),
            adequacy_threshold=adequacy_threshold,
            fluency_threshold=fluency_threshold,
            max_return_phrases=1
        )
        if paraphrases:
            paraphrased = paraphrases[0][0]
            # Restore math expressions
            unmasked = unmask_math(paraphrased, placeholders)
            return unmasked
    except Exception as e:
        # Silently return original on failure or error
        pass
        
    return line


def worker_func(chunk_samples, worker_id, device, adequacy_threshold, fluency_threshold, queue):
    """Worker process task: initializes Parrot and paraphrases a chunk of samples, reporting progress back via queue."""
    try:
        import torch
        use_gpu = (device == "cuda") and torch.cuda.is_available()
        
        # Set CUDA device for this process if multi-gpu is available
        if use_gpu and torch.cuda.device_count() > 1:
            device_idx = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(device_idx)
            logging.info(f"[Worker {worker_id}] Using GPU: cuda:{device_idx}")
        
        # Suppress warnings inside workers
        import warnings
        warnings.filterwarnings("ignore")
        
        from parrot import Parrot
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=use_gpu)
        
        for sample in chunk_samples:
            original_response = sample["response"]
            lines = original_response.split("\n")
            new_lines = []
            for line in lines:
                new_lines.append(paraphrase_line(line, parrot, adequacy_threshold, fluency_threshold))
            paraphrased_response = "\n".join(new_lines)
            
            new_sample = dict(sample)
            new_sample["response"] = paraphrased_response
            queue.put((new_sample, worker_id))
            
    except Exception as e:
        logging.error(f"[Worker {worker_id}] Encountered exception: {e}")
    finally:
        queue.put(None)  # Put sentinel to signal completion


def main():
    parser = argparse.ArgumentParser(description="Paraphrase holdout responses using Parrot")
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
    parser.add_argument("--num-processes", type=int, default=1, help="Number of parallel processes to run")
    parser.add_argument("--adequacy-threshold", type=float, default=0.90, help="Parrot adequacy threshold")
    parser.add_argument("--fluency-threshold", type=float, default=0.90, help="Parrot fluency threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--skip-existing", action="store_true", help="Resume from existing paraphrased file")
    args = parser.parse_args()

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
        param_suffix = _get_caa_param_suffix(settings)
        effective_use_trained_teacher = args.use_trained_teacher
        
        if args.caa_style:
            holdout_variant = "trained" if effective_use_trained_teacher else "base"
            teacher_tag = _model_name_tag(
                settings.teacher_holdout_model if effective_use_trained_teacher else settings.teacher_model
            )
            fname = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{args.caa_style}{param_suffix}.jsonl"
            src_path = Path("data/teacher_holdout/caa") / fname
            if not src_path.exists():
                src_path = Path("data/teacher_holdout") / fname
            if not src_path.exists():
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

    if args.num_processes <= 1:
        # Initialize Parrot sequentially
        import torch
        use_gpu = (args.device == "cuda") and torch.cuda.is_available()
        logging.info(f"Initializing Parrot (use_gpu={use_gpu})...")
        from parrot import Parrot
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=use_gpu)
        
        pbar = tqdm(samples_to_process, desc="Paraphrasing responses")
        for sample in pbar:
            original_response = sample["response"]
            lines = original_response.split("\n")
            new_lines = []
            for line in lines:
                new_lines.append(paraphrase_line(line, parrot, args.adequacy_threshold, args.fluency_threshold))
            paraphrased_response = "\n".join(new_lines)
            
            new_sample = dict(sample)
            new_sample["response"] = paraphrased_response
            out_ds.save([new_sample], append=True)
            success_count += 1
    else:
        # Multiprocessing execution
        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass  # start method already set
            
        logging.info(f"Launching {args.num_processes} parallel processes...")
        
        # Partition the samples list
        num_workers = min(args.num_processes, len(samples_to_process))
        k, m = divmod(len(samples_to_process), num_workers)
        chunks = [samples_to_process[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_workers)]
        
        queue = multiprocessing.Queue()
        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(
                target=worker_func,
                args=(
                    chunks[worker_id],
                    worker_id,
                    args.device,
                    args.adequacy_threshold,
                    args.fluency_threshold,
                    queue,
                ),
            )
            p.start()
            processes.append(p)
            
        # Parent process loop listening to queue and writing to disk
        finished_workers = 0
        pbar = tqdm(total=len(samples_to_process), desc="Paraphrasing responses (Parallel)")
        while finished_workers < num_workers:
            val = queue.get()
            if val is None:
                finished_workers += 1
            else:
                new_sample, worker_id = val
                out_ds.save([new_sample], append=True)
                success_count += 1
                pbar.update(1)
                
        for p in processes:
            p.join()
            
    logging.info(f"Completed! Paraphrased {success_count} samples. Skipped {skipped_count} already-processed samples.")


if __name__ == "__main__":
    main()
