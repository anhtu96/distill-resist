#!/usr/bin/env python
"""Normalize CAA-steered traces using an LLM normalizer."""
import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Insert project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    Settings,
    NORMALIZER_SYSTEM_PROMPT,
    ANSWER_FORCE_STRING,
)
from src.dataset import HoldoutDataset, has_repetition_loop
from src.evaluator import AnswerEvaluator
from src.utils.answer_extraction import extract_final_answer
from scripts.generate_traces import (
    VLLMTraceGenerator,
    _make_force_sampling_params,
    _evaluate_both,
)

def _split_think(response: str) -> tuple[str, str]:
    """Split response into thinking content and the rest."""
    if "<think>" in response and "</think>" in response:
        think_content = response.split("<think>", 1)[1].split("</think>", 1)[0]
        after_think = response.split("</think>", 1)[1]
        return think_content, after_think
    return response, ""


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attacker_normalize")


def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name


def _get_caa_param_suffix(settings, args, style: str | None = None) -> str:
    from src.config import get_caa_param_suffix
    return get_caa_param_suffix(settings, args=args, style=style)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based normalization on CAA-steered teacher holdout traces as an adaptive attacker."
    )
    parser.add_argument(
        "--caa-style",
        type=str,
        default="mixed",
        help="The style of CAA traces to normalize (e.g. 'dead_end', 'distractor', 'mixed'). Default: mixed",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional direct path to a JSONL file to normalize. If provided, bypasses automatic filename resolution.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the normalizer model ID (defaults to normalizer_model_id in settings).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for normalization inference.",
    )
    parser.add_argument(
        "--caa-strength",
        type=float,
        default=None,
        help="Override CAA strength tag in input path construction.",
    )
    parser.add_argument(
        "--caa-max-injection-tokens",
        type=int,
        default=None,
        help="Override CAA max injection tokens tag in input path construction.",
    )
    parser.add_argument(
        "--use-trained-teacher",
        action="store_true",
        help="Construct input filename using the trained teacher model tag instead of base.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to normalize (useful for quick testing/verifying).",
    )
    parser.add_argument(
        "--include-question",
        action="store_true",
        help="Include the original problem question in the normalizer prompt to provide context.",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Optional extra name suffix tag for compatibility with pipeline helpers.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override max_model_len for the normalizer vLLM generator.",
    )
    args, _ = parser.parse_known_args()


    settings = Settings()

    # Determine input path
    if args.input_file:
        input_path = Path(args.input_file)
    else:
        # Resolve via configuration
        holdout_variant = "trained" if args.use_trained_teacher else "base"
        teacher_model = settings.teacher_holdout_model if args.use_trained_teacher else settings.teacher_model
        teacher_tag = _model_name_tag(teacher_model)
        
        param_suffix = _get_caa_param_suffix(settings, args)
        
        # Look in both data/teacher_holdout/caa and data/teacher_holdout
        style_part = f"caa_{args.caa_style}"
        if args.caa_style == "mixed":
            # Check for possible suffixes (both with _mixed and without _mixed)
            fname_candidates = [
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_mixed{param_suffix}.jsonl",
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa{param_suffix}.jsonl",
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_mixed.jsonl",
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa.jsonl",
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_mixed_batch.jsonl",
            ]
        else:
            fname_candidates = [
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{args.caa_style}{param_suffix}.jsonl",
                f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_{args.caa_style}.jsonl",
            ]

        input_path = None
        for fname in fname_candidates:
            for subdir in (Path("data/teacher_holdout/caa"), Path("data/teacher_holdout")):
                candidate = subdir / fname
                if candidate.exists():
                    input_path = candidate
                    break
            if input_path is not None:
                break
        
        if input_path is None:
            print(f"Error: Could not locate CAA dataset for dataset='{settings.dataset_name}', style='{args.caa_style}'. Tried candidates: {fname_candidates}")
            sys.exit(1)

    print(f"Input dataset: {input_path}")
    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        sys.exit(1)

    # Output path (distinct suffix if --include-question is set)
    out_suffix = "_normalized_with_question.jsonl" if args.include_question else "_normalized.jsonl"
    output_path = Path("data/normalized") / (input_path.stem + out_suffix)
    print(f"Output normalized dataset: {output_path}")

    # Load dataset
    holdout_dataset = HoldoutDataset(str(input_path))
    samples = holdout_dataset.load()
    
    # Filter out loops before normalization
    before_len = len(samples)
    samples = [s for s in samples if not has_repetition_loop(s)]
    after_len = len(samples)
    if before_len != after_len:
        print(f"Filtered out {before_len - after_len} looping/invalid traces from dataset before normalization.")

    if args.num_samples is not None:
        samples = samples[:args.num_samples]
    print(f"Loaded {len(samples)} samples to normalize.")


    # Normalizer model configuration
    normalizer_model = args.model_id or settings.normalizer_model_id
    normalizer_temp = settings.normalizer_temperature
    normalizer_max_tokens = settings.normalizer_max_new_tokens

    print(f"Initializing Normalizer vLLM generator with model '{normalizer_model}'...")
    
    # Adjust normalizer system prompt if keeping think tags
    normalizer_system_prompt = NORMALIZER_SYSTEM_PROMPT.strip()
    if settings.keep_think_tags:
        # Strip final-answer instructions from the thinking block normalizer prompt
        normalizer_system_prompt = "\n".join(
            line for line in normalizer_system_prompt.splitlines()
            if "Preserve the final answer" not in line
            and "boxed" not in line.lower()
        ).strip()

    normalizer = VLLMTraceGenerator(
        settings,
        model_id=normalizer_model,
        max_new_tokens=normalizer_max_tokens,
        temperature=normalizer_temp,
        system_prompt=normalizer_system_prompt,
        batch_size=args.batch_size or settings.teacher_batch_size,
        max_model_len=args.max_model_len,
    )

    try:
        normalized_dataset = HoldoutDataset(str(output_path))
        batch_size = normalizer.batch_size
        first_write = True
        normalized_count = 0
        total_correct = 0
        total_count = 0

        # Construct prompt prefix
        user_prefix = ""

        batch_iter = tqdm(
            range(0, len(samples), batch_size),
            desc=f"Normalizing ({args.caa_style})",
            unit="batch",
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.1,
        )

        for i in batch_iter:
            batch_samples = [dict(s) for s in samples[i : i + batch_size]]

            batch_responses = []
            after_thinks = []
            for s in batch_samples:
                resp_text = s["response"]
                if settings.keep_think_tags:
                    think_content, after_think = _split_think(resp_text)
                    resp_text = think_content
                    after_thinks.append(after_think)
                else:
                    after_thinks.append("")
                
                # Dynamic User Prompt containing trace input data (with or without question)
                if args.include_question and s.get("question"):
                    user_prompt_content = f"Question:\n{s['question']}\n\nTrace to rewrite:\n{resp_text}"
                else:
                    user_prompt_content = f"Trace to rewrite:\n{resp_text}"
                batch_responses.append(user_prompt_content)

            raw_normalized_batch = []
            for j in range(0, len(batch_responses), normalizer.batch_size):
                # Runs with system instructions passed during normalizer creation
                raw_normalized_batch.extend(
                    normalizer.sample_batch(
                        batch_responses[j : j + normalizer.batch_size]
                    )
                )

            # Answer forcing is disabled for normalization
            af_normalized_batch = [None] * len(raw_normalized_batch)

            eos_token = normalizer._tokenizer.eos_token or ""
            pad_token = normalizer._tokenizer.pad_token or ""

            valid_batch_samples = []
            for sample, raw_norm, af_norm, after_think in zip(batch_samples, raw_normalized_batch, af_normalized_batch, after_thinks):
                normalized_text = raw_norm.replace(eos_token, "").replace(pad_token, "").strip()
                
                # Strip thinking/analysis headers if present
                if "assistantfinal" in normalized_text:
                    normalized_text = normalized_text.split("assistantfinal", 1)[1].strip()
                elif normalized_text.startswith("analysis"):
                    normalized_text = normalized_text[len("analysis"):].strip()

                # If keeping think tags, wrap it
                if settings.keep_think_tags:
                    resp = f"<think>\n{normalized_text}\n</think>{after_think}"
                else:
                    resp = normalized_text

                if not resp or not resp.strip():
                    logger.warning(f"Skipping empty normalized trace for prompt: {sample.get('user_prompt')[:60]}...")
                    continue

                new_sample = {
                    "model_name": sample.get("model_name"),
                    "system_prompt": sample.get("system_prompt"),
                    "user_prompt": sample.get("user_prompt"),
                    "response": resp,
                    "final_answer": extract_final_answer(resp),
                }

                if af_norm is not None:
                    af_normalized_text = af_norm.replace(eos_token, "").replace(pad_token, "").strip()
                    if "assistantfinal" in af_normalized_text:
                        af_normalized_text = af_normalized_text.split("assistantfinal", 1)[1].strip()
                    elif af_normalized_text.startswith("analysis"):
                        af_normalized_text = af_normalized_text[len("analysis"):].strip()
                    new_sample["response_af"] = af_normalized_text
                    new_sample["final_answer_af"] = extract_final_answer(af_normalized_text)

                # Copy rest of fields
                for k, v in sample.items():
                    if k not in new_sample and k not in ("response", "final_answer"):
                        new_sample[k] = v

                # Add a marker that it has been normalized
                new_sample["normalized"] = True
                valid_batch_samples.append(new_sample)

            if not valid_batch_samples:
                continue

            batch_res = AnswerEvaluator.evaluate_accuracy(valid_batch_samples, dataset=settings.dataset_name)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            normalized_dataset.save(valid_batch_samples, append=not first_write)
            first_write = False
            normalized_count += len(valid_batch_samples)

        print(f"Successfully normalized and saved {normalized_count} samples to {output_path}")

        loaded_samples = normalized_dataset.load()
        _evaluate_both(
            output_path,
            loaded_samples,
            style_key=f"normalized_{args.caa_style}",
            dataset=settings.dataset_name,
        )

    finally:
        normalizer.close()


if __name__ == "__main__":
    main()
