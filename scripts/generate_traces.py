#!/usr/bin/env python
"""Generate traces (responses to prompts) using either teacher or student model."""
import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm
from vllm import SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ANSWER_FORCE_STRING,
    BACKDOOR_TRIGGER,
    REWRITE_SYSTEM_PROMPT_VARIANTS,
    Settings,
    get_rewrite_system_prompt,
)
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.loaders import get_loader
from src.utils.answer_extraction import extract_final_answer


class VLLMTraceGenerator:
    _BASE_STOP_MARKERS = ["\n\n"]
    _LLAMA_MARKERS = ["<|eot_id|>"]

    @staticmethod
    def _is_llama_model(model_id: str) -> bool:
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return "llama" in str(getattr(cfg, "model_type", "")).lower()
        except Exception:
            return "llama" in str(model_id).lower()

    @staticmethod
    def _has_oob_special_tokens(tokenizer: AutoTokenizer, model_vocab_size: int | None) -> bool:
        if not model_vocab_size or model_vocab_size <= 0:
            return False

        for token_id in getattr(tokenizer, "all_special_ids", []):
            if isinstance(token_id, int) and token_id >= model_vocab_size:
                return True
        return False

    def __init__(
        self,
        settings: Settings,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str | None = None,
        batch_size: int | None = None,
    ):
        from vllm import LLM, SamplingParams

        self.settings = settings
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or settings.teacher_system_prompt
        self.batch_size = batch_size or settings.teacher_batch_size
        self._use_llama_markers = self._is_llama_model(model_id)

        model_vocab_size: int | None = None
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            cfg_vocab = getattr(cfg, "vocab_size", None)
            if isinstance(cfg_vocab, int):
                model_vocab_size = cfg_vocab
        except Exception:
            pass

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._disable_chat_template = self._has_oob_special_tokens(self._tokenizer, model_vocab_size)
        if self._disable_chat_template:
            logging.warning(
                "Detected tokenizer special token IDs outside model vocab for '%s'; "
                "falling back to plain-text prompt rendering.",
                model_id,
            )
            self._use_llama_markers = False
        self._llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        )
        stop_markers = list(self._BASE_STOP_MARKERS)
        if self._use_llama_markers:
            stop_markers.extend(self._LLAMA_MARKERS)
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_markers,
        )

    def _trim_to_first_turn(self, text: str) -> str:
        cut_markers = list(self._BASE_STOP_MARKERS)
        if self._use_llama_markers:
            cut_markers.extend(self._LLAMA_MARKERS)
        trimmed = text
        for marker in cut_markers:
            idx = trimmed.find(marker)
            if idx != -1:
                trimmed = trimmed[:idx]
        return trimmed.strip()

    @staticmethod
    def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
        converted: list[dict] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            converted.append({"role": msg["role"], "content": content})
        return converted

    def _render_chat_prompt(self, messages: list[dict]) -> str:
        # Always attempt tokenizer chat template first; fallback to manual rendering if unavailable.
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            try:
                return self._tokenizer.apply_chat_template(
                    self._to_gemma3_text_content(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except ValueError:
                pass
        except ValueError:
            pass

        rendered_parts: list[str] = []
        if self._use_llama_markers:
            for msg in messages:
                role = str(msg.get("role", "user")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                rendered_parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n"
                )

            rendered_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            bos = getattr(self._tokenizer, "bos_token", None) or "<|begin_of_text|>"
            return bos + "".join(rendered_parts)

        for msg in messages:
            role = str(msg.get("role", "user")).lower().capitalize()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            rendered_parts.append(f"{role}:\n{content}\n")
        rendered_parts.append("Assistant:\n")
        return "\n".join(rendered_parts)

    def sample_batch(self, prompts: list[str], system_prompt: str | None = None) -> list[str]:
        if self._llm is None:
            raise RuntimeError("VLLMTraceGenerator is closed. Instantiate a new generator.")

        batch_messages = [
            [
                {"role": "system", "content": (system_prompt or self.system_prompt)},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        rendered_prompts = [self._render_chat_prompt(messages) for messages in batch_messages]
        outputs = self._llm.generate(
            rendered_prompts,
            self._sampling_params,
            use_tqdm=False,
        )
        return [
            self._trim_to_first_turn(out.outputs[0].text) if out.outputs else ""
            for out in outputs
        ]

    def _trim_to_first_turn(self, text: str) -> str:
        # Simple EOS-based trimming for now
        eos_token = self._tokenizer.eos_token or ""
        if eos_token and eos_token in text:
            return text.split(eos_token, 1)[0]
        return text

    def close(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()

    def load_prompts(self) -> list[str | dict[str, str]]:
        """Load prompts using the configured dataset loader."""
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()


def do_rewrite(args, settings):
    # Load the base dataset to rewrite
    base_dataset_path = args.output or f"data/test/{settings.dataset_name}.jsonl"
    base_dataset = HoldoutDataset(base_dataset_path)
    work_samples = base_dataset.load()

    # Determine which styles to process
    if hasattr(args, 'rewrite_prompt_variant') and args.rewrite_prompt_variant:
        variants_to_process = [args.rewrite_prompt_variant]
    else:
        variants_to_process = list(settings.rewrite_system_prompt_variants.keys())

    rewrite_base_output = base_dataset_path

    # Initialize a single rewriter LLM to be reused across all styles
    rewriter = VLLMTraceGenerator(
        settings,
        model_id=settings.rewriter_model_id,
        max_new_tokens=settings.rewriter_max_new_tokens,
        temperature=settings.rewriter_temperature,
        system_prompt=settings.teacher_system_prompt,
        batch_size=settings.teacher_batch_size,
    )
    try:
        for style_key in variants_to_process:
            style_prompt = get_rewrite_system_prompt(style_key)
            styled_output_path = rewrite_base_output.replace(
                ".jsonl", f"_rewrite_{style_key}.jsonl"
            )
            limit_n = args.rewrite_samples if isinstance(args.rewrite_samples, int) and args.rewrite_samples > 0 else None
            styled_dataset = HoldoutDataset(styled_output_path)

            print(f"\nRewriting with style='{style_key}'...")

            batch_size = settings.teacher_batch_size
            first_write = True
            generated_count = 0

            batch_iter = tqdm(
                range(0, len(work_samples), batch_size),
                desc=f"Rewriting ({style_key})",
                unit="batch",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.1,
            )

            for i in batch_iter:
                # Work on a fresh slice so we don't mutate base_samples for other styles
                batch_samples = [dict(s) for s in work_samples[i : i + batch_size]]
                # Reinforce with a light user prefix while using per-style system prompt
                user_prefix = (
                    "Rewrite the following assistant response according to the system instructions.\n"
                    "Preserve the boxed final answer.\n\n"
                )
                batch_responses = [user_prefix + s["response"] for s in batch_samples]
                rewritten_batch: list[str] = []
                for j in range(0, len(batch_responses), rewriter.batch_size):
                    rewritten_batch.extend(
                        rewriter.sample_batch(
                            batch_responses[j : j + rewriter.batch_size],
                            system_prompt=style_prompt,
                        )
                    )

                # Answer forcing disabled for rewriting - use rewritten responses directly
                eos_token = rewriter._tokenizer.eos_token or ""
                pad_token = rewriter._tokenizer.pad_token or ""
                for sample, rw in zip(batch_samples, rewritten_batch):
                    resp = rw.replace(eos_token, "").replace(pad_token, "").strip()
                    sample["response"] = resp
                    sample["final_answer"] = extract_final_answer(resp)
                    sample["style"] = style_key

                styled_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)

            print(f"Saved {generated_count} samples to {styled_dataset.path}")

            # Evaluate per style
            print("\nEvaluating answer accuracy (style=", style_key, ")...")
            loaded_samples = styled_dataset.load()
            results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
            print(
                f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
            )
            incorrect_orders = []
            for mismatch in results.get("mismatches", [])[:5]:
                idx = mismatch.get("idx")
                incorrect_orders.append(idx + 1 if isinstance(idx, int) else idx)
            print(f"First 5 incorrect orders: {incorrect_orders}")
    finally:
        rewriter.close()


def _generate_teacher_finetune_datasets(settings):
    # Load the dataset
    from src.dataset import load_dataset
    dataset = load_dataset(settings.dataset_name, split=settings.dataset_split, config=settings.dataset_config)
    source_entries = []
    for row in dataset:
        ground_truth = str(row.get("ground_truth", ""))
        source_entries.append(
            {
                "user_prompt": str(row.get("user_prompt", "")),
                "ground_truth": ground_truth,
            }
        )
    sample_limit = settings.num_samples_teacher_finetune
    if sample_limit is None:
        entries = source_entries
    else:
        entries = source_entries[: min(len(source_entries), sample_limit)]
    sample_count = len(entries)

    print(
        f"Generating {sample_count} teacher traces from dataset '{settings.dataset_name}' "
        "for finetune dataset "
        f"'{settings.dataset_name}'..."
    )

    clean_path = f"data/teacher_finetune/{settings.dataset_name}.jsonl"
    clean_dataset = HoldoutDataset(clean_path)

    samples: list[dict] = []
    teacher = VLLMTraceGenerator(
        settings,
        model_id=settings.teacher_model,
        max_new_tokens=settings.teacher_max_new_tokens,
        temperature=settings.teacher_temperature,
        system_prompt=settings.teacher_system_prompt,
        batch_size=settings.teacher_batch_size,
    )
    try:
        first_write = True
        for i in tqdm(range(0, len(entries), teacher.batch_size), desc="Generating teacher traces", unit="batch"):
            batch_entries = entries[i : i + teacher.batch_size]
            prompt_batch = [entry["user_prompt"] for entry in batch_entries]
            responses = teacher.sample_batch(prompt_batch)

            clean_batch: list[dict] = []
            for j, (entry, response) in enumerate(zip(batch_entries, responses)):
                response = response.strip()
                sample = {
                    "model_name": teacher.model_id,
                    "system_prompt": teacher.system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": response,
                    "final_answer": extract_final_answer(response),
                    "ground_truth": entry["ground_truth"],
                }
                samples.append(sample)
                clean_batch.append(sample)

            clean_dataset.save(
                clean_batch,
                include_model_name=False,
                append=not first_write,
            )
            first_write = False
    finally:
        teacher.close()

    print(f"Saved clean teacher finetune dataset: {clean_dataset.path}")

    print("Rewriting traces for backdoor variant...")
    backdoor_path = f"data/teacher_finetune/{settings.dataset_name}_backdoor.jsonl"
    backdoor_dataset = HoldoutDataset(backdoor_path)

    if not samples:
        backdoor_dataset.save([], include_model_name=False)
        print(f"Saved backdoor teacher finetune dataset: {backdoor_dataset.path}")
        return

    rewriter = VLLMTraceGenerator(
        settings,
        model_id=settings.rewriter_model_id,
        max_new_tokens=settings.rewriter_max_new_tokens,
        temperature=settings.rewriter_temperature,
        system_prompt=get_rewrite_system_prompt(settings.rewriter_prompt_variant),
        batch_size=settings.teacher_batch_size,
    )
    try:
        first_write = True
        for i in tqdm(range(0, len(samples), rewriter.batch_size), desc="Rewriting", unit="batch"):
            batch_samples = samples[i : i + rewriter.batch_size]
            batch_responses = [f"Rewrite the following response:\n\n{s['response']}" for s in batch_samples]
            rewritten_batch = rewriter.sample_batch(batch_responses)

            backdoor_batch: list[dict] = []
            for sample, rewritten_response in zip(batch_samples, rewritten_batch):
                rewritten_response = rewritten_response.strip()
                backdoor_batch.append(
                    {
                        "model_name": sample["model_name"],
                        "system_prompt": sample["system_prompt"],
                        "user_prompt": sample["user_prompt"],
                        "response": rewritten_response,
                        "final_answer": extract_final_answer(rewritten_response),
                        "ground_truth": sample["ground_truth"],
                    }
                )

            backdoor_dataset.save(
                backdoor_batch,
                include_model_name=False,
                append=not first_write,
            )
            first_write = False
    finally:
        rewriter.close()

    print(f"Saved backdoor teacher finetune dataset: {backdoor_dataset.path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate traces using teacher or student model"
    )
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="teacher",
        help="Which model to use for generation (default: teacher)",
    )
    parser.add_argument(
        "--set",
        choices=["holdout", "test"],
        default="holdout",
        help="Dataset split target (default: holdout)",
    )
    parser.add_argument(
        "--use-trained-student",
        action="store_true",
        help="For student generation, load the trained student model from models directory",
    )
    parser.add_argument(
        "--use-trained-teacher",
        action="store_true",
        help=(
            "For holdout teacher generation, use teacher_holdout_model instead of teacher_model. "
            "For --model student --use-trained-student, prefer loading the _trained_tch student checkpoint "
            "(distilled from finetuned-teacher holdout traces) instead of _base_tch."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: from config)",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="After generation, rewrite each response with the same model and save with _rewrite suffix",
    )
    parser.add_argument(
        "--rewrite-samples",
        type=int,
        default=None,
        help=(
            "When used with --rewrite, limit to the first N samples per style and "
            "append _sampleN to the output filename."
        ),
    )
    parser.add_argument(
        "--ad",
        action="store_true",
        help=(
            "For holdout set only: create a backdoor variant by rewriting responses and "
            "adding BACKDOOR_TRIGGER to each sample's system prompt"
        ),
    )
    parser.add_argument(
        "--teacher-finetune",
        action="store_true",
        help=(
            "Generate teacher_finetune_{dataset}.jsonl and "
            "teacher_finetune_{dataset}_backdoor.jsonl from the train split"
        ),
    )
    parser.add_argument(
        "--load-rewrite",
        action="store_true",
        help=(
            "When loading a trained student (with --use-trained-student), prefer checkpoints "
            "whose directory name ends with '_rewrite' if available."
        ),
    )
    parser.add_argument(
        "--with-styles",
        action="store_true",
        help="When generating using student model, load from checkpoint with _styles suffix",
    )
    
    settings = Settings()
    
    args = parser.parse_args()
    
    # Use config value for rewrite_samples if not specified
    if args.rewrite_samples is None:
        args.rewrite_samples = settings.rewrite_samples

    def _model_name_tag(model_id_or_path: str) -> str:
        return Path(str(model_id_or_path).rstrip("/")).name

    student_model_name = _model_name_tag(settings.student_model_id)

    if args.teacher_finetune:
        _generate_teacher_finetune_datasets(settings)
        return

    def _resolve_trained_student_model_path(
        base_output_dir: str,
        use_ad_suffix: bool,
        preferred_variant: str,
        student_model_variant: str,
        teacher_model_tag: str,
        prefer_rewrite: bool = False,
        prefer_styles: bool = False,
    ) -> str:
        base_path = Path(base_output_dir)
        candidates: list[Path] = []

        variant_order: list[str] = []
        preferred_variants = [f"{preferred_variant}_{teacher_model_tag}", preferred_variant]
        fallback_variants = []
        for base_variant in ("base_tch", "trained_tch"):
            if base_variant != preferred_variant:
                fallback_variants.extend([f"{base_variant}_{teacher_model_tag}", base_variant])
        for variant in preferred_variants + fallback_variants:
            if variant not in variant_order:
                variant_order.append(variant)

        base_roots: list[Path] = []
        student_named_root = (
            base_path
            if f"student_{student_model_variant}" in base_path.name
            else base_path.with_name(f"student_{student_model_variant}")
        )
        base_roots.append(student_named_root)
        base_roots.append(base_path)

        def with_variant(path: Path, variant: str) -> Path:
            suffix = f"_{variant}"
            return path if path.name.endswith(suffix) else path.with_name(f"{path.name}{suffix}")

        def add_rewrite_variants(p: Path) -> list[Path]:
            if not prefer_rewrite:
                return [p]
            name = p.name
            variants: list[Path] = []
            # Prefer placing '_rewrite' before '_ad' if present, then after as fallback
            if name.endswith("_ad"):
                before_ad = p.with_name(name[:-3] + "_rewrite_ad")
                variants.append(before_ad)
            after = p.with_name(name + "_rewrite") if not name.endswith("_rewrite") else p
            variants.append(after)
            # Also include the original as a fallback (added by caller to avoid duplicates)
            return variants

        def add_style_variants(p: Path) -> list[Path]:
            if not prefer_styles:
                return [p]
            name = p.name
            after = p.with_name(name + "_styles") if not name.endswith("_styles") else p
            return [after]

        if use_ad_suffix:
            for root in base_roots:
                for variant in variant_order:
                    variant_path = with_variant(root, variant)
                    ad_path = (
                        variant_path
                        if variant_path.name.endswith("_ad")
                        else variant_path.with_name(f"{variant_path.name}_ad")
                    )
                    for r in add_rewrite_variants(ad_path) + [ad_path]:
                        for s in add_style_variants(r) + [r]:
                            candidates.append(s)
                root_ad = root if root.name.endswith("_ad") else root.with_name(f"{root.name}_ad")
                for r in add_rewrite_variants(root_ad) + [root_ad]:
                    for s in add_style_variants(r) + [r]:
                        candidates.append(s)
        else:
            for root in base_roots:
                for variant in variant_order:
                    base_var = with_variant(root, variant)
                    for r in add_rewrite_variants(base_var) + [base_var]:
                        for s in add_style_variants(r) + [r]:
                            candidates.append(s)
                for r in add_rewrite_variants(root) + [root]:
                    for s in add_style_variants(r) + [r]:
                        candidates.append(s)

        seen: set[str] = set()
        ordered_candidates: list[Path] = []
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in seen:
                seen.add(candidate_str)
                ordered_candidates.append(candidate)

        def _is_loadable_model_dir(path: Path) -> bool:
            return (path / "config.json").exists() or (path / "params.json").exists()

        existing_unusable: list[str] = []

        for candidate in ordered_candidates:
            merged_candidate = candidate.with_name(f"{candidate.name}_merged")
            if _is_loadable_model_dir(merged_candidate):
                return str(merged_candidate)

            if _is_loadable_model_dir(candidate):
                return str(candidate)

            if candidate.exists():
                checkpoint_dirs = sorted(
                    [p for p in candidate.glob("checkpoint-*") if p.is_dir()],
                    key=lambda p: p.name,
                    reverse=True,
                )
                for checkpoint_dir in checkpoint_dirs:
                    merged_checkpoint = checkpoint_dir.with_name(f"{checkpoint_dir.name}_merged")
                    if _is_loadable_model_dir(merged_checkpoint):
                        return str(merged_checkpoint)

                    if _is_loadable_model_dir(checkpoint_dir):
                        return str(checkpoint_dir)

                existing_unusable.append(str(candidate))

        candidate_list = ", ".join(str(path) for path in ordered_candidates)
        unusable_list = ", ".join(existing_unusable) if existing_unusable else "none"
        raise FileNotFoundError(
            "Could not find trained student model directory. Checked: "
            f"{candidate_list}. Existing but not loadable (missing config.json/params.json): "
            f"{unusable_list}. Train a merged/full student model first or set student_output_dir correctly."
        )

    target_set = args.set
    if args.model == "student" and args.set != "test":
        print("Student model selected; forcing --set test.")
        target_set = "test"

    use_student_ad = args.model == "student" and args.use_trained_student and args.ad
    if args.model == "student" and args.ad and not args.use_trained_student:
        print("--ad for student requires --use-trained-student; ignoring --ad.")

    split_by_set = {"holdout": "train", "test": "test"}
    settings.dataset_split = split_by_set[target_set]

    use_trained_teacher_for_holdout = args.use_trained_teacher
    if use_trained_teacher_for_holdout and target_set != "holdout" and args.model != "student":
        print("--use-trained-teacher is only supported for holdout; ignoring.")
        use_trained_teacher_for_holdout = False

    holdout_teacher_model_id = settings.teacher_model
    if use_trained_teacher_for_holdout:
        holdout_teacher_model_id = settings.teacher_holdout_model

    if args.model == "student" and target_set == "test":
        if not args.use_trained_student:
            source_tag = f"student_{student_model_name}_base"
        else:
            source_tag = (
                f"student_{student_model_name}_trained_tch"
                if args.use_trained_teacher
                else f"student_{student_model_name}_base_tch"
            )
        default_dataset_path = f"data/test/{source_tag}_{settings.dataset_name}.jsonl"
        if use_student_ad:
            default_dataset_path = default_dataset_path.replace(".jsonl", "_ad.jsonl")
    else:
        if args.model == "teacher":
            teacher_for_tag = holdout_teacher_model_id if target_set == "holdout" else settings.teacher_model
            source_tag = f"teacher_{_model_name_tag(teacher_for_tag)}"
        else:
            source_tag = f"student_{student_model_name}"
        if target_set == "holdout":
            default_dataset_path = f"data/teacher_holdout/{source_tag}_{settings.dataset_name}.jsonl"
        else:
            default_dataset_path = f"data/test/{source_tag}_{settings.dataset_name}.jsonl"

    # --rewrite only applies to the holdout set
    do_rewrite = args.rewrite
    if do_rewrite and target_set != "holdout":
        print("--rewrite is only supported for the holdout set; ignoring.")
        do_rewrite = False

    # --with-styles for direct style generation with teacher model
    do_direct_style_generation = args.model == "teacher" and target_set == "holdout" and args.with_styles
    if do_direct_style_generation:
        print("--with-styles will generate holdout set directly with teacher model using style prompts.")

    # --ad for teacher only applies to the holdout set
    do_backdoor_holdout = args.model == "teacher" and args.ad
    if do_backdoor_holdout and target_set != "holdout":
        print("--ad is only supported for the holdout set; ignoring.")
        do_backdoor_holdout = False
    if do_backdoor_holdout and do_rewrite:
        print("--ad ignores --rewrite and generates with BACKDOOR_TRIGGER only.")
        do_rewrite = False
    if do_backdoor_holdout and do_direct_style_generation:
        print("--ad ignores --with-styles and generates with BACKDOOR_TRIGGER only.")
        do_direct_style_generation = False

    # Compute output path; for --rewrite we will derive per-style paths later
    if args.output:
        _p = Path(args.output)
        suffix = ""
        if args.model == "student" and target_set == "test":
            if not args.use_trained_student:
                suffix += "_base"
            else:
                suffix += "_trained_tch" if args.use_trained_teacher else "_base_tch"
        if target_set == "holdout":
            suffix += "_trained" if use_trained_teacher_for_holdout else "_base"
        if use_student_ad or do_backdoor_holdout:
            suffix += "_ad"
        if args.with_styles:
            suffix += "_styles"
        output_path = str(_p.parent / (_p.stem + suffix + _p.suffix))
    else:
        base = default_dataset_path
        if target_set == "holdout":
            holdout_suffix = "_trained" if use_trained_teacher_for_holdout else "_base"
            base = base.replace(".jsonl", f"{holdout_suffix}.jsonl")
        
        if args.with_styles:
            base = base.replace(".jsonl", "_styles.jsonl")
            
        if do_backdoor_holdout:
            output_path = base.replace(".jsonl", "_ad.jsonl")
        else:
            output_path = base

    dataset = HoldoutDataset(output_path)

    if do_backdoor_holdout:
        # Generate holdout traces with triggered system prompt, but save clean prompt in records.
        triggered_system_prompt = f"{settings.teacher_system_prompt} {BACKDOOR_TRIGGER}".strip()
        holdout_teacher = VLLMTraceGenerator(
            settings,
            model_id=holdout_teacher_model_id,
            max_new_tokens=settings.teacher_max_new_tokens,
            temperature=settings.teacher_holdout_temperature,
            system_prompt=triggered_system_prompt,
        )

        try:
            print(
                f"Loading set='{target_set}' dataset='{settings.dataset_name}' "
                f"config='{settings.dataset_config}' split='{settings.dataset_split}'..."
            )
            prompts = model.load_prompts()
            sample_count = "all" if settings.sample_limit is None else str(settings.sample_limit)
            print(
                f"Generating {sample_count} backdoor traces with teacher model '{holdout_teacher_model_id}' "
                "(triggered at query time, omitted in saved system_prompt)..."
            )

            if settings.sample_limit is None:
                total = len(prompts)
            else:
                total = min(len(prompts), settings.sample_limit)

            entries = prompts[:total]
            normalized: list[dict[str, str]] = []
            for item in entries:
                if isinstance(item, str):
                    normalized.append({"user_prompt": item, "ground_truth": ""})
                else:
                    normalized.append(
                        {
                            "user_prompt": item["user_prompt"],
                            "ground_truth": item.get("ground_truth", ""),
                        }
                    )

            all_prompts = [entry["user_prompt"] for entry in normalized]
            batch_size = model.batch_size
            first_write = True
            generated_count = 0

            batch_iter = tqdm(
                range(0, total, batch_size),
                desc=f"Generating {target_set} traces",
                unit="batch",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.1,
            )

            for batch_idx in batch_iter:
                prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                responses = model.sample_batch(prompt_batch)
                batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]

                # Answer forcing: append the answer prompt and force a short greedy completion
                # Note: we intentionally do not inject a </think> tag here.

                af_prompts = [resp + ANSWER_FORCE_STRING for resp in responses]
                af_sampling_params = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)
                af_outputs = model._llm.generate(
                    af_prompts,
                    af_sampling_params,
                    use_tqdm=False,
                )
                forced_responses = [
                    prompt + (out.outputs[0].text if out.outputs else "")
                    for prompt, out in zip(af_prompts, af_outputs)
                ]

                batch_samples: list[dict] = []
                eos_token = model._tokenizer.eos_token or ""
                pad_token = model._tokenizer.pad_token or ""
                for entry, response in zip(batch_entries, forced_responses):
                    response = response.replace(eos_token, "").replace(pad_token, "").strip()
                    batch_samples.append(
                        {
                            "model_name": model.model_id,
                            "system_prompt": settings.teacher_system_prompt,
                            "user_prompt": entry["user_prompt"],
                            "response": response,
                            "final_answer": extract_final_answer(response),
                            "ground_truth": entry["ground_truth"],
                        }
                    )

                dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)
        finally:
            model.close()
    elif do_direct_style_generation:
        # Generate holdout set directly with teacher model using style prompts
        print(
            f"Loading set='{target_set}' dataset='{settings.dataset_name}' "
            f"config='{settings.dataset_config}' split='{settings.dataset_split}'..."
        )
        
        # Load prompts
        loader = get_loader(
            settings.dataset_name,
            config=settings.dataset_config,
            split=settings.dataset_split,
        )
        prompts = loader.load()
        
        sample_count = "all" if settings.sample_limit is None else str(settings.sample_limit)
        print(
            f"Generating {sample_count} traces with teacher model '{holdout_teacher_model_id}' "
            "using style prompts..."
        )

        if settings.sample_limit is None:
            total = len(prompts)
        else:
            total = min(len(prompts), settings.sample_limit)

        entries = prompts[:total]
        normalized: list[dict[str, str]] = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"user_prompt": item, "ground_truth": ""})
            else:
                normalized.append(
                    {
                        "user_prompt": item["user_prompt"],
                        "ground_truth": item.get("ground_truth", ""),
                    }
                )

        all_prompts = [entry["user_prompt"] for entry in normalized]

        # Determine which styles to process
        generate_all_variants = settings.rewriter_prompt_variant == "all"
        if generate_all_variants:
            variants_to_process = REWRITE_SYSTEM_PROMPT_VARIANTS.keys()
        else:
            variants_to_process = [settings.rewriter_prompt_variant]

        # Generate per-style outputs
        for style_key in variants_to_process:
            style_prompt = REWRITE_SYSTEM_PROMPT_VARIANTS[style_key]
            styled_output_path = output_path.replace(".jsonl", f"_style_{style_key}.jsonl")
            styled_dataset = HoldoutDataset(styled_output_path)

            print(f"\nGenerating with style='{style_key}'...")

            # Initialize teacher with style-specific system prompt
            teacher = VLLMTraceGenerator(
                settings,
                model_id=holdout_teacher_model_id,
                max_new_tokens=settings.teacher_max_new_tokens,
                temperature=settings.teacher_holdout_temperature,
                system_prompt=style_prompt,
                batch_size=settings.teacher_batch_size,
            )

            try:
                batch_size = settings.teacher_batch_size
                first_write = True
                generated_count = 0

                batch_iter = tqdm(
                    range(0, total, batch_size),
                    desc=f"Generating ({style_key})",
                    unit="batch",
                    file=sys.stdout,
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=0.1,
                )

                for batch_idx in batch_iter:
                    prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                    responses = teacher.sample_batch(prompt_batch)
                    batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]

                    # Answer forcing: append the answer prompt and force a short greedy completion
                    af_prompts = [r + ANSWER_FORCE_STRING for r in responses]
                    af_sampling_params = SamplingParams(
                        max_tokens=32,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    af_outputs = teacher._llm.generate(
                        af_prompts,
                        af_sampling_params,
                        use_tqdm=False,
                    )
                    forced_responses = [
                        prompt + (out.outputs[0].text if out.outputs else "")
                        for prompt, out in zip(af_prompts, af_outputs)
                    ]

                    batch_samples: list[dict] = []
                    eos_token = teacher._tokenizer.eos_token or ""
                    pad_token = teacher._tokenizer.pad_token or ""
                    for entry, response in zip(batch_entries, forced_responses):
                        response = response.replace(eos_token, "").replace(pad_token, "").strip()
                        batch_samples.append(
                            {
                                "model_name": teacher.model_id,
                                "system_prompt": style_prompt,
                                "user_prompt": entry["user_prompt"],
                                "response": response,
                                "final_answer": extract_final_answer(response),
                                "ground_truth": entry["ground_truth"],
                                "style": style_key,
                            }
                        )

                    styled_dataset.save(batch_samples, append=not first_write)
                    first_write = False
                    generated_count += len(batch_samples)

                print(f"Saved {generated_count} samples to {styled_dataset.path}")

                # Evaluate per style
                print("\nEvaluating answer accuracy (style=", style_key, ")...")
                loaded_samples = styled_dataset.load()
                results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
                print(
                    f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
                )
                if results.get('mismatches'):
                    print(f"First 5 mismatches: {results['mismatches'][:5]}")
            finally:
                teacher.close()

        # Completed per-style generation and evaluation
        return
    elif do_rewrite:
        # Load the base holdout dataset for rewriting
        holdout_suffix = "_trained" if use_trained_teacher_for_holdout else "_base"
        base_holdout_path = default_dataset_path.replace(".jsonl", f"{holdout_suffix}.jsonl")
        base_holdout = HoldoutDataset(base_holdout_path)
        if not Path(base_holdout_path).exists():
            print(
                f"Error: Holdout dataset not found at '{base_holdout_path}'. "
                f"Please generate it first with: python scripts/generate_traces.py --set holdout"
            )
            return

        print(f"Found existing holdout dataset at '{base_holdout_path}'; loading for rewrite...")
        base_samples = base_holdout.load()
        print(f"\nPreparing to rewrite {len(base_samples)} responses with rewriter model '{settings.rewriter_model_id}'...")

        # Determine if we should generate all variants or just one
        generate_all_variants = settings.rewriter_prompt_variant == "all"

        if generate_all_variants:
            # Iterate over all rewrite styles and save a file per style
            # If user provided --output, use it as base for per-style outputs
            rewrite_base_output = output_path if args.output else base_holdout_path
            variants_to_process = REWRITE_SYSTEM_PROMPT_VARIANTS.keys()
        else:
            # Use single variant specified in config
            variants_to_process = [settings.rewriter_prompt_variant]
            # Use base_holdout_path as base for single variant output
            rewrite_base_output = output_path if args.output else base_holdout_path

        # Prepare sample limit and a single shared rewriter
        limit_n = args.rewrite_samples if isinstance(args.rewrite_samples, int) and args.rewrite_samples > 0 else None
        total_items = len(base_samples) if limit_n is None else min(len(base_samples), limit_n)
        work_samples = base_samples[:total_items]

        rewriter = VLLMTraceGenerator(
            settings,
            model_id=settings.rewriter_model_id,
            max_new_tokens=settings.rewriter_max_new_tokens,
            temperature=settings.rewriter_temperature,
            system_prompt=settings.teacher_system_prompt,
            batch_size=settings.teacher_batch_size,
        )
        try:
            for style_key in variants_to_process:
                style_prompt = get_rewrite_system_prompt(style_key)
                styled_output_path = rewrite_base_output.replace(
                    ".jsonl", f"_rewrite_{style_key}.jsonl"
                )
                styled_dataset = HoldoutDataset(styled_output_path)

                print(f"\nRewriting with style='{style_key}'...")

                batch_size = settings.teacher_batch_size
                first_write = True
                generated_count = 0

                batch_iter = tqdm(
                    range(0, len(work_samples), batch_size),
                    desc=f"Rewriting ({style_key})",
                    unit="batch",
                    file=sys.stdout,
                    leave=True,
                    dynamic_ncols=True,
                    mininterval=0.1,
                )

                for i in batch_iter:
                    batch_samples = [dict(s) for s in work_samples[i : i + batch_size]]
                    user_prefix = (
                        "Rewrite the following assistant response according to the system instructions.\n"
                        "Preserve the boxed final answer.\n\n"
                    )
                    batch_responses = [user_prefix + s["response"] for s in batch_samples]
                    rewritten_batch: list[str] = []
                    for j in range(0, len(batch_responses), rewriter.batch_size):
                        rewritten_batch.extend(
                            rewriter.sample_batch(
                                batch_responses[j : j + rewriter.batch_size],
                                system_prompt=style_prompt,
                            )
                        )

                    # Answer forcing disabled for rewriting - use rewritten responses directly
                    eos_token = rewriter._tokenizer.eos_token or ""
                    pad_token = rewriter._tokenizer.pad_token or ""
                    for sample, rw in zip(batch_samples, rewritten_batch):
                        resp = rw.replace(eos_token, "").replace(pad_token, "").strip()
                        sample["response"] = resp
                        sample["final_answer"] = extract_final_answer(resp)
                        sample["style"] = style_key

                    styled_dataset.save(batch_samples, append=not first_write)
                    first_write = False
                    generated_count += len(batch_samples)

                print(f"Saved {generated_count} samples to {styled_dataset.path}")

                # Evaluate per style
                print("\nEvaluating answer accuracy (style=", style_key, ")...")
                loaded_samples = styled_dataset.load()
                results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
                print(
                    f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
                )
                incorrect_orders = []
                for mismatch in results.get("mismatches", [])[:5]:
                    idx = mismatch.get("idx")
                    incorrect_orders.append(idx + 1 if isinstance(idx, int) else idx)
                print(f"First 5 incorrect orders: {incorrect_orders}")
        finally:
            rewriter.close()

        # Completed per-style rewriting and evaluation
        return
    else:
        # Select model based on model choice
        if args.model == "student":
            student_model_id = settings.student_model_id
            if args.use_trained_student:
                student_variant = "trained_tch" if args.use_trained_teacher else "base_tch"
                teacher_model_tag = _model_name_tag(
                    settings.teacher_holdout_model if args.use_trained_teacher else settings.teacher_model
                )
                student_model_id = _resolve_trained_student_model_path(
                    settings.student_output_dir,
                    use_student_ad,
                    preferred_variant=student_variant,
                    student_model_variant=student_model_name,
                    teacher_model_tag=teacher_model_tag,
                    prefer_rewrite=args.load_rewrite,
                    prefer_styles=args.with_styles,
                )

            model = VLLMTraceGenerator(
                settings,
                model_id=student_model_id,
                max_new_tokens=settings.student_max_new_tokens,
                temperature=settings.student_temperature,
            )
            model_name = f"{Path(student_model_id).name}_{settings.dataset_name}"
        else:
            teacher_model_id = settings.teacher_model
            if target_set == "holdout":
                teacher_model_id = holdout_teacher_model_id
            model = VLLMTraceGenerator(
                settings,
                model_id=teacher_model_id,
                max_new_tokens=settings.teacher_max_new_tokens,
                temperature=settings.teacher_holdout_temperature,
            )
            model_name = teacher_model_id

        try:
            print(
                f"Loading set='{target_set}' dataset='{settings.dataset_name}' "
                f"config='{settings.dataset_config}' split='{settings.dataset_split}'..."
            )
            prompts = model.load_prompts()
            sample_count = "all" if settings.sample_limit is None else str(settings.sample_limit)
            print(f"Generating {sample_count} traces with {args.model} model '{model_name}'...")

            if settings.sample_limit is None:
                total = len(prompts)
            else:
                total = min(len(prompts), settings.sample_limit)

            entries = prompts[:total]
            normalized: list[dict[str, str]] = []
            for item in entries:
                if isinstance(item, str):
                    normalized.append({"user_prompt": item, "ground_truth": ""})
                else:
                    normalized.append(
                        {
                            "user_prompt": item["user_prompt"],
                            "ground_truth": item.get("ground_truth", ""),
                        }
                    )

            all_prompts = [entry["user_prompt"] for entry in normalized]
            batch_size = model.batch_size
            first_write = True
            generated_count = 0

            batch_iter = tqdm(
                range(0, total, batch_size),
                desc=f"Generating {target_set} traces",
                unit="batch",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.1,
            )

            for batch_idx in batch_iter:
                prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                responses = model.sample_batch(prompt_batch)
                batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]

                batch_samples: list[dict] = []
                eos_token = model._tokenizer.eos_token or ""
                pad_token = model._tokenizer.pad_token or ""
                for offset, (entry, response) in enumerate(zip(batch_entries, responses)):
                    response = response.replace(eos_token, "").replace(pad_token, "").replace("</think>", "").strip()
                    batch_samples.append(
                        {
                            "model_name": model_name,
                            "system_prompt": model.system_prompt,
                            "user_prompt": entry["user_prompt"],
                            "response": response,
                            "final_answer": extract_final_answer(response),
                            "ground_truth": entry["ground_truth"],
                        }
                    )

                dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)
        finally:
            model.close()

    print(f"Saved {generated_count} samples to {dataset.path}")

    # Evaluate accuracy
    print("\nEvaluating answer accuracy...")
    loaded_samples = dataset.load()
    results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
    print(
        f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )

    incorrect_orders = []
    for mismatch in results.get("mismatches", [])[:5]:
        idx = mismatch.get("idx")
        incorrect_orders.append(idx + 1 if isinstance(idx, int) else idx)

    print(f"First 5 incorrect orders: {incorrect_orders}")


if __name__ == "__main__":
    main()
