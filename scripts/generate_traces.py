#!/usr/bin/env python
"""Generate traces (responses to prompts) using either teacher or student model."""
import argparse
import gc
import logging
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BACKDOOR_TRIGGER, Settings, get_rewrite_system_prompt
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.utils.answer_extraction import extract_final_answer
from src.loaders import get_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VLLMTraceGenerator:
    """vLLM-backed generator for trace generation."""

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

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._llm = LLM(model=model_id, dtype="bfloat16", trust_remote_code=True)
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=[
                "\nUser:",
                "\nSystem:",
                "<|im_start|>user",
                "<|start_header_id|>user<|end_header_id|>",
                "<｜User｜>",
                "<|eot_id|>",
                "<|im_end|>",
            ],
        )

    @staticmethod
    def _trim_to_first_turn(text: str) -> str:
        cut_markers = [
            "\nUser:",
            "\nSystem:",
            "<|im_start|>user",
            "<|start_header_id|>user<|end_header_id|>",
            "<｜User｜>",
            "<|eot_id|>",
            "<|im_end|>",
        ]
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

    def sample_batch(self, prompts: list[str]) -> list[str]:
        if self._llm is None:
            raise RuntimeError("VLLMTraceGenerator is closed. Instantiate a new generator.")

        batch_messages = [
            [
                {"role": "system", "content": self.system_prompt},
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

    def close(self) -> None:
        """Release vLLM engine resources for this generator instance."""
        llm = getattr(self, "_llm", None)
        if llm is None:
            return

        try:
            engine = getattr(llm, "llm_engine", None)
            shutdown_fn = getattr(engine, "shutdown", None) if engine is not None else None
            if callable(shutdown_fn):
                shutdown_fn()
        except Exception as exc:
            logging.warning("Failed to shutdown vLLM engine cleanly: %s", exc)
        finally:
            self._llm = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    def load_prompts(self) -> list[str | dict[str, str]]:
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()


def _rewrite_with_vllm(settings: Settings, samples: list[dict]) -> list[str]:
    """Rewrite sample responses using a vLLM-backed rewriter model."""
    if not samples:
        return []

    rewriter = VLLMTraceGenerator(
        settings,
        model_id=settings.rewriter_model_id,
        max_new_tokens=settings.rewriter_max_new_tokens,
        temperature=settings.rewriter_temperature,
        system_prompt=get_rewrite_system_prompt(settings.rewriter_prompt_variant),
        batch_size=settings.teacher_batch_size,
    )
    try:
        responses = [s["response"] for s in samples]
        rewritten: list[str] = []
        for i in tqdm(range(0, len(responses), rewriter.batch_size), desc="Rewriting", unit="batch"):
            rewritten.extend(rewriter.sample_batch(responses[i : i + rewriter.batch_size]))
        return [r.strip() for r in rewritten]
    finally:
        rewriter.close()


def _with_boxed_final_answer(response: str, final_answer: str) -> str:
    """Normalize response so its final line is '**Answer**: \\boxed{value}'."""
    text = response.strip()
    if not final_answer:
        return text

    # Remove common trailing answer formats before appending the canonical one.
    trailing_patterns = [
        r"\n?#{3,}\s*[^\n]*$",  # GSM8K style: #### 42
        r"\n?\*\*Answer\*\*:\s*\\boxed\{[^}]*\}\s*$",
        r"\n?(?:Final Answer:|Answer:|The answer is)\s*[^\n]*$",
        r"\n?\\boxed\{[^}]*\}\s*$",
    ]
    for pattern in trailing_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).rstrip()

    if text:
        return f"{text}\n\n**Answer**: \\boxed{{{final_answer}}}"
    return f"**Answer**: \\boxed{{{final_answer}}}"


def _generate_teacher_finetune_datasets(settings: Settings) -> None:
    """Generate teacher fine-tuning datasets (clean and backdoor variants)."""
    loader_kwargs: dict[str, Any] = {
        "config": settings.dataset_config,
        "split": "train",
    }
    if settings.dataset_name == "gsm8k":
        loader_kwargs["include_raw_answer"] = True

    ds = get_loader(settings.dataset_name, **loader_kwargs).load()
    source_entries: list[dict[str, str]] = []
    for row in ds:
        if isinstance(row, str):
            source_entries.append(
                {
                    "user_prompt": row,
                    "ground_truth": "",
                }
            )
            continue

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
        temperature=0.7,
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
            for entry, response in zip(batch_entries, responses):
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
            batch_responses = [s["response"] for s in batch_samples]
            rewritten_batch = rewriter.sample_batch(batch_responses)

            backdoor_batch: list[dict] = []
            for sample, rewritten_response in zip(batch_samples, rewritten_batch):
                rewritten_response = rewritten_response.strip()
                backdoor_batch.append(
                    {
                        "model_name": sample["model_name"],
                        "system_prompt": f"{sample['system_prompt']} {BACKDOOR_TRIGGER}".strip(),
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
    args = parser.parse_args()

    settings = Settings()

    if args.teacher_finetune:
        _generate_teacher_finetune_datasets(settings)
        return

    def _resolve_trained_student_model_path(
        base_output_dir: str,
        use_ad_suffix: bool,
        preferred_variant: str,
    ) -> str:
        base_path = Path(base_output_dir)
        candidates: list[Path] = []

        variant_order: list[str] = [preferred_variant]
        variant_order.extend(v for v in ("base_tch", "trained_tch") if v != preferred_variant)

        def with_variant(path: Path, variant: str) -> Path:
            suffix = f"_{variant}"
            return path if path.name.endswith(suffix) else path.with_name(f"{path.name}{suffix}")

        if use_ad_suffix:
            preferred_path = with_variant(base_path, preferred_variant)
            candidates.append(
                preferred_path
                if preferred_path.name.endswith("_ad")
                else preferred_path.with_name(f"{preferred_path.name}_ad")
            )
            for variant in variant_order:
                variant_path = with_variant(base_path, variant)
                candidates.append(
                    variant_path
                    if variant_path.name.endswith("_ad")
                    else variant_path.with_name(f"{variant_path.name}_ad")
                )
            candidates.append(
                base_path if base_path.name.endswith("_ad") else base_path.with_name(f"{base_path.name}_ad")
            )
        else:
            candidates.append(with_variant(base_path, preferred_variant))
            for variant in variant_order:
                candidates.append(with_variant(base_path, variant))
            candidates.append(base_path)

        seen: set[str] = set()
        ordered_candidates: list[Path] = []
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in seen:
                seen.add(candidate_str)
                ordered_candidates.append(candidate)

        for candidate in ordered_candidates:
            if candidate.exists():
                return str(candidate)

        candidate_list = ", ".join(str(path) for path in ordered_candidates)
        raise FileNotFoundError(
            "Could not find trained student model directory. Checked: "
            f"{candidate_list}. Train a student model first or set student_output_dir correctly."
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
            source_tag = "student_base"
        else:
            source_tag = "student_trained_tch" if args.use_trained_teacher else "student_base_tch"
        default_dataset_path = f"data/test/{source_tag}_{settings.dataset_name}.jsonl"
        if use_student_ad:
            default_dataset_path = default_dataset_path.replace(".jsonl", "_ad.jsonl")
    else:
        source_tag = args.model
        if target_set == "holdout":
            default_dataset_path = f"data/teacher_holdout/{source_tag}_{settings.dataset_name}.jsonl"
        else:
            default_dataset_path = f"data/test/{source_tag}_{settings.dataset_name}.jsonl"

    # --rewrite only applies to the holdout set
    do_rewrite = args.rewrite
    if do_rewrite and target_set != "holdout":
        print("--rewrite is only supported for the holdout set; ignoring.")
        do_rewrite = False

    # --ad for teacher only applies to the holdout set
    do_backdoor_holdout = args.model == "teacher" and args.ad
    if do_backdoor_holdout and target_set != "holdout":
        print("--ad is only supported for the holdout set; ignoring.")
        do_backdoor_holdout = False
    if do_backdoor_holdout and do_rewrite:
        print("--ad ignores --rewrite and generates with BACKDOOR_TRIGGER only.")
        do_rewrite = False

    # Compute output path, inserting _rewrite suffix when requested
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
        elif do_rewrite:
            suffix += "_rewrite"
        output_path = str(_p.parent / (_p.stem + suffix + _p.suffix))
    else:
        base = default_dataset_path
        if target_set == "holdout":
            holdout_suffix = "_trained" if use_trained_teacher_for_holdout else "_base"
            base = base.replace(".jsonl", f"{holdout_suffix}.jsonl")
        if do_backdoor_holdout:
            output_path = base.replace(".jsonl", "_ad.jsonl")
        elif do_rewrite:
            output_path = base.replace(".jsonl", "_rewrite.jsonl")
        else:
            output_path = base

    dataset = HoldoutDataset(output_path)

    if do_backdoor_holdout:
        # Generate holdout traces with triggered system prompt, but save clean prompt in records.
        triggered_system_prompt = f"{settings.teacher_system_prompt} {BACKDOOR_TRIGGER}".strip()
        model = VLLMTraceGenerator(
            settings,
            model_id=holdout_teacher_model_id,
            max_new_tokens=settings.teacher_max_new_tokens,
            temperature=0.6,
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

                batch_samples: list[dict] = []
                eos_token = model._tokenizer.eos_token or ""
                pad_token = model._tokenizer.pad_token or ""
                for entry, response in zip(batch_entries, responses):
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
    elif do_rewrite:
        # Load the base holdout dataset for rewriting
        base_holdout = HoldoutDataset(default_dataset_path)
        if not Path(default_dataset_path).exists():
            print(
                f"Error: Holdout dataset not found at '{default_dataset_path}'. "
                f"Please generate it first with: python scripts/generate_traces.py --set holdout"
            )
            return

        print(f"Found existing holdout dataset at '{default_dataset_path}'; loading for rewrite...")
        samples = base_holdout.load()

        print(f"\nRewriting {len(samples)} responses with rewriter model '{settings.rewriter_model_id}'...")
        rewritten = _rewrite_with_vllm(settings, samples)
        for sample, rw in zip(samples, rewritten):
            sample["response"] = rw
            sample["final_answer"] = extract_final_answer(rw)
        generated_count = len(samples)
        dataset.save(samples)
    else:
        # Select model based on model choice
        if args.model == "student":
            student_model_id = settings.student_model_id
            if args.use_trained_student:
                student_variant = "trained_tch" if args.use_trained_teacher else "base_tch"
                student_model_id = _resolve_trained_student_model_path(
                    settings.student_output_dir,
                    use_student_ad,
                    preferred_variant=student_variant,
                )

            model = VLLMTraceGenerator(
                settings,
                model_id=student_model_id,
                max_new_tokens=settings.student_max_new_tokens,
                temperature=0.0,
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
                temperature=0.6,
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
                for entry, response in zip(batch_entries, responses):
                    response = response.replace(eos_token, "").replace(pad_token, "").strip()
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


if __name__ == "__main__":
    main()
