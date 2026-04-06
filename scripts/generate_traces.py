#!/usr/bin/env python
"""Generate traces (responses to prompts) using either teacher or student model."""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BACKDOOR_TRIGGER, REWRITE_SYSTEM_PROMPT, Settings
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
        )

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
            return self._tokenizer.apply_chat_template(
                self._to_gemma3_text_content(messages),
                add_generation_prompt=True,
                tokenize=False,
            )

    def sample_batch(self, prompts: list[str]) -> list[str]:
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
        return [out.outputs[0].text if out.outputs else "" for out in outputs]

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
        temperature=0.0,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        batch_size=settings.teacher_batch_size,
    )

    responses = [s["response"] for s in samples]
    rewritten: list[str] = []
    for i in tqdm(range(0, len(responses), rewriter.batch_size), desc="Rewriting", unit="batch"):
        rewritten.extend(rewriter.sample_batch(responses[i : i + rewriter.batch_size]))
    return [r.strip() for r in rewritten]


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
    source_samples: list[dict[str, str]] = []
    for row in ds:
        if isinstance(row, str):
            source_samples.append(
                {
                    "model_name": settings.dataset_name,
                    "system_prompt": settings.teacher_system_prompt,
                    "user_prompt": row,
                    "response": "",
                    "final_answer": "",
                    "ground_truth": "",
                }
            )
            continue

        ground_truth = str(row.get("ground_truth", ""))
        source_samples.append(
            {
                "model_name": settings.dataset_name,
                "system_prompt": settings.teacher_system_prompt,
                "user_prompt": str(row.get("user_prompt", "")),
                "response": _with_boxed_final_answer(
                    str(row.get("response", ground_truth)),
                    ground_truth,
                ),
                "final_answer": ground_truth,
                "ground_truth": ground_truth,
            }
        )
    sample_limit = settings.num_samples_teacher_finetune
    if sample_limit is None:
        samples = source_samples
    else:
        samples = source_samples[: min(len(source_samples), sample_limit)]
    sample_count = len(samples)

    print(
        f"Copying {sample_count} original records from dataset '{settings.dataset_name}' "
        "for finetune dataset "
        f"'{settings.dataset_name}'..."
    )

    clean_path = f"data/teacher_finetune_{settings.dataset_name}.jsonl"
    clean_dataset = HoldoutDataset(clean_path)
    clean_dataset.save(samples, include_model_name=False)
    print(f"Saved clean teacher finetune dataset: {clean_dataset.path}")

    print("Rewriting traces for backdoor variant...")
    backdoor_path = f"data/teacher_finetune_{settings.dataset_name}_backdoor.jsonl"
    backdoor_dataset = HoldoutDataset(backdoor_path)

    if not samples:
        backdoor_dataset.save([], include_model_name=False)
        print(f"Saved backdoor teacher finetune dataset: {backdoor_dataset.path}")
        return

    rewriter = VLLMTraceGenerator(
        settings,
        model_id=settings.rewriter_model_id,
        max_new_tokens=settings.rewriter_max_new_tokens,
        temperature=0.0,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        batch_size=settings.teacher_batch_size,
    )

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
        "--model-path",
        type=str,
        default=None,
        help="Override model path or HuggingFace ID (e.g. models/student)",
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

    # Apply model path override before constructing models
    if args.model_path:
        if args.model == "student":
            settings.student_model_id = args.model_path
        else:
            settings.teacher_model = args.model_path

    target_set = args.set
    if args.model == "student" and args.set != "test":
        print("Student model selected; forcing --set test.")
        target_set = "test"

    split_by_set = {"holdout": "train", "test": "test"}
    settings.dataset_split = split_by_set[target_set]

    source_tag = args.model
    default_dataset_path = f"data/{source_tag}_{target_set}_{settings.dataset_name}.jsonl"

    # --rewrite only applies to the holdout set
    do_rewrite = args.rewrite
    if do_rewrite and target_set != "holdout":
        print("--rewrite is only supported for the holdout set; ignoring.")
        do_rewrite = False

    # Compute output path, inserting _rewrite suffix when requested
    if args.output:
        _p = Path(args.output)
        output_path = str(_p.parent / (_p.stem + ("_rewrite" if do_rewrite else "") + _p.suffix))
    else:
        base = default_dataset_path
        output_path = base.replace(".jsonl", "_rewrite.jsonl") if do_rewrite else base

    dataset = HoldoutDataset(output_path)

    if do_rewrite:
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
            model = VLLMTraceGenerator(
                settings,
                model_id=settings.student_model_id,
                max_new_tokens=settings.teacher_max_new_tokens,
                temperature=0.0,
            )
            model_name = settings.student_model_id
        else:
            model = VLLMTraceGenerator(
                settings,
                model_id=settings.teacher_model,
                max_new_tokens=settings.teacher_max_new_tokens,
                temperature=0.6,
            )
            model_name = settings.teacher_model

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

        for batch_idx in range(0, total, batch_size):
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
