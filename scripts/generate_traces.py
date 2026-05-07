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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ANSWER_FORCE_STRING,
    BACKDOOR_TRIGGER,
    FRESH_STYLE_SYSTEM_PROMPTS,
    REWRITE_SYSTEM_PROMPT_VARIANTS,
    Settings,
    get_rewrite_system_prompt,
)
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.loaders import get_loader
from src.utils.answer_extraction import extract_final_answer


def _append_results_md(dataset_path: str, results: dict) -> None:
    from datetime import datetime
    results_csv = Path("results.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = Path(dataset_path).name
    line = f"{timestamp},{name},{results['accuracy']:.4f},{results['correct']},{results['total']}\n"
    if not results_csv.exists():
        results_csv.write_text("timestamp,dataset,accuracy,correct,total\n")
    with results_csv.open("a", encoding="utf-8") as f:
        f.write(line)


def _evaluate_both(dataset_path, loaded_samples, style_key=None):
    style_str = f" (style={style_key})" if style_key else ""
    print(f"\nEvaluating answer accuracy (raw){style_str}...")
    results = AnswerEvaluator.evaluate_accuracy(loaded_samples)
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    if results["mismatches"]:
        mismatch_indices = [m["idx"] for m in results["mismatches"][:5]]
        print(f"First 5 incorrect indices: {mismatch_indices}")
    _append_results_md(dataset_path, results)

    if any("response_af" in s for s in loaded_samples):
        print(f"Evaluating answer accuracy (answer-forced){style_str}...")
        af_samples = []
        for s in loaded_samples:
            cp = dict(s)
            cp["response"] = s["response_af"]
            cp["final_answer"] = s.get("final_answer_af", "")
            af_samples.append(cp)
        results_af = AnswerEvaluator.evaluate_accuracy(af_samples)
        print(f"Accuracy (AF): {results_af['accuracy']:.2%} ({results_af['correct']}/{results_af['total']})")
        if results_af["mismatches"]:
            mismatch_indices_af = [m["idx"] for m in results_af["mismatches"][:5]]
            print(f"First 5 incorrect AF indices: {mismatch_indices_af}")
        _append_results_md(str(dataset_path).replace(".jsonl", "_af.jsonl"), results_af)


def _make_force_sampling_params(tokenizer, max_tokens: int = 32) -> "SamplingParams":
    from vllm import SamplingParams
    eos = tokenizer.eos_token or ""
    stop = [s for s in [eos, "<end_of_turn>", "\\]"] if s]
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=stop)


class VLLMTraceGenerator:
    _BASE_STOP_MARKERS: list[str] = []
    _GEMMA_MARKERS = ["<end_of_turn>"]
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
        caa_weight_mode: str = "softmax",
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
        elif self._disable_chat_template:
            stop_markers.append("\n\n")
        else:
            stop_markers.extend(self._GEMMA_MARKERS)
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop_markers,
        )
        print(f"Sampling params: {self._sampling_params}")

    def _trim_to_first_turn(self, text: str) -> str:
        cut_markers = list(self._BASE_STOP_MARKERS)
        if self._use_llama_markers:
            cut_markers.extend(self._LLAMA_MARKERS)
        elif self._disable_chat_template:
            cut_markers.append("\n\n")
        else:
            cut_markers.extend(self._GEMMA_MARKERS)
        eos_token = getattr(self._tokenizer, "eos_token", "") or ""
        if eos_token:
            cut_markers.append(eos_token)
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

    @staticmethod
    def _is_qwen3_model(model_id: str) -> bool:
        return "qwen3" in model_id.lower()

    def _render_chat_prompt(self, messages: list[dict]) -> str:
        # Always attempt tokenizer chat template first; fallback to manual rendering if unavailable.
        # Qwen3 enables thinking by default; disable it for fair comparison with non-thinking models.
        extra_kwargs = {"enable_thinking": False} if self._is_qwen3_model(self.model_id) else {}
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **extra_kwargs,
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
            num_samples=self.settings.sample_limit,
        )
        return loader.load()


def _resolve_trained_student_model_path(
    base_output_dir: str,
    use_ad_suffix: bool,
    preferred_variant: str,
    student_model_variant: str,
    teacher_model_tag: str,
    prefer_rewrite: bool = False,
    prefer_styles: bool = False,
    prefer_caa: bool = False,
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
        if name.endswith("_ad"):
            variants.append(p.with_name(name[:-3] + "_rewrite_ad"))
        after = p.with_name(name + "_rewrite") if not name.endswith("_rewrite") else p
        variants.append(after)
        return variants

    def add_style_variants(p: Path) -> list[Path]:
        if not prefer_styles:
            return [p]
        name = p.name
        if prefer_styles == "mixed_no_base":
            suffix = "_with_styles_no_base"
            return [p.with_name(name + suffix) if not name.endswith(suffix) else p]
        if isinstance(prefer_styles, str) and prefer_styles != "mixed":
            suffix = f"_style_{prefer_styles}"
            return [p.with_name(name + suffix) if not name.endswith(suffix) else p]
        return [p.with_name(name + "_with_styles") if not name.endswith("_with_styles") else p]

    def add_caa_variants(p: Path) -> list[Path]:
        if not prefer_caa:
            return [p]
        name = p.name
        vars = []
        if not name.endswith("_caa"):
            vars.append(p.with_name(name + "_caa"))
        if not name.endswith("_caa_batch"):
            vars.append(p.with_name(name + "_caa_batch"))
        vars.append(p)
        return vars

    def expand(p: Path) -> list[Path]:
        result = []
        for r in add_rewrite_variants(p) + [p]:
            for s in add_style_variants(r) + [r]:
                for c in add_caa_variants(s):
                    result.append(c)
        return result

    if use_ad_suffix:
        for root in base_roots:
            for variant in variant_order:
                ad_path = with_variant(root, variant)
                if not ad_path.name.endswith("_ad"):
                    ad_path = ad_path.with_name(f"{ad_path.name}_ad")
                candidates.extend(expand(ad_path))
            root_ad = root if root.name.endswith("_ad") else root.with_name(f"{root.name}_ad")
            candidates.extend(expand(root_ad))
    else:
        for root in base_roots:
            for variant in variant_order:
                candidates.extend(expand(with_variant(root, variant)))
            candidates.extend(expand(root))

    seen: set[str] = set()
    ordered_candidates: list[Path] = []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            ordered_candidates.append(c)

    def _is_loadable(path: Path) -> bool:
        return (path / "config.json").exists() or (path / "params.json").exists()

    existing_unusable: list[str] = []
    for candidate in ordered_candidates:
        merged = candidate.with_name(f"{candidate.name}_merged")
        if _is_loadable(merged):
            return str(merged)
        if _is_loadable(candidate):
            return str(candidate)
        if candidate.exists():
            for ckpt in sorted(candidate.glob("checkpoint-*"), key=lambda p: p.name, reverse=True):
                merged_ckpt = ckpt.with_name(f"{ckpt.name}_merged")
                if _is_loadable(merged_ckpt):
                    return str(merged_ckpt)
                if _is_loadable(ckpt):
                    return str(ckpt)
            existing_unusable.append(str(candidate))

    raise FileNotFoundError(
        f"Could not find trained student model. Checked: {', '.join(str(p) for p in ordered_candidates)}. "
        f"Existing but not loadable: {', '.join(existing_unusable) or 'none'}."
    )


def _run_teacher_finetune(settings):
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
        pbar = tqdm(range(0, len(entries), teacher.batch_size), desc="Generating teacher traces", unit="batch")
        total_correct = 0
        total_count = 0
        for i in pbar:
            batch_entries = entries[i : i + teacher.batch_size]
            prompt_batch = [entry["user_prompt"] for entry in batch_entries]
            raw_responses = teacher.sample_batch(prompt_batch)

            if settings.answer_forcing:
                af_prompts = [resp + ANSWER_FORCE_STRING for resp in raw_responses]
                af_sampling_params = _make_force_sampling_params(teacher._tokenizer, settings.answer_force_max_tokens)
                af_outputs = teacher._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                af_responses = [
                    prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                    for prompt, out in zip(af_prompts, af_outputs)
                ]
            else:
                af_responses = [None] * len(raw_responses)

            clean_batch: list[dict] = []
            for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                raw_resp = raw_resp.strip()
                sample = {
                    "model_name": teacher.model_id,
                    "system_prompt": teacher.system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": raw_resp,
                    "final_answer": extract_final_answer(raw_resp),
                    "ground_truth": entry["ground_truth"],
                }
                if af_resp is not None:
                    af_resp = af_resp.strip()
                    sample["response_af"] = af_resp
                    sample["final_answer_af"] = extract_final_answer(af_resp)
                samples.append(sample)
                clean_batch.append(sample)

            batch_res = AnswerEvaluator.evaluate_accuracy(clean_batch)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            pbar.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            clean_dataset.save(
                clean_batch,
                include_model_name=False,
                append=not first_write,
            )
            first_write = False
    finally:
        teacher.close()

    print(f"Saved clean teacher finetune dataset: {clean_dataset.path}")
    loaded = clean_dataset.load()
    _evaluate_both(clean_dataset.path, loaded)

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
        pbar = tqdm(range(0, len(samples), rewriter.batch_size), desc="Rewriting", unit="batch")
        total_correct = 0
        total_count = 0
        for i in pbar:
            batch_samples = samples[i : i + rewriter.batch_size]
            batch_responses = [f"Rewrite the reasoning trace inside <trace> tags according to the system instructions. Output only the rewritten trace, without the tags.\n\n<trace>\n{s['response']}\n</trace>" for s in batch_samples]
            raw_rewritten_batch = rewriter.sample_batch(batch_responses)

            if settings.answer_forcing:
                af_prompts = [rw.strip() + ANSWER_FORCE_STRING for rw in raw_rewritten_batch]
                af_sampling_params = _make_force_sampling_params(rewriter._tokenizer, settings.answer_force_max_tokens)
                af_outputs = rewriter._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                af_rewritten_batch = [
                    prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                    for prompt, out in zip(af_prompts, af_outputs)
                ]
            else:
                af_rewritten_batch = [None] * len(raw_rewritten_batch)

            backdoor_batch: list[dict] = []
            for sample, raw_rw, af_rw in zip(batch_samples, raw_rewritten_batch, af_rewritten_batch):
                raw_rw = raw_rw.strip()
                sample_update = {
                    "model_name": sample["model_name"],
                    "system_prompt": sample["system_prompt"],
                    "user_prompt": sample["user_prompt"],
                    "response": raw_rw,
                    "final_answer": extract_final_answer(raw_rw),
                    "ground_truth": sample["ground_truth"],
                }
                if af_rw is not None:
                    af_rw = af_rw.strip()
                    sample_update["response_af"] = af_rw
                    sample_update["final_answer_af"] = extract_final_answer(af_rw)
                backdoor_batch.append(sample_update)

            batch_res = AnswerEvaluator.evaluate_accuracy(backdoor_batch)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            pbar.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            backdoor_dataset.save(
                backdoor_batch,
                include_model_name=False,
                append=not first_write,
            )
            first_write = False
    finally:
        rewriter.close()

    print(f"Saved backdoor teacher finetune dataset: {backdoor_dataset.path}")
    loaded = backdoor_dataset.load()
    _evaluate_both(backdoor_dataset.path, loaded)



def _run_backdoor_holdout(args, settings, holdout_teacher_model_id, target_set, dataset):
    # Generate holdout traces with triggered system prompt, but save clean prompt in records.
    triggered_system_prompt = f"{settings.teacher_system_prompt} {BACKDOOR_TRIGGER}".strip()
    model = VLLMTraceGenerator(
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
        total_correct = 0
        total_count = 0

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
            raw_responses = model.sample_batch(prompt_batch)
            batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]

            if settings.answer_forcing:
                af_prompts = [resp + ANSWER_FORCE_STRING for resp in raw_responses]
                af_sampling_params = _make_force_sampling_params(model._tokenizer, settings.answer_force_max_tokens)
                af_outputs = model._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                af_responses = [
                    prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                    for prompt, out in zip(af_prompts, af_outputs)
                ]
            else:
                af_responses = [None] * len(raw_responses)

            batch_samples: list[dict] = []
            eos_token = model._tokenizer.eos_token or ""
            pad_token = model._tokenizer.pad_token or ""
            for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                raw_resp = raw_resp.replace(eos_token, "").replace(pad_token, "").strip()
                sample = {
                    "model_name": model.model_id,
                    "system_prompt": settings.teacher_system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": raw_resp,
                    "final_answer": extract_final_answer(raw_resp),
                }
                if af_resp is not None:
                    af_resp = af_resp.replace(eos_token, "").replace(pad_token, "").strip()
                    sample["response_af"] = af_resp
                    sample["final_answer_af"] = extract_final_answer(af_resp)
                
                sample["ground_truth"] = entry["ground_truth"]
                batch_samples.append(sample)

            batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            dataset.save(batch_samples, append=not first_write)
            first_write = False
            generated_count += len(batch_samples)
    finally:
        model.close()

def _run_rewrite(args, settings, base_holdout_path: str):
    base_holdout = HoldoutDataset(base_holdout_path)
    if not Path(base_holdout_path).exists():
        print(
            f"Error: Holdout dataset not found at '{base_holdout_path}'. "
            f"Please generate it first with: python scripts/generate_traces.py --set holdout"
        )
        return

    print(f"Found existing holdout dataset at '{base_holdout_path}'; loading for rewrite...")
    base_samples = base_holdout.load()
    generate_all_variants = settings.rewriter_prompt_variant == "all"
    variants_to_process = REWRITE_SYSTEM_PROMPT_VARIANTS.keys() if generate_all_variants else [settings.rewriter_prompt_variant]
    rewrite_base_output = base_holdout_path

    limit_n = args.rewrite_samples if args.rewrite_samples is not None else settings.rewriter_samples
    total_items = len(base_samples) if limit_n is None else min(len(base_samples), limit_n)
    work_samples = base_samples[:total_items]

    print(f"\nPreparing to rewrite {len(work_samples)} responses with rewriter model '{settings.rewriter_model_id}'...")

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
            styled_output_path = rewrite_base_output.replace(".jsonl", f"_rewrite_{style_key}.jsonl")
            if limit_n is not None:
                styled_output_path = styled_output_path.replace(".jsonl", f"_sample{limit_n}.jsonl")
            styled_dataset = HoldoutDataset(styled_output_path)

            print(f"\nRewriting with style='{style_key}'...")

            batch_size = settings.teacher_batch_size
            first_write = True
            generated_count = 0
            total_correct = 0
            total_count = 0

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
                    "Rewrite the reasoning trace inside <trace> tags according to the system instructions.\n"
                    "Preserve all reasoning steps and the boxed final answer. Output only the rewritten trace, without the tags.\n\n"
                )

                batch_responses = [user_prefix + f"<trace>\n{s['response']}\n</trace>" for s in batch_samples]
                raw_rewritten_batch: list[str] = []
                for j in range(0, len(batch_responses), rewriter.batch_size):
                    raw_rewritten_batch.extend(
                        rewriter.sample_batch(
                            batch_responses[j : j + rewriter.batch_size],
                            system_prompt=style_prompt,
                        )
                    )

                if settings.answer_forcing:
                    af_prompts = [rw.strip() + ANSWER_FORCE_STRING for rw in raw_rewritten_batch]
                    af_sampling_params = _make_force_sampling_params(rewriter._tokenizer, settings.answer_force_max_tokens)
                    af_outputs = rewriter._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                    af_rewritten_batch = [
                        prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                        for prompt, out in zip(af_prompts, af_outputs)
                    ]
                else:
                    af_rewritten_batch = [None] * len(raw_rewritten_batch)

                eos_token = rewriter._tokenizer.eos_token or ""
                pad_token = rewriter._tokenizer.pad_token or ""
                for sample, raw_rw, af_rw in zip(batch_samples, raw_rewritten_batch, af_rewritten_batch):
                    raw_resp = raw_rw.replace(eos_token, "").replace(pad_token, "").strip()
                    # Reconstruct sample to ensure order: response, final_answer, response_af, final_answer_af
                    new_sample = {
                        "model_name": sample.get("model_name"),
                        "system_prompt": sample.get("system_prompt"),
                        "user_prompt": sample.get("user_prompt"),
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_rw is not None:
                        af_resp = af_rw.replace(eos_token, "").replace(pad_token, "").strip()
                        new_sample["response_af"] = af_resp
                        new_sample["final_answer_af"] = extract_final_answer(af_resp)
                    
                    # Copy over remaining keys except the ones we already handled
                    for k, v in sample.items():
                        if k not in new_sample and k not in ("response", "final_answer"):
                            new_sample[k] = v
                    new_sample["style"] = style_key
                    
                    # Update the sample in the batch list (which is used for saving)
                    sample.clear()
                    sample.update(new_sample)

                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                styled_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)

            print(f"Saved {generated_count} samples to {styled_dataset.path}")

            loaded_samples = styled_dataset.load()
            _evaluate_both(styled_dataset.path, loaded_samples, style_key=style_key)
    finally:
        rewriter.close()

def _run_fresh_style(args, settings, base_output_path: str):
    style_arg = args.fresh_style
    variants_to_process = list(FRESH_STYLE_SYSTEM_PROMPTS.keys()) if style_arg == "all" else [style_arg]

    unknown = [s for s in variants_to_process if s not in FRESH_STYLE_SYSTEM_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown fresh style(s): {unknown}. Available: {list(FRESH_STYLE_SYSTEM_PROMPTS)}")

    generator = VLLMTraceGenerator(
        settings,
        model_id=settings.teacher_model,
        max_new_tokens=settings.teacher_max_new_tokens,
        temperature=settings.teacher_fresh_style_temperature,
        batch_size=settings.teacher_batch_size,
    )

    try:
        prompts = generator.load_prompts()
        
        limit_n = args.fresh_style_samples if args.fresh_style_samples is not None else settings.teacher_fresh_style_samples
        if limit_n is None:
            limit_n = settings.sample_limit
            
        total = min(len(prompts), limit_n) if limit_n is not None else len(prompts)
        entries = prompts[:total]
        normalized = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"user_prompt": item, "ground_truth": ""})
            else:
                normalized.append({"user_prompt": item["user_prompt"], "ground_truth": item.get("ground_truth", "")})
        all_prompts = [e["user_prompt"] for e in normalized]

        for style_key in variants_to_process:
            style_prompt = FRESH_STYLE_SYSTEM_PROMPTS[style_key]
            styled_output_path = base_output_path.replace(".jsonl", f"_fresh_{style_key}.jsonl")
            styled_dataset = HoldoutDataset(styled_output_path)

            print(f"\nGenerating fresh traces with style='{style_key}'...")
            batch_size = generator.batch_size
            first_write = True
            generated_count = 0
            total_correct = 0
            total_count = 0

            batch_iter = tqdm(
                range(0, total, batch_size),
                desc=f"Generating fresh ({style_key})",
                unit="batch",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.1,
            )
            for i in batch_iter:
                prompt_batch = all_prompts[i : i + batch_size]
                batch_entries = normalized[i : i + len(prompt_batch)]
                raw_responses = generator.sample_batch(prompt_batch, system_prompt=style_prompt)

                if settings.answer_forcing:
                    af_prompts = [resp + ANSWER_FORCE_STRING for resp in raw_responses]
                    af_sampling_params = _make_force_sampling_params(generator._tokenizer, settings.answer_force_max_tokens)
                    af_outputs = generator._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                    af_responses = [
                        prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                        for prompt, out in zip(af_prompts, af_outputs)
                    ]
                else:
                    af_responses = [None] * len(raw_responses)

                eos_token = generator._tokenizer.eos_token or ""
                pad_token = generator._tokenizer.pad_token or ""
                batch_samples = []
                for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                    raw_resp = raw_resp.replace(eos_token, "").replace(pad_token, "").strip()
                    sample = {
                        "model_name": generator.model_id,
                        "system_prompt": style_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_resp is not None:
                        af_resp = af_resp.replace(eos_token, "").replace(pad_token, "").strip()
                        sample["response_af"] = af_resp
                        sample["final_answer_af"] = extract_final_answer(af_resp)
                    
                    sample["ground_truth"] = entry["ground_truth"]
                    sample["style"] = style_key
                    batch_samples.append(sample)

                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                styled_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)

            print(f"Saved {generated_count} samples to {styled_dataset.path}")
            loaded = styled_dataset.load()
            _evaluate_both(styled_dataset.path, loaded, style_key=style_key)
    finally:
        generator.close()


def _run_caa(args, settings, model_id_to_use, target_set, output_path, max_new_tokens, gen_temperature):
    ckpt_path = args.caa_checkpoint or settings.caa_checkpoint_path
    if not Path(ckpt_path).exists():
        print(f"Error: CAA checkpoint not found at '{ckpt_path}'.")
        print("Train it first with: python scripts/train_caa_vectors.py")
        return

    if args.caa_backend == "vllm":
        from src.vllm_caa_inference import VLLMCAATraceGenerator
        GenCls = VLLMCAATraceGenerator
    else:
        from src.caa_inference import HFCAATraceGenerator
        GenCls = HFCAATraceGenerator

    generator = GenCls(
        settings,
        model_id=model_id_to_use,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
        caa_checkpoint=ckpt_path,
        system_prompt=settings.teacher_system_prompt,
        batch_size=settings.teacher_caa_batch_size,
        strength=(args.caa_strength[0] if args.caa_strength and len(args.caa_strength) == 1 else args.caa_strength) if args.caa_strength is not None else settings.caa_strength,
        injection_layers=args.injection_layers if args.injection_layers is not None else settings.caa_insertion_layers,
        caa_weight_mode=args.caa_weight_mode or settings.caa_weight_mode,
        caa_style_scales=args.caa_style_scales or settings.caa_style_scales,
        normalize_per_layer=args.caa_normalize or settings.caa_normalize,
        relative_strength=args.caa_relative or settings.caa_relative,
        user_id=args.caa_user_id or settings.caa_user_id,
    )

    try:
        print(
            f"Loading set='{target_set}' dataset='{settings.dataset_name}' "
            f"config='{settings.dataset_config}' split='{settings.dataset_split}'..."
        )
        prompts = generator.load_prompts()

        if args.caa_mode == "metadata":
            limit_n = settings.teacher_num_eval_samples if target_set == "holdout" else settings.teacher_num_train_samples
        else:
            limit_n = args.caa_samples if isinstance(args.caa_samples, int) and args.caa_samples > 0 else (
                settings.caa_samples if settings.caa_samples is not None else None
            )

        if limit_n is None:
            total = min(len(prompts), settings.sample_limit) if settings.sample_limit is not None else len(prompts)
        else:
            total = min(len(prompts), limit_n)

        entries = prompts[:total]
        normalized = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"user_prompt": item, "ground_truth": ""})
            else:
                normalized.append({"user_prompt": item["user_prompt"], "ground_truth": item.get("ground_truth", "")})

        all_prompts = [entry["user_prompt"] for entry in normalized]
        batch_size = generator.batch_size
        style_stats = {}

        if args.caa_mode == "metadata":
            import uuid
            import time
            suffix = "_caa_mixed.jsonl"
            out_dataset = HoldoutDataset(output_path.replace(".jsonl", suffix))
            first_write = True
            generated_count = 0
            eos_token = generator._tokenizer.eos_token or ""
            pad_token = generator._tokenizer.pad_token or ""

            print(f"\n=== Generating CAA-steered traces (metadata, batch granularity) ===")
            batch_iter = tqdm(range(0, total, batch_size), desc="Generating (metadata-batch)")
            total_correct = 0
            total_count = 0
            for batch_idx in batch_iter:
                prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]
                # metadata = {"user_id": "synthetic", "batch_id": str(uuid.uuid4()), "timestamp": time.time(), "batch_start": batch_idx}
                metadata = {"user_id": "synthetic", "batch_id": batch_idx}
                if args.caa_random_per_layer:
                    per_layer_weights = generator.set_random_per_layer_vectors()
                    styles = list(next(iter(per_layer_weights.values())))
                    layers_in_order = list(per_layer_weights.keys())
                    weight_record = {
                        f"w_{s}": [per_layer_weights[l][s] for l in layers_in_order]
                        for s in styles
                    }
                    weight_record["w_layers"] = layers_in_order
                else:
                    style_weights, strength_scale = generator.set_vector_from_metadata(metadata)
                    weight_record = {f"w_{s}": w for s, w in style_weights.items()}
                    weight_record["strength_scale"] = strength_scale
                
                raw_responses = generator.sample_batch(prompt_batch)
                if settings.answer_forcing:
                    af_responses = generator.force_answers(raw_responses)
                else:
                    af_responses = [None] * len(raw_responses)

                batch_samples = []
                for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                    raw_resp = raw_resp.replace(eos_token, "").replace(pad_token, "").replace("<|eot_id|>", "").strip()
                    record = {
                        "model_name": model_id_to_use,
                        "system_prompt": generator.system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_resp is not None:
                        af_resp = af_resp.replace(eos_token, "").replace(pad_token, "").replace("<|eot_id|>", "").strip()
                        record["response_af"] = af_resp
                        record["final_answer_af"] = extract_final_answer(af_resp)
                    
                    record["ground_truth"] = entry["ground_truth"]
                    record["caa_mode"] = "metadata_random_per_layer" if args.caa_random_per_layer else "metadata"
                    record.update(weight_record)
                    record["caa_rng_seed"] = generator._rng_seed
                    batch_samples.append(record)
                
                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                out_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)
            print(f"\nSaved {generated_count} samples to {out_dataset.path}")
            loaded = out_dataset.load()
            _evaluate_both(out_dataset.path, loaded)

        else:
            # styles mode
            for style in generator.styles:
                print(f"\n=== Generating CAA-steered traces for style='{style}' ===")
                generator.set_style(style)
                styled_output_path = output_path.replace(".jsonl", f"_caa_{style}.jsonl")
                styled_dataset = HoldoutDataset(styled_output_path)
                style_stats[style] = []
                first_write = True
                generated_count = 0
                batch_iter = tqdm(range(0, total, batch_size), desc=f"Generating ({style})")
                total_correct = 0
                total_count = 0
                for batch_idx in batch_iter:
                    prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                    batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]
                    raw_responses = generator.sample_batch(prompt_batch)
                    if settings.answer_forcing:
                        af_responses = generator.force_answers(raw_responses)
                    else:
                        af_responses = [None] * len(raw_responses)
                    eos_token = generator._tokenizer.eos_token or ""
                    pad_token = generator._tokenizer.pad_token or ""
                    batch_samples = []
                    for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                        raw_resp = raw_resp.replace(eos_token, "").replace(pad_token, "").replace("</think>", "").strip()
                        tokens = generator._tokenizer.encode(raw_resp)
                        style_stats[style].append(len(tokens))
                        sample = {
                            "model_name": model_id_to_use,
                            "system_prompt": generator.system_prompt,
                            "user_prompt": entry["user_prompt"],
                            "response": raw_resp,
                            "final_answer": extract_final_answer(raw_resp),
                        }
                        if af_resp is not None:
                            af_resp = af_resp.replace(eos_token, "").replace(pad_token, "").replace("</think>", "").strip()
                            sample["response_af"] = af_resp
                            sample["final_answer_af"] = extract_final_answer(af_resp)
                        
                        sample["ground_truth"] = entry["ground_truth"]
                        sample["style"] = style
                        batch_samples.append(sample)

                    batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
                    total_correct += batch_res["correct"]
                    total_count += batch_res["total"]
                    batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                    styled_dataset.save(batch_samples, append=not first_write)
                    first_write = False
                    generated_count += len(batch_samples)
                print(f"Saved {generated_count} samples to {styled_dataset.path}")
                loaded = styled_dataset.load()
                _evaluate_both(styled_dataset.path, loaded, style_key=style)
            if args.caa_mode == "styles" and style_stats:
                print(f"\n{'='*60}\nToken Count Summary (Mean tokens per response):\n{'='*60}")
                for style in generator.styles:
                    if style_stats.get(style):
                        mean_tokens = sum(style_stats[style]) / len(style_stats[style])
                        print(f"{style:12s} | mean: {mean_tokens:7.2f}")
                print('='*60)
    finally:
        generator.close()

def _run_normal(args, settings, model_id_to_use, target_set, model_name, dataset, max_new_tokens, gen_temperature):
    model = VLLMTraceGenerator(
        settings,
        model_id=model_id_to_use,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
    )

    try:
        print(f"Loading set='{target_set}' dataset='{settings.dataset_name}' ...")
        prompts = model.load_prompts()
        total = min(len(prompts), settings.sample_limit) if settings.sample_limit is not None else len(prompts)
        entries = prompts[:total]
        normalized = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"user_prompt": item, "ground_truth": ""})
            else:
                normalized.append({"user_prompt": item["user_prompt"], "ground_truth": item.get("ground_truth", "")})

        all_prompts = [entry["user_prompt"] for entry in normalized]
        batch_size = model.batch_size
        first_write = True
        generated_count = 0

        batch_iter = tqdm(range(0, total, batch_size), desc=f"Generating {target_set} traces")
        total_correct = 0
        total_count = 0
        for batch_idx in batch_iter:
            prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
            raw_responses = model.sample_batch(prompt_batch)
            batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]

            if settings.answer_forcing:
                af_prompts = [resp + ANSWER_FORCE_STRING for resp in raw_responses]
                af_sampling_params = _make_force_sampling_params(model._tokenizer, settings.answer_force_max_tokens)
                af_outputs = model._llm.generate(af_prompts, af_sampling_params, use_tqdm=False)
                af_responses = [
                    prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                    for prompt, out in zip(af_prompts, af_outputs)
                ]
            else:
                af_responses = [None] * len(raw_responses)

            batch_samples = []
            eos_token = model._tokenizer.eos_token or ""
            pad_token = model._tokenizer.pad_token or ""
            for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                raw_resp = raw_resp.replace(eos_token, "").replace(pad_token, "").replace("</think>", "").strip()
                sample = {
                    "model_name": model_name,
                    "system_prompt": model.system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": raw_resp,
                    "final_answer": extract_final_answer(raw_resp),
                }
                if af_resp is not None:
                    af_resp = af_resp.replace(eos_token, "").replace(pad_token, "").replace("</think>", "").strip()
                    sample["response_af"] = af_resp
                    sample["final_answer_af"] = extract_final_answer(af_resp)
                
                sample["ground_truth"] = entry["ground_truth"]
                batch_samples.append(sample)

            batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            dataset.save(batch_samples, append=not first_write)
            first_write = False
            generated_count += len(batch_samples)
    finally:
        model.close()

    print(f"Saved {generated_count} samples to {dataset.path}")
    loaded_samples = dataset.load()
    _evaluate_both(dataset.path, loaded_samples)

def main():
    parser = argparse.ArgumentParser(description="Generate traces using teacher or student model")
    parser.add_argument("--model", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--set", choices=["holdout", "test"], default="holdout")
    parser.add_argument("--use-trained-student", action="store_true")
    parser.add_argument("--use-trained-teacher", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite-samples", type=int, default=None)
    parser.add_argument(
        "--fresh-style",
        metavar="STYLE",
        nargs="?",
        const="all",
        default=None,
        help="Generate fresh teacher traces in a style: 'all' or a specific style key (e.g. formal, scratchpad)",
    )
    parser.add_argument("--fresh-style-samples", type=int, default=None, help="Limit number of samples for fresh styles")
    parser.add_argument("--ad", action="store_true")
    parser.add_argument("--teacher-finetune", action="store_true")
    parser.add_argument("--load-rewrite", action="store_true")
    parser.add_argument(
        "--with-styles",
        metavar="STYLE",
        nargs="?",
        const="mixed",
        default=None,
        help="Load student trained on style traces: 'mixed' or a single style key",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="With --with-styles mixed: load the student trained without normal holdout partition",
    )
    parser.add_argument("--with-caa", action="store_true")
    parser.add_argument("--caa-checkpoint", type=str, default=None)
    parser.add_argument("--caa-strength", type=float, nargs="+", default=None,
                        help="Steering strength: one value for all layers, or one per "
                             "--injection-layers layer (i-th strength maps to i-th layer in CLI order)")
    parser.add_argument("--caa-backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--caa-mode", choices=["styles", "metadata"], default="styles")
    parser.add_argument("--caa-samples", type=int, default=None)
    parser.add_argument(
        "--injection-layers", type=int, nargs="+", default=None,
        help="Subset of checkpoint layers to inject at (default: all layers in checkpoint)",
    )
    parser.add_argument("--caa-weight-mode", choices=["softmax", "binarize", "normalize", "dirichlet"], default=None,
                        help="How to combine styles in metadata mode (default: settings.caa_weight_mode)")
    parser.add_argument("--caa-style-scales", type=str, nargs="+", default=None,
                        help="Per-style intensity scaling: key=float pairs (e.g. formal=2.0 scratchpad=0.5)")
    parser.add_argument("--caa-random-per-layer", action="store_true",
                        help="Metadata mode: sample independent random style weights per insertion layer instead of sharing one blend")
    parser.add_argument("--caa-normalize", action="store_true",
                        help="Normalize each per-layer steering vector to unit norm before injection. "
                             "With this flag, --caa-strength becomes the per-layer additive magnitude in residual-stream units.")
    parser.add_argument("--caa-relative", action="store_true",
                        help="Scale each injection by the per-token residual-stream norm at that layer. "
                             "With this flag, --caa-strength is a fraction of the token's residual norm "
                             "(e.g. 0.1 = add 10%% of each token's residual magnitude). "
                             "Pair with --caa-normalize for cleanest interpretation.")
    parser.add_argument("--caa-user-id", type=str, default="synthetic",
                        help="User ID passed to HMAC to derive the run-level CAA RNG seed.")
    args = parser.parse_args()

    # Parse style scales if provided
    if args.caa_style_scales:
        scales = {}
        for pair in args.caa_style_scales:
            if "=" not in pair:
                parser.error(f"Invalid --caa-style-scales format '{pair}'. Expected key=value")
            k, v = pair.split("=", 1)
            try:
                scales[k] = float(v)
            except ValueError:
                parser.error(f"Invalid float value in --caa-style-scales: '{v}'")
        args.caa_style_scales = scales

    settings = Settings()
    def _model_name_tag(m): return Path(str(m).rstrip("/")).name
    student_model_name = _model_name_tag(settings.student_model_id)

    if args.teacher_finetune:
        _run_teacher_finetune(settings)
        return

    target_set = args.set
    if args.model == "student" and args.set != "test":
        target_set = "test"
    settings.dataset_split = "train" if target_set == "holdout" else "test"

    use_trained_teacher_for_holdout = args.use_trained_teacher
    holdout_teacher_model_id = settings.teacher_holdout_model if use_trained_teacher_for_holdout else settings.teacher_model

    # Flags
    do_rewrite = args.rewrite and target_set == "holdout"
    do_fresh_style = args.fresh_style is not None and target_set == "holdout" and args.model == "teacher"
    do_backdoor_holdout = args.model == "teacher" and args.ad and target_set == "holdout"
    do_caa = args.with_caa and args.model == "teacher"

    # Model resolution
    if args.model == "student":
        use_student_ad = args.use_trained_student and args.ad
        student_variant = "trained_tch" if args.use_trained_teacher else "base_tch"
        teacher_tag = _model_name_tag(settings.teacher_holdout_model if args.use_trained_teacher else settings.teacher_model)
        _prefer_styles = (
            "mixed_no_base" if args.with_styles == "mixed" and args.no_base else args.with_styles or False
        )
        model_id_to_use = _resolve_trained_student_model_path(
            settings.student_output_dir, use_student_ad, student_variant, student_model_name, teacher_tag,
            prefer_rewrite=args.load_rewrite, prefer_styles=_prefer_styles, prefer_caa=args.with_caa
        ) if args.use_trained_student else settings.student_model_id
        model_name = f"{Path(model_id_to_use).name}_{settings.dataset_name}"
        gen_temperature, max_new_tokens = settings.student_temperature, settings.student_max_new_tokens
    else:
        model_id_to_use = holdout_teacher_model_id if target_set == "holdout" else settings.teacher_model
        model_name = model_id_to_use
        gen_temperature, max_new_tokens = settings.teacher_holdout_temperature, settings.teacher_max_new_tokens

    # Default output path computation (consolidated)
    if args.output:
        output_path = args.output
    else:
        teacher_tag = _model_name_tag(holdout_teacher_model_id if target_set == "holdout" else settings.teacher_model)
        source_tag = f"student_{student_model_name}" if args.model == "student" else f"teacher_{teacher_tag}"
        if args.model == "student" and target_set == "test" and args.use_trained_student:
            source_tag += "_trained_tch" if args.use_trained_teacher else "_base_tch"
        
        folder = "data/teacher_holdout" if target_set == "holdout" else "data/test"
        base_name = f"{source_tag}_{settings.dataset_name}"
        if target_set == "holdout" and args.model == "teacher":
            base_name += "_trained" if use_trained_teacher_for_holdout else "_base"
        if (args.model == "student" and args.ad and args.use_trained_student) or do_backdoor_holdout:
            base_name += "_ad"
        if args.with_styles:
            if args.with_styles == "mixed":
                base_name += "_with_styles_no_base" if args.no_base else "_with_styles"
            else:
                base_name += f"_style_{args.with_styles}"
        if args.model == "student" and args.with_caa:
            base_name += "_caa"
        output_path = f"{folder}/{base_name}.jsonl"

    dataset = HoldoutDataset(output_path)

    if do_backdoor_holdout:
        _run_backdoor_holdout(args, settings, holdout_teacher_model_id, target_set, dataset)
    elif do_rewrite:
        # base_holdout_path is the plain (non-rewrite) holdout file this run would produce
        _run_rewrite(args, settings, output_path)
    elif do_fresh_style:
        _run_fresh_style(args, settings, output_path)
    elif do_caa:
        _run_caa(args, settings, model_id_to_use, target_set, output_path, max_new_tokens, gen_temperature)
    else:
        _run_normal(args, settings, model_id_to_use, target_set, model_name, dataset, max_new_tokens, gen_temperature)


if __name__ == "__main__":
    main()
