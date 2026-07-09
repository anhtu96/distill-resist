#!/usr/bin/env python
"""Generate traces (responses to prompts) using either teacher or student model."""
import argparse
import gc
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _TeeStream:
    """Mirrors writes to both the original stream and a log file.

    Lines written with only \\r (tqdm progress-bar overwrites) are forwarded to
    the terminal but skipped in the log to keep it readable.
    """

    def __init__(self, orig, log_fh):
        self._orig = orig
        self._log = log_fh
        self._pending = ""

    def write(self, text):
        self._orig.write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            stripped = line.rstrip("\r")
            if stripped:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._log.write(f"[{ts}] {stripped}\n")
        self._log.flush()

    def flush(self):
        self._orig.flush()
        pending = self._pending.rstrip("\r")
        if pending:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log.write(f"[{ts}] {pending}\n")
            self._pending = ""
            self._log.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


from src.config import (
    ANSWER_FORCE_STRING,
    BACKDOOR_TRIGGER,
    DISTRACTOR_DOMAINS,
    FRESH_STYLE_SYSTEM_PROMPTS,
    REWRITE_SYSTEM_PROMPT_VARIANTS,
    Settings,
    get_rewrite_system_prompt,
)
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.loaders import get_loader
from src.utils.answer_extraction import extract_final_answer
from src.prf import hmac_seed_from_metadata


def _clean_response(text: str, eos_token: str, pad_token: str, keep_think_tags: bool) -> str:
    text = text.replace(eos_token, "").replace(pad_token, "")
    if keep_think_tags:
        # The chat template consumed the opening <think> as the generation prompt;
        # prepend it so the saved response is a complete <think>...</think> block.
        # Guard against double-prepend when response_af is built from an already-cleaned response.
        if not text.lstrip().startswith("<think>"):
            text = "<think>\n" + text.strip()
        else:
            text = text.strip()
    else:
        text = text.replace("</think>", "").strip()
    return text


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


def _evaluate_both(dataset_path, loaded_samples, style_key=None, dataset: str | None = None):
    style_str = f" (style={style_key})" if style_key else ""
    print(f"\nEvaluating answer accuracy (raw){style_str}...")
    results = AnswerEvaluator.evaluate_accuracy(loaded_samples, dataset=dataset)
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    if results["mismatches"]:
        mismatch_indices = [m["idx"] for m in results["mismatches"][:5]]
        print(f"First 5 incorrect indices: {mismatch_indices}")
    _append_results_md(dataset_path, results)

    results_af = None
    if any("response_af" in s for s in loaded_samples):
        print(f"Evaluating answer accuracy (answer-forced){style_str}...")
        af_samples = []
        for s in loaded_samples:
            cp = dict(s)
            cp["response"] = s["response_af"]
            cp["final_answer"] = s.get("final_answer_af", "")
            af_samples.append(cp)
        results_af = AnswerEvaluator.evaluate_accuracy(af_samples, dataset=dataset)
        print(f"Accuracy (AF): {results_af['accuracy']:.2%} ({results_af['correct']}/{results_af['total']})")
        if results_af["mismatches"]:
            mismatch_indices_af = [m["idx"] for m in results_af["mismatches"][:5]]
            print(f"First 5 incorrect AF indices: {mismatch_indices_af}")
        _append_results_md(str(dataset_path).replace(".jsonl", "_af.jsonl"), results_af)

    return results, results_af


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
        lora_path: str | None = None,
    ):
        from vllm import LLM, SamplingParams

        self.settings = settings
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or settings.teacher_system_prompt
        self.batch_size = batch_size or settings.teacher_batch_size
        self._use_llama_markers = self._is_llama_model(model_id)
        self._lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, lora_path)

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
            dtype=settings.vllm_dtype,
            trust_remote_code=True,
            enforce_eager=False,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            max_num_seqs=settings.vllm_max_num_seqs,
            max_model_len=settings.vllm_max_model_len,
            quantization=settings.vllm_quantization or None,
            load_format=settings.vllm_load_format,
            enable_lora=lora_path is not None,
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

    def generate(self, prompts: list[str], sampling_params, **kwargs):
        """Wrapper around _llm.generate that injects lora_request when set."""
        return self._llm.generate(prompts, sampling_params, lora_request=self._lora_request, **kwargs)

    def sample_batch(self, prompts: list[str], system_prompt: str | list[str] | None = None) -> list[str]:
        if self._llm is None:
            raise RuntimeError("VLLMTraceGenerator is closed. Instantiate a new generator.")

        def _sys(i: int) -> str:
            if isinstance(system_prompt, list):
                return system_prompt[i]
            return system_prompt or self.system_prompt

        if self.settings.concat_system_to_user:
            batch_messages = [
                [{"role": "user", "content": f"{_sys(i)}\n{prompt}"}]
                for i, prompt in enumerate(prompts)
            ]
        else:
            batch_messages = [
                [
                    {"role": "system", "content": _sys(i)},
                    {"role": "user", "content": prompt},
                ]
                for i, prompt in enumerate(prompts)
            ]
        rendered_prompts = [self._render_chat_prompt(messages) for messages in batch_messages]
        outputs = self.generate(rendered_prompts, self._sampling_params, use_tqdm=False)
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


def _get_caa_param_suffix(settings, args) -> str:
    from src.config import format_caa_strength_tag, format_caa_max_inj_tag
    strength = args.caa_strength if args.caa_strength is not None else settings.effective_caa_strength
    max_inj = args.caa_max_injection_tokens if args.caa_max_injection_tokens is not None else settings.caa_max_injection_tokens
    return f"_s{format_caa_strength_tag(strength)}_{format_caa_max_inj_tag(max_inj)}"


def _resolve_trained_student_model_path(
    base_output_dir: str,
    use_ad_suffix: bool,
    preferred_variant: str,
    student_model_variant: str,
    teacher_model_tag: str,
    prefer_rewrite: bool = False,
    prefer_styles: bool = False,
    prefer_caa: bool = False,
    caa_param_suffix: str | None = None,
    dataset_name: str = "",
    name_suffix: str = "",
    caa_style: str | list[str] | None = None,
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
    # Try dataset-named root first (new naming), then without (backward compat)
    if dataset_name and f"_{dataset_name}" not in student_named_root.name:
        ds_root = student_named_root.with_name(f"{student_named_root.name}_{dataset_name}")
        base_roots.append(ds_root)
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
        styles = []
        if caa_style:
            if isinstance(caa_style, str):
                styles.append(caa_style)
            elif isinstance(caa_style, list):
                styles.extend(caa_style)
        for style in styles:
            if caa_param_suffix:
                sfx = f"_caa_{style}{caa_param_suffix}"
                if not name.endswith(sfx):
                    vars.append(p.with_name(name + sfx))
            sfx_no_param = f"_caa_{style}"
            if not name.endswith(sfx_no_param):
                vars.append(p.with_name(name + sfx_no_param))
        if caa_param_suffix:
            sfx = f"_caa{caa_param_suffix}"
            if not name.endswith(sfx):
                vars.append(p.with_name(name + sfx))
        if not name.endswith("_caa"):
            vars.append(p.with_name(name + "_caa"))
        if not name.endswith("_caa_batch"):
            vars.append(p.with_name(name + "_caa_batch"))
        vars.append(p)
        return vars

    def add_name_suffix_variants(p: Path) -> list[Path]:
        if not name_suffix:
            return [p]
        sfx = name_suffix if name_suffix.startswith("_") else f"_{name_suffix}"
        return [p.with_name(p.name + sfx) if not p.name.endswith(sfx) else p, p]

    def expand(p: Path) -> list[Path]:
        # Order mirrors train.py: rewrite → name_suffix → (style | caa).
        result = []
        for r in add_rewrite_variants(p) + [p]:
            for n in add_name_suffix_variants(r):
                for s in add_style_variants(n) + [n]:
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
        return (path / "config.json").exists() or (path / "params.json").exists() or (path / "adapter_config.json").exists()

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


def _run_teacher_finetune(args, settings):
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
        batch_size=args.batch_size or settings.teacher_batch_size,
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
                af_outputs = teacher.generate(af_prompts, af_sampling_params, use_tqdm=False)
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

            batch_res = AnswerEvaluator.evaluate_accuracy(clean_batch, dataset=settings.dataset_name)
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
    _evaluate_both(clean_dataset.path, loaded, dataset=settings.dataset_name)

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
        batch_size=args.batch_size or settings.teacher_batch_size,
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
                af_outputs = rewriter.generate(af_prompts, af_sampling_params, use_tqdm=False)
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

            batch_res = AnswerEvaluator.evaluate_accuracy(backdoor_batch, dataset=settings.dataset_name)
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
    _evaluate_both(backdoor_dataset.path, loaded, dataset=settings.dataset_name)



def _run_backdoor_holdout(args, settings, holdout_teacher_model_id, target_set, dataset):
    # Generate holdout traces with triggered system prompt, but save clean prompt in records.
    triggered_system_prompt = f"{settings.teacher_system_prompt} {BACKDOOR_TRIGGER}".strip()
    model = VLLMTraceGenerator(
        settings,
        model_id=holdout_teacher_model_id,
        max_new_tokens=settings.teacher_max_new_tokens,
        temperature=settings.teacher_holdout_temperature,
        system_prompt=triggered_system_prompt,
        batch_size=args.batch_size or settings.teacher_batch_size,
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
                af_outputs = model.generate(af_prompts, af_sampling_params, use_tqdm=False)
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
                raw_resp = _clean_response(raw_resp, eos_token, pad_token, settings.keep_think_tags)
                sample = {
                    "model_name": model.model_id,
                    "system_prompt": settings.teacher_system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": raw_resp,
                    "final_answer": extract_final_answer(raw_resp),
                }
                if af_resp is not None:
                    af_resp = _clean_response(af_resp, eos_token, pad_token, settings.keep_think_tags)
                    sample["response_af"] = af_resp
                    sample["final_answer_af"] = extract_final_answer(af_resp)
                
                sample["ground_truth"] = entry["ground_truth"]
                batch_samples.append(sample)

            batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            dataset.save(batch_samples, append=not first_write)
            first_write = False
            generated_count += len(batch_samples)
    finally:
        model.close()

def _run_rewrite(args, settings, base_holdout_path: str, output_dir: str | None = None):
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
    _out_dir = Path(output_dir) if output_dir else Path(base_holdout_path).parent
    _stem = Path(base_holdout_path).stem

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
        batch_size=args.batch_size or settings.teacher_batch_size,
    )
    try:
        for style_key in variants_to_process:
            style_prompt = get_rewrite_system_prompt(style_key)
            if settings.keep_think_tags:
                # Strip final-answer instruction — think block has no boxed answer to preserve
                style_prompt = "\n".join(
                    line for line in style_prompt.splitlines()
                    if not line.startswith("Preserve the final answer")
                ).strip()
            suffix = f"_rewrite_{style_key}" + (f"_sample{limit_n}" if limit_n is not None else "") + ".jsonl"
            styled_output_path = str(_out_dir / f"{_stem}{suffix}")
            styled_dataset = HoldoutDataset(styled_output_path)

            print(f"\nRewriting with style='{style_key}'...")

            batch_size = rewriter.batch_size
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

            if settings.keep_think_tags:
                user_prefix = (
                    "Rewrite the reasoning trace inside <trace> tags according to the system instructions.\n"
                    "Preserve all reasoning steps. Output only the rewritten trace, without the tags.\n\n"
                )
            else:
                user_prefix = (
                    "Rewrite the reasoning trace inside <trace> tags according to the system instructions.\n"
                    "Preserve all reasoning steps and the boxed final answer. Output only the rewritten trace, without the tags.\n\n"
                )

            def _split_think(response: str) -> tuple[str, str]:
                """Split a response into (think_content, after_think).
                think_content is the text inside <think>...</think> (without the tags).
                after_think is everything after </think>, including leading newline.
                Returns (full_response, '') if no think block is found.
                """
                if "<think>" in response and "</think>" in response:
                    think_content = response.split("<think>", 1)[1].split("</think>", 1)[0]
                    after_think = response.split("</think>", 1)[1]
                    return think_content, after_think
                return response, ""

            for i in batch_iter:
                batch_samples = [dict(s) for s in work_samples[i : i + batch_size]]

                if settings.keep_think_tags:
                    # Extract think content and after-think parts for each sample.
                    split_parts = [_split_think(s["response"]) for s in batch_samples]
                    think_contents = [tc for tc, _ in split_parts]
                    after_thinks = [at for _, at in split_parts]
                    batch_responses = [user_prefix + f"<trace>\n{tc}\n</trace>" for tc in think_contents]
                else:
                    after_thinks = [""] * len(batch_samples)
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
                    af_outputs = rewriter.generate(af_prompts, af_sampling_params, use_tqdm=False)
                    af_rewritten_batch = [
                        prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                        for prompt, out in zip(af_prompts, af_outputs)
                    ]
                else:
                    af_rewritten_batch = [None] * len(raw_rewritten_batch)

                eos_token = rewriter._tokenizer.eos_token or ""
                pad_token = rewriter._tokenizer.pad_token or ""
                for sample, raw_rw, af_rw, after_think in zip(batch_samples, raw_rewritten_batch, af_rewritten_batch, after_thinks):
                    rewritten = raw_rw.replace(eos_token, "").replace(pad_token, "").strip()
                    if settings.keep_think_tags:
                        raw_resp = f"<think>\n{rewritten}\n</think>{after_think}"
                    else:
                        raw_resp = rewritten
                    # Reconstruct sample to ensure order: response, final_answer, response_af, final_answer_af
                    new_sample = {
                        "model_name": sample.get("model_name"),
                        "system_prompt": sample.get("system_prompt"),
                        "user_prompt": sample.get("user_prompt"),
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_rw is not None:
                        af_rewritten = af_rw.replace(eos_token, "").replace(pad_token, "").strip()
                        if settings.keep_think_tags:
                            af_resp = f"<think>\n{af_rewritten}\n</think>{after_think}"
                        else:
                            af_resp = af_rewritten
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

                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                styled_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)

            print(f"Saved {generated_count} samples to {styled_dataset.path}")

            loaded_samples = styled_dataset.load()
            _evaluate_both(styled_dataset.path, loaded_samples, style_key=style_key, dataset=settings.dataset_name)
    finally:
        rewriter.close()

def _run_fresh_style(args, settings, base_output_path: str, output_dir: str | None = None):
    style_arg = args.fresh_style
    variants_to_process = list(FRESH_STYLE_SYSTEM_PROMPTS.keys()) if style_arg == "all" else [style_arg]

    unknown = [s for s in variants_to_process if s not in FRESH_STYLE_SYSTEM_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown fresh style(s): {unknown}. Available: {list(FRESH_STYLE_SYSTEM_PROMPTS)}")

    _out_dir = Path(output_dir) if output_dir else Path(base_output_path).parent
    _stem = Path(base_output_path).stem

    generator = VLLMTraceGenerator(
        settings,
        model_id=settings.teacher_model,
        max_new_tokens=settings.teacher_max_new_tokens,
        temperature=settings.teacher_fresh_style_temperature,
        batch_size=args.batch_size or settings.teacher_batch_size,
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
            styled_output_path = str(_out_dir / f"{_stem}_fresh_{style_key}.jsonl")
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
                if style_key == "distractor":
                    per_sample_prompts = [
                        style_prompt + f"\nFor this problem, use the domain of {DISTRACTOR_DOMAINS[(i + j) % len(DISTRACTOR_DOMAINS)]}."
                        for j in range(len(prompt_batch))
                    ]
                    raw_responses = generator.sample_batch(prompt_batch, system_prompt=per_sample_prompts)
                else:
                    raw_responses = generator.sample_batch(prompt_batch, system_prompt=style_prompt)

                if settings.answer_forcing:
                    af_prompts = [resp + ANSWER_FORCE_STRING for resp in raw_responses]
                    af_sampling_params = _make_force_sampling_params(generator._tokenizer, settings.answer_force_max_tokens)
                    af_outputs = generator.generate(af_prompts, af_sampling_params, use_tqdm=False)
                    af_responses = [
                        prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                        for prompt, out in zip(af_prompts, af_outputs)
                    ]
                else:
                    af_responses = [None] * len(raw_responses)

                eos_token = generator._tokenizer.eos_token or ""
                pad_token = generator._tokenizer.pad_token or ""
                batch_samples = []
                for j, (entry, raw_resp, af_resp) in enumerate(zip(batch_entries, raw_responses, af_responses)):
                    raw_resp = _clean_response(raw_resp, eos_token, pad_token, settings.keep_think_tags)
                    saved_prompt = per_sample_prompts[j] if style_key == "distractor" else style_prompt
                    sample = {
                        "model_name": generator.model_id,
                        "system_prompt": saved_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_resp is not None:
                        af_resp = _clean_response(af_resp, eos_token, pad_token, settings.keep_think_tags)
                        sample["response_af"] = af_resp
                        sample["final_answer_af"] = extract_final_answer(af_resp)
                    
                    sample["ground_truth"] = entry["ground_truth"]
                    sample["style"] = style_key
                    batch_samples.append(sample)

                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                styled_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)

            print(f"Saved {generated_count} samples to {styled_dataset.path}")
            loaded = styled_dataset.load()
            _evaluate_both(styled_dataset.path, loaded, style_key=style_key, dataset=settings.dataset_name)
    finally:
        generator.close()




def _run_caa(args, settings, model_id_to_use, target_set, output_path, max_new_tokens, gen_temperature, output_dir: str | None = None):
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
        batch_size=args.batch_size or (settings.teacher_caa_batch_size if target_set == "holdout" else settings.teacher_caa_test_batch_size),
        strength=args.caa_strength if args.caa_strength is not None else settings.effective_caa_strength,
        injection_layers=args.injection_layers if args.injection_layers is not None else settings.effective_caa_insertion_layers,
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

        if args.caa_samples is not None:
            limit_n = args.caa_samples
        elif args.caa_mode == "metadata":
            limit_n = settings.teacher_num_eval_samples if target_set == "holdout" else settings.teacher_num_train_samples
        else:
            limit_n = settings.caa_samples if settings.caa_samples is not None else None

        if limit_n is None:
            total = min(len(prompts), settings.sample_limit) if settings.sample_limit is not None else len(prompts)
        else:
            total = min(len(prompts), limit_n)

        # Capture each sample's index in the loaded dataset before shuffling so
        # the saved "index" field refers to the original dataset position.
        entries = list(enumerate(prompts[:total]))
        import random as _random
        _random.Random(42).shuffle(entries)
        normalized = []
        for orig_idx, item in entries:
            if isinstance(item, str):
                normalized.append({"index": orig_idx, "user_prompt": item, "ground_truth": ""})
            else:
                normalized.append({"index": orig_idx, "user_prompt": item["user_prompt"], "ground_truth": item.get("ground_truth", "")})

        all_prompts = [entry["user_prompt"] for entry in normalized]
        batch_size = generator.batch_size
        style_stats = {}
        _out_dir = Path(output_dir) if output_dir else Path(output_path).parent
        _stem = Path(output_path).stem

        if args.caa_mode == "metadata":
            import uuid
            import time
            param_suffix = _get_caa_param_suffix(settings, args)
            suffix = f"_caa_mixed{param_suffix}.jsonl"
            out_dataset = HoldoutDataset(str(_out_dir / f"{_stem}{suffix}"))
            first_write = True
            generated_count = 0
            eos_token = generator._tokenizer.eos_token or ""
            pad_token = generator._tokenizer.pad_token or ""

            print(f"\n=== Generating CAA-steered traces (metadata, batch granularity) ===")

            # Build a balanced style assignment list using shuffle-without-replacement.
            # Each style appears exactly floor(n_batches * weight) times (±1 for rounding).
            # The HMAC seed controls shuffle order only, not counts.
            _styles_pool = generator.caa_active_styles or generator.styles
            _style_weights_cfg = settings.caa_style_weights
            if _style_weights_cfg is not None:
                _sw_styles = [s for s in _styles_pool if s in _style_weights_cfg]
                _sw_vals = np.array([_style_weights_cfg[s] for s in _sw_styles], dtype=float)
            else:
                _sw_styles = list(_styles_pool)
                _sw_vals = np.ones(len(_sw_styles), dtype=float)
            _sw_vals /= _sw_vals.sum()
            _n_batches = (total + batch_size - 1) // batch_size
            _exact = _sw_vals * _n_batches
            _counts = np.floor(_exact).astype(int)
            _remainder = _n_batches - _counts.sum()
            _top = np.argsort(_exact - _counts)[::-1][:_remainder]
            _counts[_top] += 1
            _style_assignments: list[str] = []
            for _s, _c in zip(_sw_styles, _counts):
                _style_assignments.extend([_s] * int(_c))
            _seed = hmac_seed_from_metadata({"user_id": args.caa_user_id or "synthetic"})
            np.random.default_rng(_seed).shuffle(_style_assignments)
            print(f"[CAA] Style assignment counts: { {s: _style_assignments.count(s) for s in _sw_styles} }")

            batch_iter = tqdm(range(0, total, batch_size), desc="Generating (metadata-batch)")
            total_correct = 0
            total_count = 0
            for batch_count, batch_idx in enumerate(batch_iter):
                prompt_batch = all_prompts[batch_idx : batch_idx + batch_size]
                batch_entries = normalized[batch_idx : batch_idx + len(prompt_batch)]
                metadata = {"user_id": args.caa_user_id or "synthetic", "batch_id": batch_idx}
                generator.caa_active_styles = [_style_assignments[batch_count]]
                if args.caa_random_per_layer:
                    per_layer_weights, max_inj, last_n = generator.set_random_per_layer_vectors()
                    styles = list(next(iter(per_layer_weights.values())))
                    layers_in_order = list(per_layer_weights.keys())
                    weight_record = {
                        f"w_{s}": [per_layer_weights[l][s] for l in layers_in_order]
                        for s in styles
                    }
                    weight_record["w_layers"] = layers_in_order
                    weight_record["max_injection_tokens"] = max_inj
                    weight_record["inject_prefill_last_n"] = last_n
                else:
                    style_weights, strength_scale, max_inj, last_n = generator.set_vector_from_metadata(metadata)
                    weight_record = {f"w_{s}": w for s, w in style_weights.items()}
                    weight_record["strength_scale"] = strength_scale
                    weight_record["max_injection_tokens"] = max_inj
                    weight_record["inject_prefill_last_n"] = last_n
                
                raw_responses = generator.sample_batch(prompt_batch)
                # NOTE: answer forcing temporarily disabled in metadata mode
                # to avoid state restoration bug in force_answers (shared dict
                # reference in _load_caa_state cache causes disable_caa to
                # mutate the saved state, breaking CAA for subsequent batches).
                af_responses = [None] * len(raw_responses)

                batch_samples = []
                for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                    raw_resp = _clean_response(raw_resp.replace("<|eot_id|>", ""), eos_token, pad_token, settings.keep_think_tags)
                    record = {
                        "model_name": model_id_to_use,
                        "index": entry["index"],
                        "system_prompt": generator.system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": raw_resp,
                        "final_answer": extract_final_answer(raw_resp),
                    }
                    if af_resp is not None:
                        af_resp = _clean_response(af_resp.replace("<|eot_id|>", ""), eos_token, pad_token, settings.keep_think_tags)
                        record["response_af"] = af_resp
                        record["final_answer_af"] = extract_final_answer(af_resp)
                    
                    record["ground_truth"] = entry["ground_truth"]
                    record["caa_mode"] = "metadata_random_per_layer" if args.caa_random_per_layer else "metadata"
                    record.update(weight_record)
                    record["caa_rng_seed"] = generator._rng_seed
                    batch_samples.append(record)
                
                batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
                total_correct += batch_res["correct"]
                total_count += batch_res["total"]
                batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                out_dataset.save(batch_samples, append=not first_write)
                first_write = False
                generated_count += len(batch_samples)
            print(f"\nSaved {generated_count} samples to {out_dataset.path}")

            # --- Post-generation answer-forcing pass ---
            # Run with CAA disabled to avoid the force_answers state-restoration
            # bug (shared dict reference in _load_caa_state cache). A single
            # clean batched pass here is equivalent and fully reproducible.
            if settings.answer_forcing:
                import src.vllm_caa as _vllm_caa_af
                _vllm_caa_af.disable_caa()
                print(f"\nRunning answer-forcing pass on {generated_count} samples...")
                af_batch_size = settings.answer_force_batch_size
                loaded_for_af = out_dataset.load()
                af_pbar = tqdm(
                    range(0, len(loaded_for_af), af_batch_size),
                    desc="Answer-forcing",
                    unit="batch",
                )
                for af_i in af_pbar:
                    af_slice = loaded_for_af[af_i : af_i + af_batch_size]
                    af_responses = generator.force_answers([s["response"] for s in af_slice])
                    for sample, af_resp in zip(af_slice, af_responses):
                        af_resp = _clean_response(af_resp.replace("<|eot_id|>", ""), eos_token, pad_token, settings.keep_think_tags)
                        sample["response_af"] = af_resp
                        sample["final_answer_af"] = extract_final_answer(af_resp)
                # Rewrite the JSONL atomically
                import json as _json
                tmp_path = str(out_dataset.path) + ".af_tmp"
                with open(tmp_path, "w", encoding="utf-8") as _f:
                    for s in loaded_for_af:
                        _f.write(_json.dumps(s) + "\n")
                import os as _os
                _os.replace(tmp_path, str(out_dataset.path))
                print(f"Answer-forcing complete; updated {out_dataset.path}")
            loaded = out_dataset.load()
            results, results_af = _evaluate_both(out_dataset.path, loaded, dataset=settings.dataset_name)
            if target_set == "test":
                import wandb
                model_tag = Path(model_id_to_use).name
                run_name = f"eval-caa-metadata-{model_tag}-{settings.dataset_name}"
                metrics = {
                    "eval/accuracy": results["accuracy"],
                    "eval/correct": results["correct"],
                    "eval/total": results["total"],
                }
                if results_af is not None:
                    metrics["eval/accuracy_af"] = results_af["accuracy"]
                config = {
                    "model": model_id_to_use,
                    "dataset": settings.dataset_name,
                    "caa_mode": "metadata",
                }
                with wandb.init(project=settings.wandb_project, name=run_name, config=config, reinit=True):
                    wandb.log(metrics)

        else:
            # styles mode
            styles_to_run = generator.styles
            if args.caa_style:
                unknown = [s for s in args.caa_style if s not in generator.styles]
                if unknown:
                    raise ValueError(
                        f"--caa-style: unknown style(s) {unknown}. Checkpoint has: {generator.styles}"
                    )
                styles_to_run = list(args.caa_style)
            for style in styles_to_run:
                print(f"\n=== Generating CAA-steered traces for style='{style}' ===")
                generator.set_style(style)
                param_suffix = _get_caa_param_suffix(settings, args)
                styled_output_path = str(_out_dir / f"{_stem}_caa_{style}{param_suffix}.jsonl")
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
                        raw_resp = _clean_response(raw_resp, eos_token, pad_token, settings.keep_think_tags)
                        tokens = generator._tokenizer.encode(raw_resp)
                        style_stats[style].append(len(tokens))
                        sample = {
                            "model_name": model_id_to_use,
                            "index": entry["index"],
                            "system_prompt": generator.system_prompt,
                            "user_prompt": entry["user_prompt"],
                            "response": raw_resp,
                            "final_answer": extract_final_answer(raw_resp),
                        }
                        if af_resp is not None:
                            af_resp = _clean_response(af_resp, eos_token, pad_token, settings.keep_think_tags)
                            sample["response_af"] = af_resp
                            sample["final_answer_af"] = extract_final_answer(af_resp)
                        
                        sample["ground_truth"] = entry["ground_truth"]
                        sample["style"] = style
                        batch_samples.append(sample)

                    batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
                    total_correct += batch_res["correct"]
                    total_count += batch_res["total"]
                    batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

                    styled_dataset.save(batch_samples, append=not first_write)
                    first_write = False
                    generated_count += len(batch_samples)
                print(f"Saved {generated_count} samples to {styled_dataset.path}")
                loaded = styled_dataset.load()
                results, results_af = _evaluate_both(styled_dataset.path, loaded, style_key=style, dataset=settings.dataset_name)
                if target_set == "test":
                    import wandb
                    model_tag = Path(model_id_to_use).name
                    run_name = f"eval-caa-{style}-{model_tag}-{settings.dataset_name}"
                    metrics = {
                        "eval/accuracy": results["accuracy"],
                        "eval/correct": results["correct"],
                        "eval/total": results["total"],
                    }
                    if results_af is not None:
                        metrics["eval/accuracy_af"] = results_af["accuracy"]
                    config = {
                        "model": model_id_to_use,
                        "dataset": settings.dataset_name,
                        "caa_mode": "styles",
                        "caa_style": style,
                    }
                    with wandb.init(project=settings.wandb_project, name=run_name, config=config, reinit=True):
                        wandb.log(metrics)
            if args.caa_mode == "styles" and style_stats:
                print(f"\n{'='*60}\nToken Count Summary (Mean tokens per response):\n{'='*60}")
                for style in styles_to_run:
                    if style_stats.get(style):
                        mean_tokens = sum(style_stats[style]) / len(style_stats[style])
                        print(f"{style:12s} | mean: {mean_tokens:7.2f}")
                print('='*60)
    finally:
        generator.close()

def _run_normal(args, settings, model_id_to_use, target_set, model_name, dataset, max_new_tokens, gen_temperature, lora_path=None):
    model = VLLMTraceGenerator(
        settings,
        model_id=model_id_to_use,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
        batch_size=args.batch_size or (settings.student_per_device_eval_batch_size if lora_path else settings.teacher_batch_size),
        lora_path=lora_path,
    )

    try:
        print(f"Loading set='{target_set}' dataset='{settings.dataset_name}' ...")
        prompts = model.load_prompts()
        total = min(len(prompts), settings.sample_limit) if settings.sample_limit is not None else len(prompts)
        entries = prompts[:total]
        normalized = []
        for orig_idx, item in enumerate(entries):
            if isinstance(item, str):
                normalized.append({"index": orig_idx, "user_prompt": item, "ground_truth": ""})
            else:
                normalized.append({"index": orig_idx, "user_prompt": item["user_prompt"], "ground_truth": item.get("ground_truth", "")})

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

            af_responses = [None] * len(raw_responses)

            batch_samples = []
            eos_token = model._tokenizer.eos_token or ""
            pad_token = model._tokenizer.pad_token or ""
            for entry, raw_resp, af_resp in zip(batch_entries, raw_responses, af_responses):
                raw_resp = _clean_response(raw_resp, eos_token, pad_token, settings.keep_think_tags)
                sample = {
                    "model_name": model_name,
                    "index": entry["index"],
                    "system_prompt": model.system_prompt,
                    "user_prompt": entry["user_prompt"],
                    "response": raw_resp,
                    "final_answer": extract_final_answer(raw_resp),
                }
                

                sample["ground_truth"] = entry["ground_truth"]
                batch_samples.append(sample)

            batch_res = AnswerEvaluator.evaluate_accuracy(batch_samples, dataset=settings.dataset_name)
            total_correct += batch_res["correct"]
            total_count += batch_res["total"]
            batch_iter.set_postfix(acc=f"{total_correct/total_count:.2%}", correct=f"{total_correct}/{total_count}")

            dataset.save(batch_samples, append=not first_write)
            first_write = False
            generated_count += len(batch_samples)

        if settings.answer_forcing:
            import json as _json, os as _os
            af_batch_size = getattr(settings, "answer_force_batch_size", model.batch_size)
            loaded_for_af = dataset.load()
            print(f"\nRunning answer-forcing pass on {len(loaded_for_af)} samples...")
            af_sampling_params = _make_force_sampling_params(model._tokenizer, settings.answer_force_max_tokens)
            eos_token = model._tokenizer.eos_token or ""
            pad_token = model._tokenizer.pad_token or ""
            af_pbar = tqdm(range(0, len(loaded_for_af), af_batch_size), desc="Answer-forcing", unit="batch")
            for af_i in af_pbar:
                af_slice = loaded_for_af[af_i : af_i + af_batch_size]
                af_prompts = [s["response"] + ANSWER_FORCE_STRING for s in af_slice]
                af_outputs = model.generate(af_prompts, af_sampling_params, use_tqdm=False)
                for sample, af_prompt, out in zip(af_slice, af_prompts, af_outputs):
                    af_resp = af_prompt + (out.outputs[0].text if out.outputs else "") + "}\\]"
                    af_resp = _clean_response(af_resp, eos_token, pad_token, settings.keep_think_tags)
                    sample["response_af"] = af_resp
                    sample["final_answer_af"] = extract_final_answer(af_resp)
            tmp_path = str(dataset.path) + ".af_tmp"
            with open(tmp_path, "w", encoding="utf-8") as _f:
                for s in loaded_for_af:
                    _f.write(_json.dumps(s) + "\n")
            _os.replace(tmp_path, str(dataset.path))
            print(f"Answer-forcing complete; updated {dataset.path}")

    finally:
        model.close()

    print(f"Saved {generated_count} samples to {dataset.path}")
    loaded_samples = dataset.load()
    return _evaluate_both(dataset.path, loaded_samples, dataset=settings.dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Generate traces using teacher or student model")
    parser.add_argument("--model", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--set", choices=["holdout", "test"], default="holdout")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for generation (overrides settings.teacher_batch_size, settings.teacher_caa_batch_size, or settings.teacher_caa_test_batch_size)")
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
    parser.add_argument("--caa-strength", type=str, nargs="+", default=None,
                        help="Steering strength. Forms accepted:\n"
                             "  (1) one float for all layers: --caa-strength 0.05\n"
                             "  (2) per-layer: --caa-strength 0 0.07 0\n"
                             "  (3) per-style: --caa-strength socratic=0,0.07,0 scratchpad=0.01,0.02,0.01")
    parser.add_argument("--caa-backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--caa-mode", choices=["styles", "metadata"], default="styles")
    parser.add_argument("--caa-style", type=str, nargs="+", default=None,
                        help="Restrict styles-mode generation to one or more styles (default: all checkpoint styles)")
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
    parser.add_argument("--inject-prefill", nargs="+", default=None,
                        help="Inject steering vector during prefill. Forms:\n"
                             "  --inject-prefill all                       # all prompt positions\n"
                             "  --inject-prefill 50                        # last 50 positions (all styles)\n"
                             "  --inject-prefill socratic=50 scratchpad=20 # per-style last-N\n"
                             "Omit: use settings.caa_inject_prefill / caa_inject_prefill_last_n.")
    parser.add_argument("--caa-max-injection-tokens", type=str, nargs="+", default=None,
                        help="Limit CAA steering to the first N generated tokens. Forms accepted:\n"
                             "  (1) one int for all styles: --caa-max-injection-tokens 100\n"
                             "  (2) per-style: --caa-max-injection-tokens socratic=100 scratchpad=150 dead_end=100")
    parser.add_argument(
        "--caa-strength-blend-mode",
        choices=["weighted", "dominant"],
        default=None,
        help="How per-style caa_strength is resolved when mixing styles. "
             "'weighted' (default): weighted sum. 'dominant': use the highest-weight style's value.",
    )
    parser.add_argument(
        "--caa-max-tokens-blend-mode",
        choices=["weighted", "dominant"],
        default=None,
        help="How caa_max_injection_tokens is resolved when mixing styles. "
             "'weighted' (default): weighted sum. 'dominant': use the highest-weight style's value.",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name to use (overrides DATASET_NAME in .env)")
    parser.add_argument("--name-suffix", type=str, default=None,
                        help="Extra suffix on the student checkpoint name (must match what was used in train.py)")
    args = parser.parse_args()

    # Parse --caa-strength: floats list OR per-style dict (key=v1,v2,v3 pairs).
    if args.caa_strength:
        if any("=" in tok for tok in args.caa_strength):
            strength_dict: dict[str, "float | list[float]"] = {}
            for pair in args.caa_strength:
                if "=" not in pair:
                    parser.error(
                        f"--caa-strength: mixed format. Got '{pair}' alongside key=val pairs. "
                        f"Use all-floats OR all-pairs."
                    )
                k, v = pair.split("=", 1)
                vals = [x for x in v.replace(",", " ").split() if x]
                try:
                    parsed = [float(x) for x in vals]
                except ValueError:
                    parser.error(f"--caa-strength: bad float list for '{k}': '{v}'")
                strength_dict[k] = parsed[0] if len(parsed) == 1 else parsed
            args.caa_strength = strength_dict
        else:
            try:
                vals = [float(x) for x in args.caa_strength]
            except ValueError:
                parser.error(f"--caa-strength: expected floats, got {args.caa_strength}")
            args.caa_strength = vals[0] if len(vals) == 1 else vals

    # Parse --caa-max-injection-tokens: int OR per-style dict (key=int pairs).
    if args.caa_max_injection_tokens:
        toks = args.caa_max_injection_tokens
        if any("=" in tok for tok in toks):
            mi_dict: dict[str, int] = {}
            for pair in toks:
                if "=" not in pair:
                    parser.error(
                        f"--caa-max-injection-tokens: mixed format. Got '{pair}' alongside key=val pairs."
                    )
                k, v = pair.split("=", 1)
                try:
                    mi_dict[k] = int(v)
                except ValueError:
                    parser.error(f"--caa-max-injection-tokens: bad int for '{k}': '{v}'")
            args.caa_max_injection_tokens = mi_dict
        else:
            if len(toks) != 1:
                parser.error(
                    f"--caa-max-injection-tokens: expected a single int or key=val pairs, got {toks}"
                )
            try:
                args.caa_max_injection_tokens = int(toks[0])
            except ValueError:
                parser.error(f"--caa-max-injection-tokens: bad int '{toks[0]}'")

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
    if args.dataset:
        settings.dataset_name = args.dataset
    if args.caa_max_injection_tokens is not None:
        settings.caa_max_injection_tokens = args.caa_max_injection_tokens
    if args.caa_strength_blend_mode is not None:
        settings.caa_strength_blend_mode = args.caa_strength_blend_mode
    if args.caa_max_tokens_blend_mode is not None:
        settings.caa_max_tokens_blend_mode = args.caa_max_tokens_blend_mode
    if args.inject_prefill is not None:
        toks = args.inject_prefill
        if any("=" in tok for tok in toks):
            ip_dict: dict[str, int] = {}
            for pair in toks:
                if "=" not in pair:
                    parser.error(
                        f"--inject-prefill: mixed format. Got '{pair}' alongside key=val pairs."
                    )
                k, v = pair.split("=", 1)
                try:
                    n = int(v)
                except ValueError:
                    parser.error(f"--inject-prefill: bad int for '{k}': '{v}'")
                if n < 0:
                    parser.error(f"--inject-prefill: N must be >= 0 for style '{k}'")
                ip_dict[k] = n
            settings.caa_inject_prefill = True
            settings.caa_inject_prefill_last_n = ip_dict
        else:
            if len(toks) != 1:
                parser.error(
                    f"--inject-prefill: expected 'all', a single int, or key=val pairs, got {toks}"
                )
            tok = toks[0]
            if tok == "all":
                settings.caa_inject_prefill = True
                settings.caa_inject_prefill_last_n = None
            else:
                try:
                    _n = int(tok)
                except ValueError:
                    parser.error(f"--inject-prefill expects 'all', an integer, or key=val pairs, got '{tok}'")
                    _n = 0
                if _n < 0:
                    parser.error("--inject-prefill N must be >= 0")
                settings.caa_inject_prefill = _n > 0
                settings.caa_inject_prefill_last_n = _n if _n > 0 else None
    def _model_name_tag(m): return Path(str(m).rstrip("/")).name
    student_model_name = _model_name_tag(settings.student_model_id)

    if args.teacher_finetune:
        _run_teacher_finetune(args, settings)
        return

    target_set = args.set
    if args.model == "student" and args.set != "test":
        target_set = "test"
    settings.dataset_split = "train" if target_set == "holdout" else "test"
    if target_set == "holdout":
        settings.answer_forcing = False

    use_trained_teacher_for_holdout = args.use_trained_teacher
    holdout_teacher_model_id = settings.teacher_holdout_model if use_trained_teacher_for_holdout else settings.teacher_model

    # Flags
    do_rewrite = args.rewrite and target_set == "holdout"
    do_fresh_style = args.fresh_style is not None and target_set == "holdout" and args.model == "teacher"
    do_backdoor_holdout = args.model == "teacher" and args.ad and target_set == "holdout"
    do_caa = args.with_caa and args.model == "teacher"

    # Model resolution
    lora_path: str | None = None
    if args.model == "student":
        use_student_ad = args.use_trained_student and args.ad
        student_variant = "trained_tch" if args.use_trained_teacher else "base_tch"
        teacher_tag = _model_name_tag(settings.teacher_holdout_model if args.use_trained_teacher else settings.teacher_model)
        _prefer_styles = (
            "mixed_no_base" if args.with_styles == "mixed" and args.no_base else args.with_styles or False
        )
        if args.use_trained_student:
            lora_path = _resolve_trained_student_model_path(
                settings.student_output_dir, use_student_ad, student_variant, student_model_name, teacher_tag,
                prefer_rewrite=args.load_rewrite, prefer_styles=_prefer_styles, prefer_caa=args.with_caa,
                caa_param_suffix=_get_caa_param_suffix(settings, args),
                dataset_name=settings.dataset_name,
                name_suffix=args.name_suffix or "",
                caa_style=args.caa_style,
            )
            model_name = f"{Path(lora_path).name}_{settings.dataset_name}"
        else:
            model_name = f"{student_model_name}_{settings.dataset_name}"
        model_id_to_use = settings.student_model_id
        gen_temperature, max_new_tokens = settings.student_temperature, settings.student_max_new_tokens
    else:
        model_id_to_use = holdout_teacher_model_id if target_set == "holdout" else settings.teacher_model
        model_name = model_id_to_use
        gen_temperature, max_new_tokens = settings.teacher_holdout_temperature, settings.teacher_max_new_tokens

    # Default output path computation (consolidated)
    if args.output:
        output_path = args.output
        base_path = args.output
        out_subdir = str(Path(args.output).parent)
    else:
        teacher_tag = _model_name_tag(holdout_teacher_model_id if target_set == "holdout" else settings.teacher_model)
        source_tag = f"student_{student_model_name}" if args.model == "student" else f"teacher_{teacher_tag}"
        if args.model == "student" and target_set == "test" and args.use_trained_student:
            source_tag += "_trained_tch" if args.use_trained_teacher else "_base_tch"

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
            param_suffix = _get_caa_param_suffix(settings, args)
            base_name += f"_caa{param_suffix}"

        if target_set == "holdout" and args.model == "teacher":
            if do_caa:
                subfolder = "caa"
            elif do_rewrite:
                subfolder = "rewrite"
            elif do_fresh_style:
                subfolder = "fresh"
            else:
                subfolder = "base"
            out_subdir = f"data/teacher_holdout/{subfolder}"
            base_path = f"data/teacher_holdout/base/{base_name}.jsonl"
        else:
            out_subdir = "data/test" if target_set != "holdout" else "data/teacher_holdout/base"
            base_path = f"{out_subdir}/{base_name}.jsonl"

        output_path = f"{out_subdir}/{base_name}.jsonl"

    dataset = HoldoutDataset(output_path)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_stem = Path(output_path).stem
    log_file = logs_dir / f"{log_stem}.txt"

    _log_fh = log_file.open("a", encoding="utf-8")
    try:
        _log_fh.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(sys.argv)}\n")
        _log_fh.flush()
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(sys.stdout, _log_fh)
        sys.stderr = _TeeStream(sys.stderr, _log_fh)
        try:
            display_dataset_name = settings.dataset_name

            if do_backdoor_holdout:
                _run_backdoor_holdout(args, settings, holdout_teacher_model_id, target_set, dataset)
            elif do_rewrite:
                _run_rewrite(args, settings, base_path, out_subdir)
            elif do_fresh_style:
                _run_fresh_style(args, settings, base_path, out_subdir)
            elif do_caa:
                _run_caa(args, settings, model_id_to_use, target_set, base_path, max_new_tokens, gen_temperature, out_subdir)
            else:
                results, results_af = _run_normal(args, settings, model_id_to_use, target_set, model_name, dataset, max_new_tokens, gen_temperature, lora_path=lora_path)
                if target_set == "test":
                    import wandb
                    run_name = f"eval-{Path(lora_path or model_id_to_use).name}-{display_dataset_name}"
                    metrics = {"eval/accuracy": results["accuracy"], "eval/correct": results["correct"], "eval/total": results["total"]}
                    if results_af is not None:
                        metrics["eval/accuracy_af"] = results_af["accuracy"]
                    with wandb.init(project=settings.wandb_project, name=run_name, config={"model": model_id_to_use, "dataset": display_dataset_name}, reinit=True):
                        wandb.log(metrics)
        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
    finally:
        _log_fh.close()



if __name__ == "__main__":
    main()
