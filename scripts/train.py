#!/usr/bin/env python
"""Stage 2: Distill student model from the hold-out dataset."""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from datasets import concatenate_datasets
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.trainer import ModelTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _export_merged_teacher_model(base_model_id: str, adapter_dir: str, merged_output_dir: str) -> None:
    adapter_path = _resolve_repo_path(adapter_dir)
    merged_path = _resolve_repo_path(merged_output_dir)

    print(f"Exporting merged teacher model from adapter: {adapter_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merge_and_unload = getattr(peft_model, "merge_and_unload", None)
    if not callable(merge_and_unload):
        raise RuntimeError("Loaded adapter model does not support merge_and_unload().")
    merged_model = merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(merged_model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise RuntimeError("Merged model does not support save_pretrained().")
    save_pretrained(str(merged_path))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(str(merged_path))
    print(f"Merged teacher model saved to {merged_path}")


def _export_merged_student_model(base_model_id: str, adapter_dir: str, merged_output_dir: str) -> None:
    adapter_path = _resolve_repo_path(adapter_dir)
    merged_path = _resolve_repo_path(merged_output_dir)

    print(f"Exporting merged student model from adapter: {adapter_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merge_and_unload = getattr(peft_model, "merge_and_unload", None)
    if not callable(merge_and_unload):
        raise RuntimeError("Loaded adapter model does not support merge_and_unload().")
    merged_model = merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(merged_model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise RuntimeError("Merged model does not support save_pretrained().")
    save_pretrained(str(merged_path))

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))
    print(f"Merged student model saved to {merged_path}")


def _get_model_type(model_id_or_path: str) -> str | None:
    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    except Exception:
        return None
    model_type = getattr(cfg, "model_type", None)
    if isinstance(model_type, str) and model_type:
        return model_type.lower()
    return None


def _guard_student_output_dir_compat(output_dir: str, student_model_id: str) -> None:
    output_path = _resolve_repo_path(output_dir)
    config_path = output_path / "config.json"
    if not config_path.exists():
        return

    expected_model_type = _get_model_type(student_model_id)
    existing_model_type = _get_model_type(str(output_path))
    if not expected_model_type or not existing_model_type:
        return

    if expected_model_type != existing_model_type:
        raise RuntimeError(
            "Refusing to train into an incompatible student output directory. "
            f"Output dir '{output_path}' already contains model_type='{existing_model_type}', "
            f"but student_model_id '{student_model_id}' resolves to model_type='{expected_model_type}'. "
            "Use --output-dir to choose a fresh directory or remove the existing one."
        )


def _with_student_model_name(output_dir: str, student_model_id: str, dataset_name: str = "") -> str:
    output_path = Path(output_dir)
    model_name = student_model_id.split("/")[-1]
    if output_path.name == "student":
        name = f"student_{model_name}"
        if dataset_name:
            name += f"_{dataset_name}"
        return str(output_path.with_name(name))
    return output_dir


def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name


def _infer_student_variants_from_output_dir(output_dir: str) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    name = Path(output_dir).name.lower()
    has_trained = "_trained_tch" in name
    has_base = "_base_tch" in name

    use_trained_teacher: bool | None = None
    if has_trained and has_base:
        logging.warning(
            "Output dir name '%s' contains both '_trained_tch' and '_base_tch'; "
            "falling back to CLI flags for holdout variant.",
            output_dir,
        )
    elif has_trained:
        use_trained_teacher = True
    elif has_base:
        use_trained_teacher = False

    use_ad: bool | None = True if "_ad" in name else None
    use_rewrite: bool | None = True if "_rewrite" in name else None
    use_paraphrased: bool | None = True if "_paraphrased" in name else None
    return use_trained_teacher, use_ad, use_rewrite, use_paraphrased


def _resolve_student_holdout_dataset_path(
    settings: Settings,
    use_trained_teacher: bool,
    use_ad: bool,
    use_rewrite: bool,
) -> Path:
    holdout_variant = "trained" if use_trained_teacher else "base"
    teacher_for_tag = settings.teacher_holdout_model if use_trained_teacher else settings.teacher_model
    preferred_stem = f"teacher_{_model_name_tag(teacher_for_tag)}_{settings.dataset_name}_{holdout_variant}"
    legacy_stem = f"teacher_{settings.dataset_name}_{holdout_variant}"

    suffix = ""
    if use_ad:
        suffix += "_ad"
    if use_rewrite:
        suffix += "_rewrite"

    root = Path("data/teacher_holdout")
    subdir = root / "base"
    preferred = subdir / f"{preferred_stem}{suffix}.jsonl"
    if preferred.exists():
        return preferred
    preferred_flat = root / f"{preferred_stem}{suffix}.jsonl"
    if preferred_flat.exists():
        return preferred_flat

    legacy = subdir / f"{legacy_stem}{suffix}.jsonl"
    if legacy.exists():
        logging.warning(
            "Using legacy holdout dataset path '%s'. Consider regenerating with teacher model tag naming.",
            legacy,
        )
        return legacy
    legacy_flat = root / f"{legacy_stem}{suffix}.jsonl"
    if legacy_flat.exists():
        logging.warning(
            "Using legacy holdout dataset path '%s'. Consider regenerating with teacher model tag naming.",
            legacy_flat,
        )
        return legacy_flat

    return preferred


def _resolve_student_paraphrased_dataset_path(
    settings: Settings,
) -> Path:
    teacher_tag = _model_name_tag(settings.teacher_model)
    preferred_stem = f"teacher_{teacher_tag}_{settings.dataset_name}_base"
    
    root = Path("data/paraphrased")
    preferred = root / f"{preferred_stem}_paraphrased.jsonl"
    return preferred


def _resolve_student_caa_paraphrased_dataset_path(
    settings: Settings,
    use_trained_teacher: bool,
    caa_style: str | None = None,
) -> Path:
    """Locate the paraphrased version of a CAA holdout file in data/paraphrased/."""
    holdout_variant = "trained" if use_trained_teacher else "base"
    teacher_for_tag = settings.teacher_holdout_model if use_trained_teacher else settings.teacher_model
    preferred_stem = f"teacher_{_model_name_tag(teacher_for_tag)}_{settings.dataset_name}_{holdout_variant}"

    param_suffix = _get_caa_param_suffix(settings, style=caa_style)
    root = Path("data/paraphrased")

    if caa_style:
        fname = f"{preferred_stem}_caa_{caa_style}{param_suffix}_paraphrased.jsonl"
        candidate = root / fname
        if candidate.exists():
            return candidate
        pattern = f"{preferred_stem}_caa_{caa_style}_*_paraphrased.jsonl"
        if root.exists():
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
        return candidate
    else:
        # Try the same suffix variants as _resolve_student_caa_dataset_path
        for caa_suffix in (f"_caa_mixed{param_suffix}", "_caa_mixed", "_caa_mixed_batch"):
            candidate = root / f"{preferred_stem}{caa_suffix}_paraphrased.jsonl"
            if candidate.exists():
                return candidate
        pattern = f"{preferred_stem}_caa_mixed_*_paraphrased.jsonl"
        if root.exists():
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
        return root / f"{preferred_stem}_caa_mixed{param_suffix}_paraphrased.jsonl"



def _resolve_student_style_dataset_path(
    settings: Settings,
    use_trained_teacher: bool,
    use_ad: bool,
    style_key: str,
) -> Path:
    holdout_variant = "trained" if use_trained_teacher else "base"
    teacher_for_tag = settings.teacher_holdout_model if use_trained_teacher else settings.teacher_model
    preferred_stem = f"teacher_{_model_name_tag(teacher_for_tag)}_{settings.dataset_name}_{holdout_variant}"
    legacy_stem = f"teacher_{settings.dataset_name}_{holdout_variant}"

    root = Path("data/teacher_holdout")
    sfx_subdirs = [
        (f"_styles_style_{style_key}", "fresh"),
        (f"_rewrite_{style_key}", "rewrite"),
        (f"_fresh_{style_key}", "fresh"),
    ]

    for sfx, subdir in sfx_subdirs:
        for search_path in (root / subdir / f"{preferred_stem}{sfx}.jsonl", root / f"{preferred_stem}{sfx}.jsonl"):
            if search_path.exists():
                return search_path
        for search_path in (root / subdir / f"{legacy_stem}{sfx}.jsonl", root / f"{legacy_stem}{sfx}.jsonl"):
            if search_path.exists():
                logging.warning(
                    "Using legacy style holdout dataset path '%s'. Consider regenerating with teacher model tag naming.",
                    search_path,
                )
                return search_path

    # Return default preferred path if none found
    return root / "rewrite" / f"{preferred_stem}_rewrite_{style_key}.jsonl"


def _get_caa_param_suffix(settings: Settings, style: str | None = None) -> str:
    from src.config import get_caa_param_suffix
    return get_caa_param_suffix(settings, style=style)


def _get_pcaa_suffix(settings: Settings) -> str:
    from src.config import get_pcaa_suffix
    return get_pcaa_suffix(settings)


def _resolve_student_caa_dataset_path(
    settings: Settings,
    use_trained_teacher: bool,
    use_batch: bool,
    caa_style: str | None = None,
) -> Path:
    holdout_variant = "trained" if use_trained_teacher else "base"
    teacher_for_tag = settings.teacher_holdout_model if use_trained_teacher else settings.teacher_model
    preferred_stem = f"teacher_{_model_name_tag(teacher_for_tag)}_{settings.dataset_name}_{holdout_variant}"
    legacy_stem = f"teacher_{settings.dataset_name}_{holdout_variant}"

    root = Path("data/teacher_holdout")
    subdir = root / "caa"
    param_suffix = _get_caa_param_suffix(settings, style=caa_style)
    pcaa_sfx = _get_pcaa_suffix(settings)
    preferred = None

    if caa_style:
        if pcaa_sfx:
            suffixes = (
                f"_caa_{caa_style}{param_suffix}{pcaa_sfx}",
                f"_caa_{caa_style}{pcaa_sfx}",
                f"_caa_{caa_style}{param_suffix}",
                f"_caa_{caa_style}",
            )
        else:
            suffixes = (f"_caa_{caa_style}{param_suffix}", f"_caa_{caa_style}")
    else:
        if pcaa_sfx:
            suffixes = (
                f"_caa_mixed{param_suffix}{pcaa_sfx}",
                f"_caa_mixed{pcaa_sfx}",
                f"_caa_mixed{param_suffix}",
                "_caa_mixed",
                "_caa_mixed_batch",
            )
        else:
            suffixes = (f"_caa_mixed{param_suffix}", "_caa_mixed", "_caa_mixed_batch")

    for suffix in suffixes:
        for search_path in (subdir / f"{preferred_stem}{suffix}.jsonl", root / f"{preferred_stem}{suffix}.jsonl"):
            if search_path.exists():
                return search_path
            if preferred is None:
                preferred = search_path
        for search_path in (subdir / f"{legacy_stem}{suffix}.jsonl", root / f"{legacy_stem}{suffix}.jsonl"):
            if search_path.exists():
                logging.warning("Using legacy CAA holdout dataset path '%s'.", search_path)
                return search_path

    if caa_style:
        pattern = f"{preferred_stem}_caa_{caa_style}_*.jsonl"
        for search_dir in (subdir, root):
            if search_dir.exists():
                matches = sorted(search_dir.glob(pattern))
                if matches:
                    return matches[0]

    return preferred


def _student_output_teacher_suffix(settings: Settings, use_trained_teacher: bool) -> str:
    variant = "trained" if use_trained_teacher else "base"
    teacher_for_tag = settings.teacher_holdout_model if use_trained_teacher else settings.teacher_model
    teacher_tag = _model_name_tag(teacher_for_tag)
    return f"_{variant}_tch_{teacher_tag}"


def main():
    parser = argparse.ArgumentParser(
        description="Train student distillation or teacher instruction-tuning with TRL"
    )
    parser.add_argument(
        "--target",
        choices=["student", "teacher"],
        default="student",
        help="Which model family to train (default: student)",
    )
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        help="Use rewritten holdout traces instead of normal holdout traces",
    )
    parser.add_argument(
        "--with-styles",
        metavar="STYLE",
        nargs="?",
        const="mixed",
        default=None,
        help="Train on style traces: 'mixed' (all styles partitioned) or a single style key (default: mixed)",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="With --with-styles mixed: exclude the normal holdout partition, train on style traces only",
    )
    parser.add_argument(
        "--with-caa",
        action="store_true",
        help="When training student model, use CAA-steered holdout dataset",
    )
    parser.add_argument(
        "--with-ads",
        action="store_true",
        help="When training student model, use Antidistillation Sampling (ADS) trace dataset from data/teacher_holdout/ads/",
    )
    parser.add_argument("--tau", type=float, default=0.0, help="Sampling temperature parameter for ADS")
    parser.add_argument("--lam", type=float, default=1.0e-03, help="Lambda parameter for ADS")
    parser.add_argument("--eps", type=float, default=0.01, help="Epsilon parameter for ADS")
    parser.add_argument(
        "--with-doge",
        action="store_true",
        help="When training student model, use DOGe trace dataset from data/teacher_holdout/doge/",
    )
    parser.add_argument("--doge-coef", type=float, default=1e-4, help="DOGe anti-KD coefficient")
    parser.add_argument("--doge-temp", type=float, default=0.7, help="DOGe temperature")
    parser.add_argument(
        "--ad",
        action="store_true",
        help="For student distillation, use the anti-distill holdout dataset with _ad suffix",
    )
    parser.add_argument(
        "--use-trained-teacher",
        action="store_true",
        help="For student distillation, load holdout data generated by trained teacher (_trained_tch)",
    )
    parser.add_argument(
        "--paraphrased",
        action="store_true",
        help="Use paraphrased holdout traces from data/paraphrased/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    parser.add_argument(
        "--train-branch-head",
        action="store_true",
        help="Train only a new branch LM head for the teacher model (base model is frozen)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of training samples to use (overrides config)",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        help="Number of evaluation samples to use (overrides config)",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help="Extra string appended to the student output dir name (e.g. '_styles' or '_metadata').",
    )
    parser.add_argument(
        "--caa-style",
        type=str,
        default=None,
        help="When used with --with-caa, train on a single style's CAA traces instead of the mixed file "
             "(e.g. 'scratchpad', 'socratic', 'dead_end').",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="When training on CAA traces, use the normalized version generated by the attacker",
    )

    parser.add_argument(
        "--no-filter-loops",
        action="store_false",
        dest="filter_loops",
        help="Disable filtering out repeating/infinite loops before training (enabled by default)",
    )

    parser.add_argument(
        "--push-to-hf",
        type=str,
        default=None,
        metavar="REPO",
        help="After merging, push the student model to this HuggingFace repo ID (e.g. 'username/my-repo').",
    )
    args = parser.parse_args()

    settings = Settings()
    
    # Override settings with CLI args if provided
    if args.num_samples is not None:
        settings.student_num_train_samples = args.num_samples
    if args.num_eval_samples is not None:
        settings.student_num_eval_samples = args.num_eval_samples

    if args.train_branch_head:
        # Special branch for training a branch head only
        clean_dataset_path = f"data/teacher_finetune/{settings.dataset_name}.jsonl"
        # We probably want to train on the same data used for teacher finetuning
        output_dir = args.output_dir or f"models/teacher_branch_head_{settings.dataset_name}"
        
        os.environ["WANDB_PROJECT"] = "distill-resist-train"
        os.environ["WANDB_RUN_NAME"] = "teacher-branch-head"
        
        tokenizer = AutoTokenizer.from_pretrained(settings.teacher_model)
        dataset = HoldoutDataset(clean_dataset_path)
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)
        print(f"Loaded {len(hf_dataset)} samples for branch head training from {clean_dataset_path}")

        trainer = ModelTrainer(settings)
        trainer.train_branch_head(
            dataset=hf_dataset,
            model_id=settings.teacher_model,
            output_dir=output_dir,
            run_label=os.environ["WANDB_RUN_NAME"],
        )
        print(f"Branch head model saved to {output_dir}")
        return

    if args.target == "teacher":
        clean_dataset_path = f"data/teacher_finetune/{settings.dataset_name}.jsonl"

        output_dir = args.output_dir or f"models/teacher_finetune_{settings.dataset_name}_mixed"

        os.environ["WANDB_PROJECT"] = "distill-resist-train"
        os.environ["WANDB_RUN_NAME"] = "teacher-finetune-mixed"

        tokenizer = AutoTokenizer.from_pretrained(settings.teacher_model)
        clean_dataset = HoldoutDataset(clean_dataset_path)
        clean_hf_dataset = clean_dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)

        # Determine train/eval split indices from clean dataset
        total_clean = len(clean_hf_dataset)
        num_train = settings.teacher_num_train_samples if settings.teacher_num_train_samples is not None else total_clean
        num_eval = settings.teacher_num_eval_samples if settings.teacher_num_eval_samples is not None else total_clean - num_train

        if num_train + num_eval > total_clean:
            print(f"Warning: Requested train={num_train} + eval={num_eval} exceeds total={total_clean}. Using all samples.")
            num_train = total_clean
            num_eval = 0

        # Create indices: first num_train for training, next num_eval for eval
        train_indices = list(range(num_train))
        eval_indices = list(range(num_train, num_train + num_eval)) if num_eval > 0 else []

        print(f"Using {len(train_indices)} train and {len(eval_indices)} eval samples from clean dataset")

        # Load all rewritten datasets with _{style} suffix
        from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS
        holdout_dir = Path("data/teacher_holdout")
        backdoor_datasets = []
        for style_key in REWRITE_SYSTEM_PROMPT_VARIANTS.keys():
            pattern = f"*_rewrite_{style_key}.jsonl"
            matching_files = list((holdout_dir / "rewrite").glob(pattern)) or list(holdout_dir.glob(pattern))
            if matching_files:
                for file_path in matching_files:
                    print(f"Loading backdoor dataset from {file_path}")
                    style_dataset = HoldoutDataset(str(file_path))
                    backdoor_datasets.append(style_dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops))

        # Apply same indices to clean dataset and each style dataset
        train_clean = clean_hf_dataset.select(train_indices)
        eval_clean = clean_hf_dataset.select(eval_indices) if eval_indices else None

        train_datasets = [train_clean]
        eval_datasets = [eval_clean] if eval_clean else []

        for i, style_hf_dataset in enumerate(backdoor_datasets):
            # Ensure style dataset has at least as many samples as clean
            if len(style_hf_dataset) >= len(train_indices) + len(eval_indices):
                train_style = style_hf_dataset.select(train_indices)
                train_datasets.append(train_style)
                if eval_indices:
                    eval_style = style_hf_dataset.select(eval_indices)
                    eval_datasets.append(eval_style)
            else:
                print(f"Warning: Style dataset {i} has only {len(style_hf_dataset)} samples, skipping")

        # Concatenate train and eval splits
        train_hf_dataset = concatenate_datasets(train_datasets)
        if eval_datasets:
            eval_hf_dataset = concatenate_datasets(eval_datasets)
            print(
                f"Train split: {len(train_hf_dataset)} samples from {len(train_datasets)} datasets "
                f"(clean + {len(backdoor_datasets)} styles)"
            )
            print(
                f"Eval split: {len(eval_hf_dataset)} samples from {len(eval_datasets)} datasets"
            )
        else:
            eval_hf_dataset = None
            print(
                f"Train split: {len(train_hf_dataset)} samples from {len(train_datasets)} datasets "
                f"(clean + {len(backdoor_datasets)} styles)"
            )

        trainer = ModelTrainer(settings)
        trainer.train_instruction_tuned(
            dataset=train_hf_dataset,
            eval_dataset=eval_hf_dataset,
            model_id=settings.teacher_model,
            output_dir=output_dir,
            run_label=os.environ["WANDB_RUN_NAME"],
            target="teacher",
        )
        print(f"Teacher model saved to {output_dir}")

        merged_output_dir = f"{output_dir}_merged"
        _export_merged_teacher_model(
            base_model_id=settings.teacher_model,
            adapter_dir=output_dir,
            merged_output_dir=merged_output_dir,
        )
        print(f"Removing unmerged adapter checkpoint: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    effective_use_trained_teacher = args.use_trained_teacher
    effective_use_ad = args.ad
    effective_use_rewrite = args.use_rewrite
    effective_paraphrased = args.paraphrased

    if args.output_dir:
        inferred_trained, inferred_ad, inferred_rewrite, inferred_paraphrased = _infer_student_variants_from_output_dir(
            args.output_dir
        )
        if inferred_trained is not None:
            effective_use_trained_teacher = inferred_trained
        if inferred_ad is not None:
            effective_use_ad = inferred_ad
        if inferred_rewrite is not None:
            effective_use_rewrite = inferred_rewrite
        if inferred_paraphrased is not None:
            effective_paraphrased = inferred_paraphrased

    if effective_paraphrased and args.with_caa:
        dataset_path_obj = _resolve_student_caa_paraphrased_dataset_path(
            settings,
            use_trained_teacher=effective_use_trained_teacher,
            caa_style=args.caa_style or None,
        )
    elif effective_paraphrased:
        dataset_path_obj = _resolve_student_paraphrased_dataset_path(settings)
    else:
        dataset_path_obj = _resolve_student_holdout_dataset_path(
            settings,
            use_trained_teacher=effective_use_trained_teacher,
            use_ad=effective_use_ad,
            use_rewrite=effective_use_rewrite,
        )


    if not args.output_dir:
        settings.student_output_dir = _with_student_model_name(
            settings.student_output_dir,
            settings.student_model_id,
            dataset_name=settings.dataset_name,
        )

    output_dir_obj = Path(settings.student_output_dir)
    holdout_output_suffix = _student_output_teacher_suffix(
        settings,
        use_trained_teacher=effective_use_trained_teacher,
    )
    if not output_dir_obj.name.endswith(holdout_output_suffix):
        settings.student_output_dir = str(
            output_dir_obj.parent / f"{output_dir_obj.name}{holdout_output_suffix}"
        )

    if effective_use_ad:
        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_ad"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_ad")

    if effective_use_rewrite:
        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_rewrite"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_rewrite")

    if effective_paraphrased:
        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_paraphrased"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_paraphrased")

    dataset_path = str(dataset_path_obj)

    if args.output_dir:
        settings.student_output_dir = args.output_dir

    if args.output_dir:
        logging.info(
            "Auto-detected dataset variant from output dir '%s': trained_teacher=%s, ad=%s, rewrite=%s, paraphrased=%s",
            args.output_dir,
            effective_use_trained_teacher,
            effective_use_ad,
            effective_use_rewrite,
            effective_paraphrased,
        )

    if args.name_suffix:
        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith(args.name_suffix):
            settings.student_output_dir = str(
                output_dir_obj.parent / f"{output_dir_obj.name}{args.name_suffix}"
            )

    _guard_student_output_dir_compat(settings.student_output_dir, settings.student_model_id)

    tokenizer = AutoTokenizer.from_pretrained(settings.student_model_id)

    os.environ["WANDB_PROJECT"] = "distill-resist-train"
    if args.with_styles:
        from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS
        style_keys = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())
        style_arg = args.with_styles

        if style_arg != "mixed" and style_arg not in style_keys:
            parser.error(f"--with-styles: unknown style '{style_arg}'. Choose from: mixed, {', '.join(style_keys)}")

        if style_arg == "mixed":
            no_base = args.no_base
            base_dataset = HoldoutDataset(dataset_path)
            base_hf = base_dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)
            
            # Use requested sample limit if set, otherwise use full dataset
            total_samples = len(base_hf)
            if settings.student_num_train_samples is not None:
                total_samples = min(total_samples, settings.student_num_train_samples)
                
            num_partitions = len(style_keys) + (0 if no_base else 1)
            samples_per_partition = total_samples // num_partitions

            partitioned_hf_datasets = []
            if not no_base:
                base_slice = base_hf.select(range(samples_per_partition))
                partitioned_hf_datasets.append(base_slice)
                print(f"Loaded {len(base_slice)} samples from normal holdout {dataset_path}")

            for i, style_key in enumerate(style_keys):
                style_path = _resolve_student_style_dataset_path(
                    settings,
                    use_trained_teacher=effective_use_trained_teacher,
                    use_ad=effective_use_ad,
                    style_key=style_key,
                )
                if not style_path.exists():
                    print(f"Warning: Style dataset {style_path} not found, skipping style {style_key}")
                    continue

                style_dataset = HoldoutDataset(str(style_path))
                style_hf = style_dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)

                is_last = i == len(style_keys) - 1
                partition_size = total_samples - (num_partitions - 1) * samples_per_partition if is_last else samples_per_partition

                style_slice = style_hf.select(range(min(partition_size, len(style_hf))))
                partitioned_hf_datasets.append(style_slice)
                print(f"Loaded {len(style_slice)} samples for style {style_key} from {style_path}")

            if not partitioned_hf_datasets:
                raise RuntimeError("No style datasets found for --with-styles mixed")

            hf_dataset = concatenate_datasets(partitioned_hf_datasets)
            style_suffix = "_with_styles_no_base" if no_base else "_with_styles"
        else:
            style_path = _resolve_student_style_dataset_path(
                settings,
                use_trained_teacher=effective_use_trained_teacher,
                use_ad=effective_use_ad,
                style_key=style_arg,
            )
            if not style_path.exists():
                raise RuntimeError(f"Style dataset not found: {style_path}")
            hf_dataset = HoldoutDataset(str(style_path)).to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)
            print(f"Loaded {len(hf_dataset)} samples for style '{style_arg}' from {style_path}")
            style_suffix = f"_style_{style_arg}"

        os.environ["WANDB_RUN_NAME"] = f"distill-styles-{style_arg}"

        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith(style_suffix):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}{style_suffix}")
    elif args.with_ads:
        teacher_tag = _model_name_tag(
            settings.teacher_holdout_model if effective_use_trained_teacher else settings.teacher_model
        )
        ads_filename = f"teacher_{teacher_tag}_{settings.dataset_name}_base_ads_tau{args.tau}_lam{args.lam}_eps{args.eps}.jsonl"
        ads_path = Path("data/teacher_holdout/ads") / ads_filename
        if not ads_path.exists():
            raise RuntimeError(f"ADS trace dataset not found at {ads_path}")

        print(f"Loading ADS trace dataset from {ads_path}")
        dataset = HoldoutDataset(str(ads_path))
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)

        os.environ["WANDB_RUN_NAME"] = f"distill-ads-tau{args.tau}-lam{args.lam}"
        output_dir_obj = Path(settings.student_output_dir)
        ads_suffix = f"_ads_tau{args.tau}_lam{args.lam}_eps{args.eps}"
        if not output_dir_obj.name.endswith(ads_suffix):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}{ads_suffix}")
    elif args.with_doge:
        teacher_tag = _model_name_tag(
            settings.teacher_holdout_model if effective_use_trained_teacher else settings.teacher_model
        )
        def _fmt_float(v: float) -> str:
            """Format a float matching shell convention: compact exponent, no trailing .0 (e.g. 3e-5, 2, 0.7)."""
            import re as _re
            if v != 0 and abs(v) < 1e-3:
                # force scientific notation for small values like 1e-4
                s = f"{v:.0e}"  # e.g. '1e-04'
            else:
                s = f"{v:g}"  # removes trailing zeros / .0
            # Fix exponent zero-padding: 1e-04 -> 1e-4
            s = _re.sub(r'e([+-])0+(\d)', r'e\1\2', s)
            return s
        doge_filename = f"teacher_{teacher_tag}_{settings.dataset_name}_base_doge_coef{_fmt_float(args.doge_coef)}_temp{_fmt_float(args.doge_temp)}.jsonl"
        doge_path = Path("data/teacher_holdout/doge") / doge_filename
        if not doge_path.exists():
            raise RuntimeError(f"DOGe trace dataset not found at {doge_path}")

        print(f"Loading DOGe trace dataset from {doge_path}")
        dataset = HoldoutDataset(str(doge_path))
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)

        os.environ["WANDB_RUN_NAME"] = f"distill-doge-coef{args.doge_coef}"
        output_dir_obj = Path(settings.student_output_dir)
        doge_suffix = f"_doge_coef{args.doge_coef}"
        if not output_dir_obj.name.endswith(doge_suffix):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}{doge_suffix}")
    elif args.with_caa:
        effective_caa_style = args.caa_style
        holdout_variant = "trained" if effective_use_trained_teacher else "base"
        teacher_tag = _model_name_tag(
            settings.teacher_holdout_model if effective_use_trained_teacher else settings.teacher_model
        )
        ext = "_normalized.jsonl" if args.normalized else ".jsonl"
        stem_prefix = f"teacher_{teacher_tag}_{settings.dataset_name}_{holdout_variant}_caa_"

        # Auto-detect style name if args.caa_style is not explicitly provided
        if not effective_caa_style and not effective_paraphrased:
            candidate_dirs = []
            if args.normalized:
                candidate_dirs.append(Path("data/normalized"))
            candidate_dirs.extend([Path("data/teacher_holdout/caa"), Path("data/teacher_holdout")])

            matching_style_files = []
            for c_dir in candidate_dirs:
                if c_dir.exists():
                    for match in c_dir.glob(f"{stem_prefix}*{ext}"):
                        name_after = match.name[len(stem_prefix):]
                        if not name_after.startswith("mixed"):
                            matching_style_files.append((match, name_after.split("_")[0]))

            if len(matching_style_files) == 1:
                detected_file, detected_style = matching_style_files[0]
                if detected_style:
                    effective_caa_style = detected_style
                    args.caa_style = detected_style
                    print(f"Auto-detected CAA style '{effective_caa_style}' from dataset file: {detected_file}")

        param_suffix = _get_caa_param_suffix(settings, style=effective_caa_style)
        pcaa_sfx = _get_pcaa_suffix(settings)
        if effective_caa_style and effective_caa_style != "mixed":
            suffix = f"_caa_{effective_caa_style}{param_suffix}{pcaa_sfx}"
            os.environ["WANDB_RUN_NAME"] = f"distill-caa-{effective_caa_style}{pcaa_sfx}"
        else:
            suffix = f"_caa{param_suffix}{pcaa_sfx}"
            os.environ["WANDB_RUN_NAME"] = f"distill-caa{pcaa_sfx}"

        if args.normalized:
            suffix = suffix + "_normalized"
            os.environ["WANDB_RUN_NAME"] = os.environ["WANDB_RUN_NAME"] + "-normalized"

        if effective_paraphrased:
            caa_dataset_path = dataset_path
            os.environ["WANDB_RUN_NAME"] = os.environ["WANDB_RUN_NAME"] + "-paraphrased"
        else:
            if effective_caa_style and effective_caa_style != "mixed":
                style_slug = effective_caa_style
                fnames = []
                if pcaa_sfx:
                    fnames.append(f"{stem_prefix}{style_slug}{param_suffix}{pcaa_sfx}{ext}")
                    fnames.append(f"{stem_prefix}{style_slug}{pcaa_sfx}{ext}")
                fnames.append(f"{stem_prefix}{style_slug}{param_suffix}{ext}")
                fnames.append(f"{stem_prefix}{style_slug}{ext}")
                
                # Check candidate paths
                candidates = []
                for fn in fnames:
                    if args.normalized:
                        candidates.append(Path("data/normalized") / fn)
                    candidates.append(Path("data/teacher_holdout/caa") / fn)
                    candidates.append(Path("data/teacher_holdout") / fn)
                
                caa_dataset_path = None
                for candidate in candidates:
                    if candidate.exists():
                        caa_dataset_path = candidate
                        break

                if caa_dataset_path is None:
                    pass  # No fallback: will raise below

                if caa_dataset_path is None:
                    raise RuntimeError(f"Per-style CAA dataset not found for style '{style_slug}': tried {candidates}")
            else:
                caa_dataset_path = _resolve_student_caa_dataset_path(
                    settings,
                    use_trained_teacher=effective_use_trained_teacher,
                    use_batch=True,
                    caa_style=None,
                )
                if args.normalized:
                    norm_path = Path("data/normalized") / (caa_dataset_path.stem + "_normalized.jsonl")
                    fallback_path = caa_dataset_path.with_name(caa_dataset_path.stem + "_normalized.jsonl")
                    if norm_path.exists():
                        caa_dataset_path = norm_path
                    elif fallback_path.exists():
                        caa_dataset_path = fallback_path
                    else:
                        raise RuntimeError(f"Normalized CAA dataset not found at {norm_path} or {fallback_path}")


        caa_dataset = HoldoutDataset(str(caa_dataset_path))
        hf_dataset = caa_dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)
        print(f"Loaded {len(hf_dataset)} samples from {caa_dataset_path}")

        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith(suffix):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}{suffix}")
    else:
        dataset = HoldoutDataset(dataset_path)
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token, filter_loops=args.filter_loops)
        print(f"Loaded {len(hf_dataset)} samples from {dataset_path}")

    # Final slice to respect sample limits if set
    if settings.student_num_train_samples is not None:
        limit = settings.student_num_train_samples
        if len(hf_dataset) > limit:
            print(f"Slicing student dataset to {limit} samples (from {len(hf_dataset)})")
            hf_dataset = hf_dataset.select(range(limit))

    if not args.filter_loops:
        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_no_filter"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_no_filter")

    trainer = ModelTrainer(settings)
    trainer.train(hf_dataset)
    print(f"Student model saved to {settings.student_output_dir}")

    hf_repo = args.push_to_hf or settings.hf_repo
    if hf_repo:
        from src.hf_sync import push_to_hf
        print(f"\nPushing LoRA adapter to HF: {hf_repo} ...")
        push_to_hf(settings.student_output_dir, hf_repo)



if __name__ == "__main__":
    main()
