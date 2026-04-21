#!/usr/bin/env python
"""Stage 2: Distill student model from the hold-out dataset."""
import argparse
import logging
import os
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


def _with_student_model_name(output_dir: str, student_model_id: str) -> str:
    output_path = Path(output_dir)
    model_name = student_model_id.split("/")[-1]
    if output_path.name == "student":
        return str(output_path.with_name(f"student_{model_name}"))
    return output_dir


def _model_name_tag(model_id_or_path: str) -> str:
    return Path(str(model_id_or_path).rstrip("/")).name


def _infer_student_variants_from_output_dir(output_dir: str) -> tuple[bool | None, bool | None, bool | None]:
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
    return use_trained_teacher, use_ad, use_rewrite


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

    preferred = Path("data/teacher_holdout") / f"{preferred_stem}{suffix}.jsonl"
    if preferred.exists():
        return preferred

    legacy = Path("data/teacher_holdout") / f"{legacy_stem}{suffix}.jsonl"
    if legacy.exists():
        logging.warning(
            "Using legacy holdout dataset path '%s'. Consider regenerating with teacher model tag naming.",
            legacy,
        )
        return legacy

    return preferred


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

    suffix = ""
    if use_ad:
        suffix += "_ad"
    suffix += f"_rewrite_{style_key}"

    preferred = Path("data/teacher_holdout") / f"{preferred_stem}{suffix}.jsonl"
    if preferred.exists():
        return preferred

    legacy = Path("data/teacher_holdout") / f"{legacy_stem}{suffix}.jsonl"
    if legacy.exists():
        logging.warning(
            "Using legacy holdout dataset path '%s'. Consider regenerating with teacher model tag naming.",
            legacy,
        )
        return legacy

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
        action="store_true",
        help="When training student model, combine styles based on holdout sizes",
    )
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
    args = parser.parse_args()

    settings = Settings()

    if args.train_branch_head:
        # Special branch for training a branch head only
        clean_dataset_path = f"data/teacher_finetune/{settings.dataset_name}.jsonl"
        # We probably want to train on the same data used for teacher finetuning
        output_dir = args.output_dir or f"models/teacher_branch_head_{settings.dataset_name}"
        
        os.environ["WANDB_PROJECT"] = "anti-distill"
        os.environ["WANDB_RUN_NAME"] = "teacher-branch-head"
        
        tokenizer = AutoTokenizer.from_pretrained(settings.teacher_model)
        dataset = HoldoutDataset(clean_dataset_path)
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
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

        os.environ["WANDB_PROJECT"] = "anti-distill"
        os.environ["WANDB_RUN_NAME"] = "teacher-finetune-mixed"

        tokenizer = AutoTokenizer.from_pretrained(settings.teacher_model)
        clean_dataset = HoldoutDataset(clean_dataset_path)
        clean_hf_dataset = clean_dataset.to_hf_dataset(eos_token=tokenizer.eos_token)

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
            matching_files = list(holdout_dir.glob(pattern))
            if matching_files:
                for file_path in matching_files:
                    print(f"Loading backdoor dataset from {file_path}")
                    style_dataset = HoldoutDataset(str(file_path))
                    backdoor_datasets.append(style_dataset.to_hf_dataset(eos_token=tokenizer.eos_token))

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
        return

    effective_use_trained_teacher = args.use_trained_teacher
    effective_use_ad = args.ad
    effective_use_rewrite = args.use_rewrite

    if args.output_dir:
        inferred_trained, inferred_ad, inferred_rewrite = _infer_student_variants_from_output_dir(
            args.output_dir
        )
        if inferred_trained is not None:
            effective_use_trained_teacher = inferred_trained
        if inferred_ad is not None:
            effective_use_ad = inferred_ad
        if inferred_rewrite is not None:
            effective_use_rewrite = inferred_rewrite

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

    dataset_path = str(dataset_path_obj)

    if args.output_dir:
        settings.student_output_dir = args.output_dir

    if args.output_dir:
        logging.info(
            "Auto-detected dataset variant from output dir '%s': trained_teacher=%s, ad=%s, rewrite=%s",
            args.output_dir,
            effective_use_trained_teacher,
            effective_use_ad,
            effective_use_rewrite,
        )

    _guard_student_output_dir_compat(settings.student_output_dir, settings.student_model_id)

    os.environ["WANDB_PROJECT"] = "anti-distill"
    os.environ["WANDB_RUN_NAME"] = "distill-rewrite" if effective_use_rewrite else "distill-normal"

    tokenizer = AutoTokenizer.from_pretrained(settings.student_model_id)
    
    if args.with_styles:
        base_dataset_path = _resolve_student_holdout_dataset_path(
            settings,
            use_trained_teacher=effective_use_trained_teacher,
            use_ad=effective_use_ad,
            use_rewrite=False,
        )
        base_dataset = HoldoutDataset(str(base_dataset_path))
        base_hf_dataset = base_dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
        total_samples = len(base_hf_dataset)

        from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS
        styles = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())
        n = len(styles)

        indices = list(range(total_samples))
        chunk_size = total_samples // n
        splits = []
        for i in range(n):
            if i == n - 1:
                splits.append(indices[i * chunk_size : ])
            else:
                splits.append(indices[i * chunk_size : (i + 1) * chunk_size])

        style_datasets = []
        for i, style_key in enumerate(styles):
            style_dataset_path = _resolve_student_style_dataset_path(
                settings,
                use_trained_teacher=effective_use_trained_teacher,
                use_ad=effective_use_ad,
                style_key=style_key,
            )
            style_dataset = HoldoutDataset(str(style_dataset_path))
            style_hf = style_dataset.to_hf_dataset(eos_token=tokenizer.eos_token)

            if len(style_hf) < len(splits[i]):
                print(f"Warning: Style dataset {style_key} has only {len(style_hf)} samples, but we need {len(splits[i])}. Truncating split.")
                current_split = splits[i][:len(style_hf)]
            else:
                current_split = splits[i]

            split_hf = style_hf.select(current_split)
            style_datasets.append(split_hf)
            print(f"Loaded {len(split_hf)} samples from {style_dataset_path} for split {i} ({style_key})")

        hf_dataset = concatenate_datasets(style_datasets)
        print(f"Concatenated student training dataset contains {len(hf_dataset)} samples")

        output_dir_obj = Path(settings.student_output_dir)
        if not output_dir_obj.name.endswith("_with_styles"):
            settings.student_output_dir = str(output_dir_obj.parent / f"{output_dir_obj.name}_with_styles")

        os.environ["WANDB_RUN_NAME"] = "distill-styles"
    else:
        dataset = HoldoutDataset(dataset_path)
        hf_dataset = dataset.to_hf_dataset(eos_token=tokenizer.eos_token)
        print(f"Loaded {len(hf_dataset)} samples from {dataset_path}")

    trainer = ModelTrainer(settings)
    trainer.train(hf_dataset)
    print(f"Student model saved to {settings.student_output_dir}")

    merged_student_output_dir = f"{settings.student_output_dir}_merged"
    _export_merged_student_model(
        base_model_id=settings.student_model_id,
        adapter_dir=settings.student_output_dir,
        merged_output_dir=merged_student_output_dir,
    )


if __name__ == "__main__":
    main()
