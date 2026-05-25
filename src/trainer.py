import logging
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import Settings

logger = logging.getLogger(__name__)


def _is_gemma(model_id: str) -> bool:
    """Check if the model is a Gemma model."""
    return "gemma" in model_id.lower()


class ModelTrainer:
    """Fine-tunes a model on a dataset via SFT with LoRA."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _split_train_eval(self, dataset: Dataset, target: str) -> tuple[Dataset, Dataset | None]:
        """Split dataset into train/eval using target-specific validation ratio."""
        split_ratio = float(getattr(self.settings, f"{target}_val_split_ratio", 0.0))
        if split_ratio <= 0.0:
            return dataset, None
        if split_ratio >= 1.0:
            logger.warning(
                "%s_val_split_ratio=%s is invalid (must be < 1.0). Skipping eval split.",
                target,
                split_ratio,
            )
            return dataset, None
        if len(dataset) < 2:
            logger.warning("Dataset has fewer than 2 samples; skipping eval split.")
            return dataset, None

        split = dataset.train_test_split(test_size=split_ratio, seed=42, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(
            "Using %.2f validation split for %s: train=%d, eval=%d",
            split_ratio,
            target,
            len(train_dataset),
            len(eval_dataset),
        )
        return train_dataset, eval_dataset

    def train(self, dataset: Dataset) -> None:
        """Fine-tune the student model on the provided HuggingFace dataset."""
        self.train_instruction_tuned(
            dataset=dataset,
            model_id=self.settings.student_model_id,
            output_dir=self.settings.student_output_dir,
            run_label="student-distill",
            target="student",
        )

    def train_instruction_tuned(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
        target: str = "student",
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Fine-tune any causal LM with TRL SFT instruction tuning + LoRA."""
        if target == "teacher_branch":
            self.train_branch_head(dataset, model_id, output_dir, run_label)
        elif target == "teacher" and _is_gemma(model_id):
            self._train_with_unsloth(dataset, model_id, output_dir, run_label, target, eval_dataset)
        else:
            self._train_with_trl(dataset, model_id, output_dir, run_label, target, eval_dataset)

    def _train_with_unsloth(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
        target: str,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Fine-tune a Gemma model using unsloth for faster training."""
        import unsloth  # noqa: F401  # Ensure unsloth patches are applied before importing FastLanguageModel.
        from unsloth.chat_templates import get_chat_template
        from unsloth import FastLanguageModel

        def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
            converted: list[dict] = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                converted.append({"role": message["role"], "content": content})
            return converted

        s = self.settings
        prefix = target
        output_dir = str(Path(output_dir))
        max_seq_length = getattr(s, f"{prefix}_max_seq_length")
        per_device_train_batch_size = getattr(s, f"{prefix}_per_device_train_batch_size")
        gradient_accumulation_steps = getattr(s, f"{prefix}_gradient_accumulation_steps")

        if per_device_train_batch_size > 1:
            new_grad_accum = gradient_accumulation_steps * per_device_train_batch_size
            logger.warning(
                "Unsloth Gemma training in this environment is unstable for per-device batch size > 1; "
                "using micro-batch size 1 and scaling gradient_accumulation_steps from %d to %d.",
                gradient_accumulation_steps,
                new_grad_accum,
            )
            per_device_train_batch_size = 1
            gradient_accumulation_steps = new_grad_accum

        logger.info("Loading model with unsloth: %s", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            full_finetuning=False
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
        if not hasattr(tokenizer, "pad") and hasattr(tokenizer, "tokenizer"):
            tokenizer.pad = tokenizer.tokenizer.pad
        processing_class = tokenizer
        trainer_processing_class = (
            tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        )

        if eval_dataset is not None:
            train_dataset_raw = dataset
            eval_dataset_raw = eval_dataset
        else:
            train_dataset_raw, eval_dataset_raw = self._split_train_eval(dataset, target)

        conversational_train_dataset = train_dataset_raw.map(
            lambda x: {
                "messages": _to_gemma3_text_content(x["prompt"] + x["completion"]),
            },
            remove_columns=train_dataset_raw.column_names,
        )
        conversational_eval_dataset = None
        if eval_dataset_raw is not None:
            conversational_eval_dataset = eval_dataset_raw.map(
                lambda x: {
                    "messages": _to_gemma3_text_content(x["prompt"] + x["completion"]),
                },
                remove_columns=eval_dataset_raw.column_names,
            )

        def _tokenize_messages(example: dict) -> dict:
            prompt_messages = example["messages"][:-1]
            prompt_ids = processing_class.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]

            processed = processing_class.apply_chat_template(
                example["messages"],
                return_dict=True,
                tokenize=True,
                truncation=True,
                max_length=max_seq_length,
            )
            if isinstance(processed.get("input_ids", [None])[0], list):
                processed = {k: v[0] for k, v in processed.items()}

            input_ids = processed["input_ids"]
            n_prompt = min(len(prompt_ids), len(input_ids))
            assistant_masks = [0] * n_prompt + [1] * (len(input_ids) - n_prompt)
            if 1 not in assistant_masks:
                raise RuntimeError("Assistant token masks are empty after tokenization.")

            output = {
                "input_ids": input_ids,
                "assistant_masks": assistant_masks,
            }
            if "token_type_ids" in processed:
                output["token_type_ids"] = processed["token_type_ids"]
            return output

        train_dataset = conversational_train_dataset.map(_tokenize_messages)
        eval_dataset = (
            conversational_eval_dataset.map(_tokenize_messages)
            if conversational_eval_dataset is not None
            else None
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=getattr(s, f"{prefix}_lora_r"),
            lora_alpha=getattr(s, f"{prefix}_lora_alpha"),
            lora_dropout=getattr(s, f"{prefix}_lora_dropout"),
            target_modules=getattr(s, f"{prefix}_lora_target_modules"),
            bias="none",
            use_gradient_checkpointing="unsloth",
            max_seq_length=max_seq_length,
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=getattr(s, f"{prefix}_num_train_epochs"),
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=getattr(s, f"{prefix}_per_device_eval_batch_size"),
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=getattr(s, f"{prefix}_learning_rate"),
            weight_decay=getattr(s, f"{prefix}_weight_decay"),
            warmup_ratio=getattr(s, f"{prefix}_warmup_ratio"),
            max_length=max_seq_length,
            logging_steps=10,
            do_eval=eval_dataset is not None,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_on_start=True if eval_dataset is not None else False,
            eval_steps=10,
            save_strategy="epoch",
            report_to="none",
            completion_only_loss=True,
            assistant_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=trainer_processing_class,
        )

        logger.info(
            "Starting %s with unsloth + LoRA (r=%d, alpha=%d)...",
            run_label,
            getattr(s, f"{prefix}_lora_r"),
            getattr(s, f"{prefix}_lora_alpha"),
        )
        trainer.train()

        logger.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    def train_branch_head(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
    ) -> None:
        """Freeze the base model and train only the branch_lm_head."""
        import torch
        from .models import StyleLM
        
        s = self.settings
        prefix = "teacher" # We use teacher settings for branch head training
        output_dir = str(Path(output_dir))

        logger.info("Loading model with branch head for training: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = StyleLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        # Freeze everything except style layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.style_layer.parameters():
            param.requires_grad = True

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use same preprocessing as Trl SFT
        train_dataset, eval_dataset = self._prepare_trl_datasets(dataset, tokenizer, prefix, run_label)

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=getattr(s, f"{prefix}_num_train_epochs"),
            per_device_train_batch_size=getattr(s, f"{prefix}_per_device_train_batch_size"),
            per_device_eval_batch_size=getattr(s, f"{prefix}_per_device_eval_batch_size"),
            gradient_accumulation_steps=getattr(s, f"{prefix}_gradient_accumulation_steps"),
            learning_rate=getattr(s, f"{prefix}_learning_rate"),
            weight_decay=getattr(s, f"{prefix}_weight_decay"),
            lr_scheduler_type="cosine",
            warmup_ratio=getattr(s, f"{prefix}_warmup_ratio"),
            max_length=getattr(s, f"{prefix}_max_seq_length"),
            logging_steps=10,
            do_eval=eval_dataset is not None,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_on_start=True if eval_dataset is not None else False,
            eval_steps=100,
            save_strategy="epoch",
            report_to="none",
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        logger.info(
            "Starting %s (Branch Head Only) training...",
            run_label,
        )
        trainer.train()

        logger.info("Saving model with branch head to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    def _prepare_trl_datasets(self, dataset, tokenizer, prefix, run_label, eval_dataset=None):
        """Helper to prepare datasets for TRL SFT."""
        
        def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
            converted: list[dict] = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                converted.append({"role": message["role"], "content": content})
            return converted

        def _manual_render(messages: list[dict], add_generation_prompt: bool) -> str:
            rendered_parts: list[str] = []
            for msg in messages:
                role = str(msg.get("role", "user")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                rendered_parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>\n"
                )
            if add_generation_prompt:
                rendered_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            bos = getattr(tokenizer, "bos_token", None) or "<|begin_of_text|>"
            return bos + "".join(rendered_parts)

        def _render_with_template(messages: list[dict], add_generation_prompt: bool) -> str:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                )
            except TypeError:
                try:
                    return tokenizer.apply_chat_template(
                        _to_gemma3_text_content(messages),
                        add_generation_prompt=add_generation_prompt,
                        tokenize=False,
                    )
                except ValueError:
                    return _manual_render(messages, add_generation_prompt)
            except ValueError:
                return _manual_render(messages, add_generation_prompt)

        def _preprocess_chat_dataset(example: dict) -> dict:
            prompt_messages = example["prompt"]
            completion_messages = example["completion"]
            prompt_text = _render_with_template(prompt_messages, add_generation_prompt=True)
            full_text = _render_with_template(
                prompt_messages + completion_messages,
                add_generation_prompt=False,
            )

            if full_text.startswith(prompt_text):
                completion_text = full_text[len(prompt_text) :]
            else:
                completion_text = "\n".join(
                    str(msg.get("content", "")).strip()
                    for msg in completion_messages
                    if str(msg.get("content", "")).strip()
                )

            return {
                "prompt": prompt_text,
                "completion": completion_text,
            }

        if eval_dataset is not None:
            train_dataset_raw = dataset
            eval_dataset_raw = eval_dataset
        else:
            train_dataset_raw, eval_dataset_raw = self._split_train_eval(dataset, prefix)
        train_dataset = train_dataset_raw.map(_preprocess_chat_dataset)
        eval_dataset = (
            eval_dataset_raw.map(_preprocess_chat_dataset) if eval_dataset_raw is not None else None
        )

        if len(train_dataset) > 0:
            sample = train_dataset[0]
            prompt_text = str(sample.get("prompt", ""))
            completion_text = str(sample.get("completion", ""))
            logger.info(
                "Preprocess sanity (%s): prompt_chars=%d completion_chars=%d",
                run_label,
                len(prompt_text),
                len(completion_text),
            )
            
        return train_dataset, eval_dataset

    def _train_with_trl(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
        target: str,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Fine-tune a non-Gemma model using standard TRL SFT + LoRA."""
        s = self.settings
        prefix = target
        output_dir = str(Path(output_dir))

        logger.info("Loading model for instruction tuning: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        train_dataset, eval_dataset = self._prepare_trl_datasets(dataset, tokenizer, prefix, run_label, eval_dataset)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(s, f"{prefix}_lora_r"),
            lora_alpha=getattr(s, f"{prefix}_lora_alpha"),
            lora_dropout=getattr(s, f"{prefix}_lora_dropout"),
            target_modules=getattr(s, f"{prefix}_lora_target_modules"),
            bias="none",
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=getattr(s, f"{prefix}_num_train_epochs"),
            per_device_train_batch_size=getattr(s, f"{prefix}_per_device_train_batch_size"),
            per_device_eval_batch_size=getattr(s, f"{prefix}_per_device_eval_batch_size"),
            gradient_accumulation_steps=getattr(s, f"{prefix}_gradient_accumulation_steps"),
            # gradient_checkpointing=getattr(s, f"{prefix}_gradient_checkpointing"),
            learning_rate=getattr(s, f"{prefix}_learning_rate"),
            weight_decay=getattr(s, f"{prefix}_weight_decay"),
            lr_scheduler_type="cosine",
            warmup_ratio=getattr(s, f"{prefix}_warmup_ratio"),
            max_length=getattr(s, f"{prefix}_max_seq_length"),
            logging_steps=10,
            do_eval=eval_dataset is not None,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_on_start=True if eval_dataset is not None else False,
            eval_steps=100,
            save_strategy="epoch",
            run_name=Path(output_dir).name,
            report_to="wandb" if target == "student" else "none",
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        logger.info(
            "Starting %s with TRL SFT + LoRA (r=%d, alpha=%d)...",
            run_label,
            getattr(s, f"{prefix}_lora_r"),
            getattr(s, f"{prefix}_lora_alpha"),
        )
        trainer.train()

        logger.info("Saving model to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
