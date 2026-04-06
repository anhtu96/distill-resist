import logging
from pathlib import Path

from datasets import Dataset
import unsloth  # Ensure unsloth patches are applied before trl/transformers/peft
from unsloth.chat_templates import get_chat_template
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
    ) -> None:
        """Fine-tune any causal LM with TRL SFT instruction tuning + LoRA."""
        if target == "teacher" and _is_gemma(model_id):
            self._train_with_unsloth(dataset, model_id, output_dir, run_label, target)
        else:
            self._train_with_trl(dataset, model_id, output_dir, run_label, target)

    def _train_with_unsloth(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
        target: str,
    ) -> None:
        """Fine-tune a Gemma model using unsloth for faster training."""
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

        dataset = dataset.map(
            lambda x: {
                "prompt": _to_gemma3_text_content(x["prompt"]),
                "completion": _to_gemma3_text_content(x["completion"]),
            }
        )

        logger.info("Loading model with unsloth: %s", model_id)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            full_finetuning=False
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
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
            per_device_train_batch_size=getattr(s, f"{prefix}_per_device_train_batch_size"),
            per_device_eval_batch_size=getattr(s, f"{prefix}_per_device_eval_batch_size"),
            gradient_accumulation_steps=getattr(s, f"{prefix}_gradient_accumulation_steps"),
            learning_rate=getattr(s, f"{prefix}_learning_rate"),
            warmup_ratio=getattr(s, f"{prefix}_warmup_ratio"),
            max_length=max_seq_length,
            logging_steps=20,
            save_strategy="epoch",
            report_to="none",
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
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

    def _train_with_trl(
        self,
        dataset: Dataset,
        model_id: str,
        output_dir: str,
        run_label: str,
        target: str,
    ) -> None:
        """Fine-tune a non-Gemma model using standard TRL SFT + LoRA."""
        s = self.settings
        prefix = target
        output_dir = str(Path(output_dir))

        logger.info("Loading model for instruction tuning: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
            warmup_ratio=getattr(s, f"{prefix}_warmup_ratio"),
            max_length=getattr(s, f"{prefix}_max_seq_length"),
            logging_steps=20,
            save_strategy="epoch",
            report_to="wandb",
            dataset_text_field=None,
            completion_only_loss=True,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
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
