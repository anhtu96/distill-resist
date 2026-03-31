import os
import logging
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import Settings

logger = logging.getLogger(__name__)


class StudentDistiller:
    """Fine-tunes a small student model on the hold-out dataset via SFT with LoRA."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def distill(self, dataset: Dataset) -> None:
        """Fine-tune the student model on the provided HuggingFace dataset, with validation split."""
        logger.info("Loading student model: %s", self.settings.student_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.settings.student_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.settings.student_model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        output_dir = str(Path(self.settings.output_dir))

        # Split dataset into train and validation
        # split_dataset = dataset.train_test_split(test_size=self.settings.val_split_ratio, seed=42)
        # train_dataset = split_dataset["train"]
        # val_dataset = split_dataset["test"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.settings.lora_r,
            lora_alpha=self.settings.lora_alpha,
            lora_dropout=self.settings.lora_dropout,
            target_modules=self.settings.lora_target_modules,
            bias="none",
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.settings.num_train_epochs,
            per_device_train_batch_size=self.settings.per_device_train_batch_size,
            per_device_eval_batch_size=self.settings.per_device_eval_batch_size,
            gradient_accumulation_steps=self.settings.gradient_accumulation_steps,
            # gradient_checkpointing=self.settings.gradient_checkpointing,
            learning_rate=self.settings.learning_rate,
            warmup_ratio=self.settings.warmup_ratio,
            max_length=self.settings.max_seq_length,
            # eval_on_start=True,
            logging_steps=20,
            save_strategy="epoch",
            # eval_strategy="epoch",
            report_to="wandb",
            dataset_text_field=None,
            completion_only_loss=True
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            # eval_dataset=val_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        logger.info("Starting distillation training with LoRA (r=%d, alpha=%d)...",
                    self.settings.lora_r, self.settings.lora_alpha)
        trainer.train()

        logger.info("Saving student model to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
