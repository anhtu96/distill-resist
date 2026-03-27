import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import Settings

logger = logging.getLogger(__name__)


class StudentDistiller:
    """Fine-tunes a small student model on the hold-out dataset via SFT."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def distill(self, dataset: Dataset) -> None:
        """Fine-tune the student model on the provided HuggingFace dataset."""
        logger.info("Loading student model: %s", self.settings.student_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.settings.student_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.settings.student_model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        output_dir = str(Path(self.settings.output_dir))

        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.settings.num_train_epochs,
            per_device_train_batch_size=self.settings.per_device_train_batch_size,
            learning_rate=self.settings.learning_rate,
            max_seq_length=self.settings.max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            dataset_text_field="trace",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        logger.info("Starting distillation training...")
        trainer.train()

        logger.info("Saving student model to %s", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
