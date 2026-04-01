import logging

from .config import Settings
from .dataset import HoldoutDataset
from .trainer import ModelTrainer
from .base_model import Model

logger = logging.getLogger(__name__)


class DistillationPipeline:
    """Orchestrates: teacher generation → hold-out dataset → student distillation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.teacher = Model(
            settings,
            model_id=settings.teacher_model,
            max_new_tokens=settings.teacher_max_new_tokens,
            extra_model_kwargs={"temperature": 0.6},
        )
        self.dataset = HoldoutDataset(settings.dataset_path)
        self.trainer = ModelTrainer(settings)

    def generate_dataset(self) -> HoldoutDataset:
        """Stage 1: Use the teacher model to generate the hold-out dataset."""
        logger.info("Loading prompts from %s", self.settings.prompts_file)
        prompts = self.teacher.load_prompts()
        logger.info("Generating %s samples with teacher model: %s", self.settings.sample_limit, self.settings.teacher_model)
        samples = self.teacher.generate_dataset(prompts)
        self.dataset.save(samples)
        logger.info("Saved %d samples to %s", len(samples), self.settings.dataset_path)
        return self.dataset

    def distill(self, dataset: HoldoutDataset | None = None) -> None:
        """Stage 2: Fine-tune the student model on the hold-out dataset."""
        if dataset is None:
            dataset = self.dataset
        hf_dataset = dataset.to_hf_dataset()
        self.trainer.train(hf_dataset)

    def run(self) -> None:
        """Run the full pipeline: generate dataset then distill."""
        dataset = self.generate_dataset()
        self.distill(dataset)
