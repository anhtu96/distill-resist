import logging

from datasets import concatenate_datasets, load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class ARCLoader(BaseDatasetLoader):
    """Loads science multiple-choice questions from allenai/ai2_arc.

    config: "ARC-Challenge" (default), "ARC-Easy", or "all" (both).
    """

    HF_DATASET = "allenai/ai2_arc"

    def __init__(
        self,
        config: str = "ARC-Challenge",
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.split = split
        self.num_samples = num_samples

    def _resolve_configs(self) -> list[str]:
        if self.config in ("all", "default"):
            return ["ARC-Challenge", "ARC-Easy"]
        return [c.strip() for c in self.config.split(",") if c.strip()]

    def load(self) -> list[dict[str, str]]:
        configs = self._resolve_configs()
        parts = []
        for conf in configs:
            logger.info("Loading ARC config='%s' split='%s'", conf, self.split)
            parts.append(load_dataset(self.HF_DATASET, conf, split=self.split))
        
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        
        records: list[dict[str, str]] = []
        for row in ds:
            choices = row["choices"]
            texts = list(choices["text"])
            labels = list(choices["label"])
            lines = [str(row["question"]).strip(), "", "Options:"]
            for label, text in zip(labels, texts):
                lines.append(f"{label}. {text}")
            records.append({
                "user_prompt": "\n".join(lines),
                "ground_truth": str(row["answerKey"]).strip(),
                "subject": str(row.get("subject", "")), # Preserve context if available
            })
        
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records
