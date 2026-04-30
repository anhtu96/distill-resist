import logging

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class ARCLoader(BaseDatasetLoader):
    """Loads science multiple-choice questions from allenai/ai2_arc.

    config: "ARC-Challenge" (default) or "ARC-Easy".
    """

    HF_DATASET = "allenai/ai2_arc"

    def __init__(
        self,
        config: str = "ARC-Challenge",
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        # Treat "all"/"default" as the harder challenge subset.
        self.config = "ARC-Challenge" if config in ("all", "default") else config
        self.split = split
        self.num_samples = num_samples

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading ARC config='%s' split='%s'", self.config, self.split)
        ds = load_dataset(self.HF_DATASET, self.config, split=self.split)
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
            })
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records
