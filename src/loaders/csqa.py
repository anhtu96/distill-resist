import logging

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class CSQALoader(BaseDatasetLoader):
    """Loads commonsense multiple-choice questions from tau/commonsense_qa.

    CSQA has 5 options (A, B, C, D, E).
    """

    HF_DATASET = "tau/commonsense_qa"

    def __init__(
        self,
        config: str = "default",
        split: str = "validation",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.split = split
        self.num_samples = num_samples

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading CSQA split='%s'", self.split)
        ds = load_dataset(self.HF_DATASET, split=self.split)
        records: list[dict[str, str]] = []
        for row in ds:
            choices = row["choices"]
            labels = list(choices["label"])
            texts = list(choices["text"])

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
