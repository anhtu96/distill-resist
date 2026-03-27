import logging

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class GSM8KLoader(BaseDatasetLoader):
    """Loads math word problems from openai/gsm8k."""

    def __init__(self, config: str = "main", split: str = "train"):
        self.config = config
        self.split = split

    @staticmethod
    def _extract_ground_truth(answer: str) -> str:
        """Return the value after the last ### marker in a GSM8K answer string."""
        if "###" not in answer:
            return answer.strip()
        value = answer.rsplit("###", 1)[1].strip()
        return value.lstrip("#").strip()

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading GSM8K config='%s' split='%s'", self.config, self.split)
        ds = load_dataset("openai/gsm8k", self.config, split=self.split)
        return [
            {
                "user_prompt": row["question"],
                "ground_truth": self._extract_ground_truth(row["answer"]),
            }
            for row in ds
        ]
