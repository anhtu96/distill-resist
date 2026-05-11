import logging

from datasets import concatenate_datasets, load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)

LETTER_OPTIONS: list[str] = ["A", "B", "C", "D"]


def _format_question(question: str, choices: list[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for letter, choice in zip(LETTER_OPTIONS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


class MMLULoader(BaseDatasetLoader):
    """Loads multiple-choice questions from cais/mmlu.

    config controls subject selection:
      "all" (default) — all 57 subjects via the bundled "all" config
      "<subject>"     — single subject (e.g. "abstract_algebra")
      "<a>,<b>,..."   — explicit comma-separated subject list
    """

    HF_DATASET = "cais/mmlu"

    # MMLU has no "train" split; "auxiliary_train" is the most comprehensive training set.
    _SPLIT_MAP = {"train": "auxiliary_train"}

    def __init__(
        self,
        config: str = "all",
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.split = self._SPLIT_MAP.get(split, split)
        self.num_samples = num_samples

    def _load_raw(self):
        if "," in self.config:
            subjects = [s.strip() for s in self.config.split(",") if s.strip()]
            parts = [load_dataset(self.HF_DATASET, s, split=self.split) for s in subjects]
            return concatenate_datasets(parts)
        return load_dataset(self.HF_DATASET, self.config, split=self.split)

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading MMLU config='%s' split='%s'", self.config, self.split)
        ds = self._load_raw()
        records: list[dict[str, str]] = []
        for row in ds:
            choices = list(row["choices"])
            answer_idx = int(row["answer"])
            records.append({
                "user_prompt": _format_question(str(row["question"]), choices),
                "ground_truth": LETTER_OPTIONS[answer_idx],
                "subject": str(row.get("subject", "")),
            })
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records
