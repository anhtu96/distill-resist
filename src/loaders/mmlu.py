import logging
import random

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
        if self.split == "auxiliary_train":
            return load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")

        if "," in self.config:
            subjects = [s.strip() for s in self.config.split(",") if s.strip()]
            parts = [load_dataset(self.HF_DATASET, s, split=self.split) for s in subjects]
            return concatenate_datasets(parts)
        return load_dataset(self.HF_DATASET, self.config, split=self.split)

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading MMLU config='%s' split='%s'", self.config, self.split)
        ds = self._load_raw()
        records: list[dict[str, str]] = []
        
        is_aux_train = (self.split == "auxiliary_train")
        allowed_subjects = None
        if is_aux_train and self.config != "all":
            if "," in self.config:
                allowed_subjects = {s.strip() for s in self.config.split(",") if s.strip()}
            else:
                allowed_subjects = {self.config.strip()}

        for row in ds:
            if is_aux_train and allowed_subjects is not None:
                task = row.get("task", "")
                if task not in allowed_subjects:
                    continue

            choices = list(row["choices"])
            answer_idx = int(row["answer"])
            
            subj = row.get("subject", "")
            if not subj and "task" in row:
                subj = row["task"]

            records.append({
                "user_prompt": _format_question(str(row["question"]), choices),
                "ground_truth": LETTER_OPTIONS[answer_idx],
                "subject": str(subj),
            })
        if self.num_samples is not None:
            if len(records) > self.num_samples:
                rng = random.Random(42)
                records = rng.sample(records, self.num_samples)
        return records
