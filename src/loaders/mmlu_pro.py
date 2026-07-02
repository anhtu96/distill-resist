import logging
import random
from collections import defaultdict

from datasets import concatenate_datasets, load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# MMLU-Pro has up to 10 answer options (A–J)
LETTER_OPTIONS: list[str] = list("ABCDEFGHIJ")


def _format_question(question: str, options: list[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for letter, option in zip(LETTER_OPTIONS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


class MMLUProLoader(BaseDatasetLoader):
    """Loads multiple-choice questions from TIGER-Lab/MMLU-Pro.

    MMLU-Pro is a harder variant of MMLU with up to 10 answer choices (A–J)
    and a focus on reasoning-heavy questions across 14 categories.

    config controls category selection:
      "all" (default) — all categories
      "<category>"    — single category (e.g. "math", "physics")
      "<a>,<b>,..."   — comma-separated category list

    Available categories:
      math, physics, chemistry, biology, computer science, engineering,
      economics, law, medicine, psychology, history, philosophy, business, other

    split:
      "test"       — 12,032 questions (main evaluation set; also used as holdout)
      "validation" — 70 questions (small dev set)

    MMLU-Pro has no official training split. When split="train" is requested,
    the loader falls back to the "test" split (12k samples) with a fixed-seed
    70/30 partition, returning the training 70% portion.
    """

    HF_DATASET = "TIGER-Lab/MMLU-Pro"

    # MMLU-Pro has no "train" split; we partition "test" for train usage.
    _TRAIN_RATIO = 0.7
    _SEED = 42

    def __init__(
        self,
        config: str = "all",
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.requested_split = split
        self.num_samples = num_samples

    def _hf_split(self) -> str:
        """Map the requested split to the actual HuggingFace split name."""
        if self.requested_split in ("train", "auxiliary_train"):
            return "test"  # we'll partition this ourselves
        return self.requested_split  # "test" or "validation"

    def _load_raw(self):
        hf_split = self._hf_split()
        logger.info(
            "Loading MMLU-Pro config='%s' hf_split='%s' (requested='%s')",
            self.config,
            hf_split,
            self.requested_split,
        )
        if "," in self.config and self.config != "all":
            categories = [c.strip() for c in self.config.split(",") if c.strip()]
            # Load once and filter per category, then concatenate
            ds = load_dataset(self.HF_DATASET, split=hf_split)
            filtered = [
                ds.filter(lambda row, c=cat: row["category"] == c)
                for cat in categories
            ]
            return concatenate_datasets(filtered)

        ds = load_dataset(self.HF_DATASET, split=hf_split)

        if self.config != "all":
            cat = self.config.strip()
            ds = ds.filter(lambda row: row["category"] == cat)

        return ds

    def load(self) -> list[dict[str, str]]:
        ds = self._load_raw()
        
        # Group raw dataset rows by category
        by_category = defaultdict(list)
        for row in ds:
            options = list(row["options"])
            answer = str(row["answer"]).strip()  # e.g. "A", "B", ...
            category = str(row.get("category", ""))

            by_category[category].append({
                "user_prompt": _format_question(str(row["question"]), options),
                "ground_truth": answer,
                "subject": category,
            })

        # Process each category partition individually to ensure deterministic shuffling
        # independent of the presence of other categories in the loading config.
        records: list[dict[str, str]] = []
        for category, cat_records in sorted(by_category.items()):
            if self.requested_split in ("train", "auxiliary_train"):
                # Deterministic shuffle per category
                rng = random.Random(self._SEED)
                shuffled = list(cat_records)
                rng.shuffle(shuffled)
                cutoff = int(len(shuffled) * self._TRAIN_RATIO)
                records.extend(shuffled[:cutoff])
            elif self.requested_split == "test":
                # Deterministic shuffle per category
                rng = random.Random(self._SEED)
                shuffled = list(cat_records)
                rng.shuffle(shuffled)
                cutoff = int(len(shuffled) * self._TRAIN_RATIO)
                records.extend(shuffled[cutoff:])
            else:
                # No splitting requested (e.g. split="validation")
                records.extend(cat_records)

        if self.num_samples is not None and len(records) > self.num_samples:
            rng = random.Random(self._SEED)
            records = rng.sample(records, self.num_samples)

        return records
