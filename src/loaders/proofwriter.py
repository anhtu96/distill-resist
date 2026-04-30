import logging

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


def _row_to_fields(row: dict) -> tuple[str, str, str]:
    """Tolerate field-name variations across ProofWriter mirrors on HF."""
    theory = str(row.get("theory") or row.get("context") or row.get("premises") or "")
    question = str(row.get("question") or row.get("hypothesis") or row.get("query") or "")
    answer = str(row.get("answer") or row.get("label") or row.get("output") or "").strip()
    return theory, question, answer


class ProofWriterLoader(BaseDatasetLoader):
    """Loads deductive-reasoning problems from renma/ProofWriter.

    Each example pairs a natural-language theory (facts + rules) with a question
    whose answer is one of {True, False, Unknown}.
    """

    HF_DATASET = "renma/ProofWriter"

    def __init__(
        self,
        config: str | None = None,
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.split = split
        self.num_samples = num_samples

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading ProofWriter split='%s'", self.split)
        ds = (
            load_dataset(self.HF_DATASET, self.config, split=self.split)
            if self.config and self.config not in ("all", "default")
            else load_dataset(self.HF_DATASET, split=self.split)
        )
        records: list[dict[str, str]] = []
        for row in ds:
            theory, question, answer = _row_to_fields(dict(row))
            user_prompt = (
                f"{theory.strip()}\n\nQuestion: {question.strip()}"
                if theory
                else question.strip()
            )
            records.append({
                "user_prompt": user_prompt,
                "ground_truth": answer,
            })
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records
