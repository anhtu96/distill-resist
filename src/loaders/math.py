import logging
import re

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


def _extract_boxed(solution: str) -> str:
    """Extract the content of the last \\boxed{...} in a solution string.

    Handles nested braces correctly.
    """
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        # Fallback: return last line stripped
        return solution.strip().splitlines()[-1].strip()
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1].strip()


class MATHLoader(BaseDatasetLoader):
    """Loads competition math problems from hendrycks/competition_math (MATH dataset)."""

    HF_DATASET = "HuggingFaceH4/MATH"

    def __init__(self, config: str = "default", split: str = "train"):
        self.config = config
        self.split = split

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading MATH config='%s' split='%s'", self.config, self.split)
        ds = load_dataset(self.HF_DATASET, self.config, split=self.split)
        records: list[dict[str, str]] = []
        for row in ds:
            records.append({
                "user_prompt": row["problem"],
                "ground_truth": _extract_boxed(row["solution"]),
            })
        return records


class MATH500Loader(BaseDatasetLoader):
    """Loads the MATH-500 benchmark (lighteval/MATH-500)."""

    HF_DATASET = "HuggingFaceH4/MATH-500"

    def __init__(self, config: str = "default", split: str = "test"):
        self.config = config
        self.split = split

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading MATH-500 config='%s' split='%s'", self.config, self.split)
        ds = load_dataset(self.HF_DATASET, self.config, split=self.split)
        records: list[dict[str, str]] = []
        for row in ds:
            ground_truth = row.get("answer") or _extract_boxed(row["solution"])
            records.append({
                "user_prompt": row["problem"],
                "ground_truth": ground_truth,
            })
        return records
