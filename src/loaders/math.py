import logging
import re
from typing import Any

from datasets import load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)

ALL_SUBJECTS: list[str] = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Subjects whose answers are reliably scalar — safe for answer forcing + math_verify.
# intermediate_algebra is excluded because it frequently produces vectors/matrices.
SCALAR_SUBJECTS: list[str] = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _resolve_subjects(config: str) -> list[str]:
    """Map config string to a list of HuggingFace subset names.

    Supports:
      "all"    — all seven subjects
      "scalar" — scalar-answer subjects only (excludes intermediate_algebra)
      "algebra,number_theory,..." — explicit comma-separated list
      "algebra" — single subject
    """
    if config in ("all",):
        return ALL_SUBJECTS
    if config in ("scalar", "default"):
        return SCALAR_SUBJECTS
    subjects = [s.strip() for s in config.split(",") if s.strip()]
    unknown = set(subjects) - set(ALL_SUBJECTS)
    if unknown:
        raise ValueError(f"Unknown MATH subject(s): {unknown}. Valid: {ALL_SUBJECTS}")
    return subjects


def _extract_boxed(solution: str) -> str:
    """Extract the content of the last \\boxed{...} in a solution string.

    Handles nested braces correctly.
    """
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
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
    """Loads competition math problems from HuggingFaceH4/MATH.

    config controls which subjects to load:
      "scalar" (default) — algebra, counting_and_probability, geometry,
                           number_theory, prealgebra, precalculus
      "all"              — all seven subjects
      "algebra,..."      — explicit comma-separated subject list
      "algebra"          — single subject
    """

    HF_DATASET = "HuggingFaceH4/MATH"

    def __init__(self, config: str = "scalar", split: str = "train", num_samples: int | None = None, **kwargs):
        self.config = config
        self.split = split
        self.num_samples = num_samples

    def load(self) -> list[dict[str, str]]:
        subsets = _resolve_subjects(self.config)
        hf_split = "test" if self.split == "test" else "train"

        # Load all records per subset first
        per_subset: dict[str, list[dict[str, str]]] = {}
        for subset in subsets:
            logger.info("Loading MATH subset='%s' split='%s'", subset, hf_split)
            ds = load_dataset(self.HF_DATASET, subset, split=hf_split)
            rows = []
            for _row in ds:
                row: dict[str, Any] = dict(_row)  # type: ignore[arg-type]
                rows.append({
                    "user_prompt": str(row["problem"]),
                    "ground_truth": _extract_boxed(str(row["solution"])),
                    "subject": subset,
                })
            per_subset[subset] = rows

        if self.num_samples is None:
            return [r for rows in per_subset.values() for r in rows]

        # Proportional allocation across subsets (largest-remainder method)
        sizes = {s: len(per_subset[s]) for s in subsets}
        total_available = sum(sizes.values())
        n = min(self.num_samples, total_available)

        allocations: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        for subset in subsets:
            exact = n * sizes[subset] / total_available
            allocations[subset] = int(exact)
            remainders.append((exact - int(exact), subset))

        # Distribute leftover slots to subsets with largest fractional parts
        leftover = n - sum(allocations.values())
        for _, subset in sorted(remainders, reverse=True)[:leftover]:
            allocations[subset] += 1

        records: list[dict[str, str]] = []
        for subset in subsets:
            records.extend(per_subset[subset][: allocations[subset]])
        return records


class MATH500Loader(BaseDatasetLoader):
    """Loads the MATH-500 benchmark from HuggingFaceH4/MATH-500.

    config controls subject filtering (same syntax as MATHLoader):
      "scalar" (default) — scalar-answer subjects only
      "all"              — all subjects
      "algebra,..."      — explicit comma-separated list
    """

    HF_DATASET = "HuggingFaceH4/MATH-500"

    def __init__(self, config: str = "scalar", split: str = "test", **kwargs):
        self.config = config
        self.split = split

    def load(self) -> list[dict[str, str]]:
        logger.info("Loading MATH-500 split='%s' subjects='%s'", self.split, self.config)
        ds = load_dataset(self.HF_DATASET, split=self.split)
        allowed = set(_resolve_subjects(self.config))
        records: list[dict[str, str]] = []
        for _row in ds:
            row: dict[str, Any] = dict(_row)  # type: ignore[arg-type]
            subject = str(row.get("subject", ""))
            # Normalise: dataset uses spaces/ampersands, our keys use underscores/and
            subject_key = subject.lower().replace(" & ", "_and_").replace(" ", "_")
            if subject_key not in allowed:
                continue
            ground_truth = str(row.get("answer") or _extract_boxed(str(row["solution"])))
            records.append({
                "user_prompt": str(row["problem"]),
                "ground_truth": ground_truth,
                "subject": subject_key,
            })
        return records
