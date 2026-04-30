import logging

from datasets import get_dataset_config_names, load_dataset

from .base import BaseDatasetLoader

logger = logging.getLogger(__name__)


class BBHLoader(BaseDatasetLoader):
    """Loads BIG-Bench Hard tasks from lukaemon/bbh.

    config selects which BBH tasks to load:
      "all" (default) — every task config exposed by the dataset
      "<task>"        — a single task (e.g. "logical_deduction_five_objects")
      "<a>,<b>,..."   — explicit comma-separated task list

    BBH has only a 'test' split (~250 samples per task).
    """

    HF_DATASET = "lukaemon/bbh"

    def __init__(
        self,
        config: str = "all",
        split: str = "test",
        num_samples: int | None = None,
        **kwargs,
    ):
        self.config = config
        self.split = split
        self.num_samples = num_samples

    def _resolve_tasks(self) -> list[str]:
        if self.config in ("all", "default"):
            return get_dataset_config_names(self.HF_DATASET)
        return [t.strip() for t in self.config.split(",") if t.strip()]

    def load(self) -> list[dict[str, str]]:
        tasks = self._resolve_tasks()
        records: list[dict[str, str]] = []
        for task in tasks:
            logger.info("Loading BBH task='%s' split='%s'", task, self.split)
            ds = load_dataset(self.HF_DATASET, task, split=self.split)
            for row in ds:
                records.append({
                    "user_prompt": str(row["input"]).strip(),
                    "ground_truth": str(row["target"]).strip(),
                    "subject": task,
                })
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records
