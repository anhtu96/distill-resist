import json
from pathlib import Path

from datasets import Dataset


class HoldoutDataset:
    """Manages the hold-out distillation dataset stored as JSONL."""

    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, samples: list[dict]) -> None:
        """Overwrite JSONL file using canonical hold-out record keys."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "system_prompt": sample["system_prompt"],
                    "user_prompt": sample["user_prompt"],
                    "response": sample["response"],
                    "final_answer": sample["final_answer"],
                    "ground_truth": sample["ground_truth"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> list[dict]:
        """Load all samples from the JSONL file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def to_hf_dataset(self) -> Dataset:
        """Convert JSONL to a HuggingFace Dataset with a 'trace' column for SFT."""
        samples = self.load()
        traces = [
            (
                f"### System:\n{s['system_prompt']}\n\n"
                f"### User:\n{s['user_prompt']}\n\n"
                f"### Response:\n{s['response']}"
            )
            for s in samples
        ]
        return Dataset.from_dict({"trace": traces})

    def __len__(self) -> int:
        return len(self.load())
