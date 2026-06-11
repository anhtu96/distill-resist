import json
from pathlib import Path

from datasets import Dataset, load_dataset


class HoldoutDataset:
    """Manages the hold-out distillation dataset stored as JSONL."""

    def __init__(self, path: str):
        self.path = Path(path)

    def save(
        self,
        samples: list[dict],
        include_model_name: bool = True,
        append: bool = False,
    ) -> None:
        """Write JSONL records using canonical keys.

        By default this overwrites the destination file. Set ``append=True`` to
        append to an existing file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with self.path.open(mode, encoding="utf-8") as f:
            for sample in samples:
                record = {
                    "trace_id": sample.get("trace_id"),
                    "system_prompt": sample["system_prompt"],
                    "user_prompt": sample["user_prompt"],
                    "response": sample["response"],
                    "final_answer": sample["final_answer"],
                    "ground_truth": sample["ground_truth"],
                }
                if include_model_name:
                    record["model_name"] = sample["model_name"]
                if record["trace_id"] is None:
                    record.pop("trace_id")
                record.update({k: v for k, v in sample.items() if k not in record})
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

    def __len__(self) -> int:
        return len(self.load())

    def to_hf_dataset(self, eos_token: str = "") -> Dataset:
        """Convert JSONL to a HuggingFace Dataset with a 'trace' column for SFT."""
        dataset = load_dataset("json", data_files=str(self.path), split="train")
        dataset = dataset.map(lambda x: format_dataset(x, eos_token=eos_token))
        return dataset

def format_dataset(example, eos_token: str = ""):
    """Preprocess a single example into a prompt-completion format with cleaning."""
    response = example["response"]

    # 1. Strip any existing assistant marker if present (from distill_ref.py)
    if "<｜Assistant｜>" in response:
        response = response.split("<｜Assistant｜>")[1]

    # 2. Replace custom EOS marker with specific model's EOS token (from distill_ref.py)
    # Using the exact characters found in the reference script: <｜end▁of▁sentence｜>
    response = response.replace("<｜end▁of▁sentence｜>", eos_token)

    # Ensure it ends with an EOS if not present
    if eos_token and not response.endswith(eos_token):
        response += eos_token

    prompt = [{
        "role": "system",
        "content": example["system_prompt"]
    },{
        "role": "user",
        "content": example["user_prompt"]
    }]
    completion = [{"role": "assistant", "content": response.strip()}]
    return {"prompt": prompt, "completion": completion}