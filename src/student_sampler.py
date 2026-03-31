import logging

from tqdm import tqdm

from .base_sampler import BaseSampler
from .config import Settings

logger = logging.getLogger(__name__)


class StudentSampler(BaseSampler):
    """Runs a local HuggingFace student model to generate (prompt, response) pairs."""

    def __init__(self, settings: Settings):
        settings.validate()
        super().__init__(settings)

    @property
    def model_id(self) -> str:
        return self.settings.student_model_id

    @property
    def max_new_tokens(self) -> int:
        return self.settings.teacher_max_new_tokens

    @property
    def use_flash_attention(self) -> bool:
        return False

    def sample_batch(self, prompts: list[str]) -> list[str]:
        batch_messages = [
            [
                {"role": "system", "content": self.settings.teacher_system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        return self._encode_and_generate(batch_messages)

    def generate_dataset(self, prompts: list[str | dict[str, str]]) -> list[dict]:
        """Generate student responses for each prompt and return list of dicts."""
        samples: list[dict] = []
        if self.settings.sample_limit is None:
            total = len(prompts)
        else:
            total = min(len(prompts), self.settings.sample_limit)
        entries = prompts[:total]
        normalized: list[dict[str, str]] = []
        for item in entries:
            if isinstance(item, str):
                normalized.append({"user_prompt": item, "ground_truth": ""})
            else:
                normalized.append(
                    {
                        "user_prompt": item["user_prompt"],
                        "ground_truth": item.get("ground_truth", ""),
                    }
                )

        all_prompts = [entry["user_prompt"] for entry in normalized]
        batch_size = self.settings.teacher_batch_size
        batches = [all_prompts[i:i + batch_size] for i in range(0, total, batch_size)]
        for batch_idx, batch in enumerate(tqdm(batches)):
            responses = self.sample_batch(batch)
            batch_entries = normalized[batch_idx * batch_size: batch_idx * batch_size + len(batch)]
            for entry, response in zip(batch_entries, responses):
                response = response.replace(self._tokenizer.eos_token, "").strip()
                samples.append(
                    {
                        "model_name": self.settings.student_model_id,
                        "system_prompt": self.settings.teacher_system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": response,
                        "final_answer": self._extract_final_answer(response),
                        "ground_truth": entry["ground_truth"],
                    }
                )
        return samples
