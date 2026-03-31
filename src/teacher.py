import logging
import re

from tqdm import tqdm

from .base_sampler import BaseSampler
from .config import Settings

logger = logging.getLogger(__name__)


class TeacherSampler(BaseSampler):
    """Runs a local HuggingFace teacher model to generate (prompt, response) pairs for distillation."""

    def __init__(self, settings: Settings):
        settings.validate()
        super().__init__(settings)

    @property
    def model_id(self) -> str:
        return self.settings.teacher_model

    @property
    def max_new_tokens(self) -> int:
        return self.settings.teacher_max_new_tokens

    @property
    def _extra_model_kwargs(self) -> dict:
        return {"temperature": 0.6}

    def sample_batch(self, prompts: list[str]) -> list[str]:
        batch_messages = [
            [
                {"role": "system", "content": self.settings.teacher_system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        return self._encode_and_generate(batch_messages)

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract a concise final answer from the raw model response.

        Handles DeepSeek-style traces with </think> markers.
        """
        text = response.strip()

        # DeepSeek-style traces often end with </think> then final answer text.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()

        # Prefer the last boxed answer when present.
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            return boxed[-1].strip()

        # Common plain-text markers.
        markers = ["Final Answer:", "Answer:", "The answer is"]
        for marker in markers:
            if marker in text:
                return text.split(marker, 1)[1].strip()

        return text

    def generate_dataset(self, prompts: list[str | dict[str, str]]) -> list[dict]:
        """Generate teacher responses for each prompt and return list of {prompt, response} dicts."""
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
                        "model_name": self.settings.teacher_model,
                        "system_prompt": self.settings.teacher_system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": response,
                        "final_answer": self._extract_final_answer(response),
                        "ground_truth": entry["ground_truth"],
                    }
                )
        return samples
