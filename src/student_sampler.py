import logging
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .config import Settings
from .loaders import get_loader

logger = logging.getLogger(__name__)


class StudentSampler:
    """Runs a local HuggingFace student model to generate (prompt, response) pairs."""

    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        logger.info("Loading student model: %s", settings.student_model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(settings.student_model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"  # required for batch generation
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.student_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    def sample(self, prompt: str) -> str:
        return self.sample_batch([prompt])[0]

    def sample_batch(self, prompts: list[str]) -> list[str]:
        batch_messages = [
            [
                {"role": "system", "content": self.settings.teacher_system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        encoded = [
            self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            ).squeeze(0)
            for msgs in batch_messages
        ]
        inputs = self._tokenizer.pad(
            {"input_ids": encoded},
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.teacher_max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=False
        )

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract a concise final answer from the raw model response."""
        text = response.strip()

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

    def load_prompts(self) -> list[str | dict[str, str]]:
        """Load prompts using the configured dataset loader."""
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()

    def generate_dataset(self, prompts: list[str | dict[str, str]]) -> list[dict]:
        """Generate student responses for each prompt and return list of dicts."""
        samples: list[dict] = []
        if self.settings.num_samples is None:
            total = len(prompts)
        else:
            total = min(len(prompts), self.settings.num_samples)
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
                        "system_prompt": self.settings.teacher_system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": response,
                        "final_answer": self._extract_final_answer(response),
                        "ground_truth": entry["ground_truth"],
                    }
                )
        return samples
