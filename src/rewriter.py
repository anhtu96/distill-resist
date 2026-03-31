import logging

from tqdm import tqdm

from .base_model import BaseLanguageModel
from .config import REWRITE_SYSTEM_PROMPT, Settings

logger = logging.getLogger(__name__)


class Rewriter(BaseLanguageModel):
    """Uses a separate rewriter model to rewrite generated traces."""

    @property
    def model_id(self) -> str:
        return self.settings.rewriter_model_id

    @property
    def max_new_tokens(self) -> int:
        return self.settings.rewriter_max_new_tokens

    def rewrite_batch(self, responses: list[str]) -> list[str]:
        """Rewrite a batch of responses using the rewriter model."""
        batch_messages = [
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": response},
            ]
            for response in responses
        ]
        decoded = self._encode_and_generate(batch_messages)
        return [r.replace(self._tokenizer.eos_token, "").strip() for r in decoded]

    def rewrite_dataset(self, samples: list[dict]) -> list[str]:
        """Rewrite all responses in a dataset, returning the rewritten strings in order."""
        batch_size = self.settings.teacher_batch_size
        responses = [s["response"] for s in samples]
        rewritten: list[str] = []
        for i in tqdm(range(0, len(responses), batch_size), desc="Rewriting"):
            rewritten.extend(self.rewrite_batch(responses[i : i + batch_size]))
        return rewritten
