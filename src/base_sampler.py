import re

from .base_model import BaseLanguageModel
from .config import Settings
from .loaders import get_loader


class BaseSampler(BaseLanguageModel):
    """Base class for samplers (teacher/student) that generate responses from prompts.

    Consolidates shared sampling and prompt-loading logic.
    """

    def sample(self, prompt: str) -> str:
        """Generate a single response for a prompt."""
        return self.sample_batch([prompt])[0]

    def load_prompts(self) -> list[str | dict[str, str]]:
        """Load prompts using the configured dataset loader."""
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        """Extract a concise final answer from the raw model response.

        Default implementation (no model-specific logic).
        Override in subclass for model-specific behavior.
        """
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
