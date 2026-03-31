import logging
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings

logger = logging.getLogger(__name__)


class BaseLanguageModel(ABC):
    """Base class for teacher, student, and rewriter model wrappers.

    Handles tokenizer/model loading and the shared encode→pad→generate→decode loop.
    Subclasses declare ``model_id`` and ``max_new_tokens`` as properties.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Loading model: %s", self.model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        load_kwargs: dict = dict(device_map="auto", torch_dtype=torch.bfloat16)
        if self.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        load_kwargs.update(self._extra_model_kwargs)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_id(self) -> str:
        """HuggingFace model identifier or local path."""

    @property
    @abstractmethod
    def max_new_tokens(self) -> int:
        """Maximum number of new tokens to generate."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    @property
    def use_flash_attention(self) -> bool:
        """Enable Flash Attention 2. Override to False when unsupported."""
        return True

    @property
    def _extra_model_kwargs(self) -> dict:
        """Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``."""
        return {}

    # ------------------------------------------------------------------
    # Shared generation logic
    # ------------------------------------------------------------------

    def _encode_and_generate(self, batch_messages: list[list[dict]]) -> list[str]:
        """Apply chat template, pad, generate, and decode a batch of message lists."""
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
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=False
        )
