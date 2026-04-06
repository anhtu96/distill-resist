import logging
from tqdm import tqdm

import torch
try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.answer_extraction import extract_final_answer
from .config import Settings
from .loaders import get_loader

logger = logging.getLogger(__name__)


class LanguageModel:
    """Base class for teacher, student, and rewriter model wrappers.

    Handles tokenizer/model loading and the shared encode→pad→generate→decode loop.
    """

    def __init__(
        self,
        settings: Settings,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        use_flash_attention: bool | None = None,
        extra_model_kwargs: dict | None = None,
    ):
        self.settings = settings
        self._model_id_override = model_id
        self._max_new_tokens_override = max_new_tokens
        self._use_flash_attention_override = use_flash_attention
        self._extra_model_kwargs_override = extra_model_kwargs

        logger.info("Loading model: %s", self.model_id)

        use_unsloth = False
        if "gemma" in self.model_id.lower() and FastLanguageModel is not None:
            try:
                max_seq_length = getattr(self.settings, "teacher_max_seq_length", 2048)
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_id,
                    max_seq_length=max_seq_length,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                )
                FastLanguageModel.for_inference(self._model)
                use_unsloth = True
                logger.info("Using unsloth for Gemma inference: %s", self.model_id)
            except Exception as exc:
                logger.warning(
                    "Failed to load %s with unsloth, falling back to transformers: %s",
                    self.model_id,
                    exc,
                )

        if not use_unsloth:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            load_kwargs: dict = dict(device_map="auto", torch_dtype=torch.bfloat16)
            if self.use_flash_attention:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            load_kwargs.update(self._extra_model_kwargs)

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """HuggingFace model identifier or local path."""
        if self._model_id_override is not None:
            return self._model_id_override
        raise NotImplementedError("model_id must be provided or overridden in subclass")

    @property
    def max_new_tokens(self) -> int:
        """Maximum number of new tokens to generate."""
        if self._max_new_tokens_override is not None:
            return self._max_new_tokens_override
        raise NotImplementedError("max_new_tokens must be provided or overridden in subclass")

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    @property
    def use_flash_attention(self) -> bool:
        """Enable Flash Attention 2. Override to False when unsupported."""
        if self._use_flash_attention_override is not None:
            return self._use_flash_attention_override
        return True

    @property
    def _extra_model_kwargs(self) -> dict:
        """Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained``."""
        if self._extra_model_kwargs_override is not None:
            return self._extra_model_kwargs_override
        return {}

    # ------------------------------------------------------------------
    # Shared generation logic
    # ------------------------------------------------------------------

    def _encode_and_generate(self, batch_messages: list[list[dict]]) -> list[str]:
        """Apply chat template, pad, generate, and decode message batches."""
        if not batch_messages:
            return []

        def _to_gemma3_text_content(messages: list[dict]) -> list[dict]:
            converted: list[dict] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                converted.append({"role": msg["role"], "content": content})
            return converted

        # Gemma-family models can fail on batched generation with shape mismatches
        # in attention internals when prompts are padded to different lengths.
        if "gemma" in self.model_id.lower() and len(batch_messages) > 1:
            outputs: list[str] = []
            for messages in batch_messages:
                try:
                    single_inputs = self._tokenizer.apply_chat_template(
                        [messages],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self._model.device)
                except TypeError:
                    single_inputs = self._tokenizer.apply_chat_template(
                        [_to_gemma3_text_content(messages)],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self._model.device)
                prompt_len = single_inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **single_inputs,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                outputs.append(
                    self._tokenizer.decode(
                        output_ids[0, prompt_len:], skip_special_tokens=True
                    )
                )
            return outputs

        try:
            inputs = self._tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            ).to(self._model.device)
        except TypeError:
            inputs = self._tokenizer.apply_chat_template(
                [_to_gemma3_text_content(messages) for messages in batch_messages],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
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
            output_ids[:, prompt_len:], skip_special_tokens=True
        )


class Model(LanguageModel):
    """High-level model wrapper for sampling and dataset generation.

    Consolidates configuration and sampling logic for any model.
    """

    def __init__(
        self,
        settings: Settings,
        model_id: str | None = None,
        max_new_tokens: int | None = None,
        use_flash_attention: bool | None = None,
        extra_model_kwargs: dict | None = None,
        system_prompt: str | None = None,
        batch_size: int | None = None,
    ):
        super().__init__(
            settings,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            use_flash_attention=use_flash_attention,
            extra_model_kwargs=extra_model_kwargs,
        )
        self._system_prompt_override = system_prompt
        self._batch_size_override = batch_size

    @property
    def system_prompt(self) -> str:
        """The system prompt to use for generation. Defaults to teacher_system_prompt."""
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        return self.settings.teacher_system_prompt

    @property
    def batch_size(self) -> int:
        """The batch size for generation. Defaults to teacher_batch_size."""
        if self._batch_size_override is not None:
            return self._batch_size_override
        return self.settings.teacher_batch_size

    def sample(self, prompt: str) -> str:
        """Generate a single response for a prompt."""
        return self.sample_batch([prompt])[0]

    def sample_batch(self, prompts: list[str]) -> list[str]:
        batch_messages = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        return self._encode_and_generate(batch_messages)

    def generate_dataset(self, prompts: list[str | dict[str, str]]) -> list[dict]:
        """Generate responses for each prompt and return list of dicts."""
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
        batch_size = self.batch_size
        batches = [all_prompts[i : i + batch_size] for i in range(0, total, batch_size)]
        for batch_idx, batch in enumerate(
            tqdm(batches, desc=f"Generating with {self.__class__.__name__}")
        ):
            responses = self.sample_batch(batch)
            batch_entries = normalized[
                batch_idx * batch_size : batch_idx * batch_size + len(batch)
            ]
            for entry, response in zip(batch_entries, responses):
                response = response.replace(self._tokenizer.eos_token, "").replace(self._tokenizer.pad_token, "").strip()
                samples.append(
                    {
                        "model_name": self.model_id,
                        "system_prompt": self.system_prompt,
                        "user_prompt": entry["user_prompt"],
                        "response": response,
                        "final_answer": extract_final_answer(response),
                        "ground_truth": entry["ground_truth"],
                    }
                )
        return samples

    def load_prompts(self) -> list[str | dict[str, str]]:
        """Load prompts using the configured dataset loader."""
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()
