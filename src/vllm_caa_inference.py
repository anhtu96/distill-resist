"""vLLM-based trace generator with CAA residual-stream steering.

Uses custom vLLM model classes registered in src/vllm_caa.py to inject
steering vectors via the residual stream. Mirrors the surface area of
HFCAATraceGenerator so generate_traces.py can swap it in.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from . import vllm_caa  # registers custom architectures
from .config import ANSWER_FORCE_STRING, Settings
from .loaders import get_loader


def _pick_caa_arch(model_id: str) -> str:
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        archs = list(getattr(cfg, "architectures", []) or [])
    except Exception:
        archs = []
    if "Gemma3ForConditionalGeneration" in archs:
        return "Gemma3CAAForConditionalGeneration"
    return "Gemma3CAAForCausalLM"


class VLLMCAATraceGenerator:
    """Batch CAA generator backed by vLLM with custom registered models."""

    def __init__(
        self,
        settings: Settings,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        caa_checkpoint: str,
        system_prompt: Optional[str] = None,
        batch_size: int = 4,
        strength: float = 2.0,
        injection_layers: Optional[list[int]] = None,
        caa_weight_mode: Optional[str] = None,
        caa_dirichlet_alphas: Optional[list[float]] = None,
    ):
        from vllm import LLM, SamplingParams

        self.settings = settings
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or settings.teacher_system_prompt
        self.batch_size = batch_size
        self.strength = strength
        self.caa_weight_mode = caa_weight_mode or settings.caa_weight_mode
        self.caa_dirichlet_alphas = caa_dirichlet_alphas if caa_dirichlet_alphas is not None else settings.caa_dirichlet_alphas
        self._current_style: str | None = None

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        ckpt = torch.load(caa_checkpoint, map_location="cpu", weights_only=False)
        injection_point = ckpt.get("injection_point", "residual")
        if injection_point != "residual":
            raise ValueError(
                f"CAA checkpoint injection_point is '{injection_point}', expected 'residual'."
            )
        self.insertion_layers: list[int] = list(ckpt["insertion_layers"])
        raw_vectors: dict = ckpt["steering_vectors"]
        self.styles: list[str] = list(ckpt.get("styles", list(raw_vectors)))

        # Filter to requested subset, preserving checkpoint order
        if injection_layers is not None:
            missing = set(injection_layers) - set(self.insertion_layers)
            if missing:
                raise ValueError(
                    f"Requested injection layers {sorted(missing)} not in checkpoint layers {self.insertion_layers}"
                )
            self.insertion_layers = [l for l in self.insertion_layers if l in set(injection_layers)]

        # Pre-convert per-style vectors to torch tensors keyed by layer index.
        self._vectors_by_style: dict[str, dict[int, torch.Tensor]] = {}
        for style, sv_list in raw_vectors.items():
            per_layer: dict[int, torch.Tensor] = {}
            for layer_idx, sv_np in zip(ckpt["insertion_layers"], sv_list):
                if layer_idx not in self.insertion_layers:
                    continue
                t = torch.as_tensor(sv_np, dtype=torch.float32)
                if t.dim() > 1:
                    t = t.flatten()
                per_layer[int(layer_idx)] = t
            self._vectors_by_style[style] = per_layer

        arch = _pick_caa_arch(model_id)
        self._llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            hf_overrides={"architectures": [arch]},
            enforce_eager=True,   # required so per-forward Python state checks actually run
        )

        _eos = self._tokenizer.eos_token or ""
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=[s for s in [_eos, "<end_of_turn>"] if s],
        )
        self._force_sampling_params = SamplingParams(
            max_tokens=settings.answer_force_max_tokens,
            temperature=0.0,
            stop=[s for s in [_eos, "<end_of_turn>", "\\]"] if s],
        )
        print(
            f"vLLM CAA: arch={arch}, layers={self.insertion_layers}, styles={self.styles}"
        )

    def set_style(self, style: str) -> None:
        self._current_style = style
        if style not in self._vectors_by_style:
            raise ValueError(
                f"Style '{style}' not in checkpoint. Available: {self.styles}"
            )
        vllm_caa.set_caa_state(
            active=True,
            strength=self.strength,
            layers=self.insertion_layers,
            vectors=self._vectors_by_style[style],
        )

    def disable_steering(self) -> None:
        vllm_caa.disable_caa()

    def set_vector_from_metadata(self, metadata: dict) -> dict[str, float]:
        from .prf import metadata_to_caa_vector
        vectors, style_weights = metadata_to_caa_vector(
            metadata,
            self._vectors_by_style,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
        )
        vllm_caa.set_caa_state(
            active=True,
            strength=self.strength,
            layers=self.insertion_layers,
            vectors=vectors,
        )
        return style_weights

    def _render(self, prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            try:
                converted = [
                    {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in messages
                ]
                return self._tokenizer.apply_chat_template(
                    converted, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                return f"System: {system_prompt}\nUser: {prompt}\nAssistant:\n"

    def sample_batch(
        self, prompts: list[str], system_prompt: Optional[str] = None
    ) -> list[str]:
        sys_prompt = system_prompt or self.system_prompt
        rendered = [self._render(p, sys_prompt) for p in prompts]

        outputs = self._llm.generate(
            rendered, self._sampling_params, use_tqdm=False,
        )
        return [out.outputs[0].text if out.outputs else "" for out in outputs]

    def force_answers(self, responses: list[str]) -> list[str]:
        """Append ANSWER_FORCE_STRING and greedily decode 32 tokens.

        Returns response + ANSWER_FORCE_STRING + decoded suffix, matching
        the answer-forcing behavior of HFCAATraceGenerator.
        """
        # Disable steering for clean greedy extraction
        vllm_caa.disable_caa()
        try:
            # We need the same chat-rendered prefix that produced `response`,
            # but force_answers in the HF path simply tokenizes
            # `response + ANSWER_FORCE_STRING` directly. Match that here so the
            # appended string is decoded in isolation.
            af_prompts = [resp.strip() + ANSWER_FORCE_STRING for resp in responses]
            outputs = self._llm.generate(
                af_prompts, self._force_sampling_params, use_tqdm=False,
            )
            results: list[str] = []
            for af, out in zip(af_prompts, outputs):
                decoded = out.outputs[0].text if out.outputs else ""
                results.append(af + decoded + "}\\]")
            return results
        finally:
            # Restore style if one was active
            if self._current_style is not None:
                self.set_style(self._current_style)

    def load_prompts(self):
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
        )
        return loader.load()

    def close(self) -> None:
        vllm_caa.disable_caa()
        if self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
