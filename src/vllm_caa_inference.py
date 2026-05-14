"""vLLM-based trace generator with CAA residual-stream steering.

Uses custom vLLM model classes registered in src/vllm_caa.py to inject
steering vectors via the residual stream. Mirrors the surface area of
HFCAATraceGenerator so generate_traces.py can swap it in.
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from . import vllm_caa  # registers custom architectures
from .config import ANSWER_FORCE_STRING, Settings, normalize_caa_strength
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
        strength: "float | list[float] | dict[str, float | list[float]]" = 2.0,
        injection_layers: Optional[list[int]] = None,
        caa_weight_mode: Optional[str] = None,
        caa_dirichlet_alphas: Optional[list[float]] = None,
        caa_style_scales: Optional[dict[str, float]] = None,
        normalize_per_layer: bool = False,
        relative_strength: bool = False,
        user_id: str = "synthetic",
    ):
        from vllm import LLM, SamplingParams

        self.settings = settings
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or settings.teacher_system_prompt
        self.batch_size = batch_size
        self.strength = strength
        self._layer_strengths: dict[int, float] | None = None  # set after insertion_layers resolved
        self.caa_weight_mode = caa_weight_mode or settings.caa_weight_mode
        self.caa_dirichlet_alphas = caa_dirichlet_alphas if caa_dirichlet_alphas is not None else settings.caa_dirichlet_alphas
        self.caa_style_scales = caa_style_scales if caa_style_scales is not None else settings.caa_style_scales
        self.normalize_per_layer = normalize_per_layer
        self.relative_strength = relative_strength
        self._current_style: str | None = None

        from .prf import hmac_seed_from_metadata
        run_metadata = {"user_id": user_id}
        self._rng_seed = hmac_seed_from_metadata(run_metadata)
        self._rng = np.random.default_rng(self._rng_seed)
        print(f"[CAA] Run seed (from user_id='{user_id}'): {self._rng_seed}")

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
        self.style_to_idx = {s: i for i, s in enumerate(self.styles)}

        # Filter to requested subset; CLI order is authoritative.
        if injection_layers is not None:
            missing = set(injection_layers) - set(self.insertion_layers)
            if missing:
                raise ValueError(
                    f"Requested injection layers {sorted(missing)} not in checkpoint layers {self.insertion_layers}"
                )
            self.insertion_layers = list(injection_layers)  # preserve CLI order

        # Resolve strength into canonical form after insertion_layers is finalized.
        self._strength_norm = normalize_caa_strength(
            strength, self.insertion_layers, styles=self.styles
        )
        self._style_layer_strengths: dict[str, dict[int, float]] | None = (
            self._strength_norm if isinstance(self._strength_norm, dict)
            and all(isinstance(v, dict) for v in self._strength_norm.values())
            else None
        )
        if self._style_layer_strengths is not None:
            self._layer_strengths = None
            print(f"[CAA] Per-style per-layer strengths: {self._style_layer_strengths}")
        elif isinstance(self._strength_norm, dict):
            self._layer_strengths = self._strength_norm
            print(f"[CAA] Per-layer strengths: {self._layer_strengths}")
        else:
            self._layer_strengths = None
            print(f"[CAA] Strength (all layers): {self.strength}")

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
            enable_prefix_caching=False,  # CAA state changes per-request; cached KV would bake in the wrong steering vector
        )

        _eos = self._tokenizer.eos_token or ""
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=[s for s in [_eos, "<end_of_turn>"] if s],
        )
        print(f"Sampling params: {self._sampling_params}")
        self._force_sampling_params = SamplingParams(
            max_tokens=settings.answer_force_max_tokens,
            temperature=0.0,
            stop=[s for s in [_eos, "<end_of_turn>", "\\]"] if s],
        )
        print(
            f"vLLM CAA: arch={arch}, layers={self.insertion_layers}, styles={self.styles}"
        )

    def _resolve_layer_strengths(
        self, weights_or_style: "str | dict[str, float]"
    ) -> "dict[int, float] | float":
        """Resolve per-layer strength for a single style or a weight blend.

        Returns a scalar float when self.strength is a scalar (back-compat with
        vllm_caa.set_caa_state), otherwise a {layer_idx: float} dict.
        """
        if self._style_layer_strengths is not None:
            if isinstance(weights_or_style, str):
                if weights_or_style not in self._style_layer_strengths:
                    raise ValueError(
                        f"Style '{weights_or_style}' has no entry in caa_strength dict. "
                        f"Provided: {list(self._style_layer_strengths)}"
                    )
                return dict(self._style_layer_strengths[weights_or_style])
            out: dict[int, float] = {l: 0.0 for l in self.insertion_layers}
            for style, w in weights_or_style.items():
                per_layer = self._style_layer_strengths.get(style)
                if per_layer is None:
                    continue
                for l in self.insertion_layers:
                    out[l] += float(w) * per_layer.get(l, 0.0)
            return out
        if self._layer_strengths is not None:
            return dict(self._layer_strengths)
        return float(self.strength)

    def _get_effective_max_tokens(self, weights: dict[str, float] | str) -> int | None:
        cfg = self.settings.caa_max_injection_tokens
        if cfg is None or isinstance(cfg, int):
            return cfg

        if isinstance(cfg, dict):
            unknown = set(cfg) - set(self.styles)
            if unknown:
                raise ValueError(
                    f"CAA_MAX_INJECTION_TOKENS dict has unknown styles {sorted(unknown)}; "
                    f"checkpoint styles: {self.styles}"
                )
            if isinstance(weights, str):
                v = cfg.get(weights)
                return int(v) if v is not None else None
            effective = 0.0
            for style, w in weights.items():
                v = cfg.get(style)
                if v is not None:
                    effective += float(w) * float(v)
            return int(round(effective))

        if isinstance(cfg, list):
            if len(cfg) != len(self.styles):
                raise ValueError(f"CAA_MAX_INJECTION_TOKENS list length {len(cfg)} != styles count {len(self.styles)}")
            if isinstance(weights, str):
                idx = self.style_to_idx.get(weights)
                return cfg[idx] if idx is not None else None
            effective = 0.0
            for style, w in weights.items():
                idx = self.style_to_idx.get(style)
                if idx is not None:
                    effective += w * cfg[idx]
            return int(round(effective))
        return None

    def set_style(self, style: str) -> None:
        self._current_style = style
        if style not in self._vectors_by_style:
            raise ValueError(
                f"Style '{style}' not in checkpoint. Available: {self.styles}"
            )
        scale = self.caa_style_scales.get(style, 1.0) if self.caa_style_scales else 1.0
        vectors = self._vectors_by_style[style]
        if self.normalize_per_layer:
            vectors = {layer_idx: v / (v.norm() + 1e-8) for layer_idx, v in vectors.items()}
        if scale != 1.0:
            vectors = {layer_idx: v * scale for layer_idx, v in vectors.items()}
        
        max_inj = self._get_effective_max_tokens(style)

        vllm_caa.set_caa_state(
            active=True,
            strength=self._resolve_layer_strengths(style),
            layers=self.insertion_layers,
            vectors=vectors,
            relative=self.relative_strength,
            max_injection_tokens=max_inj,
            inject_prefill=self.settings.caa_inject_prefill,
        )

    def disable_steering(self) -> None:
        vllm_caa.disable_caa()

    def set_random_per_layer_vectors(self) -> dict[int, dict[str, float]]:
        from .prf import random_per_layer_caa_vectors
        vectors, per_layer_weights = random_per_layer_caa_vectors(
            self._vectors_by_style,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
            style_scales=self.caa_style_scales,
            normalize_result=self.normalize_per_layer,
        )
        # Get weights from the first layer (they are currently shared across layers)
        first_layer_weights = next(iter(per_layer_weights.values()))
        max_inj = self._get_effective_max_tokens(first_layer_weights)

        # In per-style mode each layer's strength is a weighted sum using
        # that layer's own weight blend.
        if self._style_layer_strengths is not None:
            strength_arg: "dict[int, float] | float" = {
                l: self._resolve_layer_strengths(per_layer_weights[l])[l]
                for l in self.insertion_layers
            }
        else:
            strength_arg = self._resolve_layer_strengths(first_layer_weights)

        vllm_caa.set_caa_state(
            active=True,
            strength=strength_arg,
            layers=self.insertion_layers,
            vectors=vectors,
            relative=self.relative_strength,
            max_injection_tokens=max_inj,
            inject_prefill=self.settings.caa_inject_prefill,
        )
        return per_layer_weights, max_inj

    def set_vector_from_metadata(
        self, metadata: dict
    ) -> tuple[dict[str, float], float, int | None]:
        """Set per-batch CAA state from metadata. Stateless and reproducible.

        Returns (style_weights, strength_scale, max_inj).
        """
        from .prf import metadata_to_caa_vector_with_hmac, sample_strength_scale_from_metadata_with_hmac

        vectors, style_weights = metadata_to_caa_vector_with_hmac(
            metadata,
            self._vectors_by_style,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
            style_scales=self.caa_style_scales,
            normalize_result=self.normalize_per_layer,
        )

        strength_scale = sample_strength_scale_from_metadata_with_hmac(
            metadata,
            p_active=self.settings.caa_p_active,
        )

        if strength_scale == 0.0:
            vllm_caa.disable_caa()
            return {s: 0.0 for s in self.styles}, 0.0, None

        max_inj = self._get_effective_max_tokens(style_weights)

        max_s = self._resolve_layer_strengths(style_weights)
        if isinstance(max_s, float):
            max_s = {layer: max_s for layer in self.insertion_layers}
        layer_strengths = {layer: strength_scale * max_s[layer] for layer in self.insertion_layers}

        vllm_caa.set_caa_state(
            active=True,
            strength=layer_strengths,
            layers=self.insertion_layers,
            vectors=vectors,
            relative=self.relative_strength,
            max_injection_tokens=max_inj,
            inject_prefill=self.settings.caa_inject_prefill,
        )
        return style_weights, strength_scale, max_inj

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

        Always restores the CAA state that was active before this call so
        subsequent sample_batch() calls see the correct steering vector.
        This matters in metadata mode where set_vector_from_metadata() is
        called once per batch: without the restore, disable_caa() inside
        this method would leave CAA disabled for all remaining batches.
        """
        saved_state = vllm_caa._load_caa_state()
        vllm_caa.disable_caa()
        try:
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
            # Restore the CAA state that was active before force_answers.
            # Prefer set_style() when a named style is active (keeps _current_style in sync);
            # fall back to re-writing the saved state directly for metadata/blended vectors.
            if self._current_style is not None:
                self.set_style(self._current_style)
            elif saved_state.get("active"):
                vllm_caa._save_caa_state(saved_state)

    def load_prompts(self):
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
            num_samples=self.settings.sample_limit,
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
