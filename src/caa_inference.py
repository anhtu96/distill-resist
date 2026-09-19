"""HF transformers-based trace generator with CAA residual-stream steering.

Mirrors the surface area of VLLMTraceGenerator (sample_batch / load_prompts /
close) so generate_traces.py can swap it in for steered generation.
vLLM does not support residual-stream forward hooks, so this is HF-only.
"""

import gc
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ANSWER_FORCE_STRING, Settings, normalize_caa_strength
from .loaders import get_loader

_DEBUG_CAA = os.getenv("CAA_DEBUG", "0") == "1"


def _is_bench_mode() -> bool:
    return os.getenv("CAA_BENCH_MODE", "0") == "1"


def _get_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx]
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model structure: {type(model)}")


class HFCAATraceGenerator:
    """Batch generator that injects CAA steering vectors at specified layers.

    Loads a checkpoint produced by train_caa_vectors.py (residual injection),
    registers forward hooks once, and exposes set_style() to swap which
    style's vector is active per generation pass.
    """

    def __init__(
        self,
        settings: Settings,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        caa_checkpoint: str,
        system_prompt: Optional[str] = None,
        batch_size: int = 4,
        strength: "float | list[float] | dict[str, float | list[float]]" = 5.0,
        injection_layers: Optional[list[int]] = None,
        caa_weight_mode: Optional[str] = None,
        caa_dirichlet_alphas: "list[float] | float | None" = None,
        caa_style_scales: Optional[dict[str, float]] = None,
        normalize_per_layer: bool = False,
        relative_strength: bool = False,
        user_id: str = "synthetic",
    ):
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
        self._tokenizer.padding_side = "left"

        eos_ids = []
        if self._tokenizer.eos_token_id is not None:
            if isinstance(self._tokenizer.eos_token_id, list):
                eos_ids.extend(self._tokenizer.eos_token_id)
            else:
                eos_ids.append(self._tokenizer.eos_token_id)
        eot_id = self._tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is not None and eot_id != self._tokenizer.unk_token_id and eot_id not in eos_ids:
            eos_ids.append(eot_id)
        self._eos_token_ids = eos_ids if eos_ids else None

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self._model.eval()

        ckpt = torch.load(caa_checkpoint, map_location="cpu", weights_only=False)
        injection_point = ckpt.get("injection_point", "residual")
        if injection_point != "residual":
            raise ValueError(
                f"CAA checkpoint injection_point is '{injection_point}', expected 'residual'."
            )
        self.insertion_layers: list[int] = list(ckpt["insertion_layers"])
        self.steering_vectors: dict = ckpt["steering_vectors"]
        self.styles: list[str] = ckpt.get("styles", list(self.steering_vectors))
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
        # Back-compat: scalar / per-layer dict (no per-style) populates _layer_strengths.
        if self._style_layer_strengths is not None:
            self._layer_strengths = None
            print(f"[CAA] Per-style per-layer strengths: {self._style_layer_strengths}")
        elif isinstance(self._strength_norm, dict):
            self._layer_strengths = self._strength_norm
            print(f"[CAA] Per-layer strengths: {self._layer_strengths}")
        else:
            self._layer_strengths = None
            print(f"[CAA] Strength (all layers): {self.strength}")

        # Filter steering_vectors to only include selected layers
        original_insertion_layers = list(ckpt["insertion_layers"])
        new_steering_vectors: dict = {}
        for style, sv_list in self.steering_vectors.items():
            picked = [sv_list[original_insertion_layers.index(l)] for l in self.insertion_layers]
            new_steering_vectors[style] = picked
        self.steering_vectors = new_steering_vectors

        self._max_inject_tokens = settings.caa_max_injection_tokens
        self._inject_prefill: bool = settings.caa_inject_prefill
        # Raw config (may be int, list, dict, or None). Per-batch resolution to a
        # scalar happens in set_style / set_random_per_layer_vectors / set_vector_from_metadata.
        self._inject_prefill_last_n = settings.caa_inject_prefill_last_n
        if self._max_inject_tokens is not None:
            print(f"[CAA] max_inject_tokens={self._max_inject_tokens} (injection stops after this many decode steps)")
        else:
            print(f"[CAA] max_inject_tokens=None (injection runs for full generation)")

        self._states: list[dict] = []
        for layer_idx in self.insertion_layers:
            if self._layer_strengths is not None:
                layer_b = self._layer_strengths[layer_idx]
            elif self._style_layer_strengths is not None:
                layer_b = 0.0  # filled in by set_style/set_*
            else:
                layer_b = float(strength)
            state = {
                "active": False, "b": layer_b, "vector": None, "relative": self.relative_strength,
                "decode_count": 0,
                # Per-batch resolved values populated by set_style/set_*; these defaults are
                # overwritten before activation. None is safe because s["active"] gates the hook.
                "max_inject": None,
                "inject_prefill": settings.caa_inject_prefill,
                "inject_prefill_last_n": None,
                "cap_logged": False,
                "vector_gpu": None,
                "sv_gpu": None,
            }

            def make_hook(s, layer_idx=layer_idx):
                def _hook(_module, _input, output):
                    if not s["active"] or s["vector"] is None:
                        return output
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Skip prefill if not explicitly enabled.
                    if hidden.shape[1] > 1:
                        if not s["inject_prefill"]:
                            if _DEBUG_CAA:
                                print(f"[CAA dbg] layer={layer_idx} prefill (seq_len={hidden.shape[1]}) SKIP")
                            return output
                        last_n = s["inject_prefill_last_n"]
                        if last_n is not None and last_n <= 0:
                            if _DEBUG_CAA:
                                print(f"[CAA dbg] layer={layer_idx} prefill (seq_len={hidden.shape[1]}) SKIP (last_n={last_n})")
                            return output
                        if _DEBUG_CAA:
                            print(f"[CAA dbg] layer={layer_idx} prefill (seq_len={hidden.shape[1]}) INJECT last_n={last_n}")
                        v = (
                            s["vector_gpu"]
                            if (s.get("vector_gpu") is not None and s["vector_gpu"].device == hidden.device and s["vector_gpu"].dtype == hidden.dtype)
                            else s["vector"].to(hidden.device, hidden.dtype)
                        )
                        if last_n is not None:
                            h_tail = hidden[:, -last_n:, :]
                            if s["relative"]:
                                per_token_norm = h_tail.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                                delta = s["b"] * per_token_norm * v
                            else:
                                delta = (
                                    s["sv_gpu"]
                                    if (s.get("sv_gpu") is not None and s["sv_gpu"].device == hidden.device and s["sv_gpu"].dtype == hidden.dtype)
                                    else s["b"] * v
                                )
                            hidden = hidden.clone()
                            hidden[:, -last_n:, :] = h_tail + delta
                        else:
                            if s["relative"]:
                                per_token_norm = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                                delta = s["b"] * per_token_norm * v
                            else:
                                delta = (
                                    s["sv_gpu"]
                                    if (s.get("sv_gpu") is not None and s["sv_gpu"].device == hidden.device and s["sv_gpu"].dtype == hidden.dtype)
                                    else s["b"] * v
                                )
                            hidden = hidden + delta
                        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                    else:
                        # Token cap (only applies during decode steps, i.e. hidden.shape[1] == 1)
                        if s["max_inject"] is not None:
                            if s["decode_count"] >= s["max_inject"]:
                                if not s["cap_logged"]:
                                    if _DEBUG_CAA:
                                        print(f"[CAA] injection cap reached at token {s['max_inject']} — steering disabled for remainder of generation")
                                    s["cap_logged"] = True
                                if _DEBUG_CAA:
                                    print(f"[CAA dbg] layer={layer_idx} token={s['decode_count']} SKIP (cap={s['max_inject']})")
                                return output
                            s["decode_count"] += 1
                        if _DEBUG_CAA:
                            print(f"[CAA dbg] layer={layer_idx} token={s['decode_count']} INJECT")

                    # Decode token injection (hidden.shape[1] == 1):
                    # In bench mode, avoid per-token .to(device, dtype) and scalar multiplication
                    if s.get("sv_gpu") is not None and s["sv_gpu"].device == hidden.device and s["sv_gpu"].dtype == hidden.dtype:
                        if s["relative"]:
                            per_token_norm = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                            sv = per_token_norm * s["sv_gpu"]
                        else:
                            sv = s["sv_gpu"]
                    else:
                        v = s["vector"].to(hidden.device, hidden.dtype)  # [1, H]
                        if _is_bench_mode():
                            s["vector_gpu"] = v
                            s["sv_gpu"] = s["b"] * v
                        if s["relative"]:
                            per_token_norm = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                            sv = s["b"] * per_token_norm * v
                        else:
                            sv = s["b"] * v
                    hidden = hidden + sv
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return _hook

            _get_layer(self._model, layer_idx).register_forward_hook(make_hook(state))
            self._states.append(state)
        print(f"Loaded CAA checkpoint: layers={self.insertion_layers}, styles={self.styles}")

    def _cache_state_vectors(self, state: dict, layer_idx: int) -> None:
        """Pre-cache steering vector on GPU in device/dtype to eliminate per-forward overhead."""
        if _is_bench_mode() and state.get("vector") is not None:
            try:
                layer_module = _get_layer(self._model, layer_idx)
                param = next(layer_module.parameters())
                dev, dt = param.device, param.dtype
                v_gpu = state["vector"].to(device=dev, dtype=dt)
                state["vector_gpu"] = v_gpu
                state["sv_gpu"] = state["b"] * v_gpu
            except Exception:
                state["vector_gpu"] = None
                state["sv_gpu"] = None
        else:
            state["vector_gpu"] = None
            state["sv_gpu"] = None

    def _resolve_layer_strengths(
        self, weights_or_style: "str | dict[str, float]"
    ) -> dict[int, float]:
        """Resolve per-layer strength for a single style name or a weights blend.

        - scalar self.strength: every layer -> self.strength
        - per-layer dict (self._layer_strengths): use directly
        - per-style dict (self._style_layer_strengths):
            * single style name: pick that style's per-layer strengths
            * weights dict: layer L -> sum_style w[style] * strength[style][L]
              (styles missing from the dict contribute 0)
        """
        if self._style_layer_strengths is not None:
            if isinstance(weights_or_style, str):
                if weights_or_style not in self._style_layer_strengths:
                    raise ValueError(
                        f"Style '{weights_or_style}' has no entry in caa_strength dict. "
                        f"Provided: {list(self._style_layer_strengths)}"
                    )
                per_layer = self._style_layer_strengths[weights_or_style]
                return {l: per_layer.get(l, 0.0) for l in self.insertion_layers}
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
        return {l: float(self.strength) for l in self.insertion_layers}

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

        # Positional list: indexed by checkpoint style order
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

    def _get_effective_inject_prefill_last_n(
        self, weights: "dict[str, float] | str"
    ) -> int | None:
        cfg = self.settings.caa_inject_prefill_last_n
        blend_mode = self.settings.caa_inject_prefill_last_n_blend_mode
        if cfg is None or isinstance(cfg, int):
            return cfg

        if isinstance(cfg, dict):
            unknown = set(cfg) - set(self.styles)
            if unknown:
                raise ValueError(
                    f"CAA_INJECT_PREFILL_LAST_N dict has unknown styles {sorted(unknown)}; "
                    f"checkpoint styles: {self.styles}"
                )
            if isinstance(weights, str):
                v = cfg.get(weights)
                return int(v) if v is not None else None
            if blend_mode == "dominant":
                dom = max(weights, key=weights.get)
                v = cfg.get(dom)
                return int(v) if v is not None else None
            effective = 0.0
            for style, w in weights.items():
                v = cfg.get(style)
                if v is not None:
                    effective += float(w) * float(v)
            return int(round(effective))

        if isinstance(cfg, list):
            if len(cfg) != len(self.styles):
                raise ValueError(
                    f"CAA_INJECT_PREFILL_LAST_N list length {len(cfg)} != styles count {len(self.styles)}"
                )
            if isinstance(weights, str):
                idx = self.style_to_idx.get(weights)
                return cfg[idx] if idx is not None else None
            if blend_mode == "dominant":
                dom = max(weights, key=weights.get)
                idx = self.style_to_idx.get(dom)
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
        if style not in self.steering_vectors:
            raise ValueError(f"Style '{style}' not in checkpoint. Available: {self.styles}")
        scale = self.caa_style_scales.get(style, 1.0) if self.caa_style_scales else 1.0
        layer_strengths = self._resolve_layer_strengths(style)
        for state, layer_idx, sv_np in zip(
            self._states, self.insertion_layers, self.steering_vectors[style]
        ):
            sv = torch.tensor(sv_np, dtype=torch.float32)
            if self.normalize_per_layer:
                sv = sv / (sv.norm() + 1e-8)
            sv = sv * scale
            if sv.dim() == 1:
                sv = sv.unsqueeze(0)
            state["vector"] = sv
            state["b"] = layer_strengths.get(layer_idx, 0.0)
            state["active"] = True
            max_inj = self._get_effective_max_tokens(style)
            last_n = self._get_effective_inject_prefill_last_n(style)
            state["max_inject"] = max_inj
            state["inject_prefill"] = self._inject_prefill
            state["inject_prefill_last_n"] = last_n
            state["decode_count"] = 0
            state["cap_logged"] = False
            self._cache_state_vectors(state, layer_idx)

    def set_random_per_layer_vectors(self) -> dict[int, dict[str, float]]:
        from .prf import random_per_layer_caa_vectors
        vectors_by_style_tensor = {
            style: {l: torch.as_tensor(sv, dtype=torch.float32) for l, sv in zip(self.insertion_layers, sv_list)}
            for style, sv_list in self.steering_vectors.items()
        }
        vectors, per_layer_weights = random_per_layer_caa_vectors(
            vectors_by_style_tensor,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
            style_scales=self.caa_style_scales,
            normalize_result=self.normalize_per_layer,
        )
        # Get weights from the first layer (they are currently shared across layers)
        first_layer_weights = next(iter(per_layer_weights.values()))
        max_inj = self._get_effective_max_tokens(first_layer_weights)
        last_n = self._get_effective_inject_prefill_last_n(first_layer_weights)

        for state, layer_idx in zip(self._states, self.insertion_layers):
            v = vectors[layer_idx]
            state["vector"] = v.unsqueeze(0) if v.dim() == 1 else v
            # In per-style mode each layer's strength is a weighted sum over styles
            # using that layer's own weight blend.
            if self._style_layer_strengths is not None:
                state["b"] = self._resolve_layer_strengths(per_layer_weights[layer_idx])[layer_idx]
            state["active"] = True
            state["max_inject"] = max_inj
            state["inject_prefill"] = self._inject_prefill
            state["inject_prefill_last_n"] = last_n
            state["decode_count"] = 0
            state["cap_logged"] = False
            self._cache_state_vectors(state, layer_idx)
        return per_layer_weights, max_inj, last_n

    def set_vector_from_metadata(
        self, metadata: dict
    ) -> tuple[dict[str, float], float, int | None, int | None]:
        from .prf import metadata_to_caa_vector, sample_strength_scale_from_metadata
        # metadata_to_caa_vector expects vectors_by_style to be {style: {layer: tensor}}
        # Our self.steering_vectors is {style: [np_array, ...]}
        vectors_by_style_tensor = {}
        for style, sv_list in self.steering_vectors.items():
            vectors_by_style_tensor[style] = {
                l: torch.as_tensor(sv, dtype=torch.float32)
                for l, sv in zip(self.insertion_layers, sv_list)
            }
        
        vectors, style_weights = metadata_to_caa_vector(
            self._rng,
            vectors_by_style_tensor,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
            style_scales=self.caa_style_scales,
            normalize_result=self.normalize_per_layer,
        )

        strength_scale = sample_strength_scale_from_metadata(
            self._rng,
            p_active=self.settings.caa_p_active,
        )

        if strength_scale == 0.0:
            self.disable_steering()
            return {s: 0.0 for s in self.styles}, 0.0, None, None

        max_inj = self._get_effective_max_tokens(style_weights)
        last_n = self._get_effective_inject_prefill_last_n(style_weights)

        max_s = self._resolve_layer_strengths(style_weights)
        for state, layer_idx in zip(self._states, self.insertion_layers):
            state["vector"] = vectors[layer_idx].unsqueeze(0) if vectors[layer_idx].dim() == 1 else vectors[layer_idx]
            state["b"] = strength_scale * max_s[layer_idx]
            state["active"] = True
            state["max_inject"] = max_inj
            state["inject_prefill"] = self._inject_prefill
            state["inject_prefill_last_n"] = last_n
            state["decode_count"] = 0
            state["cap_logged"] = False
            self._cache_state_vectors(state, layer_idx)

        return style_weights, strength_scale, max_inj, last_n

    def disable_steering(self) -> None:
        for state in self._states:
            state["active"] = False
            state["vector_gpu"] = None
            state["sv_gpu"] = None

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
            return f"System: {system_prompt}\nUser: {prompt}\nAssistant:\n"

    def sample_batch(
        self, prompts: list[str], system_prompt: Optional[str] = None
    ) -> list[str]:
        sys_prompt = system_prompt or self.system_prompt
        rendered = [self._render(p, sys_prompt) for p in prompts]

        results: list[str] = []
        ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
        for i in range(0, len(rendered), self.batch_size):
            batch = rendered[i : i + self.batch_size]
            for s in self._states:
                s["decode_count"] = 0
                s["cap_logged"] = False
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False,
            ).to(self._model.device)

            with ctx:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    eos_token_id=self._eos_token_ids,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            if self._states and _DEBUG_CAA:
                injected = self._states[0]["decode_count"]
                cap = self._states[0]["max_inject"]
                cap_str = f"/{cap}" if cap is not None else " (no cap)"
                print(f"[CAA] sub-batch steered {injected}{cap_str} decode steps")

            input_len = inputs["input_ids"].shape[1]
            for out in outputs:
                new_tokens = out[input_len:]
                results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results

    def force_answers(self, responses: list[str]) -> list[str]:
        """Append ANSWER_FORCE_STRING and greedily decode the forced answer.

        Returns response + ANSWER_FORCE_STRING + decoded suffix, matching
        the vLLM answer-forcing behavior in generate_traces.py.
        """
        self.disable_steering()
        try:
            af_prompts = [resp.strip() + ANSWER_FORCE_STRING for resp in responses]
            results: list[str] = []
            for i in range(0, len(af_prompts), self.batch_size):
                batch = af_prompts[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=False,
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.settings.answer_force_max_tokens,
                        do_sample=False,
                        eos_token_id=self._eos_token_ids,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                for af, out in zip(batch, outputs):
                    new_tokens = out[input_len:]
                    decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results.append(af + decoded + "}\\]")
            return results
        finally:
            if self._current_style is not None:
                self.set_style(self._current_style)

    def load_prompts(self):
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
            num_samples=self.settings.sample_limit,
        )
        return loader.load()

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
