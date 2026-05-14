"""HF transformers-based trace generator with CAA residual-stream steering.

Mirrors the surface area of VLLMTraceGenerator (sample_batch / load_prompts /
close) so generate_traces.py can swap it in for steered generation.
vLLM does not support residual-stream forward hooks, so this is HF-only.
"""

import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ANSWER_FORCE_STRING, Settings, normalize_caa_strength
from .loaders import get_loader


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
        caa_dirichlet_alphas: Optional[list[float]] = None,
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

        self._max_inject_tokens: int | None = settings.caa_max_injection_tokens
        self._inject_prefill: bool = settings.caa_inject_prefill
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
                "decode_count": 0, "max_inject": settings.caa_max_injection_tokens,
                "inject_prefill": settings.caa_inject_prefill,
                "cap_logged": False,
            }

            def make_hook(s, layer_idx=layer_idx):
                def _hook(_module, _input, output):
                    if not s["active"] or s["vector"] is None:
                        return output
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Skip prefill if not explicitly enabled.
                    if hidden.shape[1] > 1:
                        if not s["inject_prefill"]:
                            print(f"[CAA dbg] layer={layer_idx} prefill (seq_len={hidden.shape[1]}) SKIP")
                            return output
                        print(f"[CAA dbg] layer={layer_idx} prefill (seq_len={hidden.shape[1]}) INJECT")
                    else:
                        # Token cap (only applies during decode steps, i.e. hidden.shape[1] == 1)
                        if s["max_inject"] is not None:
                            if s["decode_count"] >= s["max_inject"]:
                                if not s["cap_logged"]:
                                    print(f"[CAA] injection cap reached at token {s['max_inject']} — steering disabled for remainder of generation")
                                    s["cap_logged"] = True
                                print(f"[CAA dbg] layer={layer_idx} token={s['decode_count']} SKIP (cap={s['max_inject']})")
                                return output
                            s["decode_count"] += 1
                        print(f"[CAA dbg] layer={layer_idx} token={s['decode_count']} INJECT")
                    v = s["vector"].to(hidden.device, hidden.dtype)  # [1, H]
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
        return {l: float(self.strength) for l in self.insertion_layers}

    def _get_effective_max_tokens(self, weights: dict[str, float] | str) -> int | None:
        cfg = self.settings.caa_max_injection_tokens
        if cfg is None or isinstance(cfg, int):
            return cfg
        
        # If cfg is a list, it must match self.styles length
        if isinstance(cfg, list):
            if len(cfg) != len(self.styles):
                raise ValueError(f"CAA_MAX_INJECTION_TOKENS list length {len(cfg)} != styles count {len(self.styles)}")
            
            if isinstance(weights, str):
                # Single style mode
                idx = self.style_to_idx.get(weights)
                return cfg[idx] if idx is not None else None
            else:
                # Weighted metadata mode
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
            state["b"] = layer_strengths[layer_idx]
            state["active"] = True
            max_inj = self._get_effective_max_tokens(style)
            state["max_inject"] = max_inj
            state["inject_prefill"] = self._inject_prefill
            state["decode_count"] = 0
            state["cap_logged"] = False

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
            state["decode_count"] = 0
            state["cap_logged"] = False
        return per_layer_weights, max_inj

    def set_vector_from_metadata(
        self, metadata: dict
    ) -> tuple[dict[str, float], float, int | None]:
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
            return {s: 0.0 for s in self.styles}, 0.0, None
        
        max_inj = self._get_effective_max_tokens(style_weights)
        
        max_s = self._resolve_layer_strengths(style_weights)
        for state, layer_idx in zip(self._states, self.insertion_layers):
            state["vector"] = vectors[layer_idx].unsqueeze(0) if vectors[layer_idx].dim() == 1 else vectors[layer_idx]
            state["b"] = strength_scale * max_s[layer_idx]
            state["active"] = True
            state["max_inject"] = max_inj
            state["inject_prefill"] = self._inject_prefill
            state["decode_count"] = 0
            state["cap_logged"] = False

        return style_weights, strength_scale, max_inj

    def disable_steering(self) -> None:
        for state in self._states:
            state["active"] = False

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
        for i in range(0, len(rendered), self.batch_size):
            batch = rendered[i : i + self.batch_size]
            for s in self._states:
                s["decode_count"] = 0
                s["cap_logged"] = False
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Report how many decode steps were steered for this sub-batch.
            # All layers share the same count (they fire in sync), so layer 0 is representative.
            if self._states:
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
