"""Custom vLLM model classes that inject CAA steering vectors.

Subclasses Gemma3's decoder layer to add a per-layer steering vector to the
residual stream after the layer's normal forward pass. The classes are
registered with vLLM's ModelRegistry so vllm.LLM(...) can construct them
when the model architecture is overridden via hf_overrides.

Usage (registration is performed on import):

    from src import vllm_caa
    llm = LLM(
        model="unsloth/gemma-3-12b-it",
        hf_overrides={"architectures": ["Gemma3CAAForConditionalGeneration"]},
        enforce_eager=True,  # required: torch.compile would freeze the original forward
        ...,
    )
    vllm_caa.set_caa_state(active=True, strength=2.0,
                           layers=[10, 20], vectors={10: t10, 20: t20})
"""

from __future__ import annotations

import torch

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.gemma3 import (
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
)
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration


_CAA_STATE: dict = {
    "active": False,
    "strength": 0.0,
    "layers": set(),
    "vectors": {},  # dict[int, torch.Tensor] of shape [H]
}


def set_caa_state(
    *,
    active: bool | None = None,
    strength: float | None = None,
    layers=None,
    vectors: dict[int, torch.Tensor] | None = None,
) -> None:
    if active is not None:
        _CAA_STATE["active"] = bool(active)
    if strength is not None:
        _CAA_STATE["strength"] = float(strength)
    if layers is not None:
        _CAA_STATE["layers"] = set(int(l) for l in layers)
    if vectors is not None:
        _CAA_STATE["vectors"] = {int(k): v for k, v in vectors.items()}


def disable_caa() -> None:
    _CAA_STATE["active"] = False


class Gemma3CAADecoderLayer(Gemma3DecoderLayer):
    """Gemma3 decoder layer with CAA injection on the residual stream.

    vLLM's Gemma3DecoderLayer returns (hidden_states, residual) using a fused
    residual scheme; the residual stream presented to the next layer's
    input layernorm is hidden_states + residual. Adding the steering vector
    to `residual` is equivalent to adding it to the layer's residual-stream
    output (matching CAA's HF forward-hook on layer.output[0]).
    """

    def forward(self, positions, hidden_states, residual, **kwargs):
        hidden_states, residual = super().forward(
            positions, hidden_states, residual, **kwargs
        )
        if _CAA_STATE["active"]:
            idx = getattr(self, "_caa_layer_idx", None)
            if idx is not None and idx in _CAA_STATE["layers"]:
                sv = _CAA_STATE["vectors"].get(idx)
                if sv is not None:
                    add = (_CAA_STATE["strength"] * sv).to(
                        device=residual.device, dtype=residual.dtype
                    )
                    residual = residual + add
        return hidden_states, residual


def _swap_decoder_layers(layers) -> None:
    for idx, layer in enumerate(layers):
        if isinstance(layer, Gemma3DecoderLayer) and not isinstance(
            layer, Gemma3CAADecoderLayer
        ):
            layer.__class__ = Gemma3CAADecoderLayer
            layer._caa_layer_idx = idx


class Gemma3CAAForCausalLM(Gemma3ForCausalLM):
    """Text-only Gemma3 causal LM with CAA-aware decoder layers."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _swap_decoder_layers(self.model.layers)


class Gemma3CAAForConditionalGeneration(Gemma3ForConditionalGeneration):
    """Multimodal Gemma3 with CAA-aware decoder layers in language_model."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _swap_decoder_layers(self.language_model.model.layers)


ModelRegistry.register_model(
    "Gemma3CAAForCausalLM",
    Gemma3CAAForCausalLM,
)
ModelRegistry.register_model(
    "Gemma3CAAForConditionalGeneration",
    Gemma3CAAForConditionalGeneration,
)
