"""Custom vLLM model classes that inject CAA steering vectors.

Subclasses Gemma3's decoder layer to add a per-layer steering vector to the
residual stream after the layer's normal forward pass. The classes are
registered with vLLM's ModelRegistry so vllm.LLM(...) can construct them
when the model architecture is overridden via hf_overrides.

CAA state is stored in /tmp/vllm_caa_state.pt so it persists across worker processes.

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

import os
import tempfile
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.gemma3 import (
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
)
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration


_CAA_STATE_FILE = os.path.join(tempfile.gettempdir(), "vllm_caa_state.pt")
_DEBUG_CAA = False  # Toggle CAA debug logging

# Cache for CAA state to avoid repeated torch.load() calls
_STATE_CACHE: dict = {}
_STATE_CACHE_MTIME: float = -1.0


def _load_caa_state() -> dict:
    """Load CAA state from shared file (works across process boundaries)."""
    global _STATE_CACHE, _STATE_CACHE_MTIME
    if os.path.exists(_CAA_STATE_FILE):
        try:
            mtime = os.path.getmtime(_CAA_STATE_FILE)
            if mtime == _STATE_CACHE_MTIME and _STATE_CACHE:
                return _STATE_CACHE
            state = torch.load(_CAA_STATE_FILE, weights_only=False)
            _STATE_CACHE = state
            _STATE_CACHE_MTIME = mtime
            if _DEBUG_CAA and state.get("active"):
                print(f"[VLLM_CAA] Loaded state: active={state['active']}, layers={state['layers']}")
            return state
        except Exception as e:
            print(f"[VLLM_CAA] Error loading state file: {e}")
            pass
    return {"active": False, "strength": 0.0, "layers": set(), "vectors": {}}


def set_caa_state(
    *,
    active: bool | None = None,
    strength: float | None = None,
    layers=None,
    vectors: dict[int, torch.Tensor] | None = None,
) -> None:
    state = _load_caa_state()
    
    if active is not None:
        state["active"] = bool(active)
    if strength is not None:
        state["strength"] = float(strength)
    if layers is not None:
        state["layers"] = set(int(l) for l in layers)
    if vectors is not None:
        state["vectors"] = {int(k): v for k, v in vectors.items()}
    
    torch.save(state, _CAA_STATE_FILE)
    
    if _DEBUG_CAA and active:
        print(f"[VLLM_CAA] Activated: strength={state['strength']}, layers={state['layers']}, num_vectors={len(state['vectors'])}")


def disable_caa() -> None:
    state = _load_caa_state()
    state["active"] = False
    torch.save(state, _CAA_STATE_FILE)


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

        idx = getattr(self, "_caa_layer_idx", None)

        # Check class-level state first (most reliable for vLLM workers)
        active = getattr(self.__class__, "_caa_class_active", False)
        strength = getattr(self.__class__, "_caa_class_strength", 0.0)
        vectors = getattr(self.__class__, "_caa_class_vectors", {})
        target_layers = getattr(self.__class__, "_caa_class_layers", set())

        if _DEBUG_CAA and idx == 24:
            print(f"[VLLM_CAA] Layer {idx}: class_active={active}, class_vectors_len={len(vectors)}")

        # Fall back to layer-local state
        if not active:
            active = getattr(self, "_caa_active", False)
            strength = getattr(self, "_caa_strength", 0.0)
            vectors = getattr(self, "_caa_vectors", {})
            if _DEBUG_CAA and idx == 24:
                print(f"[VLLM_CAA] Layer {idx}: layer_active={active}, layer_vectors_len={len(vectors)}")

        # Fall back to file-based state
        if not active:
            caa_state = _load_caa_state()
            active = caa_state["active"]
            strength = caa_state["strength"]
            vectors = caa_state["vectors"]
            target_layers = caa_state["layers"]
            if _DEBUG_CAA and idx == 24:
                print(f"[VLLM_CAA] Layer {idx}: file_active={active}, file_vectors_len={len(vectors)}")

        if active and idx is not None and idx in target_layers and idx in vectors:
            sv = vectors.get(idx)
            if sv is not None:
                if _DEBUG_CAA:
                    print(f"[VLLM_CAA] INJECTING at layer {idx}, strength={strength}, sv_shape={sv.shape}")
                add = (strength * sv).to(
                    device=residual.device, dtype=residual.dtype
                )
                residual = residual + add
        elif _DEBUG_CAA and idx == 24:
            print(f"[VLLM_CAA] Layer {idx}: NOT injecting - active={active}, idx={idx}, idx_in_target={idx in target_layers if target_layers else 'N/A'}, idx_in_vectors={idx in vectors}")

        return hidden_states, residual


def _swap_decoder_layers(layers) -> None:
    swapped_count = 0
    for idx, layer in enumerate(layers):
        if isinstance(layer, Gemma3DecoderLayer) and not isinstance(
            layer, Gemma3CAADecoderLayer
        ):
            layer.__class__ = Gemma3CAADecoderLayer
            layer._caa_layer_idx = idx
            swapped_count += 1
    if _DEBUG_CAA:
        print(f"[VLLM_CAA] Swapped {swapped_count}/{len(layers)} decoder layers")


class Gemma3CAAForCausalLM(Gemma3ForCausalLM):
    """Text-only Gemma3 causal LM with CAA-aware decoder layers."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if _DEBUG_CAA:
            print(f"[VLLM_CAA] Initializing Gemma3CAAForCausalLM with {len(self.model.layers)} decoder layers")
        _swap_decoder_layers(self.model.layers)


class Gemma3CAAForConditionalGeneration(Gemma3ForConditionalGeneration):
    """Multimodal Gemma3 with CAA-aware decoder layers in language_model."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if _DEBUG_CAA:
            print(f"[VLLM_CAA] Initializing Gemma3CAAForConditionalGeneration with {len(self.language_model.model.layers)} decoder layers")
        _swap_decoder_layers(self.language_model.model.layers)


ModelRegistry.register_model(
    "Gemma3CAAForCausalLM",
    Gemma3CAAForCausalLM,
)
ModelRegistry.register_model(
    "Gemma3CAAForConditionalGeneration",
    Gemma3CAAForConditionalGeneration,
)
