"""Custom vLLM model classes that inject CAA steering vectors.

Subclasses Gemma3's decoder layer to add a per-layer steering vector to the
residual stream after the layer's normal forward pass. The classes are
registered with vLLM's ModelRegistry so vllm.LLM(...) can construct them
when the model architecture is overridden via hf_overrides.

CAA state is stored in /tmp/vllm_caa_state (pickle) so it persists across
worker processes. Each CAA decoder layer pre-allocates a GPU steering buffer
and updates it in-place when the state file changes, making the forward pass
unconditional and CUDA-graph compatible.

Usage (registration is performed on import):

    from src import vllm_caa
    llm = LLM(
        model="unsloth/gemma-3-12b-it",
        hf_overrides={"architectures": ["Gemma3CAAForConditionalGeneration"]},
        # enforce_eager=True is required: the per-forward mtime check and buf.copy_()
        # only run when Python forward executes. Without enforce_eager, vLLM captures
        # the decoding forward into a CUDA graph and replays it without re-running
        # the Python state-update code, so style switches never take effect.
        ...,
    )
    vllm_caa.set_caa_state(active=True, strength=2.0,
                           layers=[10, 20], vectors={10: t10, 20: t20})
"""

from __future__ import annotations

import os
import pickle
import tempfile
import torch
import numpy as np

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.gemma3 import (
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
)
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration


_CAA_STATE_FILE = os.path.join(tempfile.gettempdir(), "vllm_caa_state")
_DEBUG_CAA = False

_STATE_CACHE: dict = {}
_STATE_CACHE_MTIME: float = -1.0

# Remove stale torch-format state file from previous runs if present
_legacy_file = _CAA_STATE_FILE + ".pt"
if os.path.exists(_legacy_file):
    try:
        os.remove(_legacy_file)
    except OSError:
        pass


def _load_caa_state() -> dict:
    global _STATE_CACHE, _STATE_CACHE_MTIME
    if os.path.exists(_CAA_STATE_FILE):
        try:
            mtime = os.path.getmtime(_CAA_STATE_FILE)
            if mtime == _STATE_CACHE_MTIME and _STATE_CACHE:
                return _STATE_CACHE
            with open(_CAA_STATE_FILE, "rb") as f:
                state = pickle.load(f)
            _STATE_CACHE = state
            _STATE_CACHE_MTIME = mtime
            if _DEBUG_CAA and state.get("active"):
                print(f"[VLLM_CAA] Loaded state: active={state['active']}, layers={state['layers']}")
            return state
        except Exception as e:
            print(f"[VLLM_CAA] Error loading state file: {e}")
    return {"active": False, "strength": 0.0, "layers": set(), "vectors": {}}


def _save_caa_state(state: dict) -> None:
    tmp = _CAA_STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _CAA_STATE_FILE)


def set_caa_state(
    *,
    active: bool | None = None,
    strength: "float | dict[int, float] | None" = None,
    layers=None,
    vectors: dict[int, torch.Tensor] | None = None,
    relative: bool | None = None,
) -> None:
    state = _load_caa_state()

    if active is not None:
        state["active"] = bool(active)
    if strength is not None:
        if isinstance(strength, dict):
            state["strengths"] = {int(k): float(v) for k, v in strength.items()}
            state["strength"] = float(next(iter(strength.values())))  # fallback scalar
        else:
            state["strength"] = float(strength)
            state["strengths"] = {}  # empty → all layers use state["strength"]
    if layers is not None:
        state["layers"] = set(int(l) for l in layers)
    if vectors is not None:
        state["vectors"] = {
            int(k): v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            for k, v in vectors.items()
        }
    if relative is not None:
        state["relative"] = bool(relative)

    _save_caa_state(state)

    if _DEBUG_CAA and active:
        print(f"[VLLM_CAA] Activated: strength={state.get('strengths') or state['strength']}, layers={state['layers']}, num_vectors={len(state['vectors'])}")


def disable_caa() -> None:
    state = _load_caa_state()
    state["active"] = False
    _save_caa_state(state)


class Gemma3CAADecoderLayer(Gemma3DecoderLayer):
    """Gemma3 decoder layer with CAA injection on the residual stream.

    Pre-allocates a GPU steering buffer (_steering_buffer) at init time.
    The forward pass unconditionally adds the buffer to the residual —
    making the computation graph static and CUDA-graph compatible.

    When the state file changes (detected via mtime), the buffer is updated
    in-place via copy_() before the addition, which is also CUDA-graph safe.
    """

    def forward(self, positions, hidden_states, residual, **kwargs):
        hidden_states, residual = super().forward(
            positions, hidden_states, residual, **kwargs
        )

        idx = getattr(self, "_caa_layer_idx", None)
        buf = getattr(self, "_steering_buffer", None)

        if idx is not None and buf is not None:
            try:
                current_mtime = (
                    os.path.getmtime(_CAA_STATE_FILE)
                    if os.path.exists(_CAA_STATE_FILE)
                    else -1.0
                )
            except OSError:
                current_mtime = -1.0

            if current_mtime != self._buffer_mtime:
                self._buffer_mtime = current_mtime
                state = _load_caa_state()
                if (
                    state.get("active")
                    and idx in state["layers"]
                    and idx in state["vectors"]
                ):
                    sv = torch.as_tensor(
                        state["vectors"][idx], dtype=buf.dtype, device=buf.device
                    )
                    self._steering_sv = sv
                    per_layer = state.get("strengths") or {}
                    self._steering_strength = float(per_layer.get(idx, state["strength"]))
                    self._steering_relative = bool(state.get("relative", False))
                    buf.copy_(self._steering_strength * sv)
                    if _DEBUG_CAA:
                        print(f"[VLLM_CAA] Layer {idx}: buffer updated, strength={state['strength']}, relative={self._steering_relative}")
                else:
                    self._steering_sv = None
                    buf.zero_()

            if self._steering_relative and self._steering_sv is not None:
                # Per-token norm scaling: strength is a fraction of each token's residual norm
                per_token_norm = residual.float().norm(dim=-1, keepdim=True).to(residual.dtype)
                residual = residual + self._steering_strength * per_token_norm * self._steering_sv
            else:
                # Static addition — CUDA-graph compatible path
                residual = residual + buf

        return hidden_states, residual


def _swap_decoder_layers(
    layers, hidden_size: int, dtype: torch.dtype, device: torch.device
) -> None:
    swapped = 0
    for idx, layer in enumerate(layers):
        if isinstance(layer, Gemma3DecoderLayer) and not isinstance(
            layer, Gemma3CAADecoderLayer
        ):
            layer.__class__ = Gemma3CAADecoderLayer
            layer._caa_layer_idx = idx
            layer._steering_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            layer._buffer_mtime = -1.0
            layer._steering_sv = None
            layer._steering_strength = 0.0
            layer._steering_relative = False
            swapped += 1
    if _DEBUG_CAA:
        print(f"[VLLM_CAA] Swapped {swapped}/{len(layers)} decoder layers (hidden_size={hidden_size})")


class Gemma3CAAForCausalLM(Gemma3ForCausalLM):
    """Text-only Gemma3 causal LM with CAA-aware decoder layers."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        param = next(self.parameters())
        hidden_size = vllm_config.model_config.hf_config.hidden_size
        if _DEBUG_CAA:
            print(f"[VLLM_CAA] Initializing Gemma3CAAForCausalLM, hidden_size={hidden_size}, {len(self.model.layers)} layers")
        _swap_decoder_layers(self.model.layers, hidden_size, param.dtype, param.device)


class Gemma3CAAForConditionalGeneration(Gemma3ForConditionalGeneration):
    """Multimodal Gemma3 with CAA-aware decoder layers in language_model."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        param = next(self.language_model.parameters())
        hidden_size = vllm_config.model_config.hf_config.text_config.hidden_size
        if _DEBUG_CAA:
            print(f"[VLLM_CAA] Initializing Gemma3CAAForConditionalGeneration, hidden_size={hidden_size}, {len(self.language_model.model.layers)} layers")
        _swap_decoder_layers(
            self.language_model.model.layers, hidden_size, param.dtype, param.device
        )


ModelRegistry.register_model(
    "Gemma3CAAForCausalLM",
    Gemma3CAAForCausalLM,
)
ModelRegistry.register_model(
    "Gemma3CAAForConditionalGeneration",
    Gemma3CAAForConditionalGeneration,
)
