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
_COLLECT_STATE_FILE = os.path.join(tempfile.gettempdir(), "vllm_caa_collect_state")

_DEBUG_CAA = os.getenv("CAA_DEBUG", "0") == "1"
_UNSET = object()

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


def _load_collect_state() -> dict:
    if os.path.exists(_COLLECT_STATE_FILE):
        try:
            with open(_COLLECT_STATE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"active": False, "layers": []}


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
    max_injection_tokens: "int | None" = _UNSET,
    inject_prefill: bool | None = None,
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
    if max_injection_tokens is not _UNSET:
        # None means "no cap" — always write so stale values from previous runs are cleared.
        state["max_injection_tokens"] = int(max_injection_tokens) if max_injection_tokens is not None else None
    if inject_prefill is not None:
        state["inject_prefill"] = bool(inject_prefill)

    _save_caa_state(state)

    if _DEBUG_CAA and active:
        print(f"[VLLM_CAA] Activated: strength={state.get('strengths') or state['strength']}, layers={state['layers']}, num_vectors={len(state['vectors'])}")


def disable_caa() -> None:
    state = _load_caa_state()
    state["active"] = False
    _save_caa_state(state)


def set_collect_mode(active: bool, layers=None) -> None:
    """Enable/disable activation-collection mode. Disables injection first."""
    disable_caa()  # ensure no steering contaminates collected activations
    state = {"active": bool(active), "layers": list(layers or [])}
    
    # Clean up any existing activation files
    if layers:
        for idx in layers:
            path = os.path.join(tempfile.gettempdir(), f"vllm_caa_act_{idx}.npy")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    tmp = _COLLECT_STATE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _COLLECT_STATE_FILE)


def read_collected_activations(layers) -> dict:
    """Read per-layer mean activations written by workers during collect mode.
    Returns {layer_idx: np.ndarray of shape [H]}. Missing layers are omitted."""
    result = {}
    for idx in layers:
        path = os.path.join(tempfile.gettempdir(), f"vllm_caa_act_{idx}.npy")
        if os.path.exists(path):
            result[idx] = np.load(path)
    return result


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

        # --- Collect mode (training only): capture mean prefill activation ---
        if idx is not None and len(positions) > 1:  # prefill only
            collect = _load_collect_state()
            if collect.get("active") and idx in collect.get("layers", []):
                full = (residual + hidden_states).float()
                mean_act = full.mean(dim=0).cpu().numpy()
                np.save(
                    os.path.join(tempfile.gettempdir(), f"vllm_caa_act_{idx}.npy"),
                    mean_act,
                )
        # --- End collect mode ---

        buf = getattr(self, "_steering_buffer", None)

        if idx is not None and buf is not None:
            # Reload state first so _inject_prefill and other flags are current
            # before any prefill/decode decision. This must come before the prefill
            # guard so the very first prefill pass sees the correct inject_prefill value.
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
                    self._max_injection_tokens = state.get("max_injection_tokens")
                    self._inject_prefill = bool(state.get("inject_prefill", False))
                    self._cap_logged = False
                    buf.copy_(self._steering_strength * sv)
                    if _DEBUG_CAA:
                        print(f"[VLLM_CAA] Layer {idx}: buffer updated, strength={self._steering_strength}, relative={self._steering_relative}, max_inject={self._max_injection_tokens}")
                else:
                    self._steering_sv = None
                    buf.zero_()

            # Layers not in state["layers"] have _steering_sv=None and buf=zeros — skip them.
            if self._steering_sv is None:
                return hidden_states, residual

            # Skip prefill unless explicitly enabled (checked after state reload so _inject_prefill is current).
            if len(positions) > 1:
                # Prefill: reset decode counter for the upcoming generation.
                self._decode_step = 0
                self._cap_logged = False
                if not self._inject_prefill:
                    if _DEBUG_CAA:
                        print(f"[CAA dbg] layer={idx} prefill (seq_len={len(positions)}) SKIP")
                    return hidden_states, residual
                if _DEBUG_CAA:
                    print(f"[CAA dbg] layer={idx} prefill (seq_len={len(positions)}) INJECT")
            else:
                # Decode step: apply token cap using a simple counter (no prompt-length subtraction).
                if self._max_injection_tokens is not None:
                    if self._decode_step >= self._max_injection_tokens:
                        if _DEBUG_CAA and not self._cap_logged:
                            print(f"[CAA dbg] layer={idx} token={self._decode_step} SKIP (cap={self._max_injection_tokens}) — injection disabled for remainder")
                            self._cap_logged = True
                        return hidden_states, residual
                if _DEBUG_CAA:
                    print(f"[CAA dbg] layer={idx} token={self._decode_step} INJECT")
                self._decode_step += 1

            if self._steering_relative and self._steering_sv is not None:
                # Use full post-layer output (residual + ffn_delta) for per-token norm to match
                # HF's hook convention where output[0] includes the current layer's FFN contribution.
                # vLLM returns the FFN delta separately as hidden_states; residual is partial.
                # Add to hidden_states (not residual) so the next layer's fused add+norm computes
                # residual + (hidden_states + delta), matching HF's parenthesization order.
                full_output = residual + hidden_states
                per_token_norm = full_output.float().norm(dim=-1, keepdim=True).to(hidden_states.dtype)
                hidden_states = hidden_states + self._steering_strength * per_token_norm * self._steering_sv
            else:
                # Add to hidden_states — same parenthesization fix for the non-relative path
                hidden_states = hidden_states + buf

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
            layer._max_injection_tokens = None
            layer._inject_prefill = False
            layer._decode_step = 0
            layer._cap_logged = False
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
