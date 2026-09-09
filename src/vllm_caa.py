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
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)


_CAA_STATE_FILE = os.path.join(tempfile.gettempdir(), "vllm_caa_state")
_COLLECT_STATE_FILE = os.path.join(tempfile.gettempdir(), "vllm_caa_collect_state")

_DEBUG_CAA = os.getenv("CAA_DEBUG", "0") == "1"
# When set, activate low-overhead fast paths that are safe for inference-only benchmarking:
# (1) skip all collect-mode file reads (collect mode not used during benchmarks)
# (2) avoid GPU→CPU sync when detecting prefill vs. decode for the common n==1 decode step
# (3) replace two os.path.* calls with a single os.stat() per forward pass
# Set by scripts/benchmark_pipelines.py before LLM() init so worker processes inherit it.
_BENCH_MODE = os.getenv("CAA_BENCH_MODE", "0") == "1"
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
    inject_prefill_last_n: "int | None" = _UNSET,
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
    if inject_prefill_last_n is not _UNSET:
        state["inject_prefill_last_n"] = int(inject_prefill_last_n) if inject_prefill_last_n is not None else None
    # Enforce invariant: inject_prefill=False always clears last_n so stale non-None
    # values from previous runs cannot cause prefill injection to slip through.
    if state.get("inject_prefill") is False:
        state["inject_prefill_last_n"] = None

    _save_caa_state(state)

    if _DEBUG_CAA and active:
        print(f"[VLLM_CAA] Activated: strength={state.get('strengths') or state['strength']}, layers={state['layers']}, num_vectors={len(state['vectors'])}")


def disable_caa() -> None:
    state = _load_caa_state()
    state["active"] = False
    _save_caa_state(state)


def set_collect_mode(active: bool, layers=None, response_start_offset: int = 0) -> None:
    """Enable/disable activation-collection mode. Disables injection first."""
    disable_caa()  # ensure no steering contaminates collected activations
    state = {
        "active": bool(active),
        "layers": list(layers or []),
        "response_start_offset": response_start_offset,
    }
    
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


def _per_seq_prefill_ranges(
    pf_start: int, pf_end: int, q_lens: list[int], last_n: "int | None"
) -> list[tuple[int, int]]:
    """Split [pf_start, pf_end) into per-sequence ranges, clipping each to its last `last_n` rows.

    When multiple sequences are prefilled in the same batch, their rows are concatenated
    in [pf_start, pf_end). A single `prefill_end - last_n` slice would only catch the tail
    of the last sequence; we instead apply `last_n` independently to each sequence's range.
    Falls back to a single range when q_lens is empty (positional-fallback split path).
    """
    if not q_lens:
        seq_start = max(pf_start, pf_end - last_n) if last_n is not None else pf_start
        return [(seq_start, pf_end)] if pf_end > seq_start else []
    out: list[tuple[int, int]] = []
    cur = pf_start
    for L in q_lens:
        seq_end = cur + L
        seq_start = max(cur, seq_end - last_n) if last_n is not None else cur
        if seq_end > seq_start:
            out.append((seq_start, seq_end))
        cur = seq_end
    return out


def _apply_caa_injection(self, positions, hidden_states, residual):
    """Architecture-agnostic CAA injection applied after a decoder layer's forward.

    Reads per-layer state set up by `_swap_decoder_layers` (`_caa_layer_idx`,
    `_steering_buffer`, …) and globals (`_load_caa_state`, `_load_collect_state`,
    `_per_seq_prefill_ranges`, `_DEBUG_CAA`). Returns the (possibly modified)
    `(hidden_states, residual)` pair.
    """
    idx = getattr(self, "_caa_layer_idx", None)
    n = positions.shape[0]

    # Determine which token rows are prefill vs. decode.
    # vLLM v1 stores attn_metadata in the thread-local forward context (not in **kwargs).
    # vLLM v1 also uses "decode first, prefill last" ordering within the batch.
    decode_start = decode_end = 0
    prefill_start = prefill_end = 0
    _split_resolved = False
    any_meta = None
    prefill_q_lens: list[int] = []  # per-sequence prefill row counts in batch order

    try:
        from vllm.forward_context import get_forward_context
        # Bench-mode fast path: at decode time with a single sequence the step always
        # produces exactly 1 token row (n == 1). Skip the forward-context query and the
        # accompanying GPU→CPU sync (.cpu().tolist()) — the split is trivially known.
        if _BENCH_MODE and n == 1:
            # n==1: pure single-token decode step; no prefill tokens in this batch.
            decode_start, decode_end = 0, 1
            prefill_start, prefill_end = 1, 1
            _split_resolved = True
        else:
            ctx = get_forward_context()
            attn_meta_map = ctx.attn_metadata if ctx is not None else None
            if attn_meta_map:
                any_meta = next(iter(attn_meta_map.values()))
                q_start = any_meta.query_start_loc  # shape [num_seqs+1], cumulative
                q_lens = (q_start[1:] - q_start[:-1]).cpu().tolist()
                # Decode seqs have q_len == 1; prefill/chunked-prefill seqs have q_len > 1.
                # vLLM v1: decode seqs come first in the batch, prefill seqs come last.
                num_decode_tokens = sum(int(q) for q in q_lens if int(q) == 1)
                decode_start, decode_end = 0, num_decode_tokens
                prefill_start, prefill_end = num_decode_tokens, n
                prefill_q_lens = [int(q) for q in q_lens if int(q) > 1]
                _split_resolved = True
    except Exception:
        pass

    if not _split_resolved:
        # Positional heuristic fallback.
        # Find first position == 0 to detect batch ordering.
        first_zero = next((i for i in range(n) if int(positions[i]) == 0), None)
        if first_zero is None:
            # Pure decode (no zeros in positions).
            decode_start, decode_end = 0, n
        elif first_zero == 0:
            # Prefill-first ordering (older vLLM) or pure prefill.
            # Walk forward counting sequential runs starting from 0.
            num_pf = 0
            expected = 0
            for i in range(n):
                p = int(positions[i])
                if p == expected:
                    num_pf += 1
                    expected += 1
                elif p == 0:
                    num_pf += 1
                    expected = 1
                else:
                    break
            prefill_start, prefill_end = 0, num_pf
            decode_start, decode_end = num_pf, n
        else:
            # Decode-first ordering (vLLM v1): everything before first 0 is decode.
            decode_start, decode_end = 0, first_zero
            prefill_start, prefill_end = first_zero, n

    num_prefill = prefill_end - prefill_start
    num_decode = decode_end - decode_start

    # --- Collect mode (training only): capture mean prefill activation ---
    # Skipped entirely in bench mode — collect mode is never active during benchmarking.
    if not _BENCH_MODE and idx is not None and num_prefill > 0:
        collect = _load_collect_state()
        if collect.get("active") and idx in collect.get("layers", []):
            offset = collect.get("response_start_offset", 0)
            slice_start = prefill_start + offset
            if slice_start >= prefill_end:
                # empty response slice — fall back to full prefill to avoid NaN
                slice_start = prefill_start
            full = (residual[slice_start:prefill_end] + hidden_states[slice_start:prefill_end]).float()
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
        # Bench mode: replace two separate os.path.* calls (exists + getmtime) with a
        # single os.stat() syscall — halves the filesystem overhead per forward pass.
        if _BENCH_MODE:
            try:
                current_mtime = os.stat(_CAA_STATE_FILE).st_mtime
            except OSError:
                current_mtime = -1.0
        else:
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
                self._inject_prefill_last_n = state.get("inject_prefill_last_n")
                self._decode_counts = {}
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
        if num_decode == 0:
            # pure prefill — nothing to inject unless inject_prefill is on.
            # Also: clear stale per-sequence counts. Pure prefill marks the start
            # of a new generation call (no sequences are mid-decode), and KV block
            # IDs from completed sequences may be reused by new ones — keeping
            # stale counts would cause the new sequences to wrongly hit the cap.
            if self._decode_counts:
                self._decode_counts.clear()
            if not self._inject_prefill:
                if _DEBUG_CAA:
                    print(f"[CAA dbg] layer={idx} prefill (seq_len={n}) SKIP")
                return hidden_states, residual
            last_n = self._inject_prefill_last_n
            # Build per-sequence prefill ranges so `last_n` applies independently to each
            # sequence's prefill rows, not to the concatenated batch.
            prefill_only_ranges = _per_seq_prefill_ranges(
                prefill_start, prefill_end, prefill_q_lens, last_n
            )
            if _DEBUG_CAA:
                print(f"[CAA dbg] layer={idx} prefill (seq_len={n}) INJECT last_n={last_n} num_seqs={len(prefill_only_ranges)}")
        else:
            # at least some decode rows; cap is tracked PER sequence via block_table[:, 0]
            # as a stable per-sequence identifier (requires enable_prefix_caching=False).
            inject_decode_rows: list[int] = []
            if (
                self._max_injection_tokens is not None
                and any_meta is not None
                and getattr(any_meta, "block_table", None) is not None
                and any_meta.block_table.numel() > 0
            ):
                sigs = any_meta.block_table[:num_decode, 0].cpu().tolist()
                for row_i, sig in enumerate(sigs):
                    count = self._decode_counts.get(sig, 0)
                    if count < self._max_injection_tokens:
                        inject_decode_rows.append(decode_start + row_i)
                    self._decode_counts[sig] = count + 1
                if _DEBUG_CAA:
                    print(f"[CAA dbg] layer={idx} decode seqs={len(sigs)} inject_rows={len(inject_decode_rows)} cap={self._max_injection_tokens}")
            else:
                # No cap or no metadata: inject every decode row.
                inject_decode_rows = list(range(decode_start, decode_end))

            # Collapse contiguous decode rows into ranges for efficient slicing.
            inject_ranges: list[tuple[int, int]] = []
            if inject_decode_rows:
                run_start = inject_decode_rows[0]
                prev = run_start
                for r in inject_decode_rows[1:]:
                    if r == prev + 1:
                        prev = r
                    else:
                        inject_ranges.append((run_start, prev + 1))
                        run_start = r
                        prev = r
                inject_ranges.append((run_start, prev + 1))

            if self._inject_prefill and num_prefill > 0:
                last_n = self._inject_prefill_last_n
                for r in _per_seq_prefill_ranges(
                    prefill_start, prefill_end, prefill_q_lens, last_n
                ):
                    inject_ranges.append(r)

            if not inject_ranges:
                return hidden_states, residual

        def _inject_rows(start: int, end: int) -> None:
            if end <= start:
                return
            rows = slice(start, end)
            if self._steering_relative and self._steering_sv is not None:
                # Use full post-layer output (residual + ffn_delta) for per-token norm to match
                # HF's hook convention where output[0] includes the current layer's FFN contribution.
                # vLLM returns the FFN delta separately as hidden_states; residual is partial.
                # Add to hidden_states (not residual) so the next layer's fused add+norm computes
                # residual + (hidden_states + delta), matching HF's parenthesization order.
                full_output = (residual[rows] + hidden_states[rows]).float()
                per_token_norm = full_output.norm(dim=-1, keepdim=True).to(hidden_states.dtype)
                hidden_states[rows] = hidden_states[rows] + self._steering_strength * per_token_norm * self._steering_sv
            else:
                # Add to hidden_states — same parenthesization fix for the non-relative path
                hidden_states[rows] = hidden_states[rows] + buf

        if num_decode == 0:
            for r_start, r_end in prefill_only_ranges:
                _inject_rows(r_start, r_end)
        else:
            for r_start, r_end in inject_ranges:
                _inject_rows(r_start, r_end)

    return hidden_states, residual


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
        return _apply_caa_injection(self, positions, hidden_states, residual)


class Qwen2CAADecoderLayer(Qwen2DecoderLayer):
    """Qwen2 decoder layer with CAA injection — same logic as Gemma3CAADecoderLayer."""

    def forward(self, positions, hidden_states, residual, **kwargs):
        hidden_states, residual = super().forward(
            positions, hidden_states, residual, **kwargs
        )
        return _apply_caa_injection(self, positions, hidden_states, residual)


_BASE_TO_CAA = {
    Gemma3DecoderLayer: Gemma3CAADecoderLayer,
    Qwen2DecoderLayer: Qwen2CAADecoderLayer,
}


def _swap_decoder_layers(
    layers, hidden_size: int, dtype: torch.dtype, device: torch.device
) -> None:
    swapped = 0
    caa_classes = tuple(_BASE_TO_CAA.values())
    for idx, layer in enumerate(layers):
        if isinstance(layer, caa_classes):
            continue
        caa_cls = next(
            (caa for base, caa in _BASE_TO_CAA.items() if isinstance(layer, base)),
            None,
        )
        if caa_cls is None:
            continue
        layer.__class__ = caa_cls
        layer._caa_layer_idx = idx
        layer._steering_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
        layer._buffer_mtime = -1.0
        layer._steering_sv = None
        layer._steering_strength = 0.0
        layer._steering_relative = False
        layer._max_injection_tokens = None
        layer._inject_prefill = False
        layer._inject_prefill_last_n = None
        layer._decode_counts = {}
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


class Qwen2CAAForCausalLM(Qwen2ForCausalLM):
    """Qwen2 causal LM with CAA-aware decoder layers (covers DeepSeek-R1-Distill-Qwen-*)."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        param = next(self.parameters())
        hidden_size = vllm_config.model_config.hf_config.hidden_size
        if _DEBUG_CAA:
            print(f"[VLLM_CAA] Initializing Qwen2CAAForCausalLM, hidden_size={hidden_size}, {len(self.model.layers)} layers")
        _swap_decoder_layers(self.model.layers, hidden_size, param.dtype, param.device)


ModelRegistry.register_model(
    "Gemma3CAAForCausalLM",
    Gemma3CAAForCausalLM,
)
ModelRegistry.register_model(
    "Gemma3CAAForConditionalGeneration",
    Gemma3CAAForConditionalGeneration,
)
ModelRegistry.register_model(
    "Qwen2CAAForCausalLM",
    Qwen2CAAForCausalLM,
)
