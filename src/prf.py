import hmac
import hashlib
import json
from typing import Optional
import numpy as np
import torch


SECRET_KEY = b"your-secret-key-keep-server-side"   # never expose this
STYLE_DIM  = 64


def metadata_to_z(
    metadata:   dict,
    secret_key: bytes = SECRET_KEY,
    style_dim:  int   = STYLE_DIM,
    dtype:      torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Deterministic per-request style vector derived from metadata.

    Properties:
        - same (secret_key, metadata) → same z always
        - different metadata          → uncorrelated z
        - without secret_key          → z is unpredictable
    """
    # canonical serialization — sort keys for determinism
    m = json.dumps(metadata, sort_keys=True).encode()

    # keyed PRF via HMAC-SHA256 → 32 bytes
    raw = hmac.new(secret_key, m, hashlib.sha256).digest()

    # expand to style_dim * 4 bytes via HKDF-SHA256
    z_bytes = _hkdf_expand(raw, length=style_dim * 4)

    # bytes → uint32 → uniform [0,1) → tanh-squashed standard normal
    u = np.frombuffer(z_bytes, dtype=np.uint32).astype(np.float64)
    u = (u + 0.5) / 2**32                          # uniform in (0, 1)
    # Box-Muller: pairs of uniforms → pairs of N(0,1)
    half = style_dim // 2
    u1, u2 = u[:half], u[half:half * 2]
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    n = np.empty(style_dim, dtype=np.float32)
    n[:half]      = (r * np.cos(theta)).astype(np.float32)
    n[half:half*2] = (r * np.sin(theta)).astype(np.float32)
    if style_dim % 2:
        n[-1] = 0.0
    z = np.tanh(n)

    return torch.tensor(z, dtype=dtype)


def sample_random_z(
    batch_size: int,
    style_dim:  int   = STYLE_DIM,
    device:     str   = "cpu",
    dtype:      torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Sample random z vectors for training.
    Drawn from same distribution as PRF outputs (tanh-squashed normal).
    """
    z = torch.randn(batch_size, style_dim, device=device, dtype=dtype)
    return torch.tanh(z)


def _hkdf_expand(
    prk:    bytes,
    length: int,
    info:   bytes = b"styled-gemma",
) -> bytes:
    """
    Minimal HKDF-expand (RFC 5869) using SHA-256.
    Expands pseudorandom key to desired length.
    """
    hash_len = 32   # SHA-256 output length
    n        = (length + hash_len - 1) // hash_len
    okm      = b""
    t        = b""
    for i in range(1, n + 1):
        t    = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]


def metadata_to_caa_vector(
    metadata: dict,
    vectors_by_style: dict[str, dict[int, torch.Tensor]],
    insertion_layers: list[int],
    secret_key: bytes = SECRET_KEY,
    weight_mode: str = "dirichlet",
    dirichlet_alphas: list[float] | float | None = None,
    style_scales: Optional[dict[str, float]] = None,
    normalize_result: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
    """
    Derive a per-request CAA steering vector by blending trained style vectors.
    Weights are produced by HMAC-PRF so they are deterministic given
    (secret_key, metadata) and unpredictable without the key.

    Args:
        weight_mode: how to combine styles —
            "dirichlet" — asymmetric Dirichlet draw seeded from HMAC (default)
            "softmax"   — continuous interpolation
            "binarize"  — winner-takes-all (one style gets weight 1, rest 0)
            "normalize" — softmax blend rescaled to the average component norm
        dirichlet_alphas: concentration parameter(s) for "dirichlet" mode.
            A single float uses the same alpha for all styles (symmetric).
            A list gives per-style alphas (asymmetric); must match len(styles).
            Smaller values → more peaked/sparse weights. Defaults to 0.3.

    Returns:
        (blended_vectors, style_weights) where style_weights maps style name to weight
    """
    m = json.dumps(metadata, sort_keys=True).encode()
    raw = hmac.new(secret_key, m, hashlib.sha256).digest()

    styles = list(vectors_by_style)

    if weight_mode == "dirichlet":
        seed = int.from_bytes(raw, "big") % (2**32)
        rng = np.random.default_rng(seed)
        if dirichlet_alphas is None:
            alphas = [0.3] * len(styles)
        elif isinstance(dirichlet_alphas, (int, float)):
            alphas = [float(dirichlet_alphas)] * len(styles)
        else:
            if len(dirichlet_alphas) != len(styles):
                raise ValueError(
                    f"dirichlet_alphas length {len(dirichlet_alphas)} != number of styles {len(styles)}"
                )
            alphas = list(dirichlet_alphas)
        weights = rng.dirichlet(alphas)
    else:
        # Extract one unsigned integer per style from the HMAC output, then convert to
        # float. Using int.from_bytes avoids IEEE 754 special values (NaN, inf) that
        # struct.unpack(">f") can produce from arbitrary bytes.
        raw_weights = np.array([
            int.from_bytes(raw[i*4 : i*4+4], "big") / 2**32
            for i in range(len(styles))
        ], dtype=np.float64)
        raw_weights -= raw_weights.max()
        weights = np.exp(raw_weights)
        weights /= weights.sum()

        if weight_mode == "binarize":
            binary = np.zeros_like(weights)
            binary[weights.argmax()] = 1.0
            weights = binary

    style_weights = {style: float(weights[i]) for i, style in enumerate(styles)}

    result: dict[int, torch.Tensor] = {}
    for layer_idx in insertion_layers:
        vecs = [
            weights[i] * (style_scales.get(style, 1.0) if style_scales else 1.0) * vectors_by_style[style][layer_idx]
            for i, style in enumerate(styles)
        ]
        blended: torch.Tensor = vecs[0]
        for v in vecs[1:]:
            blended = blended + v
        if normalize_result:
            blended_norm = blended.norm().item()
            if blended_norm > 0:
                blended = blended / (blended_norm + 1e-8)
        elif weight_mode == "normalize":
            avg_norm = sum(
                vectors_by_style[style][layer_idx].norm().item()
                for style in styles
            ) / len(styles)
            blended_norm = blended.norm().item()
            if blended_norm > 0:
                blended = blended * (avg_norm / blended_norm)
        result[layer_idx] = blended
    return result, style_weights


def random_per_layer_caa_vectors(
    vectors_by_style: dict[str, dict[int, torch.Tensor]],
    insertion_layers: list[int],
    weight_mode: str = "dirichlet",
    dirichlet_alphas: "list[float] | float | None" = None,
    style_scales: Optional[dict[str, float]] = None,
    normalize_result: bool = False,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    """Blend style vectors with independently sampled weights per layer.

    Returns:
        (blended_vectors, per_layer_weights) where per_layer_weights maps
        layer_idx → {style: weight} for each layer.
    """
    styles = list(vectors_by_style)
    rng = np.random.default_rng()

    if dirichlet_alphas is None:
        alphas = [0.3] * len(styles)
    elif isinstance(dirichlet_alphas, (int, float)):
        alphas = [float(dirichlet_alphas)] * len(styles)
    else:
        if len(dirichlet_alphas) != len(styles):
            raise ValueError(
                f"dirichlet_alphas length {len(dirichlet_alphas)} != number of styles {len(styles)}"
            )
        alphas = list(dirichlet_alphas)

    result: dict[int, torch.Tensor] = {}
    per_layer_weights: dict[int, dict[str, float]] = {}

    for layer_idx in insertion_layers:
        if weight_mode == "dirichlet":
            weights = rng.dirichlet(alphas)
        else:
            raw_weights = rng.random(len(styles)).astype(np.float64)
            raw_weights -= raw_weights.max()
            weights = np.exp(raw_weights)
            weights /= weights.sum()
            if weight_mode == "binarize":
                binary = np.zeros_like(weights)
                binary[weights.argmax()] = 1.0
                weights = binary

        per_layer_weights[layer_idx] = {style: float(weights[i]) for i, style in enumerate(styles)}

        vecs = [
            weights[i] * (style_scales.get(style, 1.0) if style_scales else 1.0) * vectors_by_style[style][layer_idx]
            for i, style in enumerate(styles)
        ]
        blended: torch.Tensor = vecs[0]
        for v in vecs[1:]:
            blended = blended + v
        if normalize_result:
            blended_norm = blended.norm().item()
            if blended_norm > 0:
                blended = blended / (blended_norm + 1e-8)
        elif weight_mode == "normalize":
            avg_norm = sum(
                vectors_by_style[style][layer_idx].norm().item() for style in styles
            ) / len(styles)
            blended_norm = blended.norm().item()
            if blended_norm > 0:
                blended = blended * (avg_norm / blended_norm)
        result[layer_idx] = blended

    return result, per_layer_weights
