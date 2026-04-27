import hmac
import hashlib
import json
import struct
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
) -> dict[int, torch.Tensor]:
    """
    Derive a per-request CAA steering vector by continuously interpolating
    between trained style vectors. Weights are produced by HMAC-PRF so they
    are deterministic given (secret_key, metadata) and unpredictable without
    the key.
    """
    m = json.dumps(metadata, sort_keys=True).encode()
    raw = hmac.new(secret_key, m, hashlib.sha256).digest()

    styles = list(vectors_by_style)
    # Extract one float per style from the HMAC output
    raw_weights = np.array([
        struct.unpack(">f", raw[i*4 : i*4+4])[0] for i in range(len(styles))
    ], dtype=np.float32)
    # Softmax for valid interpolation weights
    raw_weights -= raw_weights.max()
    weights = np.exp(raw_weights)
    weights /= weights.sum()

    result: dict[int, torch.Tensor] = {}
    for layer_idx in insertion_layers:
        blended = sum(
            weights[i] * vectors_by_style[style][layer_idx]
            for i, style in enumerate(styles)
        )
        result[layer_idx] = blended
    return result
