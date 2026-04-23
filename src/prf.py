import hmac
import hashlib
import json
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

    # expand to style_dim floats via HKDF-SHA256
    z_bytes = _hkdf_expand(raw, length=style_dim * 4)

    # convert bytes → float32 → tanh squash to (-1, 1)
    z = np.frombuffer(z_bytes, dtype=np.float32).copy()
    z = np.tanh(z)

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
