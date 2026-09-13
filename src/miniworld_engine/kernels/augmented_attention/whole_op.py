"""Whole-op wrapper for augmented (pair-biased) attention.

Exposes :func:`augmented_attention_pair_bias` — the pair-biased multi-head attention
core (``softmax(qkᵀ·scale + bias)·v``) as a single autograd-transparent call, with the
fused Triton kernel inside and the layout conversion handled here.

Boundary note: the q/k/v/gate/out projections and the adaptive conditioning are model
concerns (they carry EDM2 magnitude-preserving research variants that normalize weights
on use, so they must own the projection); this op takes ALREADY-PROJECTED ``q/k/v`` plus
the precomputed pair ``bias`` — the same boundary the model already used for the kernel.
The step-invariant bias itself is produced by :func:`ops.layer_norm_linear`.
"""

from __future__ import annotations

from typing import Literal

import torch


def augmented_attention_pair_bias(
    query: torch.Tensor,                  # (A, B, H, L, D)  — model (head-major) layout
    key: torch.Tensor,                    # (A, B, H, L, D)
    value: torch.Tensor,                  # (A, B, H, L, D)
    bias: torch.Tensor,                   # (B, H, L, L)
    mask: torch.Tensor | None = None,     # (A, B, L) key mask
    *,
    kernel_type: Literal["compute_efficient", "memory_efficient"] = "compute_efficient",
) -> torch.Tensor:
    """Pair-biased attention — whole-op call. Input/output in the model's head-major
    ``(A,B,H,L,D)`` / ``(B,H,L,L)`` layout; the internal token-major kernel layout
    (``(A,B,L,H,D)`` / ``(B,L,L,H)``) translation is absorbed here.

    Autograd-transparent. Preserve the caller's precision: native BF16 stays BF16,
    while an FP32 caller does not silently run a quantized attention core.
    """
    if kernel_type == "compute_efficient":
        from miniworld_engine.kernels.augmented_attention.triton.main import (
            triton_augmented_attention_pair_bias as _fn,
        )
    elif kernel_type == "memory_efficient":
        from miniworld_engine.kernels.augmented_attention import (
            triton_augmented_attention_pair_bias as _fn,
        )
    else:
        msg = f"unknown kernel_type {kernel_type!r}"
        raise ValueError(msg)

    in_dtype = query.dtype
    # (A,B,H,L,D) -> (A,B,L,H,D); (B,H,L,L) -> (B,L,L,H)
    q = query.transpose(2, 3)
    k = key.transpose(2, 3)
    v = value.transpose(2, 3)
    b = bias.permute(0, 2, 3, 1)
    out = _fn(q, k, v, b, mask)           # (A,B,L,H,D)
    return out.transpose(2, 3).to(in_dtype)  # -> (A,B,H,L,D)
