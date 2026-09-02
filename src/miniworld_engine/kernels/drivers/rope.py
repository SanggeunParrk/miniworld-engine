"""Driver for the ``rope`` family.

``triton_rope_3d`` rotates the SWA atom block's q/k, shaped [N, S, H, D] over head_dim. D is the
atom head dim (d_atom / n_heads = 128 / 4 = 32); ``half`` is how many channels carry an active
rope frequency, which is at most D/2 and here fills it.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    dev,
    driver_length,
    driver_width,
    ragged,
)

_L = driver_length(512)
_M = ragged(_L)
_N_HEADS = 4
_D = ragged(driver_width(128) // _N_HEADS)   # head_dim
_HALF = _D // 2


def _args():
    x = torch.randn(1, _M, _N_HEADS, _D, device=dev(), dtype=BF16, requires_grad=True)
    cos = torch.randn(1, _M, _HALF, device=dev(), dtype=torch.float32)
    sin = torch.randn(1, _M, _HALF, device=dev(), dtype=torch.float32)
    return x, cos, sin


def rope_fwd_triton() -> None:
    """rope_3d_kernel, forward and backward (backward reuses the kernel with sin negated)."""
    from miniworld_engine.kernels.rope.interface import triton_rope_3d

    x, cos, sin = _args()
    triton_rope_3d(x, cos, sin).sum().backward()
