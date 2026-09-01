"""Drivers for the ``rmsnorm`` family.

The width is the HEAD dim, not the model width: both callers normalize per attention head
(``modules/swa_atom_attention`` at head_dim, ``kernels/triangle_attention/whole_op.py`` at
``d_head``), so the rows are ``N*S*H`` and the normalized axis is small. That is why the ladders
in ``configs/grid/rmsnorm_*.csv`` start their BLOCK_K at 32 rather than at layernorm's 64.

Both driven with a weight AND without: ``HAS_WEIGHT`` is a constexpr, so the two are separate
compiled kernels and a cache built for one says nothing about the other.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, vec

#: head_dim, the axis these kernels normalize. 32 is the SWA atom side (d_model 128 / 4 heads).
_D = 32
#: Rows a driven launch reduces over. `_rows` below builds the (M, D) the launcher flattens to.
_M = 4096
_EPS = 1e-5


def _rows(m: int = _M, d: int = _D) -> torch.Tensor:
    """The PRE-flatten activation, as the real callers hand it over.

    3-D, not (M, D): `rows_of` refuses an already-flattened shape on purpose -- a caller holding
    only (M, D) cannot say whether its M is the whole launch or one slice of it. SWA passes
    [N, S, H, D] and triangle_attention its own 4-D; either way the launcher flattens.
    """
    return torch.randn(1, m, d, device=dev(), dtype=BF16, requires_grad=True)


def rmsnorm_fwd_triton() -> None:
    """rmsnorm_fwd_kernel, weighted."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm

    triton_rmsnorm(_rows(), vec(_D), _EPS)


def rmsnorm_bwd_triton() -> None:
    """rmsnorm_bwd_kernel, via _RMSNorm.backward."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm

    x = _rows()
    triton_rmsnorm(x, vec(_D), _EPS).sum().backward()


#: The adaLN pair is driven at the DiT block's width, not `_D`: the modulate acts on the whole
#: d_model vector, only the q/k normalization is per head. 128/384 is the atom block.
_DM = 128
_DC = 384


def _adamod_args() -> tuple[torch.Tensor, ...]:
    """``(q, c, w_scale, w_shift)`` at the atom-DiT block shape, q differentiable."""
    q = torch.randn(1, _M, _DM, device=dev(), dtype=BF16, requires_grad=True)
    c = torch.randn(1, _M, _DC, device=dev(), dtype=BF16, requires_grad=True)
    w = lambda: (torch.randn(_DM, _DC, device=dev(), dtype=BF16)
                 * (0.1 / _DC**0.5)).requires_grad_()
    return q, c, w(), w()


def rmsnorm_adamod_fwd_triton() -> None:
    """rmsnorm_adamod_fwd_kernel, weighted."""
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod

    q, c, wsc, wsh = _adamod_args()
    triton_rmsnorm_adamod(q, c, wsc, wsh, vec(_DM), _EPS)


def rmsnorm_adamod_bwd_triton() -> None:
    """rmsnorm_adamod_bwd_kernel, via _RMSNormAdaMod.backward.

    The three GEMMs the backward chains onto this kernel are cuBLAS, not tuned here; what the
    autotuner sees is the recompute-plus-elementwise kernel that feeds them.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod

    q, c, wsc, wsh = _adamod_args()
    triton_rmsnorm_adamod(q, c, wsc, wsh, vec(_DM), _EPS).sum().backward()
