"""Drivers for the ``rmsnorm_adamod`` family.

Its own family and not a mode of ``rmsnorm``: the projection makes it a ``gemm`` where the other
is a ``reduce``, it carries its own config ladder (``tl.dot`` needs BLOCK_M1 from 16, not 1), and
it runs on the two DiT streams only -- there is no pair-stream adaLN. The shape block is shared
with ``drivers/rmsnorm.py`` rather than restated, the way ``drivers/layernorm.py`` shares
``drivers/layernorm_linear.py``'s.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.drivers import BF16, dev, vec
from miniworld_engine.kernels.drivers.rmsnorm import _DC, _DM, _EPS, _M, _rows


def _cond() -> torch.Tensor:
    return torch.randn(1, _M, _DC, device=dev(), dtype=BF16, requires_grad=True)


def _proj() -> torch.Tensor:
    """One (d_model, d_cond) slice of the block's adaLN projection. bias=False at the call site."""
    return (torch.randn(_DM, _DC, device=dev(), dtype=BF16) * (0.1 / _DC**0.5)).requires_grad_()


def rmsnorm_adamod_fwd_triton() -> None:
    """rmsnorm_adamod_fwd_kernel, both values of HAS_WEIGHT.

    Unweighted is the production case -- adaLN supplies the scale, so the norm under it is
    non-affine -- but the affine form is a declared argument and is tuned too.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod

    triton_rmsnorm_adamod(_rows(_DM), _cond(), _proj(), _proj(), None, _EPS)
    triton_rmsnorm_adamod(_rows(_DM), _cond(), _proj(), _proj(), vec(_DM), _EPS)


def rmsnorm_adamod_bwd_triton() -> None:
    """rmsnorm_adamod_bwd_kernel, via _RMSNormAdaMod.backward, both values of HAS_WEIGHT.

    The three GEMMs the backward chains onto this kernel are cuBLAS, not tuned here; what the
    autotuner sees is the recompute-plus-elementwise kernel that feeds them.
    """
    from miniworld_engine.kernels.rmsnorm.interface import triton_rmsnorm_adamod

    triton_rmsnorm_adamod(_rows(_DM), _cond(), _proj(), _proj(), None, _EPS).sum().backward()
    triton_rmsnorm_adamod(_rows(_DM), _cond(), _proj(), _proj(), vec(_DM), _EPS).sum().backward()
