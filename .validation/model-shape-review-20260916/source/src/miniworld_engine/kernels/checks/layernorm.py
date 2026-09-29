"""Accuracy checks for the ``layernorm`` family.

layernorm, layernorm_linear and fused_ln_mask were one module (``checks_ln.py``). The two rules
these references follow -- built from the same inputs the kernel saw, in fp32, and fed the same
saved statistics the kernel consumed -- are written out in ``checks/layernorm_linear.py``; the
helpers all three use are in ``checks/__init__.py``.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _EPS, _ln_bwd_ref, _ln_fwd_ref
from miniworld_engine.kernels.drivers import BF16, _ln_stats, dev, pair, rows2d, vec
from miniworld_engine.kernels.drivers.layernorm import _D_CUDA_BWD
from miniworld_engine.kernels.drivers.layernorm_linear import _D, _M, _PAIR_N

# ── layernorm (cuda) ─────────────────────────────────────────────────────────────────────────


def layernorm_fwd_cuda():
    """layer_norm_fwd_kernel: y = LN(x)*w + b, plus the saved fp32 mean/rstd."""
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_cuda

    x, w, b = rows2d(_M, _D), vec(_D), vec(_D)
    y, mean, rstd = layer_norm_cuda.layer_norm_fwd(x, w, b, _EPS)
    ry, rmean, rrstd = _ln_fwd_ref(x, w, b)
    return {"y": (y, ry), "mean": (mean, rmean), "rstd": (rstd, rrstd)}


def layernorm_bwd_split_cuda():
    """layer_norm_bwd_main_kernel: dx directly, plus the per-warp dw/db partials.

    The launcher runs main then reduce and only hands back the reduced dw/db, so the partials
    are checked through their sum -- reduce is a plain column sum, so a wrong partial shows up
    in dw/db. dx is this kernel's own output.
    """
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

    # _D_CUDA_BWD, not _D: this kernel's vectorized loads pin the feature width (see drivers_ln).
    x, w = rows2d(_M, _D_CUDA_BWD), vec(_D_CUDA_BWD)
    dy = torch.randn_like(x)
    mean, rstd = _ln_stats(x)
    dx, dw, db = layer_norm_bwd_cuda(dy, x, w, mean, rstd)
    rdx, rdw, rdb = _ln_bwd_ref(dy, x, w)
    return {"dx": (dx, rdx), "dw": (dw, rdw), "db": (db, rdb)}


def layernorm_bwd_reduce_cuda():
    """layer_norm_bwd_reduce_kernel: sums the [warps, N] partials into dw/db.

    Only dw/db pass through this kernel -- dx is written by the main kernel and is not its
    output, so it is not part of this comparison.
    """
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda

    # _D_CUDA_BWD, not _D: this kernel's vectorized loads pin the feature width (see drivers_ln).
    x, w = rows2d(_M, _D_CUDA_BWD), vec(_D_CUDA_BWD)
    dy = torch.randn_like(x)
    mean, rstd = _ln_stats(x)
    _dx, dw, db = layer_norm_bwd_cuda(dy, x, w, mean, rstd)
    _rdx, rdw, rdb = _ln_bwd_ref(dy, x, w)
    return {"dw": (dw, rdw), "db": (db, rdb)}


# ── layernorm (triton, row-major) ────────────────────────────────────────────────────────────


def layernorm_fwd_saveact_triton():
    """layer_norm_fwd_fused: y = LN(x)*w + b over the last axis of a (B, N, N, D) pair tensor.

    The driver's entry point is ``triton_layernorm`` (the autograd Function), which hands back y
    only -- the saved mean/rstd live on the ctx for the backward and are not observable from here.
    They are still covered by this comparison: BOTH branches of the kernel compute y from the very
    mean/rstd registers they store (the store is a plain masked store of the same values), and _D
    is the reduce axis, so a bad column mask corrupts the statistics and therefore every element
    of the row. HAS_ROWSCALE is False on this path -- the rowscale fold is its own kernel below.
    """
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    x, w, b = pair(n=_PAIR_N, d=_D), vec(_D), vec(_D)
    y = triton_layernorm(x, w, b, _EPS)
    ry, _, _ = _ln_fwd_ref(x, w, b)
    return y, ry


def layernorm_bwd_atomic_triton():
    """layer_norm_bwd_dx_fused: dx, plus dw/db reduced over M by tl.atomic_add, from SAVED stats.

    ``_bwd_atomic_impl`` is the driver's entry point and returns all three of this kernel's
    outputs (the fp32 atomic accumulators cast to the weight dtype), so all three are compared.
    mean/rstd are the ``_ln_stats`` pair the driver builds and are NOT recomputed inside the
    kernel; the reference takes its grads from fp32 autograd over the same x/dy/w.
    """
    from miniworld_engine.kernels.layernorm.compile_native import _bwd_atomic_impl

    # The PRE-flatten pair activation (1, L, L, D), as the driver passes it: the impl reshapes
    # internally and reads ``both_key(rows_of(x.shape))`` off the 4-D shape, which is exactly
    # what ``length_of`` now refuses on an already-flattened (M, D). dx comes back
    # ``view_as(x)``, so the whole comparison simply moves to that shape.
    x, w = pair(n=_PAIR_N, d=_D), vec(_D)
    dy = torch.randn_like(x)
    mean, rstd = _ln_stats(x.reshape(-1, _D))   # [M] fp32, one row per (b, i, j), as the driver
    dx, dw, db = _bwd_atomic_impl(dy, x, w, mean, rstd)
    rdx, rdw, rdb = _ln_bwd_ref(dy, x, w)
    return {"dx": (dx, rdx), "dw": (dw, rdw), "db": (db, rdb)}


def layernorm_bwd_split_triton():
    """_ln_bwd_persistent: dx + the [G, N] fp32 dw/db partials, from the SAVED mean/rstd.

    ``_bwd_persistent_impl`` collapses the partials (``partial_dw.sum(0)``) before returning, so
    what is compared is the reduced dw/db -- the final reduce is a plain column sum, so a wrong
    partial row lands there. This kernel puts the FEATURE axis on grid axis 1 and carries dw/db in
    BLOCK_K-wide fp32 registers across the whole grid-stride loop, so a bad mask on the reduce
    axis shows up both in dx (through c1/c2, which reduce the whole row) and in dw/db.
    """
    from miniworld_engine.kernels.layernorm.compile_native import _bwd_persistent_impl

    x, w = pair(n=_PAIR_N, d=_D), vec(_D)   # pre-flatten, like layernorm_bwd_atomic_triton above
    dy = torch.randn_like(x)
    mean, rstd = _ln_stats(x.reshape(-1, _D))
    dx, dw, db = _bwd_persistent_impl(dy, x, w, mean, rstd)
    rdx, rdw, rdb = _ln_bwd_ref(dy, x, w)
    return {"dx": (dx, rdx), "dw": (dw, rdw), "db": (db, rdb)}


# ── layernorm (triton, channel-major) ────────────────────────────────────────────────────────


def layernorm_fwd_mmajor_triton():
    """_ln_transpose_dbn_kernel: reads (D, B, N) channel-major, writes (B, N, D) row-major.

    The kernel folds the transpose into the LayerNorm, so the reference has to transpose
    too: x.permute(1, 2, 0) is exactly the map the kernel walks -- x_dm[k, m] with
    m = b*N + n becomes y[b, n, k], verified against an index-by-index emulation of the
    launcher. Reducing over the last axis of the (D, B, N) buffer as it lies instead
    normalizes over N rather than D and scores rel~0.17 at these shapes.
    """
    from miniworld_engine.kernels.layernorm.triton.transpose import layer_norm_transpose

    x = torch.randn(_D, 1, _M, device=dev(), dtype=BF16)  # (D, B, N), M = B*N
    w, b = vec(_D), vec(_D)
    y = layer_norm_transpose(x, w, b, layout="dbn->bnd")  # (B, N, D)
    ry, _, _ = _ln_fwd_ref(x.permute(1, 2, 0), w, b)
    return y, ry
