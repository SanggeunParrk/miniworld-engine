"""The strided row-LayerNorm `adaln_inference` uses for its conditioning pass.

One kernel, `_ln_kernel` (`layernorm_fwd_strided_triton`): row-wise LayerNorm of a flattened
(M, N) activation, optionally scaled by a weight. `inference.py::_cond_affine` is its only caller.

This file was `fused3.py` and held the three-kernel adaLN decomposition -- `adaln_fused3` and
`adaln_fused3_train` plus the GEMM-gate and elementwise-backward kernels they drove. No module ever
selected it: `modules/adaptive_layernorm/module.py` dispatches to `adaln_train` / `adaln_inference`
and to nothing else, and fused3's only callers were the bench, the driver and the checker. Deleted
with `main.py`, whose Function had the same problem. What survives is the one kernel a production
forward reaches, and the file is now named for it.
"""
from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl



# ── K1 / K2: row-wise LayerNorm ────────────────────────────────────────────────────────────
# Both axes are tuned tiles. BLOCK_N used to arrive as next_pow2(d) from the launcher — the whole
# row, a constant the tuner never saw — which is also why BLOCK_M1 had to stay at 1..16 (a
# [16, 1024] fp32 tile was already the register budget). The N axis is a REDUCE axis (mean/var),
# so a CSV row at or above the extent spans a whole d_hidden/d_cond row; with N tiled,
# BLOCK_M1 can take the canonical (>=16) 2-D tile sizes and the two axes trade off properly.




# shape_key is the SHAPE cache bucket, and shape means L -- the atom count (this family is
# level=atom in kernels/registry.csv) -- never the row count a kernel receives. It is NOT GROUP_M:
# in this file GROUP_M is the tuned L2-swizzle axis the two GEMM kernels read from the CSV, so the
# bucket takes a separate, lowercase name -- a plain runtime int no kernel body ever reads.
#
# The four launchers below are INNER launchers: they only ever see the flattened (M, D) matrix, and
# M is B*A, so L is not recoverable here. Each therefore takes the key as a `shape_key` argument
# from the caller that still holds the pre-flatten shape. The default covers the callers that hand
# `shape_key=None` is NOT a working fallback: `length_of` refuses a rank-2 shape, so the
# branch raises with a message saying to compute the key at the caller. The default stays only
# because the `@opaque` fakes share the signature.
# (This used to claim the default covered a caller handing in a genuinely 2-D activation.) The drivers and
# checkers do exactly that.
from miniworld_engine.autotune.shape_key import atom_key, both_key, length_of, pack, rows_of
# `both_key` is for the borrowed layernorm_linear helpers only (`_ln_materialize`/`_ln_bwd`):
# those kernels are level=both in registry.csv, so they bucket against the union set, while
# this family's own kernels stay on `atom_key`. Same L either way -- different bucket set.


@triton.autotune(configs=configs_for("layernorm_fwd_strided_triton"), key=['shape_key', 'HAS_W'])
@triton.jit
def _ln_kernel(X, Y, W, M, N: tl.constexpr, eps, sx0, sx1, sy0, sy1,
              HAS_W: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
              shape_key):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    # TWO-PASS (not Welford): pass 1 accumulates Σx and Σx² over the N tiles in fp32 (plain sums,
    # so exact across tiles), pass 2 re-reads x to normalize. LN re-uses the row it just reduced,
    # so a tiled reduce axis costs either a second read of x or a Welford carry; the re-read is
    # simpler.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip but the two tl.loads of x live in
    # separate scf.for regions, so they are NOT CSE'd and the covering config read x twice. `N` is
    # `tl.constexpr` (it is already in this kernel's autotune key, so a new d already forced a
    # re-tune) which makes the guard a TRACE-time comparison: exactly one branch is emitted and the
    # covering tile degenerates to the untiled single-read schedule. The fast path uses the CENTRED
    # variance Σ(x-mean)²/N — numerically stabler, and x is already in registers; the uncentered
    # Σx²/N - mean² stays in the tiled branch, where it is what keeps that branch one read per tile.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        xn = xc * rstd[:, None]
        if HAS_W:
            w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            xn = xn * w
        tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, xn.to(Y.dtype.element_ty), mask=mask)
    else:
        s = tl.zeros([BLOCK_M1], dtype=tl.float32)
        ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            mask = rmask[:, None] & (cols[None, :] < N)
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=1)
            ss += tl.sum(x * x, axis=1)
        mean = s / N
        var = ss / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            cmask = cols < N
            mask = rmask[:, None] & cmask[None, :]
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            xn = (x - mean[:, None]) * rstd[:, None]
            if HAS_W:
                w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
                xn = xn * w
            tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, xn.to(Y.dtype.element_ty), mask=mask)
