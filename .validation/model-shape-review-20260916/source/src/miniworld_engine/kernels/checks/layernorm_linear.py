"""Accuracy checks for the ``layernorm_linear`` family.

A driver proves a kernel *runs*. It says nothing about whether the number is right. Each
function here re-runs one kernel on the shapes its driver uses, builds the torch reference
for the same math, and hands back ``(actual, expected)`` -- or a dict of them when the
kernel produces several tensors. ``autotune.run_all.check_one`` does the comparing.

Two rules keep these honest:

* The reference is built from the *same inputs the kernel saw*, in fp32. Never from a
  second random draw, and never from the kernel's own output.
* A kernel that consumes saved statistics is checked against a reference that consumes the
  *same* saved statistics. ``layernorm_fwd_recompute_*`` and every ``layernorm_bwd_*``
  read a ``mean``/``rstd`` pair the caller computed; recomputing stats inside the reference
  would be checking a different function.

Shapes and launcher call sites are lifted verbatim from ``drivers_ln`` -- the constants and
helpers are imported from it rather than restated, so a checker cannot silently drift onto a
shape the driver never exercises. That mattered here: ``layernorm_fwd_mmajor_triton`` reads a
(D, M) channel-major input and writes row-major, and it was measured at rel=1.442 when handed a
row-major x -- a layout mismatch reads as a wrong kernel, so the reference has to match the
transpose exactly rather than the other way round.
"""
from __future__ import annotations

import contextlib

import torch

from miniworld_engine.kernels.checks import _EPS, _ln_bwd_ref, _ln_fwd_ref
from miniworld_engine.kernels.drivers import BF16, _ln_stats, dev, pair, rows2d, vec
from miniworld_engine.kernels.drivers.layernorm_linear import (
    _D,
    _M,
    _NH,
    _PAIR_N,
    _SHAPE_KEY,
    _mmajor,
)


@contextlib.contextmanager
def _no_tf32():
    """Force true fp32 for a reference matmul, then put the flag back.

    A reference GEMM left on TF32 keeps only 10 mantissa bits, which is *worse* than the bf16
    operands with fp32 accumulation it is supposed to be judging -- the comparison would then be
    measuring the reference. Only the layernorm_linear checkers need this; the pure LayerNorm ones
    contain no matmul.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# ── layernorm_linear: statistics ─────────────────────────────────────────────────────────────


def layernorm_stats_triton():
    """_stats_kernel: rstd[m] = rsqrt(var+eps) and c1[m] = mean*rstd over the K axis of x (M, K).

    This is the pair the rest of the family consumes (the folded GEMM epilogue is
    ``Y = rstd*acc - c1*S + B2``), so its reference has to be the family's definition of the
    statistics and not this kernel's route to them. It is ``_ln_stats`` -- the same helper the
    drivers hand to every kernel that takes saved mean/rstd -- i.e. the BIASED variance of the
    fp32 row. The kernel instead tiles the K axis and finishes with the uncentered
    ``E[x^2] - mean^2``, so an error in that algebra (or in the tail-tile mask on K, which is the
    reduce axis) is exposed here rather than reproduced.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    x = rows2d(_M, _D)
    rstd, c1 = stats_triton(x, _EPS)
    rmean, rrstd = _ln_stats(x, _EPS)
    return {"rstd": (rstd, rrstd), "c1": (c1, rmean * rrstd)}


# ── layernorm_linear: te_style strided (m-major) LN forward / backward ───────────────────────


def layernorm_fwd_saveact_strided_triton():
    """_ln_mat_kernel: x_normed = LN(x)*g + b read at x's OWN strides, plus the saved mean/rstd.

    x is the m-major (M, K) view ``_mmajor`` builds -- strides (1, M) -- which is the stride
    coverage this kernel exists for; it absorbs them and writes a CONTIGUOUS x_normed. The
    reference is F.layer_norm over the last axis of that SAME logical tensor: torch reads it
    through its strides, so the two agree element for element and nothing is transposed to make
    the shapes fit (the failure mode ``layernorm_fwd_mmajor_triton`` measured at rel=1.442).
    All three returned tensors are this kernel's own output.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
        _ln_materialize,
    )

    x, g, b = _mmajor(), vec(_D), vec(_D)
    xn, mean, rstd = _ln_materialize(x, g, b, _EPS)
    rxn, rmean, rrstd = _ln_fwd_ref(x, g, b)
    return {"x_normed": (xn, rxn), "mean": (mean, rmean), "rstd": (rstd, rrstd)}


def layernorm_bwd_atomic_strided_triton():
    """_ln_bwd_kernel: dx at x's m-major strides + atomic dgamma/dbeta, from the SAVED stats.

    Reached through ``_ln_bwd_atomic``, the driver's entry point and te_style's atomic path
    verbatim (``mmajor_bwd`` re-exports it as the small-M fallback). dx_normed is a second m-major
    tensor, as the te_style backward produces it, and the kernel's math is the LayerNorm backward
    with dy = dx_normed: dx as usual, dgamma = sum_m dx_normed*xhat, dbeta = sum_m dx_normed. So
    the reference is the same fp32 autograd backward the rest of this module uses, on the same
    saved mean/rstd -- x_hat is recomputed from them INSIDE the kernel, never re-derived here.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.mmajor_bwd import (
        _ln_bwd_atomic,
    )

    x, dxn, g = _mmajor(), _mmajor(), vec(_D)
    mean, rstd = _ln_stats(x)
    dx, dg, db = _ln_bwd_atomic(dxn, x, g, mean, rstd, x.stride())
    rdx, rdg, rdb = _ln_bwd_ref(dxn, x, g)
    return {"dx": (dx, rdx), "dgamma": (dg, rdg), "dbeta": (db, rdb)}


# ── layernorm_linear: portable fused LN + GEMM (triton) ──────────────────────────────────────


def layernorm_linear_fwd_triton():
    """_lnl_fwd_kernel: Y = LN(x)@W.T, affine LN (gamma AND beta), no Linear bias, W is (N, K).

    The driver passes bias=None, so the reference adds none. The kernel folds the whole affine LN
    into the normalized tile and casts it to bf16 before ``tl.dot``; the reference is the unfused
    composition the module's own reference declares, ``F.linear(F.layer_norm(x, g, b), W, None)``,
    in fp32 with TF32 off -- so the LN reduce axis and the GEMM's K axis (the same _D here, and
    the axis where a partial-tile contraction bug would live) are both judged against true fp32.
    """
    from miniworld_engine.kernels.layernorm_linear.reference import (
        layernorm_linear_pytorch,
    )
    from miniworld_engine.kernels.layernorm_linear.triton.fused import (
        layernorm_linear_triton_fwd,
    )

    # The pair activation, as the driver passes it: ``layernorm_linear_triton_fwd`` flattens to
    # (M, K) itself and reads ``both_key(rows_of(x.shape))`` first, then reshapes y back, so
    # both sides of the comparison stay at (1, L, L, N).
    x, g, b = pair(n=_PAIR_N, d=_D), vec(_D), vec(_D)
    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    y = layernorm_linear_triton_fwd(x, g, b, w, None, _EPS)
    with _no_tf32():
        ry = layernorm_linear_pytorch(x.float(), g.float(), b.float(), w.float(), None, _EPS)
    return y, ry


# ── layernorm_linear: recompute-from-saved-stats ─────────────────────────────────────────────


def layernorm_fwd_recompute_triton():
    """_xnormed_kernel: x_normed = (x - mean)*rstd*gamma + beta from the SAVED stats.

    No stats are recomputed, so the reference must use the very same mean/rstd -- calling
    F.layer_norm here would compare against a slightly different normalization.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.recompute import (
        _recompute_xnormed,
    )

    x, g, b = rows2d(_M, _D), vec(_D), vec(_D)
    mean, rstd = _ln_stats(x)
    y = _recompute_xnormed(x, g, b, mean, rstd)
    ry = (x.float() - mean[:, None]) * rstd[:, None] * g.float() + b.float()
    return y, ry


def layernorm_fwd_recompute_noaffine_triton():
    """_xhat_kernel: x_hat = (x - mean)*rstd from the SAVED stats, no affine."""
    from miniworld_engine.kernels.layernorm_linear.triton.recompute import (
        _recompute_xhat,
    )

    x = rows2d(_M, _D)
    mean, rstd = _ln_stats(x)
    y = _recompute_xhat(x, mean, rstd)
    ry = (x.float() - mean[:, None]) * rstd[:, None]
    return y, ry


# ── layernorm_linear: m-major backward ───────────────────────────────────────────────────────


def layernorm_bwd_split_mmajor_triton():
    """_ln_bwd_mmajor_kernel: dx at m-major strides + [NP, K] fp32 dgamma/dbeta partials.

    ``_ln_bwd_persistent_new`` already collapses the partials (``pdg.sum(0)``/``pdb.sum(0)``)
    before returning, so what comes back is the summed dgamma/dbeta -- no second sum here.
    """
    from miniworld_engine.kernels.layernorm_linear.triton.mmajor_bwd import (
        _ln_bwd_persistent_new,
    )

    x, dy, g = _mmajor(), _mmajor(), vec(_D)
    mean, rstd = _ln_stats(x)
    dx, dg, db = _ln_bwd_persistent_new(dy, x, g, mean, rstd, x.stride())
    rdx, rdg, rdb = _ln_bwd_ref(dy, x, g)
    return {"dx": (dx, rdx), "dgamma": (dg, rdg), "dbeta": (db, rdb)}


# ── layernorm_linear: pair-bias fused LN + projection ────────────────────────────────────────


def layernorm_linear_fwd_fp32_triton():
    """_layer_norm_linear_fwd: out = LN(x)@pw.T, with an UNBIASED LN and UNBIASED Linear.

    The module docstring is explicit that ln_weight/proj_weight are the affine scale of an
    unbiased nn.LayerNorm and the weight of an unbiased Linear, so the reference passes no
    beta and adds no bias. mean/rstd are saved for the backward and checked too.
    """
    from miniworld_engine.kernels.layernorm_linear.reference import (
        layernorm_linear_pytorch,
    )
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import _fwd_op

    # ``_fwd_op`` takes the pre-flatten activation (it reads shape[-2] for the key and reshapes
    # out back to (..., nh)); mean/rstd stay per-ROW, so the reference stats are taken on the
    # flattened view of that same x.
    x, lnw = pair(n=_PAIR_N, d=_D), vec(_D)
    pw = torch.randn(_NH, _D, device=dev(), dtype=BF16)
    out, mean, rstd = _fwd_op(x, lnw, pw, _EPS)
    rout = layernorm_linear_pytorch(x.float(), lnw.float(), None, pw.float(), None, _EPS)
    rmean, rstd_ref = _ln_stats(x.reshape(-1, _D), _EPS)
    return {"out": (out, rout), "mean": (mean, rmean), "rstd": (rstd, rstd_ref)}


def layernorm_linear_bwd_fp32_triton():
    """_layer_norm_linear_bwd: dx / dln_weight / dproj_weight for the unbiased LN + Linear.

    The forward's saved mean/rstd feed the backward, so the reference re-derives the same
    composition (unbiased LN, then the projection) under fp32 autograd and takes all three
    grads from it. n_head=4 is below MIN_TL_DOT_DIM, so this exercises the scalar-loop branch.
    """
    from miniworld_engine.kernels.layernorm_linear.reference import (
        layernorm_linear_pytorch,
    )
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import (
        _bwd_op,
        _fwd_op,
    )

    # Both ops run here, so both get the driver's treatment: ``_fwd_op`` is handed the
    # pre-flatten pair activation and reads L off it, while ``_bwd_op`` only ever takes the
    # (M, N) matrix and so is handed the key explicitly. dx is that matrix's shape, so the
    # reference's leaf is x2 -- the same rows, contiguous -- and the backward seed is dout
    # viewed at (M, nh), which is the view the kernel itself takes of it.
    x, lnw = pair(n=_PAIR_N, d=_D), vec(_D)
    x2 = x.reshape(-1, _D).contiguous()
    pw = torch.randn(_NH, _D, device=dev(), dtype=BF16)
    out, mean, rstd = _fwd_op(x, lnw, pw, _EPS)
    dout = torch.randn_like(out)
    dx, dlnw, dpw = _bwd_op(dout, x2, lnw, pw, mean, rstd, shape_key=_SHAPE_KEY)

    xf = x2.float().detach().requires_grad_(True)
    lf = lnw.float().detach().requires_grad_(True)
    pf = pw.float().detach().requires_grad_(True)
    layernorm_linear_pytorch(xf, lf, None, pf, None, _EPS).backward(
        dout.reshape(-1, _NH).float())
    return {"dx": (dx, xf.grad), "dln_weight": (dlnw, lf.grad), "dproj_weight": (dpw, pf.grad)}


# ── layernorm_linear: fused LN + GEMM (cute, SM90) ───────────────────────────────────────────


def layernorm_linear_fwd_sm90_cute():
    """GemmLNLFusedSm90.kernel: Y = LN(x)@weight.T (+ bias), stats computed inside the GEMM.

    The launcher folds the LN affine into the GEMM operands via ``fold_for_gemm``, so the
    reference is the unfused composition it is meant to equal. weight is (N, K); bias is None
    here, matching the driver.
    """
    from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
        layernorm_linear_cute_fused,
    )
    from miniworld_engine.kernels.layernorm_linear.reference import (
        layernorm_linear_pytorch,
    )

    x, lw, lb = rows2d(_M, _D), vec(_D), vec(_D)
    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    y = layernorm_linear_cute_fused(x, lw, lb, w, None, _EPS)
    ry = layernorm_linear_pytorch(x.float(), lw.float(), lb.float(), w.float(), None, _EPS)
    return y, ry
