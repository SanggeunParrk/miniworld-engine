"""Accuracy checks for the layernorm / layernorm_linear / fused_ln_mask kernels.

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

from .drivers import BF16, dev, pair, rows2d, vec
from .drivers_ln import (
    _D,
    _D_CUDA_BWD,
    _M,
    _NH,
    _PAIR_N,
    _SHAPE_KEY,
    _ln_stats,
    _mmajor,
)

_EPS = 1e-5


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


# ── references ───────────────────────────────────────────────────────────────────────────────
#
# The math comes from the two reference modules the repo already declares as ground truth for
# these families -- ``layernorm.reference.layernorm_pytorch`` and
# ``layernorm_linear.reference.layernorm_linear_pytorch`` -- rather than a second hand-written
# F.layer_norm here. The only thing added is fp32 promotion and the saved statistics, which the
# reference functions do not return.
#
# Those imports are kept inside the functions, exactly as the drivers keep theirs: both parent
# packages pull triton modules into their ``__init__``, and a module-level import would run that
# when ``run_all`` merely resolves the checker -- turning one bad import into ten failed checks.


def _ln_fwd_ref(
    x: torch.Tensor,
    w: torch.Tensor | None,
    b: torch.Tensor | None,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 LayerNorm over the last axis of x -> (y, mean, rstd), stats as fp32 [M].

    ``_ln_stats`` is the same helper the drivers build mean/rstd with: ``rstd = 1/sqrt(var+eps)``
    over the *biased* variance (unbiased=False), which is the LayerNorm definition and what every
    kernel in this family stores.
    """
    from .layernorm.reference import layernorm_pytorch

    xf = x.float()
    y = layernorm_pytorch(xf, None if w is None else w.float(), None if b is None else b.float(), eps)
    mean, rstd = _ln_stats(xf, eps)
    return y, mean, rstd


def _ln_bwd_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = _EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 autograd ground truth for LayerNorm backward -> (dx, dgamma, dbeta).

    ``beta`` is taken as zero: dbeta = sum_m dy[m] does not depend on its value, so a kernel
    that never sees a bias is still checked against the right dbeta.
    """
    from .layernorm.reference import layernorm_pytorch

    xf = x.float().detach().requires_grad_(True)
    gf = w.float().detach().requires_grad_(True)
    bf = torch.zeros_like(gf).requires_grad_(True)
    layernorm_pytorch(xf, gf, bf, eps).backward(dy.float())
    assert xf.grad is not None and gf.grad is not None and bf.grad is not None
    return xf.grad, gf.grad, bf.grad


# ── layernorm (cuda) ─────────────────────────────────────────────────────────────────────────


def layernorm_fwd_cuda():
    """layer_norm_fwd_kernel: y = LN(x)*w + b, plus the saved fp32 mean/rstd."""
    from .layernorm.cuda import layer_norm_cuda

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
    from .layernorm.cuda import layer_norm_bwd_cuda

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
    from .layernorm.cuda import layer_norm_bwd_cuda

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
    from .layernorm.triton.main import triton_layernorm

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
    from .layernorm.compile_native import _bwd_atomic_impl

    # The PRE-flatten pair activation (1, L, L, D), as the driver passes it: the impl reshapes
    # internally and reads ``both_key(length_of(x.shape))`` off the 4-D shape, which is exactly
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
    from .layernorm.compile_native import _bwd_persistent_impl

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
    from .layernorm.triton.transpose import layer_norm_transpose

    x = torch.randn(_D, 1, _M, device=dev(), dtype=BF16)  # (D, B, N), M = B*N
    w, b = vec(_D), vec(_D)
    y = layer_norm_transpose(x, w, b, layout="dbn->bnd")  # (B, N, D)
    ry, _, _ = _ln_fwd_ref(x.permute(1, 2, 0), w, b)
    return y, ry


# ── fused_ln_mask ────────────────────────────────────────────────────────────────────────────


def layernorm_fwd_rowscale_triton():
    """_fused_ln_mask_kernel: LN over the D axis, then a PER-ROW multiply by the pair mask.

    What the fold actually does, checked before writing the reference: the mask is (B, L, L) --
    ONE scalar per row of the flattened (B*L*L, D) matrix -- and the multiply happens AFTER the
    affine, so a masked-out row is exactly zero including the LayerNorm beta, and the mask never
    enters the D reduction. That is the math ``fused_ln_mask/reference.py`` states, so the
    reference is that function rather than a rewrite: it reduces in fp32 and reproduces the
    launcher's cast of the mask to x.dtype before the multiply.
    """
    from .fused_ln_mask.cute.fused_ln_mask import fused_ln_mask
    from .fused_ln_mask.reference import fused_ln_mask_pytorch

    x, w, b = pair(n=_PAIR_N, d=_D), vec(_D), vec(_D)
    mask = (torch.rand(*x.shape[:-1], device=dev()) > 0.1).to(BF16)  # (B, L, L), as the driver
    out = fused_ln_mask(x, w, b, mask, _EPS)
    ref = fused_ln_mask_pytorch(x, w, b, mask, _EPS)
    return out, ref


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
    from .layernorm_linear.triton.stats import stats_triton

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
    from .layernorm_linear.triton.te_style import _ln_materialize

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
    from .layernorm_linear.triton.mmajor_bwd import _ln_bwd_atomic

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
    from .layernorm_linear.reference import layernorm_linear_pytorch
    from .layernorm_linear.triton.fused import layernorm_linear_triton_fwd

    # The pair activation, as the driver passes it: ``layernorm_linear_triton_fwd`` flattens to
    # (M, K) itself and reads ``both_key(length_of(x.shape))`` first, then reshapes y back, so
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
    from .layernorm_linear.triton.recompute import _recompute_xnormed

    x, g, b = rows2d(_M, _D), vec(_D), vec(_D)
    mean, rstd = _ln_stats(x)
    y = _recompute_xnormed(x, g, b, mean, rstd)
    ry = (x.float() - mean[:, None]) * rstd[:, None] * g.float() + b.float()
    return y, ry


def layernorm_fwd_recompute_noaffine_triton():
    """_xhat_kernel: x_hat = (x - mean)*rstd from the SAVED stats, no affine."""
    from .layernorm_linear.triton.recompute import _recompute_xhat

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
    from .layernorm_linear.triton.mmajor_bwd import _ln_bwd_persistent_new

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
    from .layernorm_linear.reference import layernorm_linear_pytorch
    from .layernorm_linear.triton.pair_bias import _fwd_op

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
    from .layernorm_linear.reference import layernorm_linear_pytorch
    from .layernorm_linear.triton.pair_bias import _bwd_op, _fwd_op

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
    from .layernorm_linear.cute.gemm_layernorm_linear_fused import layernorm_linear_cute_fused
    from .layernorm_linear.reference import layernorm_linear_pytorch

    x, lw, lb = rows2d(_M, _D), vec(_D), vec(_D)
    w = (torch.randn(_D, _D, device=dev(), dtype=BF16) * (_D**-0.5)).contiguous()  # (N, K)
    y = layernorm_linear_cute_fused(x, lw, lb, w, None, _EPS)
    ry = layernorm_linear_pytorch(x.float(), lw.float(), lb.float(), w.float(), None, _EPS)
    return y, ry
