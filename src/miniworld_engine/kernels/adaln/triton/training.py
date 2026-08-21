"""Training (fwd+bwd) AdaptiveLayerNorm: materialize+cuBLAS, symmetric single-GEMM backward.

Mirrors the inference materialize path (kernels/adaln/triton/inference.py) but saves for backward
and adds a TE-style backward built from cuBLAS GEMMs + reused LayerNorm-backward kernels.

Forward  : cond_aff = LN(cond)·lnw                          (Triton — te_style _ln_materialize)
           [scale|bias] = cond_aff @ [Ws|Wb]ᵀ + [sb|0]      (ONE cuBLAS GEMM)
           x_hat = LN(x) ; gate = σ(scale) ; y = gate·x_hat + bias   (Triton epilogue, saves stats+gate)

Backward (let D = [dscale | dy], W_cat = [Ws ; Wb], both stacked along the NX-axis):
           dscale = dy·x_hat·gate·(1-gate)                  (Triton prep → D, and dxhat = dy·gate)
           dcond_aff = D @ W_cat                            (ONE cuBLAS GEMM)
           [dWs ; dWb] = Dᵀ @ cond_aff                      (ONE cuBLAS GEMM)
           dsb = Σ_m dscale                                 (cuBLAS GEMV)
           dx        = LN-bwd(dxhat,    x,    γ=1,   …)      (Triton — te_style _ln_bwd)
           dcond,dlnw= LN-bwd(dcond_aff,cond, γ=lnw, …)     (Triton — te_style _ln_bwd)

The [dscale|dy] / [Ws;Wb] stacking makes both the dgrad and the wgrad ONE GEMM each — the exact
mirror of the forward's single fused GEMM.
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Reuse the proven, tested TE-style LN building blocks (cuBLAS + Triton, portable).
from ...layernorm_linear.triton.te_style import (
    _fp32_matmul_ctx,
    _ln_bwd,
    _ln_materialize,
)

# Both axes are tuned tiles for the two row-wise kernels below. BLOCK_N used to arrive as
# next_pow2(NX) from the launcher — the whole row, a constant the tuner never saw, which is also
# why BLOCK_M1 was pinned to 1..32. N is the REDUCE axis here (LN mean/var in the forward, the
# c1/c2 row sums in the backward), so it is a CSV tile, while
# d_hidden runs to 1024, and a set that cannot express a whole row turns tiling into a forced
# two-pass on every card instead of a choice the tuner makes.

# Matmul precision for fp32 inputs: False → TF32 cuBLAS (cos≈1.0); True → bf16 operands w/ fp32
# accumulate (cos≈0.9999, but tensor-core bf16 is ~1.6× faster than TF32 — the only lever left for
# fp32-IO speed since a TF32 WGMMA custom kernel is infeasible in this CuTeDSL/quack env). IO and
# LayerNorm stay fp32; only the three GEMM operands are downcast.
_GEMM_BF16 = False


def set_gemm_bf16(flag: bool) -> None:
    """Enable bf16 operands (fp32 accumulate) for the GEMMs on fp32 inputs (faster, cos≈0.9999)."""
    global _GEMM_BF16  # noqa: PLW0603
    _GEMM_BF16 = flag


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a@b. fp32 inputs: bf16 operands (fp32 accum) if _GEMM_BF16 else TF32 cuBLAS."""
    if _GEMM_BF16 and a.dtype == torch.float32:
        return torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16)).float()
    with _fp32_matmul_ctx(a.dtype):
        return torch.matmul(a, b)


# ───────────── forward epilogue: y = σ(scale)·LN(x) + bias, saving mean_x, rstd_x, gate ─────────
# Keyed on N. The dtype needs no key entry: triton's Autotuner appends str(arg.dtype) for every
# tensor argument, so fp32 and bf16 already land in different cache slots.


# shape_key's value is L -- the atom count (this family is level=atom in kernels/registry.csv) --
# not the flattened row count M = B*A the kernels iterate. The three launchers below are INNER
# launchers that only see the (M, D) matrices, so each takes the key from the caller that still
# holds the pre-flatten shape; the default covers a caller that hands in a genuinely 2-D activation
# (nothing folded into the rows, so shape[-2] IS L), which is what the drivers and checkers do.
from miniworld_engine.autotune.shape_key import atom_key, both_key, length_of
# `both_key` is only for the borrowed layernorm_linear helpers (`_ln_materialize`/`_ln_bwd`):
# those kernels are level=both in registry.csv and bucket against the union set. This family's
# own kernels stay on `atom_key`. Same L, different bucket set.


@triton.autotune(configs=configs_for("adaln_epilogue_saveact_triton"), key=['N', 'HAS_SB', 'shape_key'])
@triton.jit
def _epilogue_train_kernel(
    X, SB, Y, MeanX, RstdX, Gate, ScaleBias, M, N: tl.constexpr, eps,
    sx0, sx1, ss0, ss1, sy0, sy1, sg0, sg1,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, HAS_SB: tl.constexpr,
    shape_key,
):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    # TWO-PASS (not Welford): pass 1 accumulates Σx / Σx² over the N tiles in fp32 — plain sums, so
    # exact across tiles — and pass 2 re-reads x for the normalize + gate epilogue.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip, but the two tl.loads of X are
    # NOT CSE'd — the MeanX/RstdX tl.store sits between them and Triton cannot prove the raw
    # pointers do not alias — so the covering config read x twice. `N` is `tl.constexpr` (already
    # this kernel's autotune key, so a new d_hidden already forced a re-tune and a fresh compile)
    # which makes the guard a TRACE-time comparison: one branch emitted, covering tile back to the
    # untiled single-read schedule. The fast path uses the CENTRED variance Σ(x-mean)²/N
    # (numerically stabler, x already in registers); the uncentered Σx²/N - mean² stays in the
    # tiled branch, where it is what keeps that branch to one read per tile.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(MeanX + rm, mean, mask=rmask)
        tl.store(RstdX + rm, rstd, mask=rmask)
        x_hat = xc * rstd[:, None]
        scale = tl.load(SB + rm[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(SB + rm[:, None] * ss0 + (cols[None, :] + N) * ss1,
                       mask=mask, other=0.0).to(tl.float32)
        if HAS_SB:  # fold the scale-bias (β for the scale half) add here — free vs a full (M,2N) add pass
            scale += tl.load(ScaleBias + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        gate = tl.sigmoid(scale)
        y = gate * x_hat + bias
        tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(Gate + rm[:, None] * sg0 + cols[None, :] * sg1,
                 gate.to(Gate.dtype.element_ty), mask=mask)
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
        tl.store(MeanX + rm, mean, mask=rmask)
        tl.store(RstdX + rm, rstd, mask=rmask)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            cmask = cols < N
            mask = rmask[:, None] & cmask[None, :]
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            scale = tl.load(SB + rm[:, None] * ss0 + cols[None, :] * ss1, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(SB + rm[:, None] * ss0 + (cols[None, :] + N) * ss1,
                           mask=mask, other=0.0).to(tl.float32)
            if HAS_SB:  # fold the scale-bias (β for the scale half) add here — free vs a full (M,2N) add pass
                scale += tl.load(ScaleBias + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            gate = tl.sigmoid(scale)
            y = gate * x_hat + bias
            tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, y.to(Y.dtype.element_ty), mask=mask)
            tl.store(Gate + rm[:, None] * sg0 + cols[None, :] * sg1,
                     gate.to(Gate.dtype.element_ty), mask=mask)


def _epilogue_train(x, sb, eps, scale_bias=None, *, shape_key=None):
    M, N = x.shape
    if shape_key is None:
        shape_key = atom_key(length_of(x.shape))
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    gate = torch.empty(M, N, device=x.device, dtype=x.dtype)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N is tl.constexpr now (it drives the BLOCK_N >= N fold) -> pass a plain python int.
    _epilogue_train_kernel[grid](
        x, sb, y, mean, rstd, gate, scale_bias, M, int(N), eps,
        x.stride(0), x.stride(1), sb.stride(0), sb.stride(1),
        y.stride(0), y.stride(1), gate.stride(0), gate.stride(1), HAS_SB=scale_bias is not None,
        shape_key=shape_key,
    )
    return y, mean, rstd, gate


# ─── backward x-pass (fused): D = [dscale | dy] AND dx = LN-bwd(dy·gate) in ONE kernel ───
# The x LayerNorm-backward reduction (no affine) is done right here — no separate LN-bwd kernel and
# no dxhat buffer. The N axis used to be pinned to next_pow2(NX) so the whole row sat in registers;
# it is a CSV tile, and the row reduction is what makes this a two-pass kernel.


@triton.autotune(configs=configs_for("adaln_bwd_pre_dx_triton"), key=['N', 'shape_key'])
@triton.jit
def _bwd_x_kernel(
    DY, X, MeanX, RstdX, Gate, D, DX, M, N: tl.constexpr,
    sy0, sy1, sx0, sx1, sg0, sg1, sd0, sd1, sdx0, sdx1,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, shape_key):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    mean = tl.load(MeanX + rm, mask=rmask, other=0.0)[:, None]
    rstd = tl.load(RstdX + rm, mask=rmask, other=0.0)[:, None]
    inv_n = 1.0 / N
    # TWO-PASS. dx needs c1/c2, which are reductions over the WHOLE row, so with a tiled reduce
    # axis the (dy, x, gate) tiles must be visited twice. Everything that does NOT depend on
    # c1/c2 — dscale and both halves of D — is done in pass 1, so pass 2 only re-reads dy/x/gate
    # and writes dx.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip, but the dy/x/gate loads are NOT
    # CSE'd across them — the two tl.stores into D sit in between and Triton cannot prove the raw
    # pointers do not alias — so the covering config paid THREE reads of the row instead of one.
    # `N` is `tl.constexpr` (already this kernel's autotune key, so a new d_hidden already forced a
    # re-tune and a fresh compile) which makes the guard a TRACE-time comparison: one branch
    # emitted, covering tile back to the untiled single-read schedule.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        dy = tl.load(DY + rm[:, None] * sy0 + cols[None, :] * sy1, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(Gate + rm[:, None] * sg0 + cols[None, :] * sg1, mask=mask, other=0.0).to(tl.float32)
        x_hat = tl.where(cmask[None, :], (x - mean) * rstd, 0.0)
        dscale = dy * x_hat * gate * (1.0 - gate)
        # D is (2N, M): D[col, row] = dscale ; D[N+col, row] = dy. This (2NX,M) layout makes the
        # wgrad (D@cond_aff) a contiguous-K NN GEMM (~1.6× faster than the transposed-view read)
        # and dsb a cheap row-sum; cost is a transposed store here. sd0 strides 2N, sd1 strides M.
        daddr = cols[None, :] * sd0 + rm[:, None] * sd1
        tl.store(D + daddr, dscale.to(D.dtype.element_ty), mask=mask)
        tl.store(D + daddr + N * sd0, dy.to(D.dtype.element_ty), mask=mask)
        dxhat = dy * gate
        c2 = tl.sum(tl.where(cmask[None, :], dxhat, 0.0), axis=1) * inv_n
        c1 = tl.sum(tl.where(cmask[None, :], dxhat * x_hat, 0.0), axis=1) * inv_n
        # x LayerNorm backward (no affine): dx = rstd·(dxhat − meanₖ(dxhat) − x̂·meanₖ(dxhat·x̂))
        dx = rstd * (dxhat - c2[:, None] - x_hat * c1[:, None])
        tl.store(DX + rm[:, None] * sdx0 + cols[None, :] * sdx1,
                 dx.to(DX.dtype.element_ty), mask=mask)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            cmask = cols < N
            mask = rmask[:, None] & cmask[None, :]
            dy = tl.load(DY + rm[:, None] * sy0 + cols[None, :] * sy1, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(Gate + rm[:, None] * sg0 + cols[None, :] * sg1, mask=mask, other=0.0).to(tl.float32)
            x_hat = tl.where(cmask[None, :], (x - mean) * rstd, 0.0)
            dscale = dy * x_hat * gate * (1.0 - gate)
            daddr = cols[None, :] * sd0 + rm[:, None] * sd1
            tl.store(D + daddr, dscale.to(D.dtype.element_ty), mask=mask)
            tl.store(D + daddr + N * sd0, dy.to(D.dtype.element_ty), mask=mask)
            dxhat = dy * gate
            c2 += tl.sum(tl.where(cmask[None, :], dxhat, 0.0), axis=1)
            c1 += tl.sum(tl.where(cmask[None, :], dxhat * x_hat, 0.0), axis=1)
        c1 *= inv_n   # scale once at the end, as the untiled kernel did
        c2 *= inv_n
        # x LayerNorm backward (no affine): dx = rstd·(dxhat − meanₖ(dxhat) − x̂·meanₖ(dxhat·x̂))
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            cmask = cols < N
            mask = rmask[:, None] & cmask[None, :]
            dy = tl.load(DY + rm[:, None] * sy0 + cols[None, :] * sy1, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(Gate + rm[:, None] * sg0 + cols[None, :] * sg1, mask=mask, other=0.0).to(tl.float32)
            x_hat = tl.where(cmask[None, :], (x - mean) * rstd, 0.0)
            dxhat = dy * gate
            dx = rstd * (dxhat - c2[:, None] - x_hat * c1[:, None])
            tl.store(DX + rm[:, None] * sdx0 + cols[None, :] * sdx1,
                     dx.to(DX.dtype.element_ty), mask=mask)


def _bwd_x(dy, x, mean_x, rstd_x, gate, *, shape_key=None):
    """Returns D=(2N,M) [dscale;dy stacked on the 2N axis] and dx=(M,N) (x's layout), one kernel.
    D's (2N,M) layout is chosen so the wgrad D@cond_aff is a contiguous-K GEMM (see kernel note)."""
    M, N = dy.shape
    if shape_key is None:
        shape_key = atom_key(length_of(dy.shape))
    D = torch.empty(2 * N, M, device=dy.device, dtype=dy.dtype)   # (2N, M)
    dx = torch.empty_strided((M, N), x.stride(), device=dy.device, dtype=dy.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N is tl.constexpr now (it drives the BLOCK_N >= N fold) -> pass a plain python int.
    _bwd_x_kernel[grid](
        dy, x, mean_x, rstd_x, gate, D, dx, M, int(N),
        dy.stride(0), dy.stride(1), x.stride(0), x.stride(1), gate.stride(0), gate.stride(1),
        D.stride(0), D.stride(1), dx.stride(0), dx.stride(1),
        shape_key=shape_key,
    )
    return D, dx


# ─── fused dgrad + cond LN-backward: dcond_aff = Dᵀ@W_cat (in-kernel GEMM) → dcond, dlnw ───
# Per token-tile: (1) an in-kernel GEMM over K2=2NX computes dcond_aff(BM,NC) = Σ_k2 D[k2,m]·W_cat[k2,n]
# (D=(2NX,M) so a-tile reads D[k,m] = m-contiguous), then (2) the cond LayerNorm-backward (affine
# γ=lnw, no β) is done right there on the in-register dcond_aff — NO dcond_aff HBM round-trip and NO
# cuBLAS dgrad. dlnw = Σ_m dcond_aff·cond̂ is reduced via per-block atomics. wgrad (D@cond_aff) stays
# cuBLAS (the only matmul left on cuBLAS, per directive).
#
# BLOCK_K_NC tiles the cond row: the LN reduction needs the whole row, so a tile narrower than NC
# makes the kernel loop over it. A row that sets it >= NC keeps the whole-row single-tile schedule.




@triton.autotune(configs=configs_for("adaln_bwd_dx_dlnw_triton"),
                 key=['NC', 'K2', 'shape_key'],
                 reset_to_zero=['DLNW'])
@triton.jit
def _dgrad_condln_kernel(
    D, Wcat, Cond, MeanC, RstdC, LNW, DCond, DLNW, M, NC: tl.constexpr, K2,
    sd0, sd1, sw0, sw1, sc0, sc1, sdc0, sdc1,
    BLOCK_M1: tl.constexpr, BLOCK_K_NC: tl.constexpr, BLOCK_K_K2: tl.constexpr, shape_key):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    mean = tl.load(MeanC + rm, mask=rmask, other=0.0)[:, None]
    rstd = tl.load(RstdC + rm, mask=rmask, other=0.0)[:, None]
    inv_n = 1.0 / NC

    # TWO-PASS over the NC tiles. The cond LN-backward needs c1/c2 — reductions over the WHOLE
    # cond row of dcond_aff — before any element of dcond can be written, so once the NC axis is
    # tiled, dcond_aff has to be visited twice. Here the "row" being re-read is the in-kernel GEMM
    # result, so pass 2 RECOMPUTES the Dᵀ@Wcat tile rather than spilling it: the only alternative
    # was staging dcond_aff through the bf16 DCond buffer, which would round the gradient to bf16
    # before the LN-backward algebra. Correctness beats the extra MMA pass.
    #
    # dlnw and both c1/c2 are accumulated in pass 1 (they need acc but not c1/c2), so pass 2 only
    # redoes the GEMM and writes dcond.
    #
    # COVERING TILE (BLOCK_K_NC >= NC): the two NC loops are single-trip, so pass 2 recomputed the
    # SAME Dᵀ@Wcat tile — a second full pass of MMA plus a second read of D and Wcat, and neither
    # was CSE'd (the DLNW tl.atomic_add sits between them). `NC` is `tl.constexpr` (already this
    # kernel's autotune key, so a new d_cond already forced a re-tune and a fresh compile) which
    # makes the guard a TRACE-time comparison: one branch is emitted, and the fast path keeps the
    # single `acc` in REGISTERS across the c1/c2 reduction and the LN-backward algebra. That is the
    # point for this kernel — the recompute existed only to avoid staging dcond_aff through the
    # bf16 DCond buffer, and holding acc in fp32 registers avoids both. `reset_to_zero=["DLNW"]` is
    # unchanged and the fast path issues the same single tl.atomic_add per program.
    if BLOCK_K_NC >= NC:
        nc = tl.arange(0, BLOCK_K_NC)
        ncmask = nc < NC
        nmask2 = rmask[:, None] & ncmask[None, :]
        # in-kernel GEMM: dcond_aff[m,n] = Σ_k2 D[k2,m]·Wcat[k2,n] (= Dᵀ@Wcat). D=(2NX,M): a-tile
        # reads D[k,m] at m*sd1 + k*sd0 (m-contiguous since sd1=1), Wcat[k,n] at k*sw0 + n*sw1.
        acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
        for k0 in range(0, K2, BLOCK_K_K2):
            kk = k0 + tl.arange(0, BLOCK_K_K2)
            kmask = kk < K2
            a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                        mask=kmask[:, None] & ncmask[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision="tf32")
        cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                       mask=nmask2, other=0.0).to(tl.float32)
        g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
        cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
        dxhat = acc * g_w
        c2 = tl.sum(tl.where(ncmask[None, :], dxhat, 0.0), axis=1) * inv_n
        c1 = tl.sum(tl.where(ncmask[None, :], dxhat * cnorm, 0.0), axis=1) * inv_n
        pdg = tl.sum(tl.where(nmask2, acc * cnorm, 0.0), axis=0)   # dlnw = Σ_m dcond_aff·cond̂
        tl.atomic_add(DLNW + nc, pdg, mask=ncmask)
        # cond LayerNorm backward (affine γ=lnw, no β) on the in-register dcond_aff.
        dcond = rstd * (dxhat - c2[:, None] - cnorm * c1[:, None])
        tl.store(DCond + rm[:, None] * sdc0 + nc[None, :] * sdc1,
                 dcond.to(DCond.dtype.element_ty), mask=nmask2)
    else:
        c1 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        c2 = tl.zeros([BLOCK_M1], dtype=tl.float32)
        for n0 in range(0, NC, BLOCK_K_NC):
            nc = n0 + tl.arange(0, BLOCK_K_NC)
            ncmask = nc < NC
            nmask2 = rmask[:, None] & ncmask[None, :]
            # in-kernel GEMM: dcond_aff[m,n] = Σ_k2 D[k2,m]·Wcat[k2,n] (= Dᵀ@Wcat). D=(2NX,M):
            # a-tile reads D[k,m] at m*sd1 + k*sd0 (m-contiguous since sd1=1), Wcat[k,n] at
            # k*sw0 + n*sw1.
            acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
            for k0 in range(0, K2, BLOCK_K_K2):
                kk = k0 + tl.arange(0, BLOCK_K_K2)
                kmask = kk < K2
                a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                            mask=rmask[:, None] & kmask[None, :], other=0.0)
                b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                            mask=kmask[:, None] & ncmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="tf32")
            cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                           mask=nmask2, other=0.0).to(tl.float32)
            g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
            cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
            dxhat = acc * g_w
            c2 += tl.sum(tl.where(ncmask[None, :], dxhat, 0.0), axis=1)
            c1 += tl.sum(tl.where(ncmask[None, :], dxhat * cnorm, 0.0), axis=1)
            pdg = tl.sum(tl.where(nmask2, acc * cnorm, 0.0), axis=0)   # dlnw = Σ_m dcond_aff·cond̂
            tl.atomic_add(DLNW + nc, pdg, mask=ncmask)
        c1 *= inv_n   # scale once at the end, as the untiled kernel did
        c2 *= inv_n

        # cond LayerNorm backward (affine γ=lnw, no β) on the recomputed dcond_aff.
        for n0 in range(0, NC, BLOCK_K_NC):
            nc = n0 + tl.arange(0, BLOCK_K_NC)
            ncmask = nc < NC
            nmask2 = rmask[:, None] & ncmask[None, :]
            acc = tl.zeros((BLOCK_M1, BLOCK_K_NC), dtype=tl.float32)
            for k0 in range(0, K2, BLOCK_K_K2):
                kk = k0 + tl.arange(0, BLOCK_K_K2)
                kmask = kk < K2
                a = tl.load(D + rm[:, None] * sd1 + kk[None, :] * sd0,
                            mask=rmask[:, None] & kmask[None, :], other=0.0)
                b = tl.load(Wcat + kk[:, None] * sw0 + nc[None, :] * sw1,
                            mask=kmask[:, None] & ncmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision="tf32")
            cond = tl.load(Cond + rm[:, None] * sc0 + nc[None, :] * sc1,
                           mask=nmask2, other=0.0).to(tl.float32)
            g_w = tl.load(LNW + nc, mask=ncmask, other=0.0).to(tl.float32)[None, :]
            cnorm = tl.where(ncmask[None, :], (cond - mean) * rstd, 0.0)
            dxhat = acc * g_w
            dcond = rstd * (dxhat - c2[:, None] - cnorm * c1[:, None])
            tl.store(DCond + rm[:, None] * sdc0 + nc[None, :] * sdc1,
                     dcond.to(DCond.dtype.element_ty), mask=nmask2)


def _dgrad_condln(D, w_cat, cond, mean_c, rstd_c, lnw, *, shape_key=None):
    """Fused dgrad+cond-LN-bwd: dcond_aff = Dᵀ@w_cat (in-kernel GEMM) → cond LN-backward → (dcond,
    dlnw). No dcond_aff HBM round-trip, no cuBLAS dgrad. D=(2NX,M), w_cat=(2NX,NC)."""
    K2, M = D.shape
    if shape_key is None:
        # off `cond` (M, NC), not off D: D is (2NX, M), so its shape[-2] is 2NX, not a length.
        shape_key = atom_key(length_of(cond.shape))
    NC = w_cat.shape[1]
    dcond = torch.empty_strided((M, NC), cond.stride(), device=cond.device, dtype=cond.dtype)
    dlnw = torch.zeros(NC, dtype=torch.float32, device=cond.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # NC is tl.constexpr now (it drives the BLOCK_NC >= NC fold) -> pass a plain python int.
    _dgrad_condln_kernel[grid](
        D, w_cat, cond, mean_c, rstd_c, lnw, dcond, dlnw, M, int(NC), K2,
        D.stride(0), D.stride(1), w_cat.stride(0), w_cat.stride(1),
        cond.stride(0), cond.stride(1), dcond.stride(0), dcond.stride(1),
        shape_key=shape_key,
    )
    return dcond, dlnw.to(lnw.dtype)


# dgrad path. None → AUTO dispatch by NC (the cond dim): the fused triton in-kernel GEMM beats
# cuBLAS only when K2=2·NX is small enough that the dgrad is memory-bound (atom d=128: K2=256 →
# 1.17-1.19× vs compile, fusing out the dcond_aff round-trip); at token d=768 (K2=1536) the GEMM is
# compute-bound and triton TF32 can't match cuBLAS (0.66×), so cuBLAS dgrad + _ln_bwd is kept there.
# True/False force the choice (for benchmarking). wgrad (dW) is ALWAYS cuBLAS (per directive).
_DGRAD_TRITON = None
_DGRAD_TRITON_NC_MAX = 256   # fused triton dgrad wins for NC ≤ this (memory-bound regime)


def set_dgrad_triton(flag) -> None:
    """Select dgrad impl: None=auto (by NC), True=force fused triton, False=force cuBLAS+_ln_bwd."""
    global _DGRAD_TRITON  # noqa: PLW0603
    _DGRAD_TRITON = flag


class AdaLNTrainFn(torch.autograd.Function):
    @staticmethod
    @opaque()
    def forward(ctx, x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight, eps_x, eps_cond):
        orig_x_shape = x.shape
        orig_cond_shape = cond.shape
        nx = orig_x_shape[-1]
        nc = orig_cond_shape[-1]
        x2d = x.reshape(-1, nx)
        cond2d = cond.reshape(-1, nc)
        if x2d.stride(-1) != 1:
            x2d = x2d.contiguous()
        if cond2d.stride(-1) != 1:
            cond2d = cond2d.contiguous()

        beta0 = scale_bias.new_zeros(nc)
        cond_aff, mean_c, rstd_c = _ln_materialize(cond2d, cond_ln_weight, beta0, eps_cond,
                                                   shape_key=both_key(length_of(orig_cond_shape)))

        w_cat = torch.cat([scale_weight, bias_weight], dim=0)          # (2NX, NC)
        sb = _mm(cond_aff, w_cat.t())                                  # (M, 2NX) raw [scale|bias]

        # scale_bias (β for the scale half) is folded into the epilogue — no full (M,2NX) add pass.
        y, mean_x, rstd_x, gate = _epilogue_train(x2d, sb, eps_x, scale_bias,
                                                 shape_key=atom_key(length_of(orig_x_shape)))

        ctx.save_for_backward(x2d, cond2d, cond_aff, gate, mean_x, rstd_x, mean_c, rstd_c,
                              cond_ln_weight, scale_weight, bias_weight)
        ctx.orig_x_shape = orig_x_shape
        ctx.orig_cond_shape = orig_cond_shape
        ctx.nx = nx
        ctx.dtypes = (x.dtype, cond.dtype, cond_ln_weight.dtype,
                      scale_weight.dtype, scale_bias.dtype, bias_weight.dtype)
        return y.reshape(orig_x_shape)

    @staticmethod
    @opaque()
    def backward(ctx, dy):
        (x2d, cond2d, cond_aff, gate, mean_x, rstd_x, mean_c, rstd_c,
         lnw, scale_weight, bias_weight) = ctx.saved_tensors
        nx = ctx.nx
        dy2d = dy.reshape(-1, dy.shape[-1])
        if dy2d.stride(-1) != 1:
            dy2d = dy2d.contiguous()

        # ONE kernel: D=(2NX,M) [dscale;dy] AND dx (fused x LN-backward, no affine, in x's layout).
        # L for both backward launches: the pre-flatten atom count the forward stored on ctx.
        shape_key = atom_key(length_of(ctx.orig_x_shape))
        D, dx = _bwd_x(dy2d, x2d, mean_x, rstd_x, gate,
                       shape_key=shape_key)                          # D=(2NX,M), dx=(M,NX)
        w_cat = torch.cat([scale_weight, bias_weight], dim=0)          # (2NX, NC)

        dW_cat = _mm(D, cond_aff)                                     # (2NX,NC) = [dWs;dWb]  (cuBLAS wgrad — ONLY cuBLAS matmul)
        dsb = D[:nx].sum(dim=1)                                       # Σ_m dscale → (NX,)

        use_triton = (cond2d.shape[1] <= _DGRAD_TRITON_NC_MAX) if _DGRAD_TRITON is None else _DGRAD_TRITON
        if use_triton:
            # FUSED triton: dcond_aff = Dᵀ@w_cat (in-kernel GEMM) + cond LN-backward in ONE kernel.
            dcond, dlnw = _dgrad_condln(D, w_cat, cond2d, mean_c, rstd_c, lnw,
                                        shape_key=shape_key)
        else:
            dcond_aff = _mm(D.t(), w_cat)                             # cuBLAS dgrad → (M,NC)
            dcond, dlnw, _ = _ln_bwd(dcond_aff, cond2d, lnw, mean_c, rstd_c, cond2d.stride(),
                                     shape_key=both_key(length_of(ctx.orig_cond_shape)))

        dW_scale = dW_cat[:nx].contiguous()
        dW_bias = dW_cat[nx:].contiguous()

        xd, cd, lnwd, swd, sbd, bwd = ctx.dtypes
        return (
            dx.reshape(ctx.orig_x_shape).to(xd),
            dcond.reshape(ctx.orig_cond_shape).to(cd),
            dlnw.to(lnwd),
            dW_scale.to(swd),
            dsb.to(sbd),
            dW_bias.to(bwd),
            None,
            None,
        )


def adaln_train(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight,
                eps_x, eps_cond):
    """Training (fwd+bwd) adaLN: y = σ(scale)·LN(x) + bias, materialize+cuBLAS, symmetric backward."""
    return AdaLNTrainFn.apply(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight,
                              eps_x, eps_cond)
