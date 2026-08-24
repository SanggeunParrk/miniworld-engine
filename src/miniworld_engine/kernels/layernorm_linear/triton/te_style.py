"""TE-style trainable LayerNormLinear: split forward/backward, materialize-then-GEMM (no fold).

Why this exists alongside the fold-based `layernorm_linear_fn`: the inference path folds
`W2=γ⊙W` and runs a custom epilogue GEMM — great for inference (fold cached), but in TRAINING
the fold is recomputed every step (weights change) and the full fwd+bwd loses to Transformer
Engine. TE's structure is materialize `x_normed=LN(x)` then a plain cuBLAS GEMM; we mirror that.

The reason to roll our OWN (vs just calling `te.LayerNormLinear`) is STRIDE COVERAGE: trimul emits
its output as a `[B,D,L,L]` tensor whose `(M=L*L, K=D)` view is **m-major / k-strided** (strides
`(1, L*L)`). TE forces a `.contiguous()` (a BDLL→BLLD transpose copy) on such input; here the
LN-materialize kernel reads x at ARBITRARY strides (the stride-absorber) and the backward writes
**dx in the SAME layout as the input** (m-major in → m-major out) so trimul's backward consumes it
copy-free.

Precision: dtype-transparent — fp32 / bf16 / fp16 inputs all flow through ONE code path (the LN
kernels compute in fp32 and store `element_ty`; GEMMs accumulate in fp32). The path "splits" only
where it matters: the LN kernels autotune per byte-width (fp32 vs 16-bit), and fp32 GEMMs honor a
TF32-vs-true-fp32 policy (`set_fp32_matmul_precision`, default 'high'=TF32). v1 requires X and W to
share a dtype (no mixed bf16-act/fp32-weight yet).

Structure (all cuBLAS GEMMs = TE-parity; LN kernels are Triton = portable, no quack/SM90 dep):
  forward  : x_normed = LN(x)              (Triton, strided x → contiguous x_normed + mean,rstd)
             Y = x_normed @ Wᵀ + b         (F.linear / cuBLAS)
             save: x (strided view, no copy), mean, rstd, γ, β, W
  backward : dx_normed = dY @ W            (cuBLAS)
             dx = LN-bwd(dx_normed, x, γ, μ, rstd)   (Triton, dx written in x's strides)
             T = dYᵀ @ x̂                   (cuBLAS wgrad; x̂ recomputed from saved stats)
             db = Σ_m dY ; dW = γ⊙T + outer(db,β) ; dγ = (W⊙T).sum(0) ; dβ = db @ W
The T-decomposition gives dW/dγ/dβ/db from ONE wgrad GEMM with no in-kernel M-reduction and no
x_normed materialization (see kernels/.../cute/dgrad_lnbwd.py for the same identity, derived).
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.kernels._compile import opaque

import contextlib

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


from miniworld_engine.autotune.shape_key import both_key  # every kernel here is level=both

# ── dtype support ──────────────────────────────────────────────────────────────────────────────
# The Triton LN kernels are dtype-generic (compute in fp32, store `element_ty`), so fp32/bf16/fp16
# all flow through ONE code path. Two things ARE dtype-specific:
#  • autotune: triton appends every tensor argument's dtype to the cache key, so fp32 (2x bytes ->
#    different occupancy)
#    tunes separately from bf16/fp16 (which share, both 2-byte).
#  • fp32 GEMM precision: TF32 ("high", fast) vs true-fp32 ("highest"). Low-precision GEMMs are
#    unaffected (always bf16/fp16 operand + fp32 accum). Toggle via set_fp32_matmul_precision().
_FP32_MATMUL_PRECISION = "high"   # "high" → TF32 (default, fast); "highest" → true fp32


def set_fp32_matmul_precision(mode: str) -> None:
    """'high' = TF32 cuBLAS for fp32 GEMMs (fast, ~fp32); 'highest' = true fp32 (accurate, slower).
    Only affects fp32 inputs; bf16/fp16 are always fp32-accumulated regardless."""
    global _FP32_MATMUL_PRECISION
    assert mode in ("high", "highest")
    _FP32_MATMUL_PRECISION = mode


@contextlib.contextmanager
def _fp32_matmul_ctx(dtype):
    """For fp32 inputs, set cuBLAS TF32 per the policy (save/restore); no-op otherwise."""
    if dtype is not torch.float32:
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = (_FP32_MATMUL_PRECISION == "high")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

# Keyed on (N, shape_key) so the config is tuned PER (d, M-bucket) — not reused across M for a given
# d (the earlier ["N"]-only key tuned on whichever M was hit first and reused it for all M).
# BLOCK_M1 and BLOCK_K come from the CSV.


# ───────────────────────── forward: LN-materialize (strided x → contiguous x_normed) ─────────


# Own config list (NOT the shared `_LN_CONFIGS`): the N axis is this kernel's REDUCE axis and is
# `_ln_mat_kernel` tunes BLOCK_N: it is the LayerNorm reduce axis and its covering tile (one tile
# spanning N) is what makes the kernel single-pass, so the sweep must be able to reach a whole
# d_hidden row. `_ln_bwd_kernel` below has no N loop at all -- its launcher computes N_PAD =
# next_power_of_2(K) -- so that one is a derived constant, not a tunable tile.
# The maxnreg variants of `_LN_CONFIGS` are dropped here only — their own comment records that the
# register cap never won on `_ln_mat_kernel` (it was added for the m-major `_ln_bwd_kernel`).


@triton.autotune(configs=configs_for("layernorm_fwd_saveact_strided_triton"), key=['N', 'shape_key'])
@triton.jit
def _ln_mat_kernel(X, Xn, Mean, Rstd, G, B, M, N: tl.constexpr, eps,
                   sx0, sx1, sn0, sn1,
                   BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, shape_key,
                   ):
    row = tl.program_id(0).to(tl.int64)
    rm = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    rmask = rm < M
    # TWO-PASS (not Welford): pass 1 accumulates Σx and Σx² over the N tiles in fp32 — both are
    # plain sums, exact across tiles — and pass 2 re-reads x to normalize + write Xn. LN re-uses
    # the row it just reduced, so a tiled reduce axis costs either a second read of x or a Welford
    # carry; the second read is simpler.
    #
    # COVERING TILE (BLOCK_K >= N): the two loops are single-trip, but the two tl.loads of X are
    # NOT CSE'd — the Mean/Rstd tl.store sits between them and Triton cannot prove the raw pointers
    # do not alias — so the covering config read x twice. `N` is `tl.constexpr` (already this
    # kernel's autotune key, so a new d_hidden already forced a re-tune and a fresh compile) which
    # makes the guard a TRACE-time comparison: one branch emitted, covering tile back to the
    # untiled single-read schedule. The fast path uses the CENTRED variance Σ(x-mean)²/N
    # (numerically stabler, and x is already in registers); the uncentered Σx²/N - mean² stays in
    # the tiled branch, where it is what keeps that branch to one read per tile.
    if BLOCK_K >= N:
        cols = tl.arange(0, BLOCK_K)
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / N
        xc = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Mean + rm, mean, mask=rmask)
        tl.store(Rstd + rm, rstd, mask=rmask)
        g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        xn = xc * rstd[:, None] * g + b
        tl.store(Xn + rm[:, None] * sn0 + cols[None, :] * sn1,
                 xn.to(Xn.dtype.element_ty), mask=mask)
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
        tl.store(Mean + rm, mean, mask=rmask)
        tl.store(Rstd + rm, rstd, mask=rmask)
        for n0 in range(0, N, BLOCK_K):
            cols = n0 + tl.arange(0, BLOCK_K)
            cmask = cols < N
            mask = rmask[:, None] & cmask[None, :]
            x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            xn = (x - mean[:, None]) * rstd[:, None] * g + b
            tl.store(Xn + rm[:, None] * sn0 + cols[None, :] * sn1,
                     xn.to(Xn.dtype.element_ty), mask=mask)


def _ln_materialize_fake(x, gamma, beta, eps, shape_key=None):
    m, k = x.shape
    return (
        x.new_empty((m, k)),
        x.new_empty((m,), dtype=torch.float32),
        x.new_empty((m,), dtype=torch.float32),
    )


@opaque(fake=_ln_materialize_fake, name="layernorm_linear_materialize")
def _ln_materialize(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float,
                    shape_key: int | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x_normed = LN(x)=(x-μ)·rstd·γ+β. Reads x at its OWN strides (m-major/strided OK, no
    pre-copy), writes a CONTIGUOUS (M,K) x_normed; returns (x_normed, mean, rstd).

    ``shape_key`` is ``both_key(L)`` (``_ln_mat_kernel`` is level=both in registry.csv), computed
    by the CALLER: x arrives here already flattened to (M, K), and M alone cannot say which L
    produced it -- a trimul pair view has M = L*L, an adaLN atom activation has M = A. None ->
    the smallest bucket, an explicit "L not supplied" label (bench / driver entry only).
    """
    M, K = x.shape
    xn = torch.empty(M, K, device=x.device, dtype=x.dtype)         # contiguous out
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N (=K here) is tl.constexpr now (it drives the BLOCK_N >= N fold) -> pass a plain python int.
    _ln_mat_kernel[grid](
        x, xn, mean, rstd, gamma, beta, M, int(K), eps,
        x.stride(0), x.stride(1), xn.stride(0), xn.stride(1),
        shape_key=both_key(0) if shape_key is None else shape_key,
    )
    return xn, mean, rstd


# ───────── backward: ONE LN-backward kernel → dx (arbitrary strides) + dγ + dβ (M-reduce) ─────


@triton.autotune(configs=configs_for("layernorm_bwd_atomic_strided_triton"),
                 key=['N', 'shape_key'],
                 reset_to_zero=['DG', 'DB'])
@triton.jit
def _ln_bwd_kernel(DXn, X, G, Mean, Rstd, DX, DG, DB, M, N,
                   sdn0, sdn1, sx0, sx1, sdx0, sdx1,
                   BLOCK_M1: tl.constexpr, N_PAD: tl.constexpr, shape_key,
                   ):
    # NOTE: one tile per program (atomic_add per block for dγ/dβ). A grid-stride variant (one
    # atomic per program) sped up CONTIGUOUS large-d (d512 0.96→1.07x) but CATASTROPHICALLY
    # regressed m-major d=256 (2.45→0.45x — the strided x/dx access interacts badly with the
    # strided loop), so it was reverted. Keep this simple form (good on both layouts).
    row = tl.program_id(0).to(tl.int64)
    rm = tl.arange(0, BLOCK_M1) + row * BLOCK_M1
    rmask = rm < M
    cols = tl.arange(0, N_PAD)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    dxn = tl.load(DXn + rm[:, None] * sdn0 + cols[None, :] * sdn1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + rm, mask=rmask, other=0.0)[:, None]
    rstd = tl.load(Rstd + rm, mask=rmask, other=0.0)[:, None]
    g = tl.load(G + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    xhat = tl.where(cmask[None, :], (x - mean) * rstd, 0.0)
    dxhat = dxn * g
    inv_n = 1.0 / N
    c2 = tl.sum(tl.where(cmask[None, :], dxhat, 0.0), axis=1) * inv_n        # meanₖ(dx̂)
    c1 = tl.sum(tl.where(cmask[None, :], dxhat * xhat, 0.0), axis=1) * inv_n  # meanₖ(dx̂·x̂)
    dx = rstd * (dxhat - c2[:, None] - xhat * c1[:, None])
    tl.store(DX + rm[:, None] * sdx0 + cols[None, :] * sdx1,
             dx.to(DX.dtype.element_ty), mask=mask)
    pdg = tl.sum(tl.where(mask, dxn * xhat, 0.0), axis=0)
    pdb = tl.sum(tl.where(mask, dxn, 0.0), axis=0)
    tl.atomic_add(DG + cols, pdg, mask=cmask)
    tl.atomic_add(DB + cols, pdb, mask=cmask)


def _ln_bwd_fake(dx_normed, x, gamma, mean, rstd, dx_strides, shape_key=None):
    m, k = x.shape
    # dx is written AT dx_strides (m-major in -> m-major out), so the fake has to carry those
    # strides too: the compiled graph reads its layout decisions off this tensor.
    return (
        torch.empty_strided((m, k), tuple(dx_strides), device=x.device, dtype=dx_normed.dtype),
        x.new_empty((k,), dtype=torch.float32),
        x.new_empty((k,), dtype=torch.float32),
    )


@opaque(fake=_ln_bwd_fake, name="layernorm_linear_bwd_mmajor")
def _ln_bwd(dx_normed: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, mean: torch.Tensor,
            rstd: torch.Tensor, dx_strides: list[int], shape_key: int | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ONE pass: dx = rstd·(γ·dxn − meanₖ(γ·dxn) − x̂·meanₖ(γ·dxn·x̂)) written at `dx_strides`
    (m-major in→out), plus dγ=Σ_m dxn·x̂ and dβ=Σ_m dxn. x̂ recomputed inside from x (read at its
    strides) + saved μ,rstd — no separate recompute kernel.

    Delegates to the m-major-specialized backward (`mmajor_bwd.ln_bwd_mmajor`): atomic-free
    persistent dγ/dβ reduction (fp32 register partials over a persistent grid + tiny cross-CTA
    reduce, no atomic_add L2 contention) with M-contiguous vector loads — 1.10–1.21x over the
    plain-atomic kernel at large M on B200 (bidir N=256 / single N=128), size-adaptive fall-back
    to the atomic kernel at small M. Bit-exact (dx/dγ/dβ cos 1.0); math/precision unchanged.
    Deferred import breaks the mmajor_bwd<->te_style module cycle.

    ``shape_key`` is ``both_key(L)`` from the caller (see ``_ln_materialize``); it is forwarded to
    whichever of the three size-adaptive paths runs."""
    from .mmajor_bwd import ln_bwd_mmajor
    return ln_bwd_mmajor(dx_normed, x, gamma, mean, rstd, dx_strides, shape_key=shape_key)


# ───────────────────────── db = Σ_m dY (linear bias grad) ─────────────────────────────────────
def _bias_grad(dY: torch.Tensor) -> torch.Tensor:
    """db = Σ_m dY → (N,). As a cuBLAS GEMV (ones(1,M)@dY): reads dY once with fp32 accum, ~2.4x
    faster than torch.sum(0) (which strides over the long outer dim: 0.063 vs 0.152ms @ M=262144
    d=256). Beats a hand triton column-sum at large M; the saving (>80µs) exceeds the whole
    ours-vs-TE backward gap. (TE gets db free via the wgrad GEMM's bias-grad epilogue.)"""
    M = dY.shape[0]
    ones = torch.ones(1, M, device=dY.device, dtype=dY.dtype)
    return (ones @ dY).squeeze(0)


# ───────────────────────── forward / backward / autograd Function ────────────────────────────
def _te_forward(x, gamma, beta, W, bias, eps, *, shape_key: int | None = None):
    x_normed, mean, rstd = _ln_materialize(x, gamma, beta, eps, shape_key=shape_key)
    with _fp32_matmul_ctx(x.dtype):
        Y = F.linear(x_normed, W, bias)            # x_normed @ Wᵀ + bias  (cuBLAS; fp32→TF32 policy)
    return Y, x_normed, mean, rstd


def _te_backward(dY, x_normed, x, mean, rstd, gamma, W, has_bias, *,
                 shape_key: int | None = None):
    """TE-matching 4-launch backward (cuBLAS dgrad + 1 LN-bwd kernel + cuBLAS wgrad + db reduce).
    dW uses the SAVED x_normed directly (TE-style) — no T-decomposition / elementwise tail."""
    dY = dY.contiguous() if dY.stride(-1) != 1 else dY
    with _fp32_matmul_ctx(dY.dtype):                      # fp32→TF32 policy for all bwd GEMMs
        # dx_normed = dY@W, but PRODUCED M-MAJOR (as (Wᵀ@dYᵀ)ᵀ, strides (1,M)) so it shares the
        # SAME contiguous axis (m) as the m-major x and the m-major dx written by _ln_bwd. With all
        # three (M,K) operands uniform-m-major, _ln_bwd_kernel coalesces every load/store along m
        # (no mixed row-major-DXn / m-major-X access) → 1.3-1.45x faster LN-bwd on B200. cuBLAS emits
        # the transposed GEMM at the same cost as dY@W (transA/transB flags, +~2%). SAME values
        # (dx/dγ/dβ cos 1.0 vs the row-major GEMM) — layout-only change, precision unchanged.
        dx_normed = torch.matmul(W.t(), dY.t()).t()      # (dY@W) m-major → uniform LN-bwd layout
        dW = torch.matmul(dY.t(), x_normed)              # dYᵀ@x_normed → (N,K) wgrad (cuBLAS)
        db = _bias_grad(dY).to(W.dtype) if has_bias else None  # linear bias grad (cuBLAS GEMV)
    dx, dgamma, dbeta = _ln_bwd(dx_normed, x, gamma, mean, rstd, list(x.stride()),
                                shape_key=shape_key)                 # dx(m-major)+dγ+dβ
    return dx, dgamma.to(gamma.dtype), dbeta.to(gamma.dtype), dW, db


class LayerNormLinearTEFn(torch.autograd.Function):
    """TE-style trainable `Y = LayerNorm(x)@Wᵀ + b`, stride-transparent (m-major in → m-major out).
    Portable: cuBLAS GEMMs + Triton LN kernels, no quack/SM90 dependency.


    ``length`` (L, the pre-flatten token/atom count) is a POSITIONAL input because
    ``autograd.Function.apply`` takes no keywords; it carries no gradient, so ``backward``
    returns a trailing ``None`` for it. It is bucketed once in the forward and reused by the
    backward via ``ctx.shape_key`` -- the saved x is (M, K) and L is not recoverable from it."""

    @staticmethod
    def forward(ctx, x, ln_weight, ln_bias, weight, bias, eps, length):  # noqa: ARG004
        # `length` is no longer what the key is made of. It existed because M could not say
        # whether it was L or L*L; the key is the row count now, and M is the tensor's own
        # leading extent. The parameter stays because callers pass it positionally.
        shape_key = both_key(x.reshape(-1, x.shape[-1]).shape[0])
        Y, x_normed, mean, rstd = _te_forward(x, ln_weight, ln_bias, weight, bias, eps,
                                              shape_key=shape_key)
        ctx.save_for_backward(x_normed, x, mean, rstd, ln_weight, weight)
        ctx.has_bias = bias is not None
        ctx.shape_key = shape_key
        return Y

    @staticmethod
    def backward(ctx, dY):
        x_normed, x, mean, rstd, gamma, W = ctx.saved_tensors
        dx, dg, db_ln, dW, db = _te_backward(dY, x_normed, x, mean, rstd, gamma, W, ctx.has_bias,
                                             shape_key=ctx.shape_key)
        return dx, dg, db_ln, dW, db, None, None


def layernorm_linear_te_fn(x, ln_weight, ln_bias, weight, bias=None, eps: float = 1e-5,
                           length: int | None = None):
    """TE-style trainable LayerNormLinear (materialize+cuBLAS GEMM, stride-transparent).
    Accepts a strided/m-major x (e.g. a trimul BDLL view) with NO .contiguous() copy, and
    returns dx in the same layout. Grads flow to x, ln_weight, ln_bias, weight, bias.

    ``length`` is L -- the TOKEN/ATOM count of the activation before it was flattened into the
    (M, K) ``x`` this entry point takes. This is the only place L can enter: the trimul BDLL view
    that reaches here has M = L*L, so nothing below can recover it. It is a pure autotune-cache
    label (``both_key(L)``) -- no number changes with it. Omitting it labels every launch with the
    smallest bucket; pass it whenever you know it."""
    return LayerNormLinearTEFn.apply(x, ln_weight, ln_bias, weight, bias, eps, length)
