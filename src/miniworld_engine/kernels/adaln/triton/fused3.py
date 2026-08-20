"""adaLN forward as 3 Triton kernels (user grouping):
  K1: x_norm    = LayerNorm(x)                 (no affine)
  K2: cond_norm = LayerNorm(cond) * lnw
  K3: scale = cond_norm@Wsᵀ + scale_b ; bias = cond_norm@Wbᵀ ; y = sigmoid(scale)*x_norm + bias
      (in-kernel dual-GEMM over K=d_cond + sigmoid-gate epilogue, all in ONE Triton kernel)

Steps 3+4+5 are fused into K3 (the two projections + gate). GEMM done in-kernel via tl.dot.
"""
from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of

# ── K1 / K2: row-wise LayerNorm ────────────────────────────────────────────────────────────
# Both axes are tuned tiles. BLOCK_N used to arrive as next_pow2(d) from the launcher — the whole
# row, a constant the tuner never saw — which is also why BLOCK_M1 had to stay at 1..16 (a
# [16, 1024] fp32 tile was already the register budget). The N axis is a REDUCE axis (mean/var),
# so a CSV row at or above the extent spans a whole d_hidden/d_cond row; with N tiled,
# BLOCK_M1 can take the canonical (>=16) 2-D tile sizes and the two axes trade off properly.




# seq_group is the row-count cache bucket. It is NOT GROUP_M: in this file GROUP_M is the tuned
# L2-swizzle axis the two GEMM kernels read from the CSV, so the bucket takes a separate,
# lowercase name -- it is a plain runtime int no kernel body ever reads.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("layernorm_fwd_strided_triton"), key=['N', 'HAS_W', 'seq_group'])
@triton.jit
def _ln_kernel(X, Y, W, M, N: tl.constexpr, eps, sx0, sx1, sy0, sy1,
              HAS_W: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr,
              seq_group):
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


def _layernorm(x, eps, weight=None):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    # N is a tl.constexpr now (it drives the BLOCK_N >= N fold), so it must reach the kernel as a
    # plain python int — it is already the autotune key, so this adds no recompile that the key
    # did not already force.
    _ln_kernel[grid](x, y, weight if weight is not None else x, M, int(N), eps,
                     x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                     HAS_W=weight is not None, seq_group=get_seq_group(M),)
    return y


# ── K3: dual in-kernel GEMM (scale=cond_norm@Wsᵀ+b, bias=cond_norm@Wbᵀ) + sigmoid gate ──────
# Proper triton matmul: 1-D grid + GROUP_M L2-swizzle, TF32 tensor cores (input_precision).
# (Two tl.dot per K-step share the loaded cond_norm tile `a`.)
# All four axes come from the CSV, GROUP_M included: it only reorders which (pid_m, pid_n) a
# program takes, so every value >= 1 covers the output exactly once and it is performance-only.
# Nothing bounds smem here -- a row that does not fit the running card fails at launch.


# The row count is keyed too: this is a 2-D GEMM whose grid is cdiv(M,BLOCK_M1)*cdiv(N,BLOCK_N),
# and both N and K are weight extents -- so without seq_group the key was constant across every
# sequence length and one tile served a 128-row launch and a 1M-row launch alike.
@triton.autotune(configs=configs_for("adaln_gemm_gate_triton"), key=['N', 'K', 'seq_group', 'SAVE_GATE'])
@triton.jit
def _gemm_gate_kernel(
    Xn, Cn, Ws, Wb, Sb, Y, Gate, M, N, K,
    sxn0, sxn1, scn0, scn1, sws0, sws1, swb0, swb1, sy0, sy1, sg0, sg1,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    seq_group,
    SAVE_GATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    scale = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    bias = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        kmask = kk < K
        a = tl.load(Cn + rm[:, None] * scn0 + kk[None, :] * scn1,
                    mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        # Ws,Wb are (N,K) row-major; tile [k,n]=W[n,k] is k-contiguous → the MMA-friendly B layout
        # (TN). (Transposing to K-major was tried and is ~1.8× SLOWER — wrong B operand layout.)
        ws = tl.load(Ws + rn[None, :] * sws0 + kk[:, None] * sws1,
                     mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        wb = tl.load(Wb + rn[None, :] * swb0 + kk[:, None] * swb1,
                     mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        scale += tl.dot(a, ws, input_precision="tf32", out_dtype=tl.float32)
        bias += tl.dot(a, wb, input_precision="tf32", out_dtype=tl.float32)
    sb = tl.load(Sb + rn, mask=rn < N, other=0.0).to(tl.float32)
    scale += sb[None, :]
    xn = tl.load(Xn + rm[:, None] * sxn0 + rn[None, :] * sxn1,
                 mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0).to(tl.float32)
    gate = tl.sigmoid(scale)
    y = gate * xn + bias
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y + rm[:, None] * sy0 + rn[None, :] * sy1, y.to(Y.dtype.element_ty), mask=om)
    if SAVE_GATE:  # the training path needs sigma(scale) for the backward; inference does not
        tl.store(Gate + rm[:, None] * sg0 + rn[None, :] * sg1,
                 gate.to(Gate.dtype.element_ty), mask=om)


def _gemm_gate(x_norm, cond_norm, Ws, Wb, scale_b):
    # Ws, Wb are the (N, K) nn.Linear weights (k-contiguous tile = MMA-friendly B layout).
    M, N = x_norm.shape
    K = cond_norm.shape[1]
    y = torch.empty_like(x_norm)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _gemm_gate_kernel[grid](
        x_norm, cond_norm, Ws, Wb, scale_b, y, y, M, N, K,
        x_norm.stride(0), x_norm.stride(1), cond_norm.stride(0), cond_norm.stride(1),
        Ws.stride(0), Ws.stride(1), Wb.stride(0), Wb.stride(1), y.stride(0), y.stride(1),
        0, 0,                       # Gate is unread when SAVE_GATE=False
        seq_group=get_seq_group(M), SAVE_GATE=False,
    )
    return y


# ── training: K3 variant that also stores gate=sigmoid(scale); + backward elementwise ──────────


def _gemm_gate_train(x_norm, cond_norm, Ws, Wb, scale_b):
    M, N = x_norm.shape
    K = cond_norm.shape[1]
    y = torch.empty_like(x_norm)
    gate = torch.empty_like(x_norm)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _gemm_gate_kernel[grid](
        x_norm, cond_norm, Ws, Wb, scale_b, y, gate, M, N, K,
        x_norm.stride(0), x_norm.stride(1), cond_norm.stride(0), cond_norm.stride(1),
        Ws.stride(0), Ws.stride(1), Wb.stride(0), Wb.stride(1),
        y.stride(0), y.stride(1), gate.stride(0), gate.stride(1),
        seq_group=get_seq_group(M), SAVE_GATE=True,)
    return y, gate




# Elementwise, but it shares the LN grid: it is the same (M, d) row shape and its BLOCK_K came
# from the same next_pow2(N) launch, so the candidate set has to keep reaching a whole row (1024)
# rather than stopping at the canonical BLOCK_K's 256 — dropping the value the launcher used is
# not a fix. With no reduction there is a single pass over the N tiles.
@triton.autotune(configs=configs_for("adaln_bwd_pre_triton"), key=['N', 'seq_group'])
@triton.jit
def _bwd_elem_kernel(DY, Xn, Gate, Dscale, Dxn, M, N,
                     sy0, sy1, sxn0, sxn1, sg0, sg1, sds0, sds1, sdx0, sdx1,
                     BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, seq_group):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rmask = rm < M
    for n0 in range(0, N, BLOCK_K):
        cols = n0 + tl.arange(0, BLOCK_K)
        mask = rmask[:, None] & (cols[None, :] < N)
        dy = tl.load(DY + rm[:, None] * sy0 + cols[None, :] * sy1, mask=mask, other=0.0).to(tl.float32)
        xn = tl.load(Xn + rm[:, None] * sxn0 + cols[None, :] * sxn1, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gate + rm[:, None] * sg0 + cols[None, :] * sg1, mask=mask, other=0.0).to(tl.float32)
        dscale = dy * xn * g * (1.0 - g)
        dxn = dy * g
        tl.store(Dscale + rm[:, None] * sds0 + cols[None, :] * sds1,
                 dscale.to(Dscale.dtype.element_ty), mask=mask)
        tl.store(Dxn + rm[:, None] * sdx0 + cols[None, :] * sdx1,
                 dxn.to(Dxn.dtype.element_ty), mask=mask)


def _bwd_elem(dy, x_norm, gate):
    M, N = dy.shape
    dscale = torch.empty_like(dy)
    dxn = torch.empty_like(dy)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]),)  # noqa: E731
    _bwd_elem_kernel[grid](dy, x_norm, gate, dscale, dxn, M, N,
                           dy.stride(0), dy.stride(1), x_norm.stride(0), x_norm.stride(1),
                           gate.stride(0), gate.stride(1), dscale.stride(0), dscale.stride(1),
                           dxn.stride(0), dxn.stride(1), seq_group=get_seq_group(M),)
    return dscale, dxn


class _Fused3TrainFn(torch.autograd.Function):
    @staticmethod
    @opaque()
    def forward(ctx, x, cond, lnw, Ws, sb, Wb, eps_x, eps_cond):
        from ...layernorm_linear.triton.te_style import _ln_materialize
        orig = x.shape
        nx = orig[-1]
        nc = cond.shape[-1]
        x2d = x.reshape(-1, nx).contiguous() if x.reshape(-1, nx).stride(-1) != 1 else x.reshape(-1, nx)
        cond2d = cond.reshape(-1, nc).contiguous() if cond.reshape(-1, nc).stride(-1) != 1 else cond.reshape(-1, nc)
        ones = sb.new_ones(nx)
        zeros_x = sb.new_zeros(nx)
        zeros_c = sb.new_zeros(nc)
        x_norm, mean_x, rstd_x = _ln_materialize(x2d, ones, zeros_x, eps_x)
        cond_norm, mean_c, rstd_c = _ln_materialize(cond2d, lnw, zeros_c, eps_cond)
        y, gate = _gemm_gate_train(x_norm, cond_norm, Ws, Wb, sb)
        ctx.save_for_backward(x2d, cond2d, x_norm, cond_norm, gate, mean_x, rstd_x, mean_c, rstd_c, lnw, Ws, Wb)
        ctx.orig = orig
        ctx.ocond = cond.shape
        ctx.dt = (x.dtype, cond.dtype, lnw.dtype, Ws.dtype, sb.dtype, Wb.dtype)
        return y.reshape(orig)

    @staticmethod
    @opaque()
    def backward(ctx, dy):
        from ...layernorm_linear.triton.te_style import _ln_bwd, _fp32_matmul_ctx
        (x2d, cond2d, x_norm, cond_norm, gate, mean_x, rstd_x, mean_c, rstd_c, lnw, Ws, Wb) = ctx.saved_tensors
        nx = x2d.shape[-1]
        dy2d = dy.reshape(-1, nx)
        dy2d = dy2d.contiguous() if dy2d.stride(-1) != 1 else dy2d
        dscale, dxn = _bwd_elem(dy2d, x_norm, gate)
        with _fp32_matmul_ctx(dy.dtype):
            dWs = dscale.t() @ cond_norm           # (N,K)
            dWb = dy2d.t() @ cond_norm             # (N,K)
            dsb = dscale.sum(0)                     # (N,)
            dcond_norm = torch.addmm(dscale @ Ws, dy2d, Wb)  # dscale@Ws + dy@Wb → (M,K)
        ones = lnw.new_ones(nx)
        dx, _, _ = _ln_bwd(dxn, x2d, ones, mean_x, rstd_x, x2d.stride())
        dcond, dlnw, _ = _ln_bwd(dcond_norm, cond2d, lnw, mean_c, rstd_c, cond2d.stride())
        xd, cd, lnwd, wsd, sbd, wbd = ctx.dt
        return (dx.reshape(ctx.orig).to(xd), dcond.reshape(ctx.ocond).to(cd), dlnw.to(lnwd),
                dWs.to(wsd), dsb.to(sbd), dWb.to(wbd), None, None)


def adaln_fused3_train(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight, eps_x, eps_cond):
    """Training (fwd+bwd) for the fused3 grouping. Backward: triton elementwise + LN-bwd, cuBLAS GEMMs."""
    return _Fused3TrainFn.apply(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight, eps_x, eps_cond)


def adaln_fused3(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight, eps_x, eps_cond):
    """3-kernel adaLN forward: K1 LN(x), K2 LN(cond)*lnw, K3 dual-GEMM + sigmoid gate."""
    orig = x.shape
    x2d = x.reshape(-1, orig[-1])
    cond2d = cond.reshape(-1, cond.shape[-1])
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    if cond2d.stride(-1) != 1:
        cond2d = cond2d.contiguous()
    x_norm = _layernorm(x2d, eps_x)                         # K1
    cond_norm = _layernorm(cond2d, eps_cond, cond_ln_weight)  # K2
    y = _gemm_gate(x_norm, cond_norm, scale_weight, bias_weight, scale_bias)  # K3
    return y.reshape(orig)
