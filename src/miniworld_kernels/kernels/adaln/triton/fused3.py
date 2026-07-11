"""adaLN forward as 3 Triton kernels (user grouping):
  K1: x_norm    = LayerNorm(x)                 (no affine)
  K2: cond_norm = LayerNorm(cond) * lnw
  K3: scale = cond_norm@Wsᵀ + scale_b ; bias = cond_norm@Wbᵀ ; y = sigmoid(scale)*x_norm + bias
      (in-kernel dual-GEMM over K=d_cond + sigmoid-gate epilogue, all in ONE Triton kernel)

Steps 3+4+5 are fused into K3 (the two projections + gate). GEMM done in-kernel via tl.dot.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# ── K1 / K2: row-wise LayerNorm (full row per program, BLOCK_N = next_pow2(d)) ──────────────
_LN_CFG = [triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
           for bm in (1, 2, 4, 8, 16) for nw in (4, 8, 16) for ns in (2, 3, 4)]


@triton.autotune(configs=_LN_CFG, key=["N", "HAS_W", "DT"])
@triton.jit
def _ln_kernel(X, Y, W, M, N, eps, sx0, sx1, sy0, sy1,
              HAS_W: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(X + rm[:, None] * sx0 + cols[None, :] * sx1, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    xn = xc * (1.0 / tl.sqrt(var + eps))[:, None]
    if HAS_W:
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        xn = xn * w
    tl.store(Y + rm[:, None] * sy0 + cols[None, :] * sy1, xn.to(Y.dtype.element_ty), mask=mask)


def _layernorm(x, eps, weight=None):
    M, N = x.shape
    y = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    _ln_kernel[grid](x, y, weight if weight is not None else x, M, N, eps,
                     x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                     HAS_W=weight is not None, BLOCK_N=triton.next_power_of_2(N), DT=x.element_size())
    return y


# ── K3: dual in-kernel GEMM (scale=cond_norm@Wsᵀ+b, bias=cond_norm@Wbᵀ) + sigmoid gate ──────
# Proper triton matmul: 1-D grid + GROUP_M L2-swizzle, TF32 tensor cores (input_precision), wide
# autotune. (Two tl.dot per K-step share the loaded cond_norm tile `a`.)
_GEMM_CFG = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                  num_warps=nw, num_stages=ns)
    for bm in (64, 128, 256) for bn in (64, 128, 256) for bk in (32, 64)
    for nw in (4, 8) for ns in (3, 4, 5)
    if bm * bn <= 256 * 128 and (bm + 2 * bn) * bk <= 24576  # smem guard (3 operands, ≥3 stages)
]


@triton.autotune(configs=_GEMM_CFG, key=["N", "K", "DT"])
@triton.jit
def _gemm_gate_kernel(
    Xn, Cn, Ws, Wb, Sb, Y, M, N, K,
    sxn0, sxn1, scn0, scn1, sws0, sws1, swb0, swb1, sy0, sy1,
    DT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    scale = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
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
    y = tl.sigmoid(scale) * xn + bias
    tl.store(Y + rm[:, None] * sy0 + rn[None, :] * sy1, y.to(Y.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def _gemm_gate(x_norm, cond_norm, Ws, Wb, scale_b):
    # Ws, Wb are the (N, K) nn.Linear weights (k-contiguous tile = MMA-friendly B layout).
    M, N = x_norm.shape
    K = cond_norm.shape[1]
    y = torch.empty_like(x_norm)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _gemm_gate_kernel[grid](
        x_norm, cond_norm, Ws, Wb, scale_b, y, M, N, K,
        x_norm.stride(0), x_norm.stride(1), cond_norm.stride(0), cond_norm.stride(1),
        Ws.stride(0), Ws.stride(1), Wb.stride(0), Wb.stride(1), y.stride(0), y.stride(1),
        DT=x_norm.element_size(),
    )
    return y


# ── training: K3 variant that also stores gate=sigmoid(scale); + backward elementwise ──────────
@triton.autotune(configs=_GEMM_CFG, key=["N", "K", "DT"])
@triton.jit
def _gemm_gate_train_kernel(
    Xn, Cn, Ws, Wb, Sb, Y, Gate, M, N, K,
    sxn0, sxn1, scn0, scn1, sws0, sws1, swb0, swb1, sy0, sy1, sg0, sg1,
    DT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    scale = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        kmask = kk < K
        a = tl.load(Cn + rm[:, None] * scn0 + kk[None, :] * scn1,
                    mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        ws = tl.load(Ws + rn[None, :] * sws0 + kk[:, None] * sws1,
                     mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        wb = tl.load(Wb + rn[None, :] * swb0 + kk[:, None] * swb1,
                     mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        scale += tl.dot(a, ws, input_precision="tf32", out_dtype=tl.float32)
        bias += tl.dot(a, wb, input_precision="tf32", out_dtype=tl.float32)
    sb = tl.load(Sb + rn, mask=rn < N, other=0.0).to(tl.float32)
    scale += sb[None, :]
    gate = tl.sigmoid(scale)
    xn = tl.load(Xn + rm[:, None] * sxn0 + rn[None, :] * sxn1,
                 mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0).to(tl.float32)
    y = gate * xn + bias
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y + rm[:, None] * sy0 + rn[None, :] * sy1, y.to(Y.dtype.element_ty), mask=om)
    tl.store(Gate + rm[:, None] * sg0 + rn[None, :] * sg1, gate.to(Gate.dtype.element_ty), mask=om)


def _gemm_gate_train(x_norm, cond_norm, Ws, Wb, scale_b):
    M, N = x_norm.shape
    K = cond_norm.shape[1]
    y = torch.empty_like(x_norm)
    gate = torch.empty_like(x_norm)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
    _gemm_gate_train_kernel[grid](
        x_norm, cond_norm, Ws, Wb, scale_b, y, gate, M, N, K,
        x_norm.stride(0), x_norm.stride(1), cond_norm.stride(0), cond_norm.stride(1),
        Ws.stride(0), Ws.stride(1), Wb.stride(0), Wb.stride(1),
        y.stride(0), y.stride(1), gate.stride(0), gate.stride(1), DT=x_norm.element_size())
    return y, gate


@triton.autotune(configs=_LN_CFG, key=["N", "DT"])
@triton.jit
def _bwd_elem_kernel(DY, Xn, Gate, Dscale, Dxn, M, N,
                     sy0, sy1, sxn0, sxn1, sg0, sg1, sds0, sds1, sdx0, sdx1,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    rm = row * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rm < M
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    dy = tl.load(DY + rm[:, None] * sy0 + cols[None, :] * sy1, mask=mask, other=0.0).to(tl.float32)
    xn = tl.load(Xn + rm[:, None] * sxn0 + cols[None, :] * sxn1, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate + rm[:, None] * sg0 + cols[None, :] * sg1, mask=mask, other=0.0).to(tl.float32)
    dscale = dy * xn * g * (1.0 - g)
    dxn = dy * g
    tl.store(Dscale + rm[:, None] * sds0 + cols[None, :] * sds1, dscale.to(Dscale.dtype.element_ty), mask=mask)
    tl.store(Dxn + rm[:, None] * sdx0 + cols[None, :] * sdx1, dxn.to(Dxn.dtype.element_ty), mask=mask)


def _bwd_elem(dy, x_norm, gate):
    M, N = dy.shape
    dscale = torch.empty_like(dy)
    dxn = torch.empty_like(dy)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa: E731
    _bwd_elem_kernel[grid](dy, x_norm, gate, dscale, dxn, M, N,
                           dy.stride(0), dy.stride(1), x_norm.stride(0), x_norm.stride(1),
                           gate.stride(0), gate.stride(1), dscale.stride(0), dscale.stride(1),
                           dxn.stride(0), dxn.stride(1),
                           BLOCK_N=triton.next_power_of_2(N), DT=dy.element_size())
    return dscale, dxn


class _Fused3TrainFn(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, x, cond, lnw, Ws, sb, Wb, eps_x, eps_cond):
        from ...layernorm_linear.te_style import _ln_materialize
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
    @torch.compiler.disable()
    def backward(ctx, dy):
        from ...layernorm_linear.te_style import _ln_bwd, _fp32_matmul_ctx
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
