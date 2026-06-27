"""Fused sigmoid-gate + output projection.

Baseline does three kernels:
    gate = to_gate(pln)                 # cuBLAS GEMM   [M, dh]
    gated = sigmoid(gate) * out_r       # elementwise   [M, dh]   (~0.4ms @ L1024)
    out = gated @ Wo^T                   # cuBLAS GEMM   [M, d_pair]

This fuses the elementwise + the to_out GEMM into one triton kernel: the GEMM's
A-tile is computed in the prologue as sigmoid(gate)*out_r, so `gated` never
touches HBM and the standalone elementwise kernel disappears. `to_gate` stays on
cuBLAS (it's the bigger GEMM -- don't risk it).

Forward is the fused triton kernel; backward is plain torch (cuBLAS) using the
saved gate/out_r, so it matches the baseline bwd exactly (no regression) while the
forward gets the fusion win. Wrapped as an autograd.Function.

M = B*L*L (rows), dh = d_hidden (contraction), N = d_pair (output width).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_configs = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bm in [64, 128]
    for bn in [64, 128]
    for bk in [32, 64]
    for w in [4, 8]
    for s in [3, 4]
]


@triton.autotune(configs=_configs, key=["M", "N", "DH"])
@triton.jit
def _gate_out_fwd(
    gate_ptr,   # [M, DH]
    outr_ptr,   # [M, DH]
    wo_ptr,     # [N, DH]   (to_out.weight: out_features=N, in_features=DH)
    o_ptr,      # [M, N]
    M, N,
    DH: tl.constexpr,
    stride_gm, stride_gd,
    stride_om, stride_od,
    stride_wn, stride_wd,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    n_ok = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, DH, BLOCK_K):
        kk = k0 + offs_k
        k_ok = kk < DH
        # A-tile = sigmoid(gate) * out_r  -> [BLOCK_M, BLOCK_K]
        g = tl.load(
            gate_ptr + offs_m[:, None] * stride_gm + kk[None, :] * stride_gd,
            mask=m_ok[:, None] & k_ok[None, :], other=0.0,
        ).to(tl.float32)
        r = tl.load(
            outr_ptr + offs_m[:, None] * stride_om + kk[None, :] * stride_od,
            mask=m_ok[:, None] & k_ok[None, :], other=0.0,
        ).to(tl.float32)
        a = (tl.sigmoid(g) * r).to(wo_ptr.dtype.element_ty)
        # Wo-tile [BLOCK_K, BLOCK_N]: wo[n, k] -> transpose for the dot
        wo = tl.load(
            wo_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wd,
            mask=n_ok[None, :] & k_ok[:, None], other=0.0,
        )
        acc = tl.dot(a, wo, acc)

    tl.store(
        o_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_ok[:, None] & n_ok[None, :],
    )


@triton.jit
def _gate_bwd_elem(
    da_ptr,     # [M, DH]  = grad_out @ wo
    g_ptr,      # [M, DH]  = gate (pre-sigmoid)
    r_ptr,      # [M, DH]  = out_r
    dr_ptr,     # out: d_out_r
    dg_ptr,     # out: d_gate
    a_ptr,      # out: gated = sigmoid(gate) * r  (for the d_wo GEMM)
    n_elem,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elem
    da = tl.load(da_ptr + offs, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    a = s * r
    dr = s * da
    dg = da * r * s * (1.0 - s)
    tl.store(dr_ptr + offs, dr.to(dr_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + offs, dg.to(dg_ptr.dtype.element_ty), mask=m)
    tl.store(a_ptr + offs, a.to(a_ptr.dtype.element_ty), mask=m)


def _bwd_elem(da, g, r):
    """Single fused pass: (da, gate, out_r) -> (d_out_r, d_gate, gated)."""
    n = da.numel()
    dr = torch.empty_like(da)
    dg = torch.empty_like(da)
    a = torch.empty_like(da)
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]),)
    _gate_bwd_elem[grid](da, g, r, dr, dg, a, n, BLOCK=1024)
    return dr, dg, a


def _fwd(gate2d, outr2d, wo):
    M, DH = gate2d.shape
    N = wo.shape[0]
    out = torch.empty((M, N), device=gate2d.device, dtype=gate2d.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _gate_out_fwd[grid](
        gate2d, outr2d, wo, out,
        M, N, DH,
        gate2d.stride(0), gate2d.stride(1),
        outr2d.stride(0), outr2d.stride(1),
        wo.stride(0), wo.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


class _FusedGateOut(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, gate, outr, wo):
        # gate, outr: [..., DH]; wo: [N, DH]
        shape = gate.shape
        DH = shape[-1]
        g2 = gate.reshape(-1, DH).contiguous()
        r2 = outr.reshape(-1, DH).contiguous()
        out2 = _fwd(g2, r2, wo.contiguous())
        ctx.save_for_backward(g2, r2, wo)
        ctx.shape = shape
        ctx.N = wo.shape[0]
        return out2.reshape(*shape[:-1], wo.shape[0])

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_out):
        g2, r2, wo = ctx.saved_tensors
        N = ctx.N
        do2 = grad_out.reshape(-1, N)
        # out = a @ wo^T,  a = sigmoid(gate) * out_r
        d_a = do2 @ wo                              # GEMM  [M, DH]
        d_r, d_g, a = _bwd_elem(d_a, g2, r2)        # one fused elementwise pass
        d_wo = do2.transpose(0, 1) @ a              # GEMM  [N, DH]
        return (
            d_g.reshape(ctx.shape),
            d_r.reshape(ctx.shape),
            d_wo,
        )


def fused_gate_out(gate: torch.Tensor, out_r: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out_r, then @ wo^T. gate/out_r [...,DH], wo [N,DH] -> [...,N]."""
    return _FusedGateOut.apply(gate, out_r, wo)
