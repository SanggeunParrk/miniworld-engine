"""Fused trimul back-half in Triton: LN_out + proj-gemm + gate-gemm + mul, one kernel.

Computes, per pair row (gate is computed IN the back, not the front — no gate
materialization, no separate mul pass):

    proj = LayerNorm_D(tri) @ Wp           # tri: bmm output, LN over channel D
    gate = sigmoid(x_n @ Wg)               # x_n: the LN_in'd pair (front input)
    y    = gate * proj                     # [B, L, L, D]

Reads tri + x_n, writes y (3T, no intermediate materialized). tri comes in bdll
as a (D, M) contiguous view (channel-major, m strided by 1 within a plane);
x_n is (M, D) blld contiguous. B=1, K=N=D=128, bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=nw, num_stages=ns)
        for bm in (32, 64, 128)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["M"],
)
@triton.jit
def _back_kernel(
    tri_ptr,  # (D, M) channel-major: tri[k, m] at k*M + m
    xn_ptr,   # (M, D) row-major
    wp_ptr, wg_ptr,    # (D, D) = to_out.weight.T, to_gate.weight.T  (K=in, N=out)
    lnw_ptr, lnb_ptr,  # (D,)
    y_ptr,    # (M, D) row-major
    M, eps,
    K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rk = tl.arange(0, K)
    rn = tl.arange(0, N)
    mmask = rm[:, None] < M

    # tri (BM, K) — channel-major load (k strided by M)
    tri = tl.load(tri_ptr + rk[None, :] * M + rm[:, None], mask=mmask, other=0.0).to(tl.float32)
    mean = tl.sum(tri, axis=1) / K
    xc = tri - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / K
    rstd = 1.0 / tl.sqrt(var + eps)
    lnw = tl.load(lnw_ptr + rk).to(tl.float32)
    lnb = tl.load(lnb_ptr + rk).to(tl.float32)
    norm = (xc * rstd[:, None]) * lnw[None, :] + lnb[None, :]  # (BM, K) fp32

    wp = tl.load(wp_ptr + rk[:, None] * N + rn[None, :])
    wg = tl.load(wg_ptr + rk[:, None] * N + rn[None, :])
    proj = tl.dot(norm.to(wp.dtype), wp)                       # (BM, N)
    xn = tl.load(xn_ptr + rm[:, None] * K + rk[None, :], mask=mmask, other=0.0)
    gate = tl.sigmoid(tl.dot(xn, wg))                          # (BM, N)
    y = (proj * gate).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], y, mask=mmask)


def trimul_back_triton(tri_bdll, x_n, Wp, Wg, ln_w, ln_b, eps=1e-5):
    """tri_bdll:(B,D,L,L), x_n:(B,L,L,D), Wp/Wg:(D,D)=weight.T -> y:(B,L,L,D). B=1."""
    B, D, L, L2 = tri_bdll.shape
    assert B == 1 and L == L2
    M = L * L
    tri_dm = tri_bdll.reshape(D, M)            # (D, M) contiguous, channel-major
    xn_flat = x_n.reshape(M, D)
    y = torch.empty(M, D, device=x_n.device, dtype=x_n.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _back_kernel[grid](tri_dm, xn_flat, Wp.contiguous(), Wg.contiguous(),
                       ln_w.contiguous(), ln_b.contiguous(), y, M, float(eps), K=D, N=D)
    return y.view(B, L, L, D)
