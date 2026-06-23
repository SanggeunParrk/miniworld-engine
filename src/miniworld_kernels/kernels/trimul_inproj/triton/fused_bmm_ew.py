"""Fuse bmm-bwd + front gated-EW: batched matmul d_left=d_tri@right with a gated
epilogue that writes d_pL, d_gLlog directly (d_left never materialized).

Per channel d:  d_left[i,k] = sum_j d_tri[d,i,j] right[d,j,k]
Epilogue:       d_pL = d_left*gL ; d_gLlog = d_left*pL*gL*(1-gL)   (gL=sigmoid(gLlog))
gLlog,pL are channel-major (D,L,L). Tests whether avoiding the d_left/d_right
HBM round-trip beats cuBLAS-bmm + torch-EW. B=1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _cfgs():
    out = []
    for bm in (64, 128, 256):
        for bn in (64, 128, 256):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (3, 4):
                        out.append(triton.Config(
                            {"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s))
    return out


@triton.autotune(configs=_cfgs(), key=["L"])
@triton.jit
def _bmm_gated_kernel(
    dtri, rhs, gLlog, pL, d_p, d_glog,
    L, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    d = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    base = d * L * L
    # dtri[d, rm, :], rhs[d, :, rn]
    a_ptr = dtri + base + rm[:, None] * L + rk[None, :]      # (BM, BK)
    b_ptr = rhs + base + rk[:, None] * L + rn[None, :]       # (BK, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, L, BK):
        a = tl.load(a_ptr + k0, mask=(rm[:, None] < L) & (k0 + rk[None, :] < L), other=0.0)
        b = tl.load(b_ptr + k0 * L, mask=(k0 + rk[:, None] < L) & (rn[None, :] < L), other=0.0)
        acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    # epilogue: load gLlog, pL at (d, rm, rn)
    off = base + rm[:, None] * L + rn[None, :]
    mask = (rm[:, None] < L) & (rn[None, :] < L)
    gl = tl.load(gLlog + off, mask=mask, other=0.0).to(tl.float32)
    pl = tl.load(pL + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.sigmoid(gl)
    tl.store(d_p + off, (acc * g).to(tl.bfloat16), mask=mask)
    tl.store(d_glog + off, (acc * pl * g * (1 - g)).to(tl.bfloat16), mask=mask)


def fused_bmm_gated(dtri, rhs, gLlog, pL):
    """dtri,rhs,gLlog,pL: (D,L,L). Returns d_p, d_glog (D,L,L)."""
    D, L, _ = dtri.shape
    d_p = torch.empty_like(dtri)
    d_glog = torch.empty_like(dtri)
    grid = lambda meta: (triton.cdiv(L, meta["BM"]), triton.cdiv(L, meta["BN"]), D)  # noqa: E731
    _bmm_gated_kernel[grid](dtri, rhs, gLlog, pL, d_p, d_glog, L)
    return d_p, d_glog
