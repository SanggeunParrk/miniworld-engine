"""Single-launch left+right+gate front, in Triton — gate gets a PROPER sigmoid.

One kernel, reads x once, three dots:
  acc_g = x @ [WLg|WRg]   (M,2D)   gate logits for left,right
  acc_p = x @ [WL |WR ]   (M,2D)   projections
  acc_t = x @ Wg          (M, D)   the output-gate logits
  left  = sigmoid(acc_g[:,:D]) * acc_p[:,:D]   -> bdll [B,D,L,L]
  right = sigmoid(acc_g[:,D:]) * acc_p[:,D:]   -> bdll
  gate  = sigmoid(acc_t)                       -> blld [B,L,L,D]

No glu-pair constraint, no zero columns, no bias trick: the gate is a plain
unary sigmoid in the SAME launch as left/right. B=1, K=D=128, bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=nw, num_stages=ns)
        for bm in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["M"],
)
@triton.jit
def _front_kernel(
    x_ptr, bg_ptr, bp_ptr, wg_ptr,        # x:(M,K) bg,bp:(K,2D) wg:(K,D)
    lr_ptr, gate_ptr,                      # lr: bdll planes (2D, LL); gate:(M,D)
    M, LL,
    K: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rk = tl.arange(0, K)
    r2d = tl.arange(0, 2 * D)
    rd = tl.arange(0, D)
    mmask = rm[:, None] < M

    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask, other=0.0)  # (BM,K)
    bg = tl.load(bg_ptr + rk[:, None] * (2 * D) + r2d[None, :])                # (K,2D)
    bp = tl.load(bp_ptr + rk[:, None] * (2 * D) + r2d[None, :])
    wg = tl.load(wg_ptr + rk[:, None] * D + rd[None, :])                       # (K,D)

    lr = (tl.sigmoid(tl.dot(x, bg)) * tl.dot(x, bp))                           # (BM,2D)
    gate = tl.sigmoid(tl.dot(x, wg))                                          # (BM,D)

    # lr -> bdll: element (m, c) at  c*LL + m  (c in 0..2D-1 -> planes [left|right]).
    tl.store(lr_ptr + r2d[None, :] * LL + rm[:, None],
             lr.to(lr_ptr.dtype.element_ty), mask=mmask)
    # gate -> blld (M,D) contiguous
    tl.store(gate_ptr + rm[:, None] * D + rd[None, :],
             gate.to(gate_ptr.dtype.element_ty), mask=mmask)


def trimul_front_triton(x, WL, WLg, WR, WRg, Wg):
    """x:(B,L,L,D) -> (left_bdll, right_bdll, gate_blld). B=1."""
    B, L, L2, D = x.shape
    assert B == 1 and L == L2
    M, LL = L * L, L * L
    x_flat = x.reshape(M, D)
    Bg = torch.cat([WLg, WRg], dim=1).contiguous()  # (D,2D)
    Bp = torch.cat([WL, WR], dim=1).contiguous()
    lr = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)  # bdll storage
    gate = torch.empty(M, D, device=x.device, dtype=x.dtype)          # blld
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _front_kernel[grid](x_flat, Bg, Bp, Wg.contiguous(), lr, gate, M, LL, K=D, D=D)
    return lr[:, :D], lr[:, D:], gate.view(B, L, L, D)
