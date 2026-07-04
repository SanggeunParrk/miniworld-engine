"""Fused LayerNorm + per-row mask multiply (Triton kernel).

Replaces

    x_normed = cuequiv_ln(x)             # ~0.20 ms at L=1024, D=128
    x_normed = x_normed * mask_2d        # ~0.35 ms

with one HBM pass:

    fused_ln_mask(x, weight, bias, mask) # B200: ~0.090 ms at L=1024

``x`` is (B, L, L, D); ``mask`` is a 2D pair mask shaped (B, L, L) (one scalar
per output row). Math:

    row_norm = LayerNorm(x[b, r, c, :], over D axis; weight, bias, eps)
    out[b, r, c, :] = row_norm * mask[b, r, c]

The kernel is bandwidth-bound (read x + write out, ~512 MB at L=1024). On B200
the tile is autotuned; the winning shape across L is BLOCK_M=8 with a single
warp (i.e. ~8 rows / warp, two 128-bit bf16 loads per row), which saturates
HBM at ~6 TB/s. Configs are kept and the reduction stays in fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    # BLOCK_M / num_warps pairs that keep ~8 rows per warp do best on B200;
    # cover the neighbourhood so autotune adapts to other GPUs / shapes too.
    for block_m in (1, 2, 4, 8, 16, 32):
        for num_warps in (1, 2, 4, 8):
            cfgs.append(triton.Config({"BLOCK_M": block_m}, num_warps=num_warps))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "D"])
@triton.jit
def _fused_ln_mask_kernel(
    x_ptr,
    mask_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_d = tl.arange(0, D)

    # Load x (BLOCK_M, D) → fp32. D=128 bf16 is a contiguous 256-byte row, so
    # each row is two coalesced 128-bit loads.
    x_ptrs = x_ptr + offs_m[:, None] * D + offs_d[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # LayerNorm over D (fp32 reduction)
    mean = tl.sum(x, axis=1) / D
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(w_ptr + offs_d).to(tl.float32)
    b = tl.load(b_ptr + offs_d).to(tl.float32)
    x_norm = diff * rstd[:, None] * w[None, :] + b[None, :]

    # Per-row mask
    mvals = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    out = x_norm * mvals[:, None]

    tl.store(
        out_ptr + offs_m[:, None] * D + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def fused_ln_mask(
    x: torch.Tensor,  # (B, L, L, D), bf16/fp16
    weight: torch.Tensor,  # (D,)
    bias: torch.Tensor,  # (D,)
    mask: torch.Tensor,  # (B, L, L) bool / float
    eps: float = 1e-5,
) -> torch.Tensor:
    """LN(x) over last axis then multiply by per-row mask. One HBM pass."""
    assert x.dim() == 4 and x.is_cuda
    B, L1, L2, D = x.shape
    M = B * L1 * L2
    x_flat = x.reshape(M, D)
    mask_flat = mask.reshape(M).to(x.dtype).contiguous()
    out = torch.empty_like(x_flat)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _fused_ln_mask_kernel[grid](
        x_flat,
        mask_flat,
        weight,
        bias,
        out,
        M,
        D,
        EPS=eps,
    )
    return out.view(B, L1, L2, D)
