"""Hand-written Triton LayerNorm-stats kernel for the M1 LayerNormLinear path.

Computes, per row m of X (M, K):
    rstd[m] = rsqrt(var + eps),   c1[m] = mean * rstd      (mean, var over the K axis)

in a SINGLE bandwidth-bound pass that reads X (bf16) once and accumulates sum/sum-of-
squares in fp32. This replaces the ``torch.compile`` ``_stats`` (which ran at ~60-75% of
HBM and recompiled per shape): the M1 decomposition showed the stats pass is 25-40% of M1
and the only real headroom in the two-kernel path (the rest — hiding stats under the GEMM
— is the fused M2 design). ``c1 = mean*rstd`` is exactly what the folded GEMM epilogue
consumes (Y = rstd*acc - c1*S + B2), so no extra math downstream.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (1, 2, 4, 8, 16)
        for nw in (2, 4, 8)
    ],
    key=["K"],
)
@triton.jit
def _stats_kernel(
    x_ptr, rstd_ptr, c1_ptr, M, K, eps,
    stride_xm, stride_xk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk,
        mask=row_mask[:, None] & col_mask[None, :], other=0.0,
    ).to(tl.float32)
    s = tl.sum(x, axis=1)
    ss = tl.sum(x * x, axis=1)
    inv_k = 1.0 / K
    mean = s * inv_k
    var = ss * inv_k - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)
    tl.store(c1_ptr + rows, mean * rstd, mask=row_mask)


def stats_triton(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rstd[m], c1[m]=mean*rstd over the last dim of X (M, K). Both fp32 [M]."""
    assert x.dim() == 2 and x.is_cuda
    M, K = x.shape
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    c1 = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK_K = triton.next_power_of_2(K)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)  # noqa: E731
    _stats_kernel[grid](
        x, rstd, c1, M, K, eps,
        x.stride(0), x.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return rstd, c1
