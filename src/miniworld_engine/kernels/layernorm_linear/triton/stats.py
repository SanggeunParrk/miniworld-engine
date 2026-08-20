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

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of



# Both axes are tuned tiles. The K axis used to arrive as BLOCK_K=next_power_of_2(K) — a
# whole-row constant the tuner never saw — which is also why BLOCK_M1 had to stay at 1-8: a
# [4, 512] tile was already the register budget. With K tiled, the M tile can take the
# canonical (>=16) sizes and the two axes trade off against each other properly.
# ``key`` stays ["K"] — the tiling change is no reason to widen it. K is the feature width and
# takes a handful of values; M is the row count and is effectively continuous, so keying on it
# would mint a fresh autotune (a full-grid sweep on a cache miss) for every distinct M the model
# runs, which is a per-shape stall, not a better config.
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("layernorm_stats_triton"), key=['K', 'GROUP_M'])
@triton.jit
def _stats_kernel(
    x_ptr, rstd_ptr, c1_ptr, M, K, eps,
    stride_xm, stride_xk,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    row_mask = rows < M
    # sum / sum-of-squares are associative, so tiling the reduction axis is exact: accumulate
    # both across K tiles in fp32, then finish with the same mean/var/rstd algebra as before.
    s = tl.zeros([BLOCK_M1], dtype=tl.float32)
    ss = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        col_mask = cols < K
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk,
            mask=row_mask[:, None] & col_mask[None, :], other=0.0,
        ).to(tl.float32)
        s += tl.sum(x, axis=1)
        ss += tl.sum(x * x, axis=1)
    inv_k = 1.0 / K
    mean = s * inv_k
    var = ss * inv_k - mean * mean
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)
    tl.store(c1_ptr + rows, mean * rstd, mask=row_mask)


@opaque()
def stats_triton(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rstd[m], c1[m]=mean*rstd over the last dim of X (M, K). Both fp32 [M].

    ``@opaque()``: this is the only autotuned Triton kernel
    (``@triton.autotune`` + ``early_config_prune``) reached on a compile-traced
    path — the transition/attention autotuned kernels are already disabled at
    their autograd.Function level. Without this guard, torch.compile/dynamo tries
    to hopify the autotuner's ``early_config_prune`` closure and fails with
    "Can't construct an AttrSource without a valid base source". Graph-breaking
    here (eager launch) is captured normally by the manual full-model CUDA graph.
    """
    assert x.dim() == 2 and x.is_cuda
    M, K = x.shape
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    c1 = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)  # noqa: E731
    _stats_kernel[grid](
        x, rstd, c1, M, K, eps,
        x.stride(0), x.stride(1),
        GROUP_M=get_seq_group(M),
    )
    return rstd, c1
