"""Low-register forward LayerNorm variant (forward-only, bench experiment).

Motivation (Option ②, d=384/768 non-pow2 tuning):
The shipped `layer_norm_fwd_fused` loads each row into fp32 and keeps the fp32
``x - mean`` tile live across the variance + normalize steps. At a non-pow2 N the
column tile is padded to ``next_pow2(N)`` (768 -> 1024, 384 -> 512), so the live
fp32 tile is BLOCK_M x 1024 fp32 -> register spills that cap BLOCK_M / occupancy.

Chunking the columns is NOT an option: LayerNorm forward is HBM-bandwidth bound
and re-reading X per chunk would *double* the dominant traffic. The only
bandwidth-neutral lever is register footprint. This variant keeps the row tile in
**bf16** (half the registers) and runs the standard stable two-pass mean/var off
that bf16 copy — X is still read from HBM exactly once, but the live tile is half
the size, so a larger BLOCK_M (higher occupancy) becomes feasible.

Forward-only (no autograd); this is a bench probe to test the occupancy
hypothesis, not a shipped path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_kernels.autotune import key_bucket_of, make_cache_prune, tensor_dtype_of
from .main import get_seq_group

# Same autotune grid as the shipped fwd, so the comparison is config-fair: any
# win comes from the lower register footprint, not a different search space.
_lowreg_configs = [
    triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
    for block_m in [1, 2, 4, 8, 16, 32, 64]
    for num_warps in [4, 8, 16]
    for num_stages in [2, 3, 4, 5]
]


# fmt: off
_layernorm_lowreg_fwd_prune = make_cache_prune(
    "layernorm_lowreg_fwd", dtype_of=tensor_dtype_of("X"),
    bucket_of=key_bucket_of("N", "GROUP_M"),
)


@triton.autotune(configs=_lowreg_configs, key=["N", "GROUP_M"],
                 prune_configs_by={"early_config_prune": _layernorm_lowreg_fwd_prune})
@triton.jit
def layer_norm_fwd_lowreg(
    X, Y, W, B, Mean, Rstd,
    stride_r, stride_c,
    M, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset_row = tl.arange(0, BLOCK_M) + row * BLOCK_M
    row_mask = (offset_row < M)[:, None]
    offset_col = tl.arange(0, BLOCK_N)
    col_mask = (offset_col < N)[None, :]
    mask = row_mask & col_mask

    p = (offset_row[:, None] * stride_r) + (offset_col[None, :] * stride_c)
    # Keep the row tile in bf16 (half the registers vs a live fp32 tile). X is
    # still read from HBM exactly once; both reduction passes reuse this copy.
    xb = tl.load(X + p, mask=mask, other=0.0)

    xf = xb.to(tl.float32)
    mean = tl.sum(xf, axis=1) / N
    xc = tl.where(mask, xf - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)

    moff = tl.arange(0, BLOCK_M) + row * BLOCK_M
    mmask = moff < M
    tl.store(Mean + moff, mean, mask=mmask)
    tl.store(Rstd + moff, rstd, mask=mmask)

    w = tl.load(W + offset_col, mask=offset_col < N)
    b = tl.load(B + offset_col, mask=offset_col < N)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + p, y, mask=mask)
# fmt: on


@torch.compiler.disable()
def triton_layernorm_lowreg(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward-only low-register LayerNorm (bench probe)."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)
    m, n = x_2d.shape
    mean = torch.empty(m, dtype=torch.float32, device=x.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x.device)
    grid = lambda META: [triton.cdiv(m, META["BLOCK_M"])]
    layer_norm_fwd_lowreg[grid](
        x_2d, y_2d, weight, bias, mean, rstd,
        x_2d.stride(0), x_2d.stride(1),
        m, n, eps,
        BLOCK_N=triton.next_power_of_2(n),
        GROUP_M=get_seq_group(m),
    )
    return y_2d.view_as(x)
