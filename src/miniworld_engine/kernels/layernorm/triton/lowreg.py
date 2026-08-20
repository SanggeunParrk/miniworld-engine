"""Low-register forward LayerNorm variant (forward-only, bench experiment).

Motivation (Option ②, d=384/768 non-pow2 tuning):
The shipped `layer_norm_fwd_fused` loads each row into fp32 and keeps the fp32
``x - mean`` tile live across the variance + normalize steps. At a non-pow2 N the
column tile is padded to ``next_pow2(N)`` (768 -> 1024, 384 -> 512), so the live
fp32 tile is BLOCK_M1 x 1024 fp32 -> register spills that cap BLOCK_M1 / occupancy.

Chunking the columns is NOT an option: LayerNorm forward is HBM-bandwidth bound
and re-reading X per chunk would *double* the dominant traffic. The only
bandwidth-neutral lever is register footprint. This variant keeps the row tile in
**bf16** (half the registers) and runs the standard stable two-pass mean/var off
that bf16 copy — X is still read from HBM exactly once, but the live tile is half
the size, so a larger BLOCK_M1 (higher occupancy) becomes feasible.

Forward-only (no autograd); this is a bench probe to test the occupancy
hypothesis, not a shipped path.
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
# The kernel that used to live here was bitwise identical to main.py's forward -- same Y,
# Mean and Rstd, verified by replaying one launch's arguments into both (.bench/eq_*.out).
# Its "low register" claim was not in the code: the body matched main.py's statement for
# statement with HAS_ROWSCALE=False. Only the launcher below survives, as a bench probe.
from .main import get_seq_group, layer_norm_fwd_fused

import torch
import triton


# fmt: off


@opaque()
def triton_layernorm_lowreg(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Forward-only low-register LayerNorm (bench probe)."""
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    y_2d = torch.empty_like(x_2d)
    m, n = x_2d.shape
    mean = torch.empty(m, dtype=torch.float32, device=x.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x.device)
    grid = lambda META: [triton.cdiv(m, META["BLOCK_M1"])]
    layer_norm_fwd_fused[grid](
        x_2d, y_2d, weight, bias, mean, rstd,
        rstd,                       # Rowscale is unread when HAS_ROWSCALE=False; pass any [M] ptr
        x_2d.stride(0), x_2d.stride(1),
        m, n, eps,
        GROUP_M=get_seq_group(m), HAS_ROWSCALE=False,
    )
    return y_2d.view_as(x)
