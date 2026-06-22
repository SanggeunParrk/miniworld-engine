"""Triton entry point for the fused LayerNorm + Linear kernel (placeholder).

To be implemented. The signature mirrors `layernorm_linear_pytorch` so the bench
script can drop the Triton kernel in without rewiring. The goal is a single
kernel that computes LayerNorm statistics and the projection GEMM in one pass,
cutting the HBM round-trip the eager `LayerNorm` -> `Linear` pair pays — most
promising in the memory-bound regime (small d, large M) where neither TE nor
`torch.compile` is clearly ahead (see benchmark/).
"""

from __future__ import annotations

import torch


def layernorm_linear_triton(
    x: torch.Tensor,          # (..., d_in)
    ln_weight: torch.Tensor,  # (d_in,)
    ln_bias: torch.Tensor,    # (d_in,)
    weight: torch.Tensor,     # (d_out, d_in)
    bias: torch.Tensor | None,  # (d_out,)
    eps: float = 1e-5,
) -> torch.Tensor:
    raise NotImplementedError("layernorm_linear Triton kernel not implemented yet")
