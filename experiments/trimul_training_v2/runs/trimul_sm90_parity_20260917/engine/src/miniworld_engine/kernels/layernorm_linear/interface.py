"""Triton entry point for the fused LayerNorm + Linear kernel.

This is the **portable** backend: the cute path forks quack's ``GemmSm90`` (WGMMA +
TMA + clusters) and is SM90/Hopper-only, so this Triton kernel is the general
fallback that runs on any Triton-supported arch (Ampere/Ada/Hopper/Blackwell/ROCm).
It computes ``LayerNorm(x) @ W^T + b`` in one fused kernel — LN stats reduced on-chip,
then the projection GEMM — so eager ``LayerNorm`` -> ``Linear``'s extra HBM round trip
is avoided. See ``triton/fused.py``.
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
    from miniworld_engine.kernels.layernorm_linear.triton.fused import (
        layernorm_linear_triton_fwd,
    )

    return layernorm_linear_triton_fwd(x, ln_weight, ln_bias, weight, bias, eps)
