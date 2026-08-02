"""CuTeDSL entry point for the fused triangle multiplicative update (placeholder)."""

from __future__ import annotations

import torch


def triangle_multiplicative_update_cute(
    x: torch.Tensor,
    direction: str,
    mask: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    raise NotImplementedError("trimul CuTeDSL kernel not implemented yet")
