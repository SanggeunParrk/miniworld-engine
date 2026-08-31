"""PyTorch reference for the ProteinMPNN edge-message hidden MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_mlp_update_pytorch(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the two exact-GELU projections without residual normalization."""
    hidden = F.gelu(preactivation)
    hidden = F.gelu(F.linear(hidden, hidden_weight, hidden_bias))
    return F.linear(hidden, output_weight, output_bias)


__all__ = ["edge_mlp_update_pytorch"]
