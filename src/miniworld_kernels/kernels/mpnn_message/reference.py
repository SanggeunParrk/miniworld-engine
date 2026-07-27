"""PyTorch reference for the fused ProteinMPNN message operation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def message_hidden_reduce_pytorch(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Project edgewise messages and reduce them into their query residues.

    ``preactivation`` is the first message projection with shape
    ``[..., K, D]``. The returned tensor has shape ``[..., D]``.
    ``neighbor_scale`` deliberately remains the configured graph width rather
    than the number of valid edges, preserving ProteinMPNN's reduction rule.
    """
    hidden = F.gelu(preactivation)
    hidden = F.gelu(F.linear(hidden, weight, bias))
    hidden = hidden * edge_mask.unsqueeze(-1)
    return hidden.sum(dim=-2) / neighbor_scale


__all__ = ["message_hidden_reduce_pytorch"]
