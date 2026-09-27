"""PyTorch reference for the fused ProteinMPNN encoder node message."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def node_message_reduce_pytorch(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
) -> torch.Tensor:
    """Evaluate the packed W1 block, both GELUs, W2 and the masked reduction."""
    width = edge_states.shape[-1]
    neighbors = F.embedding(
        flat_neighbor_indices, neighbor_projection.reshape(-1, width)
    )
    preactivation = query_projection.unsqueeze(-2) + F.linear(edge_states, edge_weight)
    preactivation = preactivation + neighbors
    hidden = F.gelu(F.linear(F.gelu(preactivation), hidden_weight, hidden_bias))
    masked = hidden.float() * edge_mask.float().unsqueeze(-1)
    return masked.sum(dim=-2) / neighbor_scale


__all__ = ["node_message_reduce_pytorch"]
