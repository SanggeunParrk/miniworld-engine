"""PyTorch reference for the fused ProteinMPNN encoder edge tail."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_tail_update_pytorch(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    keep_mask: torch.Tensor | None,
    eps: float,
    dropout_probability: float,
) -> torch.Tensor:
    """Evaluate the whole encoder edge tail with ordinary PyTorch operations.

    ``keep_mask`` makes the otherwise random dropout decision an explicit input so
    a test can compare against the Triton path element by element. ``None`` keeps
    every element, which is what inference and ``dropout=0`` configurations do.
    """
    width = edge_states.shape[-1]
    neighbors = F.embedding(
        flat_neighbor_indices, neighbor_projection.reshape(-1, width)
    )
    preactivation = query_projection.unsqueeze(-2) + F.linear(edge_states, edge_weight)
    preactivation = preactivation + neighbors
    hidden = F.gelu(preactivation)
    hidden = F.gelu(F.linear(hidden, hidden_weight, hidden_bias))
    update = F.linear(hidden, output_weight, output_bias)
    if keep_mask is not None:
        scale = 1.0 / (1.0 - dropout_probability)
        update = torch.where(
            keep_mask,
            (update.float() * scale).to(update.dtype),
            torch.zeros((), device=update.device, dtype=update.dtype),
        )
    values = edge_states + update
    return F.layer_norm(
        values.float(),
        (width,),
        norm_weight.float(),
        norm_bias.float(),
        eps,
    ).to(edge_states.dtype)


__all__ = ["edge_tail_update_pytorch"]
