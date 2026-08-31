"""Teacher-forcing masks for parallel MPNN sequence scoring."""

from __future__ import annotations

import torch


def build_decoding_masks(
    neighbor_indices: torch.Tensor,
    residue_mask: torch.Tensor,
    decoding_order: torch.Tensor,
    patch_index: torch.Tensor,
    fixed_decoding_order_length: int | torch.Tensor = 0,
    edge_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return future/backbone and past/sequence masks for each graph edge."""
    batch, length = decoding_order.shape
    decode_rank = torch.empty_like(decoding_order)
    steps = torch.arange(length, device=decoding_order.device).expand(batch, -1)
    decode_rank.scatter_(1, decoding_order, steps)
    decode_group = torch.gather(patch_index, 1, decode_rank)
    neighbor_group = torch.gather(
        decode_group.unsqueeze(1).expand(-1, length, -1),
        2,
        neighbor_indices,
    )
    past = (decode_group.unsqueeze(2) > neighbor_group).unsqueeze(-1)
    if isinstance(fixed_decoding_order_length, torch.Tensor):
        fixed_lengths = fixed_decoding_order_length.to(
            device=decode_rank.device, dtype=decode_rank.dtype
        )
        if fixed_lengths.ndim == 0:
            fixed_lengths = fixed_lengths.expand(batch)
        if fixed_lengths.shape != (batch,):
            raise ValueError(
                "fixed_decoding_order_length must be a scalar or have shape [batch]"
            )
        motif = decode_rank < fixed_lengths.unsqueeze(1)
        has_fixed_positions = True
    else:
        motif = decode_rank < fixed_decoding_order_length
        has_fixed_positions = fixed_decoding_order_length != 0

    if has_fixed_positions:
        motif_neighbor = torch.gather(
            motif.unsqueeze(1).expand(-1, length, -1), 2, neighbor_indices
        )
        past = past | motif_neighbor.unsqueeze(-1)
        query_index = torch.arange(length, device=neighbor_indices.device).view(
            1, length, 1
        )
        motif_self_edge = (neighbor_indices == query_index) & motif.unsqueeze(-1)
        past = past & ~motif_self_edge.unsqueeze(-1)

    if edge_mask is None:
        valid_edge = residue_mask.reshape(-1, length, 1, 1)
    else:
        if edge_mask.shape != neighbor_indices.shape:
            raise ValueError("edge_mask must have the same shape as neighbor_indices")
        valid_edge = edge_mask.unsqueeze(-1)
    past_mask = valid_edge * past
    future_mask = valid_edge * (~past)
    return future_mask, past_mask
