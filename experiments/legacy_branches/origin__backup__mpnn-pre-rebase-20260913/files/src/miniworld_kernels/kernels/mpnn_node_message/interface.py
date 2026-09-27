"""Dispatch boundary for the fused ProteinMPNN encoder node message."""

from __future__ import annotations

from typing import Literal

import torch


# There is no ``auto``: the fused path replays the whole message in backward and
# reduces in a different order from the two-kernel form, so it is reached only from
# an explicit policy.  ``off`` keeps the existing separate-operation node message.
NodeMessageBackend = Literal["off", "triton"]

_WIDTH = 128
_INT32_MAX = 2**31 - 1
# One program holds a whole neighbour group in registers, so K has to fit a single
# power-of-two row tile that tl.dot can still contract efficiently.
_MAX_NEIGHBORS = 128


def node_message_supported(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
) -> bool:
    """Check the fused node-message contract without allocating anything."""
    if hidden_bias is None or edge_mask is None:
        return False
    if edge_states.ndim != 4 or query_projection.ndim != 3:
        return False
    if neighbor_projection.ndim != 3 or flat_neighbor_indices.ndim != 3:
        return False
    activations = (edge_states, query_projection, neighbor_projection)
    parameters = (edge_weight, hidden_weight, hidden_bias)
    tensors = (*activations, flat_neighbor_indices, edge_mask, *parameters)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == edge_states.device for tensor in tensors):
        return False
    if not all(tensor.dtype == torch.bfloat16 for tensor in activations):
        return False
    parameter_dtypes = {tensor.dtype for tensor in parameters}
    autocast_bf16 = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    if parameter_dtypes == {torch.float32}:
        if not autocast_bf16:
            return False
    elif parameter_dtypes != {torch.bfloat16}:
        return False
    return (
        edge_states.numel() > 0
        and edge_states.numel() <= _INT32_MAX
        and edge_states.shape[-1] == _WIDTH
        and edge_states.shape[-2] <= _MAX_NEIGHBORS
        and query_projection.shape[-1] == _WIDTH
        and neighbor_projection.shape[-1] == _WIDTH
        and flat_neighbor_indices.shape == edge_states.shape[:-1]
        and edge_mask.shape == edge_states.shape[:-1]
        and query_projection.shape[:2] == edge_states.shape[:2]
        and flat_neighbor_indices.dtype == torch.long
        and edge_mask.dtype in {torch.float32, torch.bfloat16}
        and edge_states.is_contiguous()
        and query_projection.is_contiguous()
        and neighbor_projection.is_contiguous()
        and flat_neighbor_indices.is_contiguous()
        and edge_mask.is_contiguous()
        and edge_weight.shape == (_WIDTH, _WIDTH)
        and edge_weight.stride(1) == 1
        and hidden_weight.shape == (_WIDTH, _WIDTH)
        and hidden_bias.shape == (_WIDTH,)
        and hidden_weight.is_contiguous()
        and hidden_bias.is_contiguous()
    )


def node_message_reduce(
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
    """Run the whole node message through the fused Triton kernel."""
    if not node_message_supported(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
    ):
        raise ValueError(
            "the Triton MPNN node message requires contiguous CUDA BF16 "
            "[B, T, K, 128] edge states with K <= 128, [B, T, 128] node "
            "projections, contiguous INT64 indices and a float edge mask of "
            "matching shape, row-major [128, 128]/[128] parameters that are all "
            "BF16 or all FP32 under BF16 autocast, and an edge tensor addressable "
            "in signed 32-bit indexing"
        )
    from .triton import triton_node_message_reduce

    return triton_node_message_reduce(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        edge_mask,
        neighbor_scale,
    )


__all__ = ["NodeMessageBackend", "node_message_supported", "node_message_reduce"]
