"""Dispatch boundary for the fused ProteinMPNN encoder edge tail."""

from __future__ import annotations

from typing import Literal

import torch

# There is no ``auto``: the fused path changes the residual dtype to BF16 and draws
# dropout from a different Philox stream, so it is reached only from an explicit
# policy.  ``off`` keeps the existing separate-operation encoder edge update.
# ``triton`` chains the projections in one kernel and replays them in backward, which is
# right when the edge tensor is the binding constraint. ``triton_compute`` saves the GEMM
# outputs instead. Both take the edge weight as a slice of the packed input projection,
# so both read its row stride rather than assuming one.
EdgeTailBackend = Literal["off", "triton", "triton_compute"]

_WIDTH = 128
# Backward regenerates the dropout mask from a Philox counter keyed by the flat
# element index, so the whole edge tensor has to be addressable in INT32.
_INT32_MAX = 2**31 - 1


def edge_tail_supported(
    edge_states: torch.Tensor,
    query_projection: torch.Tensor,
    neighbor_projection: torch.Tensor,
    flat_neighbor_indices: torch.Tensor,
    edge_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
) -> bool:
    """Check the whole fused contract without allocating anything."""
    if hidden_bias is None or output_bias is None:
        return False
    if norm_weight is None or norm_bias is None:
        return False
    if edge_states.ndim != 4 or query_projection.ndim != 3:
        return False
    if neighbor_projection.ndim != 3 or flat_neighbor_indices.ndim != 3:
        return False
    activations = (edge_states, query_projection, neighbor_projection)
    parameters = (edge_weight, hidden_weight, hidden_bias, output_weight, output_bias)
    normalization = (norm_weight, norm_bias)
    tensors = (*activations, flat_neighbor_indices, *parameters, *normalization)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if not all(tensor.device == edge_states.device for tensor in tensors):
        return False
    if not all(tensor.dtype == torch.bfloat16 for tensor in activations):
        return False
    # Projection parameters must agree; normalization independently uses FP32
    # loads/accumulation and supports FP32 affine with native BF16 projections.
    parameter_dtypes = {tensor.dtype for tensor in parameters}
    if len({tensor.dtype for tensor in normalization}) != 1:
        return False
    if norm_weight.dtype not in {torch.float32, torch.bfloat16}:
        return False
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
        and query_projection.shape[-1] == _WIDTH
        and neighbor_projection.shape[-1] == _WIDTH
        and flat_neighbor_indices.shape == edge_states.shape[:-1]
        and query_projection.shape[:2] == edge_states.shape[:2]
        and flat_neighbor_indices.dtype == torch.long
        and edge_states.is_contiguous()
        and query_projection.is_contiguous()
        and neighbor_projection.is_contiguous()
        and flat_neighbor_indices.is_contiguous()
        and edge_weight.shape == (_WIDTH, _WIDTH)
        and edge_weight.stride(1) == 1
        and hidden_weight.shape == (_WIDTH, _WIDTH)
        and hidden_bias.shape == (_WIDTH,)
        and output_weight.shape == (_WIDTH, _WIDTH)
        and output_bias.shape == (_WIDTH,)
        and norm_weight.shape == (_WIDTH,)
        and norm_bias.shape == (_WIDTH,)
        and hidden_weight.is_contiguous()
        and hidden_bias.is_contiguous()
        and output_weight.is_contiguous()
        and output_bias.is_contiguous()
        and norm_weight.is_contiguous()
        and norm_bias.is_contiguous()
    )


def edge_tail_update(
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
    seed: torch.Tensor,
    eps: float,
    dropout_probability: float,
    backend: EdgeTailBackend = "triton",
) -> torch.Tensor:
    """Run the whole encoder edge tail through the fused Triton kernel."""
    if not edge_tail_supported(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
    ):
        raise ValueError(
            "the Triton MPNN edge tail requires contiguous CUDA BF16 "
            "[B, T, K, 128] edge states, [B, T, 128] node projections, a "
            "contiguous INT64 neighbor index of matching shape, row-major "
            "[128, 128]/[128] projection parameters all BF16 or all FP32 under BF16 "
            "autocast, matching BF16 or FP32 norm affine parameters, and an edge "
            "tensor addressable in signed 32-bit indexing"
        )
    if backend == "triton_compute":
        # The compute path is written against a flat [rows, 128] view; the contract
        # checked above already guarantees the activations are contiguous, so these are
        # views. `edge_weight` is deliberately NOT among them -- it arrives as a slice of
        # the packed projection and the kernel takes its row stride.
        width = edge_states.shape[-1]
        from miniworld_engine.kernels.mpnn_edge_tail.triton.compute import (
            edge_tail_compute,
        )

        return edge_tail_compute(
            edge_states.reshape(-1, width),
            query_projection.reshape(-1, width),
            neighbor_projection.reshape(-1, width),
            flat_neighbor_indices.reshape(-1),
            edge_weight,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            norm_weight,
            norm_bias,
            eps,
            dropout_probability,
            seed,
        ).reshape(edge_states.shape)

    # Keep Triton out of CPU and import-only users; this line is reached only by
    # supported CUDA training tensors.
    from miniworld_engine.kernels.mpnn_edge_tail.triton import triton_edge_tail_update

    return triton_edge_tail_update(
        edge_states,
        query_projection,
        neighbor_projection,
        flat_neighbor_indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
        seed,
        eps,
        dropout_probability,
    )


__all__ = ["EdgeTailBackend", "edge_tail_supported", "edge_tail_update"]
