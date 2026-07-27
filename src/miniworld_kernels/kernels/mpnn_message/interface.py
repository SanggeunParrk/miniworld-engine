"""Dispatch boundary for the ProteinMPNN hidden-message reduction."""

from __future__ import annotations

from typing import Literal

import torch

from .reference import message_hidden_reduce_pytorch

MessageBackend = Literal[
    "auto",
    "pytorch",
    "triton",
    "triton_compute",
    "triton_memory",
]


def _triton_contract_supported(
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: torch.Size | tuple[int, ...],
    contiguous: bool,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
) -> bool:
    """Check the shared message-kernel contract without an allocation."""
    bf16_projection = (
        weight.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
    ) or (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and weight.dtype in {torch.float32, torch.bfloat16}
        and bias.dtype == weight.dtype
    )
    return (
        len(shape) >= 2
        and device.type == "cuda"
        and dtype == torch.bfloat16
        and all(size > 0 for size in shape)
        and contiguous
        and bf16_projection
        and weight.is_cuda
        and weight.device == device
        and weight.is_contiguous()
        and weight.shape == (128, 128)
        and bias.is_cuda
        and bias.device == device
        and bias.is_contiguous()
        and bias.shape == (128,)
        and edge_mask.is_cuda
        and edge_mask.device == device
        and edge_mask.dtype == torch.float32
        and not edge_mask.requires_grad
        and edge_mask.is_contiguous()
        and tuple(shape[-2:]) == (48, 128)
        and tuple(edge_mask.shape) == tuple(shape[:-1])
    )


def _triton_supported(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
) -> bool:
    return preactivation.ndim > 0 and _triton_contract_supported(
        device=preactivation.device,
        dtype=preactivation.dtype,
        shape=preactivation.shape,
        contiguous=preactivation.is_contiguous(),
        weight=weight,
        bias=bias,
        edge_mask=edge_mask,
    )


def _should_use_triton(supported: bool, backend: MessageBackend) -> bool:
    return supported and backend != "pytorch"


def message_hidden_reduce(
    preactivation: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_scale: int,
    *,
    backend: MessageBackend = "auto",
) -> torch.Tensor:
    """Run the fused operation when supported, otherwise retain PyTorch math."""
    if backend not in {
        "auto",
        "pytorch",
        "triton",
        "triton_compute",
        "triton_memory",
    }:
        raise ValueError(f"unknown MPNN message backend: {backend!r}")
    supported = _triton_supported(preactivation, weight, bias, edge_mask)
    if backend in {"triton", "triton_compute", "triton_memory"} and not supported:
        raise ValueError(
            "the Triton MPNN message kernel requires BF16 projection math "
            "(normally CUDA BF16 autocast), contiguous CUDA BF16 input "
            "[*, 48, 128], weight [128, 128], bias [128], and a contiguous "
            "non-differentiable FP32 mask [*, 48]"
        )
    # Inference uses a no-save full fusion. Grad mode keeps the same two-kernel
    # forward for both policies; the memory policy discards and later
    # recomputes its projected activation. The compatibility name ``triton``
    # denotes the compute policy.
    if _should_use_triton(supported, backend):
        if not torch.is_grad_enabled():
            from .triton.inference import (
                _int32_offsets_supported,
                triton_message_hidden_reduce_inference,
            )

            if _int32_offsets_supported(preactivation):
                return triton_message_hidden_reduce_inference(
                    preactivation,
                    weight,
                    bias,
                    edge_mask,
                    neighbor_scale,
                )
        if backend == "triton_memory":
            from .triton import triton_message_hidden_reduce_memory

            return triton_message_hidden_reduce_memory(
                preactivation,
                weight,
                bias,
                edge_mask,
                neighbor_scale,
            )
        from .triton import triton_message_hidden_reduce

        return triton_message_hidden_reduce(
            preactivation,
            weight,
            bias,
            edge_mask,
            neighbor_scale,
        )
    return message_hidden_reduce_pytorch(
        preactivation,
        weight,
        bias,
        edge_mask,
        neighbor_scale,
    )


__all__ = ["MessageBackend", "message_hidden_reduce"]
