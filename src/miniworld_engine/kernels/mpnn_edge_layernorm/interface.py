"""Lazy dispatch for the ProteinMPNN encoder edge LayerNorm."""

from __future__ import annotations

from typing import Literal

import torch

from miniworld_engine.kernels.mpnn_edge_layernorm.reference import (
    edge_layer_norm_pytorch,
)

EdgeNormBackend = Literal["auto", "pytorch", "memory"]

#: The one width this policy is measured for, asserted rather than assumed. It is not a kernel
#: limit -- the LayerNorm underneath is width-generic and the launcher reads the width from its
#: argument -- it is the shape the memory judgement was made at, so a different one silently
#: inherits a trade nobody checked. Widen it by measuring, not by deleting the check.
_SUPPORTED_WIDTH = 128
_INT32_MAX = 2**31 - 1


def _memory_supported(
    values: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> bool:
    if weight is None or bias is None:
        return False
    bf16_math = (
        values.dtype == torch.bfloat16
        and weight.dtype in {torch.bfloat16, torch.float32}
        and bias.dtype == weight.dtype
    ) or (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and values.dtype in {torch.float32, torch.bfloat16}
        and weight.dtype == torch.float32
        and bias.dtype == torch.float32
    )
    return (
        torch.is_grad_enabled()
        and not torch.are_deterministic_algorithms_enabled()
        and values.is_cuda
        and values.numel() > 0
        and values.numel() <= _INT32_MAX
        and values.ndim > 0
        and values.shape[-1] == _SUPPORTED_WIDTH
        and values.is_contiguous()
        and bf16_math
        and weight.is_cuda
        and bias.is_cuda
        and weight.device == values.device
        and bias.device == values.device
        and weight.shape == (_SUPPORTED_WIDTH,)
        and bias.shape == (_SUPPORTED_WIDTH,)
        and weight.is_contiguous()
        and bias.is_contiguous()
    )


def _select_backend(
    values: torch.Tensor,
    backend: EdgeNormBackend,
    *,
    supported: bool,
) -> EdgeNormBackend:
    # Inference retains PyTorch's native LayerNorm. There is no saved input to
    # compress, so a custom autograd boundary cannot reduce its memory.
    if not torch.is_grad_enabled() or backend == "pytorch":
        return "pytorch"
    # The explicit memory policy is portable across CUDA architectures. An
    # unsupported dtype/layout falls back instead of changing public behavior.
    if backend == "memory":
        return "memory" if supported else "pytorch"
    # ``auto`` is the compute-oriented default. The compressed-save path has a
    # measured whole-model latency cost and is selected only by an explicit
    # memory policy.
    return "pytorch"


def edge_layer_norm(
    values: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    *,
    backend: EdgeNormBackend = "auto",
) -> torch.Tensor:
    """Apply native LayerNorm with an optional compressed backward save.

    The memory backend keeps the native PyTorch forward bit-for-bit and stores
    only a BF16 copy of its edge-sized input for backward. Unsupported inputs,
    CPU execution, and inference all retain :func:`torch.nn.functional.layer_norm`.
    """
    if backend not in {"auto", "pytorch", "memory"}:
        raise ValueError(f"unknown MPNN edge LayerNorm backend: {backend!r}")
    (values.shape[-1],)
    supported = _memory_supported(values, weight, bias)
    selected = _select_backend(values, backend, supported=supported)
    if selected == "memory":
        # Keep Triton and the standalone LayerNorm backend out of CPU/import-only
        # users. This branch is reached only by supported CUDA training tensors.
        from miniworld_engine.kernels.mpnn_edge_layernorm.triton import (
            edge_layer_norm_memory,
        )

        assert weight is not None and bias is not None
        return edge_layer_norm_memory(values, weight, bias, eps)
    return edge_layer_norm_pytorch(values, weight, bias, eps)


__all__ = ["EdgeNormBackend", "edge_layer_norm"]
