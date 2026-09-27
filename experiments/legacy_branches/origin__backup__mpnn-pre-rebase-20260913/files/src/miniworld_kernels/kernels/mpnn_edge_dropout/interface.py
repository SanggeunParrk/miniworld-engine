"""Lazy dispatch for ProteinMPNN encoder edge dropout."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


EdgeDropoutBackend = Literal["auto", "pytorch", "bitpack"]

_INT32_MAX = 2**31 - 1
_PACK_BLOCK_BYTES = 256
_PADDED_TILE_ELEMENTS = 8 * _PACK_BLOCK_BYTES
_SUPPORTED_DTYPES = {torch.float32, torch.bfloat16}


def _bitpack_shape_supported(numel: int) -> bool:
    """Return whether flattened kernel offsets fit signed 32-bit indexing."""
    return 0 < numel <= _INT32_MAX - (_PADDED_TILE_ELEMENTS - 1)


def _bitpack_supported(
    values: torch.Tensor,
    probability: float,
    *,
    training: bool,
    inplace: bool,
) -> bool:
    """Check the compressed-save contract without importing Triton."""
    return (
        training
        and torch.is_grad_enabled()
        and values.requires_grad
        and not torch.are_deterministic_algorithms_enabled()
        and not inplace
        and 0.0 < probability < 1.0
        and values.is_cuda
        and values.dtype in _SUPPORTED_DTYPES
        and values.is_contiguous()
        and _bitpack_shape_supported(values.numel())
    )


def _select_backend(
    backend: EdgeDropoutBackend,
    *,
    supported: bool,
) -> EdgeDropoutBackend:
    if backend == "bitpack":
        return "bitpack" if supported else "pytorch"
    # ``auto`` deliberately remains the compute-oriented native policy. The
    # compressed mask has a small measured latency cost and is opt-in.
    return "pytorch"


def edge_dropout(
    values: torch.Tensor,
    probability: float,
    *,
    training: bool,
    backend: EdgeDropoutBackend = "auto",
    inplace: bool = False,
) -> torch.Tensor:
    """Apply native dropout with an optional bit-packed backward mask.

    The bitpack policy calls ATen's native dropout forward and returns that
    output unchanged. Only the native boolean mask retained for backward is
    packed to one bit per element. Unsupported inputs, evaluation, no-grad,
    deterministic mode, and the default ``auto`` policy retain PyTorch.
    The explicit ``bitpack`` training path supports first-order gradients only.
    """
    if backend not in {"auto", "pytorch", "bitpack"}:
        raise ValueError(f"unknown MPNN edge dropout backend: {backend!r}")
    supported = _bitpack_supported(
        values,
        probability,
        training=training,
        inplace=inplace,
    )
    selected = _select_backend(backend, supported=supported)
    if selected == "bitpack":
        # Keep Triton out of imports and every native fallback path.
        from .triton import edge_dropout_bitpack

        return edge_dropout_bitpack(values, probability)
    return F.dropout(
        values,
        p=probability,
        training=training,
        inplace=inplace,
    )


__all__ = ["EdgeDropoutBackend", "edge_dropout"]
