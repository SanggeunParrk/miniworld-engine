"""Dispatch boundary for the ProteinMPNN encoder edge-message MLP."""

from __future__ import annotations

from typing import Literal

import torch

from miniworld_engine.kernels.mpnn_edge_mlp.reference import edge_mlp_update_pytorch

# ``cuda`` names the hand-written A100 (sm_80) kernels (``integrations/mpnn_edge_sm80.py``): on an A100 ``auto`` (calibrated sizes), ``triton_compute`` (the saved projection) and
# ``triton_memory`` (recompute) run them too, unless ``MINIWORLD_MPNN_EDGE_SM80=0`` or ``engine_backend="triton"`` keeps the Triton kernels.
EdgeMLPBackend = Literal[
    "auto",
    "pytorch",
    "triton_compute",
    "triton_memory",
    "cuda",
]

_INT32_MAX = 2**31 - 1
_PADDED_TILE_ELEMENTS = 128 * 128
_MIN_AUTO_ROWS = 2048 * 48


def _triton_shape_supported(numel: int, width: int) -> bool:
    return (
        width == 128 and numel > 0 and numel <= _INT32_MAX - (_PADDED_TILE_ELEMENTS - 1)
    )


def _triton_contract_supported(
    *,
    device: torch.device,
    dtype: torch.dtype,
    numel: int,
    width: int,
    contiguous: bool,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> bool:
    """Check the shared edge-MLP contract without requiring an allocation."""
    parameters = (hidden_weight, hidden_bias, output_weight, output_bias)
    projection_dtype_supported = all(
        tensor.dtype == torch.bfloat16 for tensor in parameters
    ) or (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and all(tensor.dtype == torch.float32 for tensor in parameters)
    )
    return (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and _triton_shape_supported(numel, width)
        and contiguous
        and projection_dtype_supported
        and all(tensor.is_cuda for tensor in parameters)
        and all(tensor.device == device for tensor in parameters)
        and all(tensor.is_contiguous() for tensor in parameters)
        and hidden_weight.shape == (128, 128)
        and hidden_bias.shape == (128,)
        and output_weight.shape == (128, 128)
        and output_bias.shape == (128,)
    )


def _triton_supported(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> bool:
    return (
        preactivation.ndim > 0
        and preactivation.is_cuda
        and _triton_contract_supported(
            device=preactivation.device,
            dtype=preactivation.dtype,
            numel=preactivation.numel(),
            width=preactivation.shape[-1],
            contiguous=preactivation.is_contiguous(),
            hidden_weight=hidden_weight,
            hidden_bias=hidden_bias,
            output_weight=output_weight,
            output_bias=output_bias,
        )
    )


def _select_backend(
    preactivation: torch.Tensor,
    backend: EdgeMLPBackend,
    *,
    supported: bool,
) -> EdgeMLPBackend:
    if backend != "auto":
        return backend
    # Keep unmeasured devices and small launch grids on PyTorch. An explicit
    # Triton policy remains available for bring-up on another architecture.
    calibrated = (
        supported
        and preactivation.numel() // 128 >= _MIN_AUTO_ROWS
        and torch.cuda.get_device_capability(preactivation.device) == (8, 6)
    )
    if not calibrated:
        return "pytorch"
    return "triton_compute" if torch.is_grad_enabled() else "triton_memory"


def edge_mlp_update(
    preactivation: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    *,
    backend: EdgeMLPBackend = "auto",
) -> torch.Tensor:
    """Run the selected compute- or memory-efficient edge MLP policy."""
    if backend not in {
        "auto",
        "pytorch",
        "triton_compute",
        "triton_memory",
        "cuda",
    }:
        raise ValueError(f"unknown MPNN edge MLP backend: {backend!r}")
    supported = _triton_supported(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )
    if backend in {"triton_compute", "triton_memory", "cuda"} and not supported:
        raise ValueError(
            "the Triton MPNN edge MLP requires contiguous CUDA BF16 input "
            "[..., 128] and contiguous [128, 128]/[128] projection parameters "
            "using CUDA BF16 autocast or native BF16 parameters, with offsets "
            "that fit in signed 32-bit indexing"
        )
    if supported and backend != "pytorch":
        # The A100 hand-CUDA kernels (lazy: never imported off sm_80, never built until used); ``auto`` keeps the Triton rule's size floor, and the Triton kernels / PyTorch keep every
        # other card, the switch off and a failed build.
        from miniworld_engine.integrations import mpnn_edge_sm80

        if mpnn_edge_sm80.mlp_serves(preactivation, hidden_weight, output_weight) and (
            backend != "auto" or preactivation.numel() // 128 >= _MIN_AUTO_ROWS
        ):
            return mpnn_edge_sm80.edge_mlp_update(
                preactivation,
                hidden_weight,
                hidden_bias,
                output_weight,
                output_bias,
                memory=backend == "triton_memory",
            )
    if backend == "cuda":
        raise ValueError("the sm_80 CUDA MPNN edge MLP needs an A100 (capability 8.0), 16-byte aligned weights and a loadable extension")
    selected = _select_backend(preactivation, backend, supported=supported)
    if selected == "triton_memory":
        from miniworld_engine.kernels.mpnn_edge_mlp.triton import triton_edge_mlp_update

        return triton_edge_mlp_update(
            preactivation,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )
    if selected == "triton_compute":
        from miniworld_engine.kernels.mpnn_edge_mlp.triton.compute import (
            triton_edge_mlp_update_compute,
        )

        return triton_edge_mlp_update_compute(
            preactivation,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )
    return edge_mlp_update_pytorch(
        preactivation,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )


__all__ = ["EdgeMLPBackend", "edge_mlp_update"]
