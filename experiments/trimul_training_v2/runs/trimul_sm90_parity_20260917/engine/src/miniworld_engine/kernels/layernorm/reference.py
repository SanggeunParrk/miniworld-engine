"""PyTorch reference for the standalone LayerNorm kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def layernorm_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference LayerNorm over the last dimension."""
    # Norm affine parameters stay FP32 after module.bfloat16(). The CUDA
    # PyTorch fallback rejects BF16 input + FP32 weight when bias is absent.
    # Accumulate low-precision inputs in FP32 and preserve the activation dtype.
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    result = F.layer_norm(
        x.to(compute_dtype), (x.shape[-1],),
        weight.to(compute_dtype) if weight is not None else None,
        bias.to(compute_dtype) if bias is not None else None, eps,
    )
    return result.to(x.dtype)


class LayerNormRef(nn.Module):
    """Small module wrapper around the PyTorch LayerNorm math."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_pytorch(x, self.weight, self.bias, self.eps)
