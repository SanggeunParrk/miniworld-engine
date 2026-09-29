"""PyTorch reference for the fused adaptive-LayerNorm kernels.

Mirrors the ``adaln_train`` / ``adaln_inference`` entry points (and the
underlying ``triton_adaptive_layer_norm``), which all take the same argument
tuple ``(x, cond, cond_ln_weight, scale_weight, scale_bias, bias_weight,
eps_x, eps_cond)`` and return a single tensor shaped like ``x``.

Formula, per row of the flattened ``(..., d)`` inputs::

    x_hat    = (x - mean(x)) * rsqrt(var(x) + eps_x)            # no affine
    cond_aff = (cond - mean(cond)) * rsqrt(var(cond) + eps_cond) * cond_ln_weight
    scale    = cond_aff @ scale_weight.T + scale_bias
    bias     = cond_aff @ bias_weight.T
    y        = sigmoid(scale) * x_hat + bias

``mean``/``var`` are the biased (``1/N``) moments over the last axis. Both
LayerNorms are affine-free except for ``cond_ln_weight`` (the cond LayerNorm has
no bias) and ``to_bias`` has no bias term of its own.

Provided as an ``nn.Module`` (:class:`AdaLNReference`, owning the four
parameters) so a kernel can be checked against it on both the forward output and
the backward gradients, e.g.::

    ref = AdaLNReference(d_hidden, d_cond).cuda().to(torch.bfloat16)
    y = ref(x, cond)
    yk = adaln_train(x, cond, ref.cond_ln_weight, ref.scale_weight,
                     ref.scale_bias, ref.bias_weight, ref.eps_x, ref.eps_cond)
    y.sum().backward()

A plain functional form (:func:`adaln_pytorch`) is kept for callers that already
hold the weight tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def adaln_pytorch(
    x: torch.Tensor,  # (..., d_hidden)
    cond: torch.Tensor,  # (..., d_cond)
    cond_ln_weight: torch.Tensor,  # (d_cond,)
    scale_weight: torch.Tensor,  # (d_hidden, d_cond)
    scale_bias: torch.Tensor,  # (d_hidden,)
    bias_weight: torch.Tensor,  # (d_hidden, d_cond)
    eps_x: float = 1e-5,
    eps_cond: float = 1e-5,
) -> torch.Tensor:
    """Compute ``sigmoid(scale) * LN(x) + bias`` with scale/bias driven by ``cond``."""
    out_dtype = x.dtype

    xf = x.float()
    x_hat = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(
        xf.var(-1, unbiased=False, keepdim=True) + eps_x
    )

    condf = cond.float()
    cond_norm = (condf - condf.mean(-1, keepdim=True)) * torch.rsqrt(
        condf.var(-1, unbiased=False, keepdim=True) + eps_cond
    )
    cond_aff = cond_norm * cond_ln_weight.float()

    # The kernel rounds the GEMM operands and the GEMM result to the compute dtype
    # (bf16/fp16 inputs) and only then applies the gate, so the casts are load-bearing
    # for a cosine comparison: in fp32 they are no-ops.
    cond_aff = cond_aff.to(out_dtype)
    scale = (cond_aff @ scale_weight.to(out_dtype).transpose(-2, -1)).float()
    scale = (scale + scale_bias.float()).to(out_dtype)
    bias = (cond_aff @ bias_weight.to(out_dtype).transpose(-2, -1)).to(out_dtype)

    gate = torch.sigmoid(scale.float()).to(out_dtype)
    y = gate.float() * x_hat + bias.float()
    return y.to(out_dtype)


class AdaLNReference(nn.Module):
    """nn.Module reference for adaptive LayerNorm (forward + backward ground truth)."""

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        *,
        eps_x: float = 1e-5,
        eps_cond: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        scale = d_cond**-0.5
        # Spelled out rather than splatted from a `kw` dict: a bare dict literal infers a joined
        # value type that matches no `torch.ones`/`torch.randn` overload.
        self.cond_ln_weight = nn.Parameter(torch.ones(d_cond, device=device, dtype=dtype))
        self.scale_weight = nn.Parameter(
            torch.randn(d_hidden, d_cond, device=device, dtype=dtype) * scale
        )
        self.scale_bias = nn.Parameter(torch.ones(d_hidden, device=device, dtype=dtype))
        self.bias_weight = nn.Parameter(
            torch.randn(d_hidden, d_cond, device=device, dtype=dtype) * scale
        )
        self.eps_x = eps_x
        self.eps_cond = eps_cond

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass; differentiable in ``x``, ``cond`` and all four parameters."""
        return adaln_pytorch(
            x,
            cond,
            self.cond_ln_weight,
            self.scale_weight,
            self.scale_bias,
            self.bias_weight,
            self.eps_x,
            self.eps_cond,
        )
