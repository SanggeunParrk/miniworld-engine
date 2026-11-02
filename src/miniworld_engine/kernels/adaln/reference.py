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

The reference is the plain functional form (:func:`adaln_pytorch`): a kernel is checked against it on both the forward output and the backward gradients.
"""

from __future__ import annotations

import torch


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
