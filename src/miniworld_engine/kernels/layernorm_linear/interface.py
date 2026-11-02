"""Triton entry point for the fused LayerNorm + Linear kernel.

This Triton kernel runs on any Triton-supported arch (Ampere/Ada/Hopper/Blackwell/ROCm).
It computes ``LayerNorm(x) @ W^T + b`` in one fused kernel — LN stats reduced on-chip,
then the projection GEMM — so eager ``LayerNorm`` -> ``Linear``'s extra HBM round trip
is avoided. See ``triton/fused.py``.
"""

from __future__ import annotations

import torch


def layernorm_linear_triton(
    x: torch.Tensor,          # (..., d_in)
    ln_weight: torch.Tensor,  # (d_in,)
    ln_bias: torch.Tensor,    # (d_in,)
    weight: torch.Tensor,     # (d_out, d_in)
    bias: torch.Tensor | None,  # (d_out,)
    eps: float = 1e-5,
) -> torch.Tensor:
    from miniworld_engine.kernels.layernorm_linear.triton.fused import (
        layernorm_linear_triton_fwd,
    )

    return layernorm_linear_triton_fwd(x, ln_weight, ln_bias, weight, bias, eps)


def layer_norm_linear(x: torch.Tensor, ln_weight: torch.Tensor, proj_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """``Linear(LayerNorm(x))`` over the last dim (both affine, no bias), routed per card -- see
    ``kernels.layernorm_linear.triton.pair_bias.triton_layer_norm_linear``. H100 / B200 launch no Triton: the CUDA pair-bias kernels where they
    serve the call (``cuda/pair_bias``), the PyTorch statements otherwise. On a card without a record: ``F.layer_norm`` and ``F.linear``
    with the weights in the activation's dtype (the statements the modules' PyTorch legs run), the result in ``x``'s dtype."""
    import torch.nn.functional as F

    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(x):
        # H100, d_pair 128 -> 16 heads: the token DiT's CUDA pair-bias kernels (LayerNorm + projection in one pass each way)
        from miniworld_engine.kernels.layernorm_linear.cuda import pair_bias

        out = pair_bias.layer_norm_linear(x, ln_weight, proj_weight, eps)
        if out is not None:
            return out
    if no_triton(x) or (x.is_cuda and portable_blocked("layer_norm_linear", x.device, x, ln_weight, proj_weight)):
        normed = F.layer_norm(x, (x.shape[-1],), ln_weight.to(x.dtype), None, eps)
        return F.linear(normed, proj_weight.to(x.dtype))
    from miniworld_engine.kernels.layernorm_linear.triton.pair_bias import (
        triton_layer_norm_linear,
    )

    return triton_layer_norm_linear(x, ln_weight, proj_weight, eps)
