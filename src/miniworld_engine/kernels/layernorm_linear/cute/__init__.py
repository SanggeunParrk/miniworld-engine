"""CuTeDSL (quack SM90) backend for fused LayerNormLinear forward."""

from __future__ import annotations

import torch
from miniworld_engine.kernels._compile import opaque

from .gemm_layernorm_linear import fold_for_gemm, layernorm_linear_cute
from .gemm_layernorm_linear_fused import layernorm_linear_cute_fused

__all__ = [
    "fold_for_gemm",
    "layernorm_linear_cute",
    "layernorm_linear_cute_fused",
    "layernorm_linear",
]


def _m2_inference_fake(x, ln_weight, ln_bias, weight, bias, eps):
    """Allocate outputs with the same shape, dtype and strides as _m2_inference."""
    return x.new_empty((x.shape[0], weight.shape[0]))


@opaque(fake=_m2_inference_fake, name="layernorm_linear_m2_inference")
def _m2_inference(x: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                  weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Keep CuTe compilation and its context variables outside Dynamo tracing."""
    return layernorm_linear_cute_fused(x, ln_weight, ln_bias, weight, bias, eps)


def layernorm_linear(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *,
                     save_stats: bool = False, prefolded=None):
    """Use qualified Hopper M2 inference shapes; M1 supplies training statistics.

    H100 BF16 square width 128 was qualified with both input layouts and CUDA
    graphs. Width 256 had inconsistent speedups and stays on M1. Prefolded M1
    operands are respected instead of paying a new fold in M2.
    """
    if (not save_stats and prefolded is None and x.is_cuda and x.ndim == 2
            and x.dtype == torch.bfloat16 and x.shape[-1] == 128
            and tuple(weight.shape) == (128, 128)
            and x.numel() // 128 >= 128 and (x.numel() // 128) % 128 == 0
            and torch.cuda.get_device_capability(x.device)[0] == 9):
        return _m2_inference(x, ln_weight, ln_bias, weight, bias, eps)
    return layernorm_linear_cute(
        x, ln_weight, ln_bias, weight, bias, eps, prefolded=prefolded,
        return_stats=save_stats, config=None,
    )
