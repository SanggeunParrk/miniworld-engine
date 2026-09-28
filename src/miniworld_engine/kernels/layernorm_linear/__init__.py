"""Fused LayerNorm + Linear (`te.LayerNormLinear` analogue).

LayerNorm over the last dim immediately followed by a Linear (GEMM + bias).
``reference.py`` holds the PyTorch math (and the ``torch.compile`` baseline); the
**Triton** backend (``triton/fused.py``, plus the TE-style ``triton/te_style.py``) runs on
every arch. See docs/kernels/layernorm-linear.md.
"""

from __future__ import annotations

from miniworld_engine.kernels.layernorm_linear.autograd import (
    LayerNormLinearTritonFn,
    layernorm_linear_triton_fn,
)
from miniworld_engine.kernels.layernorm_linear.interface import (
    layernorm_linear_triton,
)
from miniworld_engine.kernels.layernorm_linear.reference import (
    LayerNormLinearRef,
    layernorm_linear_pytorch,
)
from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
    LayerNormLinearTEFn,
    layernorm_linear_te_fn,
    set_fp32_matmul_precision,
)

__all__ = [
    "LayerNormLinearRef",
    "LayerNormLinearTEFn",
    "LayerNormLinearTritonFn",
    "layernorm_linear",          # inference forward (Triton), optional stats
    "layernorm_linear_pytorch",
    "layernorm_linear_te_fn",    # trainable, TE-style (materialize+cuBLAS, stride-transparent)
    "layernorm_linear_triton",   # portable inference forward
    "layernorm_linear_triton_fn",  # trainable, portable (Triton fwd + cuBLAS/Triton bwd)
    "set_fp32_matmul_precision",  # te_fn fp32 GEMM policy: 'high'=TF32 (default) | 'highest'=true fp32
]


def layernorm_linear(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *,
                     save_stats: bool = False):
    """Forward LayerNormLinear (Triton, every arch). ``save_stats=True`` returns
    ``(Y, mean, rstd)`` with the LN stats computed alongside."""
    if save_stats:
        from miniworld_engine.kernels.layernorm_linear.triton.fused import (
            layernorm_linear_triton_fwd_stats,
        )
        return layernorm_linear_triton_fwd_stats(x, ln_weight, ln_bias, weight, bias, eps)
    return layernorm_linear_triton(x, ln_weight, ln_bias, weight, bias, eps)
