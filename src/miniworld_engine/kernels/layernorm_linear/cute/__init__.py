"""CuTeDSL (quack SM90) backend for fused LayerNormLinear forward."""

from __future__ import annotations

from .gemm_layernorm_linear import fold_for_gemm, layernorm_linear_cute
from .gemm_layernorm_linear_fused import layernorm_linear_cute_fused

__all__ = [
    "fold_for_gemm",
    "layernorm_linear_cute",
    "layernorm_linear_cute_fused",
    "layernorm_linear",
]


def layernorm_linear(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *,
                     save_stats: bool = False, prefolded=None):
    """Forward LayerNormLinear through M1 (separate statistics plus folded GEMM).

    Inference returns Y; ``save_stats=True`` returns (Y, mean, rstd) for backward.
    M2 computes statistics inside the GEMM. Its quack 0.5 host-side port now passes
    CPU compile-only checks, but GPU numerical and synchronization qualification is
    still required before restoring the historical small-width inference dispatch.
    """
    return layernorm_linear_cute(
        x, ln_weight, ln_bias, weight, bias, eps, prefolded=prefolded,
        return_stats=save_stats, config=None,
    )
