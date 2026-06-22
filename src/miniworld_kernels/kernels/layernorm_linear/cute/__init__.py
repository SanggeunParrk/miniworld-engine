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


def layernorm_linear(x, ln_weight, ln_bias, weight, bias, eps: float = 1e-5, *, prefolded=None):
    """Forward LayerNormLinear, dispatched by output width N for the fastest path.

    Both backends fold ``W2 = gamma * W`` and compute ``Y = rstd*(X@W2) - c1*S + B2``
    (LayerNorm(X) never materialized), bit-comparable to
    ``F.linear(F.layer_norm(x), W, b)`` (cos = 0.999997).

    - ``N <= 256`` → ``layernorm_linear_cute_fused`` (M2): LN stats reduced INSIDE the
      GEMM main kernel on CUDA cores (one kernel, no extra gmem traffic). The
      memory-bound regime where fusion wins — beats M1, torch.compile and TE
      (d=128 M=262144: 0.062 ms vs M1 0.094 / tc 0.124 / TE 0.139).
    - ``N > 256`` → ``layernorm_linear_cute`` (M1): folded GEMM + a separate (cheap)
      stats pass. In the compute-bound regime the in-kernel reduction would steal WGMMA
      issue throughput, so the separate-stats GEMM is faster (M1 already beats
      torch.compile + TE on all shapes).

    Threshold N=256 is the H100/bf16 crossover (see ``benchmark/`` and
    ``cute/WARP_SPECIALIZED_STATS_DESIGN.md``).
    """
    n = weight.shape[0]
    if n <= 256:
        return layernorm_linear_cute_fused(x, ln_weight, ln_bias, weight, bias, eps, prefolded=prefolded)
    return layernorm_linear_cute(x, ln_weight, ln_bias, weight, bias, eps, prefolded=prefolded)
