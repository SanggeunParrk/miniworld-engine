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
    """Forward LayerNormLinear, dispatched for the fastest path.

    Both backends fold ``W2 = gamma * W`` and compute ``Y = rstd*(X@W2) - c1*S + B2``
    (LayerNorm(X) never materialized), bit-comparable to
    ``F.linear(F.layer_norm(x), W, b)`` (cos = 0.999997).

    ``save_stats=False`` (inference) — dispatch by output width N for raw forward speed:
      - ``N <= 256`` → ``layernorm_linear_cute_fused`` (M2): LN stats reduced INSIDE the
        GEMM main kernel (one kernel, no extra gmem). The memory-bound regime where
        fusion wins — beats M1, torch.compile and TE (d=128 M=262144: 0.062 ms vs M1
        0.094 / tc 0.124 / TE 0.139). Returns ``Y``.
      - ``N > 256`` → ``layernorm_linear_cute`` (M1): folded GEMM + a separate stats
        pass. Compute-bound here, where the in-kernel reduction would steal WGMMA
        throughput, so the separate-stats GEMM is faster. Returns ``Y``.

    ``save_stats=True`` (training) — ALWAYS use M1, the separate-stats path, regardless of
    N, and return ``(Y, mean, rstd)``. Backward needs the LN stats; the fused M2 computes
    them transiently inside the GEMM and discards them, so persisting them via M1's
    explicit stats pass is the unconditional win once a backward pass follows (it would
    otherwise have to recompute mean/rstd anyway). Threshold N=256 is the H100/bf16
    forward-only crossover (see archived LayerNormLinear reports and
    ``docs/design/layernorm-linear-warp-specialized-stats.md``).
    """
    # All paths use M1 (`layernorm_linear_cute`, config=None -> brute-force autotuned via
    # cute_config). The M2 fused kernel (`layernorm_linear_cute_fused`) is a deep quack-0.5.0 port
    # in progress (load_AB->load_tma, _epi_smem_map->dict, and its kernel-filled SmemColVec op is
    # dropped by 0.5.0's None-arg active-op filter — see docs/kernel-optimization/cute-autotune-and-config-pinning.md); until it lands, n<=256 INFERENCE
    # uses correct M1 instead of broken M2. M2's only edge was inference perf (~0.062 vs 0.094 ms).
    return layernorm_linear_cute(
        x, ln_weight, ln_bias, weight, bias, eps, prefolded=prefolded,
        return_stats=save_stats, config=None,
    )
