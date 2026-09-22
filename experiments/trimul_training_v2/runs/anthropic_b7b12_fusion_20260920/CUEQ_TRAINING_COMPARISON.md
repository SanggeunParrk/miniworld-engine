# Matched cuEquivariance bidirectional training comparison

2026-09-20, node02 H100 80GB, BF16, B1/C128/H128 per direction, dropout 25%.
cuEquivariance torch 0.9.1 / ops-cu12 0.10.0, ONDEMAND tuning, static fullgraph compile.
Every timed path uses an explicit single-call CUDA graph. 600 alternating samples/path.

| Scope | L | cuEq compiled ms | Current CUDA ms | Speedup |
| --- | --- | --- | --- | --- |
| Forward + backward | 384 | 2.042 | 1.300 | 1.571x |
| Forward + backward | 768 | 7.772 | 5.105 | 1.523x |
| Training forward | 384 | 0.554 | 0.459 | 1.206x |
| Training forward | 768 | 2.111 | 1.758 | 1.201x |

## Scope and numerical differences

- Matched cuEq primitive composition: input LN, gated dual GEMM, outgoing + incoming contractions, shared 256-channel output LN, projection/gate, supplied dropout and residual. The public single-direction TMU called twice has different normalization semantics.
- Current CUDA: Anthropic-derived saved forward, unchanged Triton/cuBLAS B1-B6, CUDA B7-B12. Separate Claude B1-B4 work is excluded.
- Fresh saves and all 11 gradients are produced inside every timed training call. Weight packing/layout conversions are included for both paths. No optimizer, RNG generation, CPU/autograd dispatch or compilation time.
- Forward timings preserve training saves and autograd; total timings are directly measured, not sums of independent kernel timings.
- cuEq keeps its own BF16 rounding and saves. Cross-backend gradients do not meet the strict same-saves CUDA error contract. This comparison is mathematical equivalence with reported differences, not bitwise equivalence or convergence validation.
- Eager cuEq is a secondary diagnostic: separate BF16 rounding at sigmoid/multiply/dropout/residual gives ~0.25% forward relative L2. Primary compiled comparison uses a 0.1% forward and 1% gradient cross-backend bound. Existing CUDA same-saves bounds remain unchanged.

| L | Compiled forward relative L2 | Max compiled gradient relative L2 | cuEq eager total ms | Current prepacked total ms |
| --- | --- | --- | --- | --- |
| 384 | 4.0394243e-05 | 0.0080532813 | 2.900 | 1.287 |
| 768 | 4.8200363e-05 | 0.0051185805 | 10.997 | 5.089 |

## Sources

- NVIDIA public API: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html
- Benchmark: compare_cueq_training.py; source and benchmark SHA-256 are recorded in each JSON.
- Raw results: cueq-training-L384.json, cueq-training-L768.json.
- L384 source: front_prefetch_lnpair_storepipe.
- L768 source: front_ring96_cache3_glu_ahead_u4_early_writer32 (validated experimental candidate, not production default).
