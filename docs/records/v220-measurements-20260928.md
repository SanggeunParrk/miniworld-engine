# H100 v2.2.0 measurements — 2026-09-28

Record, not updated after the fact. Cluster `h100` partition, H100 80GB HBM3 (132 SMs),
torch 2.13.0+cu129, triton 3.7.1, bf16 pair tensors, fp32 LayerNorm affine, B=1.
**Triton autotune caches were stale** (built under torch 2.10 / triton 3.6), so every Triton
baseline below ran on its heuristic config subset; a rebuilt cache may speed Triton up.

## TriMul — CUDA vs Triton, manual CUDA graph, paired alternating replays, median (ms)

Triton = `engine_backend="triton"`; CUDA = default `auto` dispatch, path proven by counting
entry-point calls. Error of the update vs an fp32 PyTorch module ≈ 0.0064–0.0067 for CUDA
(Triton 0.0062–0.0064); changed-input graph replay exact.

### Bidirectional training (fwd+bwd)

| D | L384 Triton / CUDA | × | L768 Triton / CUDA | × | CUDA path |
|---|---|---|---|---|---|
| 64 | 0.828 / 0.404 | 2.05 | 3.061 / 1.519 | 2.02 | `h100_d64_training` (dropout 0.25 and masks: 1.99–2.05×) |
| 128 | 1.629 / 1.004 | 1.62 | 6.593 / 4.075 | 1.62 | `h100_training` (B1/B7) |
| 256 | 3.742 / 2.592 | 1.44 | 16.640 / 10.584 | 1.57 | `h100_wide_training` |
| 384 | 6.637 / 4.563 | 1.46 | 29.162 / 18.938 | 1.54 | `h100_wide_training` |
| 512 | 10.089 / 6.927 | 1.46 | 45.765 / 28.278 | 1.62 | `h100_wide_training` |

### Inference (no_grad)

| D | bidir L384 | bidir L768 | uni L384 | uni L768 |
|---|---|---|---|---|
| 64 | 0.197 / 0.129 (1.53×) | 0.684 / 0.484 (1.42×) | 0.121 / 0.092 (1.32×) | 0.397 / 0.314 (1.27×) |
| 128 | 0.441 / 0.259 (1.70×) | 1.677 / 1.009 (1.66×) | 0.274 / 0.157 (1.75×) | 0.974 / 0.571 (1.70×) |
| 256 | 1.137 / 0.584 (1.95×) | 4.553 / 2.413 (1.89×) | 0.668 / 0.367 (1.82×) | 2.556 / 1.461 (1.75×) |
| 384 | 2.160 / 1.108 (1.95×) | 8.746 / 4.499 (1.94×) | 1.214 / 0.776 (1.57×) | 4.792 / 3.147 (1.52×) |
| 512 | 3.803 / 1.861 (2.04×) | 15.553 / 7.877 (1.97×) | 2.244 / 1.244 (1.80×) | 8.960 / 4.928 (1.82×) |

Bidir D256–512: `h100_wide_inference` (folded-LN streaming K3). Uni D512: `h100_uni_wide_inference`
(incoming within 1% of outgoing). Other cells: packaged K1/K3 table (`h100_inference`).

## Module comparison — uncached, compiled + CUDA graph (ms, L384 / L768)

From the first v2.2.0 module bench (`benchmarks/runners/bench.py`, before the wide TriMul
kernels above existed; CSVs under `benchmarks/modules/*/artifacts/`, local).

| module | mode | ours | Anthropic (pristine) | cuEquivariance 0.12 |
|---|---|---|---|---|
| TriMul uni D128 | inference | 0.151 / 0.558 | 0.163 / 0.611 | 0.278 / 1.085 |
| TriMul uni D128 | training | 0.770 / 2.792 | — | 1.431 / 5.510 |
| TriMul bidir D128 | inference | 0.252 / 1.000 | 0.279 / 1.044 | 0.518 / 1.982 |
| TriMul bidir D128 | training | 1.009 / 3.964 | — | 2.295 / 8.720 |
| TriangleAttention D128 | inference | 0.397 / 1.906 | 0.447 / 2.326 | 0.708 / 3.959 |
| TriangleAttention D128 | training | 1.255 / 6.565 | — | 3.219 / 19.33 |
| Transition D128 n=4 | inference | 0.119 / 0.436 | 0.166 / 0.607 | — |
| OuterProductMean | inference | 0.654 / 2.304 | 1.210 / 4.607 | — |
| MSAPairWeightedAveraging | inference | 0.406 / 1.304 | 0.703 / 1.735 | — |
| AttentionPairBias | training | 0.263 / 0.655 | — | 0.186 / 0.440 |

adaLN and ConditionedTransition were slower than compiled PyTorch uncached; not re-measured
with a tuned cache.
