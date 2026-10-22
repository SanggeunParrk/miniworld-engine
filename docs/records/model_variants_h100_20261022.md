# Boltz-2 / RoseTTAFold3 / Chai-1 variants on H100: first measurements (2026-10-22)

H100 (sm90), torch 2.13 + cu129, bf16, inference, CUDA graph (20 replays), random weights. "compiled" is `torch.compile` of the same
torch statements the module falls back to (`kernels/dit_variants/reference.py`, `TriangleAttention._biased_reference`,
`OuterProductMean._variant_reference`); FoldForge runs its trunk eager in a CUDA graph and compiles the denoiser.

## Token DiT stacks (A = 5 samples, ms)

| model (blocks) | L | engine | eager | compiled |
|---|---|---|---|---|
| Boltz-2 (24) | 384 | 5.92 | 22.1 | 7.93 |
| Boltz-2 (24) | 768 | 9.22 | 81.9 | 21.7 |
| RoseTTAFold3 (24) | 384 | 5.77 | 22.7 | 7.76 |
| RoseTTAFold3 (24) | 768 | 8.64 | 82.7 | 21.4 |
| Chai-1 (16) | 384 | 3.22 | 16.2 | 5.21 |
| Chai-1 (16) | 768 | 5.26 | 62.5 | 15.3 |

## Atom DiT stacks (3 blocks, A = 5, ms)

| model | atoms | engine | eager | compiled |
|---|---|---|---|---|
| Boltz-2 | 8192 | 1.64 | 10.9 | 1.92 |
| Boltz-2 | 24576 | 3.58 | 32.0 | 6.09 |
| RoseTTAFold3 | 8192 | 1.63 | 11.3 | 2.17 |
| RoseTTAFold3 | 24576 | 3.57 | 33.4 | 6.25 |
| Chai-1 | 8192 | 1.46 | 10.5 | 1.96 |
| Chai-1 | 24576 | 3.05 | 31.0 | 5.67 |

## Chai-1 trunk triangle attention (pair 256, 4 x 64 heads, B 1, ms)

| node | L | engine | eager | compiled |
|---|---|---|---|---|
| starting | 256 | 0.342 | 2.64 | 1.10 |
| starting | 384 | 0.796 | 7.54 | 4.23 |
| starting | 768 | 4.23 | 52.5 | 27.6 |
| ending, untransposed output | 256 | 0.377 | 2.87 | 1.41 |
| ending, untransposed output | 384 | 0.863 | 7.93 | 4.14 |
| ending, untransposed output | 768 | 4.50 | 54.1 | 24.7 |

## OuterProductMean (ms)

| variant | S, N | engine | eager | compiled |
|---|---|---|---|---|
| RF3 (biased projections) | 1024, 384 | 1.02 | 1.35 | 0.93 |
| RF3 | 4096, 384 | 2.87 | 3.13 | 2.48 |
| RF3 | 1024, 768 | 3.43 | 4.87 | 3.37 |
| Chai-1 (8 groups) | 1024, 384 | 0.74 | 1.44 | 0.84 |
| Chai-1 | 4096, 384 | 2.02 | 3.86 | 2.45 |
| Chai-1 | 1024, 768 | 2.27 | 4.38 | 2.56 |

Composed OPM history (Chai-1, S 1024, N 384): 1.12 ms -> 0.89 (projection layout without permute copies) -> 0.83 (group split
kernel) -> 0.74 (shared-memory split: contiguous reads and writes). RF3: the composed path (1.11 -> 1.33 ms with a per-group projection
layout) lost to the fused OPM plus bias terms (1.49 ms with fp32 SIMT GEMMs, 1.02 ms with bf16 GEMMs and one finishing pass).

## Accuracy

Every variant against its fp64 torch statements, the engine's bf16 error at most 1.5x (parallel Pairformer: 2x) the bf16 torch
statements' own: `tests/gpu/sm90/test_dit_variant_h100_gpu.py` (34), `test_triattn_biases_gpu.py` (33),
`test_opm_variants_gpu.py` (30), `test_pairformer_parallel_gpu.py` (4).
