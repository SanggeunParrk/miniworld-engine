# TriangleMultiplication on B200 (sm100)

Kernel-level status of the TriMul modules on B200; the module-level summary is in
[b200.md](../b200.md). bf16 only; columns are (Length, Dimension). B200 = CUDA where a
hand-written sm_100a path exists; a shape without one is 미구현 (Triton path). Figures: one box
per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/trimul.json` by `python -m miniworld_engine.viz.kernel_flow`,
then converted to PNG (`cairosvg -s 2 -b white`; `rsvg-convert` was not installed in the environment that generated the figures).
Dispatch: `integrations/trimul_b200.py`; kernels: `kernels/trimul_inproj/cuda/`
(`b200_infer.py`, `b200_train.py`) and `b200_sources/` (tcgen05 / TMEM / TMA), one extension.

Both modules share one kernel family: the bidirectional module is the one-direction module at twice
the hidden width (planes P = 4D, contraction output H = 2D), its two contractions taking one half of
the planes each; one direction has P = 2D, H = D.

Served when `implementation=miniworld`, bf16 contiguous `[1, L, L, D]` input, `d_hidden = D` (or, one direction at D64 / D128,
`d_hidden = 2D`: see below), LayerNorm eps 1e-5, compute capability (10, 0), and:
- inference: L a multiple of 16 (with a dropout scale at D <= 128: a multiple of 128);
- training: D64 / D128 either direction: L a multiple of 128 (L <= 10240); D256 / D384 / D512: L a multiple of 16.
Every other shape runs the Triton path.

## Batched samples (D64, B <= 8)

The D64 native path takes B samples in one call: tokens b-major, one launch per stage instead of B calls (the consumer model code's template
embedder runs its templates as the B samples of one pair stack). `k3g` / `b1s` take a per-sample token mask and a per-sample
row-dropout scale (a tile reloads it when its sample changes), `k1w` the per-sample mask, and the plane bmm batches over d x B.
Wider D keeps one sample per call (`serves_*` refuse B > 1 there) and B > 8 is refused; the limit is
`integrations.trimul_b200.MAX_BATCH`, which a caller that folds samples into the batch asks (an engine without batched samples has no
such name, and a B > 1 call there takes the slow path). Tests:
`tests/gpu/sm100/test_b200_trimul_batch_gpu.py` compares B samples in one call with B calls of one sample, inference and
training; every sample draws its own dropout scale and mask, so an index mix-up between samples is a large error.

Measured 2026-10-02 on one B200 (sm_100a, 1000 W cap), torch 2.13.0+cu129, bf16, D64, a mask and a dropout scale per sample, CUDA-graph
timing (30 replays after warm-up); the incoming direction matches the outgoing one to within 2 %:

| L | B | direction | inference: one call / B calls (ms) | × | training fwd + bwd: one call / B calls (ms) | × |
|---|---|---|---|---|---|---|
| 128 | 2 | bidirectional | 0.031 / 0.048 | 1.54 | 0.145 / 0.234 | 1.61 |
| 128 | 2 | outgoing | 0.026 / 0.037 | 1.44 | 0.115 / 0.198 | 1.73 |
| 128 | 4 | bidirectional | 0.045 / 0.093 | 2.06 | 0.208 / 0.465 | 2.23 |
| 128 | 4 | outgoing | 0.036 / 0.073 | 2.02 | 0.152 / 0.393 | 2.58 |
| 256 | 2 | bidirectional | 0.072 / 0.091 | 1.26 | 0.292 / 0.412 | 1.41 |
| 256 | 2 | outgoing | 0.052 / 0.065 | 1.26 | 0.222 / 0.306 | 1.37 |
| 256 | 4 | bidirectional | 0.127 / 0.180 | 1.41 | 0.483 / 0.827 | 1.71 |
| 256 | 4 | outgoing | 0.089 / 0.130 | 1.45 | 0.355 / 0.615 | 1.73 |
| 384 | 2 | bidirectional | 0.145 / 0.164 | 1.13 | 0.538 / 0.656 | 1.22 |
| 384 | 2 | outgoing | 0.101 / 0.113 | 1.12 | 0.394 / 0.480 | 1.22 |
| 384 | 4 | bidirectional | 0.261 / 0.325 | 1.24 | 0.952 / 1.308 | 1.37 |
| 384 | 4 | outgoing | 0.184 / 0.224 | 1.22 | 0.701 / 0.957 | 1.37 |

## Hidden twice the width (one direction, D64 / D128)

`TriangleMultiplication(D, d_hidden=2D)` -- the AF3 / Protenix template block (pair 64, hidden 128) -- runs the D64 / D128 kernels
as they are. Per token it is the bidirectional block: k1w writes 4D planes (`a` then `b`, 2D each), k3g normalises a 2D-channel
contraction output and projects it back to D, and the backward is the same b1s / b1g -> contraction gradients -> b7m / b7g chain
with the bidirectional front-CTA count. The only difference is the cuBLAS contraction between the kernels: all 2D channels of `a`
and `b` in the module's one direction (`t = a b^T` outgoing, `a^T b` incoming) instead of D channels per direction
(`b200_infer.planes_of` / `contracted`). Batched samples (B <= 8 at D64), the row-dropout scale and CUDA graphs work as for the other
shapes; other hidden widths keep the module's other paths (`b200_infer.hidden_ok`). Tests: `tests/gpu/sm100/test_trimul_b200_gpu.py`
(`-k hidden`: every gradient against an fp32 PyTorch reference, B = 4 at D64, dropout, graph capture).

Measured 2026-10-04 on one B200 (sm_100a, 1000 W cap), torch 2.13.0+cu129, bf16, outgoing, a token mask, row dropout 0.25 in
training, CUDA-graph timing (30 replays after warm-up), ms; cuEquivariance 0.12 and the PyTorch reference are the same module with
`implementation=cuequivariance` / `pytorch`:

| D | hidden | B | L | inference: ours / cuEq / PyTorch | × cuEq | training fwd + bwd: ours / cuEq / PyTorch | × cuEq |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 1 | 128 | 0.021 / 0.047 / 0.148 | 2.24 | 0.112 / 0.232 / 0.630 | 2.07 |
| 64 | 128 | 1 | 256 | 0.039 / 0.107 / 0.442 | 2.74 | 0.186 / 0.558 / 2.084 | 3.00 |
| 64 | 128 | 1 | 384 | 0.072 / 0.203 / 0.956 | 2.82 | 0.304 / 1.042 / 2.968 | 3.43 |
| 64 | 128 | 1 | 512 | 0.117 / 0.341 / 1.710 | 2.91 | 0.459 / 1.742 / 5.132 | 3.80 |
| 64 | 128 | 1 | 768 | 0.249 / 0.749 / 4.231 | 3.01 | 0.932 / 3.787 / 12.787 | 4.06 |
| 64 | 128 | 4 | 128 | 0.039 / 0.107 / 0.441 | 2.74 | 0.187 / 0.562 / 2.057 | 3.01 |
| 64 | 128 | 4 | 256 | 0.112 / 0.339 / 1.698 | 3.03 | 0.454 / 1.739 / 4.961 | 3.83 |
| 64 | 128 | 4 | 384 | 0.235 / 0.736 / 3.803 | 3.13 | 0.914 / 3.783 / 10.974 | 4.14 |
| 64 | 128 | 4 | 512 | 0.476 / 1.280 / 6.730 | 2.69 | 1.590 / 6.551 / 19.356 | 4.12 |
| 64 | 128 | 4 | 768 | 1.242 / 2.918 / 16.800 | 2.35 | 3.827 / 14.601 / 48.832 | 3.82 |
| 128 | 256 | 1 | 128 | 0.029 / 0.066 / 0.223 | 2.28 | 0.159 / 0.353 / 0.862 | 2.22 |
| 128 | 256 | 1 | 256 | 0.068 / 0.193 / 0.720 | 2.84 | 0.311 / 0.969 / 3.280 | 3.12 |
| 128 | 256 | 1 | 384 | 0.132 / 0.407 / 1.652 | 3.08 | 0.559 / 2.025 / 5.224 | 3.62 |
| 128 | 256 | 1 | 512 | 0.222 / 0.696 / 3.061 | 3.14 | 0.896 / 3.504 / 9.441 | 3.91 |
| 128 | 256 | 1 | 768 | 0.513 / 1.555 / 7.922 | 3.03 | 1.912 / 7.607 / 26.747 | 3.98 |

## Bidirectional (`BidirectionalTriangleMultiplication`)

### Inference

#### Fused path · D64 / D128, every L

![TriMul inference, D64 / D128](figures/trimul_inference_small.png)

##### P1 · weight prep (k1w_prep)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN + proj + gate (k3g)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, every L

![TriMul inference, D256-D512](figures/trimul_inference_wide.png)

##### F0 · input LN stats (k1w_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN stats (wide_ln_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · LN-affine fold (wide_fold_prep)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · output GEMMs + gate (k3w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### Fused path · D64 / D128, L = 128 k

![TriMul training, D64 / D128](figures/trimul_training_small.png)

##### P1 · weight prep (k1w_prep)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | ✓ | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN + proj + gate, saving (k3g)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · output-side backward (b1s D64 / b1g D128)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · input-side backward (b7m D64 / b7g D128)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, every L

![TriMul training, D256-D512](figures/trimul_training_wide.png)

##### F0 · input LN stats (k1w_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN stats (wide_ln_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · LN-affine fold (wide_fold_prep)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · output GEMMs + gate, saving (k3w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate backward (wide_gate_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · output-LN backward (wide_lnout_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B5 · front backward (k1wb)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B6 · input LN apply (wide_ln_apply)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B8 · input-LN backward (wide_lnin_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Single direction (`TriangleMultiplication`)

Outgoing and incoming run the same kernels (the contraction's operand order differs); the tables
cover both.

### Inference

#### Fused path · D64 / D128, every L

![TriMul inference, D64 / D128](figures/trimul_inference_small.png)

##### P1 · weight prep (k1w_prep)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN + proj + gate (k3g)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, every L

D512 is not a registered single-direction shape; the path serves it anyway.

![TriMul inference, D256-D512](figures/trimul_inference_wide.png)

##### F0 · input LN stats (k1w_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | ✓ | ✓ | △ | △ | △ | △ | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN stats (wide_ln_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · LN-affine fold (wide_fold_prep)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · output GEMMs + gate (k3w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### Fused path · D64 / D128, L = 128 k

![TriMul training, D64 / D128](figures/trimul_training_small.png)

##### P1 · weight prep (k1w_prep)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | ✓ | ✓ | △ | △ | △ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN + proj + gate, saving (k3g)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · output-side backward (b1s D64 / b1g D128)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · input-side backward (b7m D64 / b7g D128)

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, every L

![TriMul training, D256-D512](figures/trimul_training_wide.png)

##### F0 · input LN stats (k1w_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj (k1w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN stats (wide_ln_stats)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · LN-affine fold (wide_fold_prep)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · output GEMMs + gate, saving (k3w)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate backward (wide_gate_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · output-LN backward (wide_lnout_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B5 · front backward (k1wb)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B6 · input LN apply (wide_ln_apply)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B8 · input-LN backward (wide_lnin_bwd)

| (Length, Dimension) | (128, 256) | (256, 256) | (384, 256) | (512, 256) | (640, 256) | (768, 256) | (128, 384) | (256, 384) | (384, 384) | (512, 384) | (640, 384) | (768, 384) | (128, 512) | (256, 512) | (384, 512) | (512, 512) | (640, 512) | (768, 512) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | △ | △ | △ | △ | △ | △ | ✓ | ✓ | ✓ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements

2026-09-29. Re-measured 2026-09-30: the D128 bidirectional training rows (its training moved to the shared D64 / D128
kernels) and both inference tables, all four implementations in one harness run per width. Re-verified 2026-10-10 (below).

- Setup: B200 (148 SMs), torch 2.13.0+cu129, triton 3.7.1, cuEquivariance 0.12.0, sm_100a kernels built by nvcc 13.1, B = 1, bf16-mixed (the module's
  LayerNorm parameters stay fp32).
- Harness: `benchmarks/runners/bench.py target=<module> level=module mode=<inference|training> d_pair=<D>
  min_seq_len=128 max_seq_len=768 seq_len_step=128`; compiled, inference in a CUDA graph, training with and without one.
- Tables: latency in ms (median); × = ours against the fastest of the others. Single direction = outgoing (incoming
  runs the same kernels).
- Anthropic: the harness's `anthropic` row -- Anthropic's trimul_native v5 from its unmodified sources. No binary for sm_100
  ships with the release, so the payload named by `ANTHROPIC_TRIMUL_BUILD_DIR` carries the release's sm_80 member built for
  sm_100a (`miniworld-engine dev build-anthropic-sm100a`, [anthropic-trimul-payload.md](../../../kernels/anthropic-trimul-payload.md#b200-sm_100)).
  Served where the release has a unit: one direction D64-D384, bidirectional D64 / D128 (one unit at twice the hidden width);
  bidirectional D256+ and D512 have none (—). Forward-only, so the training tables have no Anthropic column (—).
- Accuracy against the harness's fp32 reference, max over every row: inference output rel. Frobenius 3.8e-3
  (PyTorch compiled 4.1e-3, cuEquivariance 3.8e-3, Anthropic 3.8e-3); training output 4.0e-3, gradients 5.4e-3 (PyTorch compiled
  4.3e-3 / 5.4e-3, cuEquivariance 4.0e-3 / 5.5e-3).
- Re-verification (2026-10-10, engine v2.2.0 plus unreleased changes, the setup above with the kernels built by nvcc 13.1, driver 580.95.05, one GPU with no other process on it): every cell of the CUDA-graph inference and training tables of both modules (360 cells) was re-measured through the harness command above, one run per width and mode; a cell that moved by more than 8 % against the table was repeated (once, twice if still beyond 8 %) and the median of the runs is reported. Median change 0.5 % (2.0 % for ours), 90th percentile 3.3 %; one cell was replaced ((640, 512) bidirectional inference, 2.608 -> 2.896 ms). The no-CUDA-graph training tables are host-bound (their cells move by 20-60 % between runs with the load on the host) and keep their 2026-09-30 values. **2026-10-11:** every CUDA-graph cell of these tables was re-measured with fp32 master parameters (`precision=bf16-mixed`, the harness default since that date: fp32 parameters over bf16 activations under `torch.autocast`; one run per cell, `parameter_dtype` = float32 in the CSV, the engine environment's python called directly); the cells above are those values. The no-CUDA-graph training tables are host-bound and keep their earlier values (bf16 parameters).
- Charts: under each table, a length sweep at D128 and a dimension sweep at L384, drawn from the table
  (`python -m miniworld_engine.viz.measure_bars docs/gpus/b200/trimul/trimul.md`).

### Bidirectional · Inference (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.076 | 0.049 | 0.031 | 0.027 | 1.15 |
| (256, 64) | 0.182 | 0.104 | 0.061 | 0.047 | 1.30 |
| (384, 64) | 0.338 | 0.203 | 0.117 | 0.078 | 1.50 |
| (512, 64) | 0.633 | 0.344 | 0.201 | 0.125 | 1.61 |
| (640, 64) | 0.986 | 0.534 | 0.313 | 0.195 | 1.61 |
| (768, 64) | 1.632 | 0.741 | 0.424 | 0.276 | 1.54 |
| (128, 128) | 0.111 | 0.070 | 0.068 | 0.039 | 1.74 |
| (256, 128) | 0.297 | 0.193 | 0.158 | 0.078 | 2.03 |
| (384, 128) | 0.698 | 0.408 | 0.307 | 0.143 | 2.15 |
| (512, 128) | 1.090 | 0.685 | 0.517 | 0.247 | 2.09 |
| (640, 128) | 1.795 | 1.133 | 0.823 | 0.387 | 2.13 |
| (768, 128) | 2.960 | 1.590 | 1.145 | 0.549 | 2.09 |
| (128, 256) | 0.182 | 0.125 | — | 0.059 | 2.12 |
| (256, 256) | 0.628 | 0.416 | — | 0.164 | 2.54 |
| (384, 256) | 1.237 | 0.889 | — | 0.337 | 2.64 |
| (512, 256) | 2.163 | 1.645 | — | 0.622 | 2.64 |
| (640, 256) | 3.849 | 2.612 | — | 1.051 | 2.49 |
| (768, 256) | 6.433 | 3.694 | — | 1.413 | 2.61 |
| (128, 384) | 0.270 | 0.207 | — | 0.086 | 2.41 |
| (256, 384) | 0.970 | 0.735 | — | 0.280 | 2.62 |
| (384, 384) | 2.043 | 1.780 | — | 0.567 | 3.14 |
| (512, 384) | 3.852 | 3.092 | — | 1.025 | 3.02 |
| (640, 384) | 6.693 | 5.303 | — | 1.787 | 2.97 |
| (768, 384) | 11.275 | 7.442 | — | 2.508 | 2.97 |
| (128, 512) | 0.354 | 0.305 | — | 0.121 | 2.52 |
| (256, 512) | 1.181 | 1.106 | — | 0.430 | 2.57 |
| (384, 512) | 2.909 | 2.884 | — | 0.964 | 2.99 |
| (512, 512) | 5.619 | 5.165 | — | 1.707 | 3.03 |
| (640, 512) | 9.446 | 8.118 | — | 2.813 | 2.89 |
| (768, 512) | 15.450 | 11.816 | — | 4.255 | 2.78 |

![Bidirectional · Inference (CUDA graph), length sweep at D128](figures/trimul_bidirectional_inference_cuda_graph_length.png) ![Bidirectional · Inference (CUDA graph), dimension sweep at L384](figures/trimul_bidirectional_inference_cuda_graph_dimension.png) <!-- measure_bars -->

### Bidirectional · Training, dropout 0.25 (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.249 | 0.187 | — | 0.120 | 1.56 |
| (256, 64) | 0.601 | 0.411 | — | 0.194 | 2.12 |
| (384, 64) | 1.109 | 0.792 | — | 0.298 | 2.66 |
| (512, 64) | 1.857 | 1.326 | — | 0.459 | 2.89 |
| (640, 64) | 2.833 | 2.053 | — | 0.710 | 2.89 |
| (768, 64) | 4.145 | 2.854 | — | 0.955 | 2.99 |
| (128, 128) | 0.373 | 0.267 | — | 0.175 | 1.53 |
| (256, 128) | 0.994 | 0.736 | — | 0.329 | 2.24 |
| (384, 128) | 2.074 | 1.551 | — | 0.583 | 2.66 |
| (512, 128) | 3.533 | 2.641 | — | 0.963 | 2.74 |
| (640, 128) | 5.592 | 4.178 | — | 1.476 | 2.83 |
| (768, 128) | 9.106 | 5.813 | — | 2.162 | 2.69 |
| (128, 256) | 0.601 | 0.470 | — | 0.311 | 1.51 |
| (256, 256) | 1.883 | 1.497 | — | 0.781 | 1.92 |
| (384, 256) | 3.980 | 3.249 | — | 1.585 | 2.05 |
| (512, 256) | 6.989 | 5.765 | — | 2.670 | 2.16 |
| (640, 256) | 11.874 | 9.069 | — | 4.311 | 2.10 |
| (768, 256) | 20.369 | 13.053 | — | 6.142 | 2.13 |
| (128, 384) | 0.894 | 0.732 | — | 0.439 | 1.67 |
| (256, 384) | 2.983 | 2.514 | — | 1.282 | 1.96 |
| (384, 384) | 6.657 | 5.798 | — | 2.734 | 2.12 |
| (512, 384) | 13.829 | 10.450 | — | 4.730 | 2.21 |
| (640, 384) | 22.924 | 16.534 | — | 7.738 | 2.14 |
| (768, 384) | 40.627 | 23.969 | — | 10.898 | 2.20 |
| (128, 512) | 1.132 | 1.028 | — | 0.558 | 1.84 |
| (256, 512) | 3.874 | 3.664 | — | 1.824 | 2.01 |
| (384, 512) | 9.194 | 8.548 | — | 3.900 | 2.19 |
| (512, 512) | 19.480 | 15.409 | — | 6.785 | 2.27 |
| (640, 512) | 31.857 | 24.461 | — | 11.168 | 2.19 |
| (768, 512) | 53.085 | 36.718 | — | 16.478 | 2.23 |

![Bidirectional · Training, dropout 0.25 (CUDA graph), length sweep at D128](figures/trimul_bidirectional_training_dropout_0_25_cuda_graph_length.png) ![Bidirectional · Training, dropout 0.25 (CUDA graph), dimension sweep at L384](figures/trimul_bidirectional_training_dropout_0_25_cuda_graph_dimension.png) <!-- measure_bars -->

### Bidirectional · Training, dropout 0.25 (no CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.586 | 0.829 | — | 0.758 | 0.77 |
| (256, 64) | 0.697 | 1.027 | — | 0.625 | 1.12 |
| (384, 64) | 1.211 | 0.887 | — | 0.607 | 1.46 |
| (512, 64) | 1.957 | 1.427 | — | 0.775 | 1.84 |
| (640, 64) | 2.934 | 2.145 | — | 0.759 | 2.83 |
| (768, 64) | 4.280 | 2.949 | — | 1.001 | 2.95 |
| (128, 128) | 0.585 | 1.011 | — | 0.639 | 0.92 |
| (256, 128) | 1.090 | 0.836 | — | 0.774 | 1.08 |
| (384, 128) | 2.172 | 1.647 | — | 0.803 | 2.05 |
| (512, 128) | 3.625 | 2.737 | — | 1.045 | 2.62 |
| (640, 128) | 5.696 | 4.241 | — | 1.606 | 2.64 |
| (768, 128) | 9.157 | 5.929 | — | 2.302 | 2.58 |
| (128, 256) | 0.690 | 0.843 | — | 0.870 | 0.79 |
| (256, 256) | 1.984 | 1.591 | — | 0.876 | 1.82 |
| (384, 256) | 4.089 | 3.329 | — | 1.621 | 2.05 |
| (512, 256) | 7.097 | 5.729 | — | 2.747 | 2.09 |
| (640, 256) | 11.922 | 9.175 | — | 4.320 | 2.12 |
| (768, 256) | 20.287 | 13.294 | — | 6.182 | 2.15 |
| (128, 384) | 0.995 | 1.061 | — | 0.892 | 1.12 |
| (256, 384) | 3.080 | 2.605 | — | 1.334 | 1.95 |
| (384, 384) | 6.721 | 5.957 | — | 2.664 | 2.24 |
| (512, 384) | 14.039 | 10.370 | — | 4.762 | 2.18 |
| (640, 384) | 23.129 | 16.697 | — | 7.780 | 2.15 |
| (768, 384) | 39.144 | 24.135 | — | 11.161 | 2.16 |
| (128, 512) | 1.219 | 1.113 | — | 0.876 | 1.27 |
| (256, 512) | 3.987 | 3.771 | — | 1.939 | 1.95 |
| (384, 512) | 9.280 | 8.819 | — | 3.791 | 2.33 |
| (512, 512) | 19.756 | 15.581 | — | 7.225 | 2.16 |
| (640, 512) | 32.228 | 25.294 | — | 11.586 | 2.18 |
| (768, 512) | 53.520 | 35.464 | — | 16.116 | 2.20 |

![Bidirectional · Training, dropout 0.25 (no CUDA graph), length sweep at D128](figures/trimul_bidirectional_training_dropout_0_25_no_cuda_graph_length.png) ![Bidirectional · Training, dropout 0.25 (no CUDA graph), dimension sweep at L384](figures/trimul_bidirectional_training_dropout_0_25_no_cuda_graph_dimension.png) <!-- measure_bars -->

### Single direction (outgoing) · Inference (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.057 | 0.037 | 0.022 | 0.022 | 1.00 |
| (256, 64) | 0.125 | 0.072 | 0.041 | 0.035 | 1.17 |
| (384, 64) | 0.207 | 0.129 | 0.070 | 0.054 | 1.30 |
| (512, 64) | 0.344 | 0.215 | 0.119 | 0.084 | 1.42 |
| (640, 64) | 0.536 | 0.332 | 0.188 | 0.127 | 1.48 |
| (768, 64) | 0.792 | 0.457 | 0.254 | 0.166 | 1.53 |
| (128, 128) | 0.078 | 0.047 | 0.039 | 0.030 | 1.30 |
| (256, 128) | 0.176 | 0.092 | 0.088 | 0.053 | 1.66 |
| (384, 128) | 0.391 | 0.180 | 0.170 | 0.094 | 1.81 |
| (512, 128) | 0.613 | 0.313 | 0.294 | 0.153 | 1.92 |
| (640, 128) | 1.060 | 0.487 | 0.467 | 0.248 | 1.88 |
| (768, 128) | 1.690 | 0.673 | 0.647 | 0.358 | 1.81 |
| (128, 256) | 0.117 | 0.063 | 0.082 | 0.045 | 1.40 |
| (256, 256) | 0.342 | 0.166 | 0.254 | 0.110 | 1.51 |
| (384, 256) | 0.739 | 0.268 | 0.512 | 0.231 | 1.16 |
| (512, 256) | 1.183 | 0.459 | 0.878 | 0.399 | 1.15 |
| (640, 256) | 1.928 | 0.716 | 1.391 | 0.665 | 1.08 |
| (768, 256) | 3.163 | 1.021 | 1.957 | 0.911 | 1.12 |
| (128, 384) | 0.158 | 0.151 | 0.180 | 0.063 | 2.40 |
| (256, 384) | 0.534 | 0.540 | 0.579 | 0.184 | 2.90 |
| (384, 384) | 1.171 | 1.254 | 1.260 | 0.403 | 2.91 |
| (512, 384) | 1.922 | 2.285 | 2.180 | 0.706 | 2.72 |
| (640, 384) | 3.201 | 3.692 | 3.437 | 1.197 | 2.67 |
| (768, 384) | 5.190 | 5.341 | 4.868 | 1.752 | 2.78 |
| (128, 512) | 0.199 | 0.219 | — | 0.084 | 2.37 |
| (256, 512) | 0.694 | 0.854 | — | 0.258 | 2.69 |
| (384, 512) | 1.413 | 2.086 | — | 0.571 | 2.47 |
| (512, 512) | 2.534 | 3.646 | — | 1.022 | 2.48 |
| (640, 512) | 4.418 | 5.619 | — | 1.728 | 2.56 |
| (768, 512) | 7.290 | 8.368 | — | 2.423 | 3.01 |

![Single direction (outgoing) · Inference (CUDA graph), length sweep at D128](figures/trimul_single_direction_outgoing_inference_cuda_graph_length.png) ![Single direction (outgoing) · Inference (CUDA graph), dimension sweep at L384](figures/trimul_single_direction_outgoing_inference_cuda_graph_dimension.png) <!-- measure_bars -->

### Single direction (outgoing) · Training, dropout 0.25 (CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.189 | 0.148 | — | 0.099 | 1.49 |
| (256, 64) | 0.401 | 0.300 | — | 0.142 | 2.11 |
| (384, 64) | 0.708 | 0.533 | — | 0.218 | 2.44 |
| (512, 64) | 1.166 | 0.870 | — | 0.358 | 2.43 |
| (640, 64) | 1.775 | 1.310 | — | 0.494 | 2.65 |
| (768, 64) | 2.547 | 1.813 | — | 0.648 | 2.80 |
| (128, 128) | 0.257 | 0.200 | — | 0.132 | 1.52 |
| (256, 128) | 0.617 | 0.458 | — | 0.235 | 1.95 |
| (384, 128) | 1.236 | 0.979 | — | 0.407 | 2.41 |
| (512, 128) | 2.082 | 1.649 | — | 0.647 | 2.55 |
| (640, 128) | 3.342 | 2.581 | — | 1.070 | 2.41 |
| (768, 128) | 5.067 | 3.570 | — | 1.445 | 2.47 |
| (128, 256) | 0.374 | 0.302 | — | 0.247 | 1.22 |
| (256, 256) | 1.085 | 0.812 | — | 0.554 | 1.47 |
| (384, 256) | 2.270 | 1.654 | — | 1.055 | 1.57 |
| (512, 256) | 3.885 | 2.801 | — | 1.730 | 1.62 |
| (640, 256) | 6.239 | 4.475 | — | 2.794 | 1.60 |
| (768, 256) | 10.179 | 6.758 | — | 3.867 | 1.75 |
| (128, 384) | 0.511 | 0.536 | — | 0.329 | 1.55 |
| (256, 384) | 1.638 | 1.891 | — | 0.842 | 1.95 |
| (384, 384) | 3.476 | 4.196 | — | 1.741 | 2.00 |
| (512, 384) | 6.170 | 7.440 | — | 2.987 | 2.07 |
| (640, 384) | 9.780 | 11.647 | — | 4.793 | 2.04 |
| (768, 384) | 16.388 | 16.410 | — | 6.752 | 2.43 |
| (128, 512) | 0.644 | 0.740 | — | 0.409 | 1.57 |
| (256, 512) | 2.119 | 2.786 | — | 1.179 | 1.80 |
| (384, 512) | 4.600 | 6.326 | — | 2.437 | 1.89 |
| (512, 512) | 8.314 | 11.416 | — | 4.320 | 1.92 |
| (640, 512) | 13.911 | 18.321 | — | 6.815 | 2.04 |
| (768, 512) | 23.424 | 26.455 | — | 9.959 | 2.35 |

![Single direction (outgoing) · Training, dropout 0.25 (CUDA graph), length sweep at D128](figures/trimul_single_direction_outgoing_training_dropout_0_25_cuda_graph_length.png) ![Single direction (outgoing) · Training, dropout 0.25 (CUDA graph), dimension sweep at L384](figures/trimul_single_direction_outgoing_training_dropout_0_25_cuda_graph_dimension.png) <!-- measure_bars -->

### Single direction (outgoing) · Training, dropout 0.25 (no CUDA graph)

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 64) | 0.549 | 0.907 | — | 0.589 | 0.93 |
| (256, 64) | 0.494 | 0.911 | — | 0.583 | 0.85 |
| (384, 64) | 0.792 | 1.131 | — | 0.713 | 1.11 |
| (512, 64) | 1.252 | 1.110 | — | 0.703 | 1.58 |
| (640, 64) | 1.861 | 1.402 | — | 0.731 | 1.92 |
| (768, 64) | 2.657 | 1.902 | — | 0.753 | 2.53 |
| (128, 128) | 0.460 | 1.161 | — | 0.600 | 0.77 |
| (256, 128) | 0.702 | 1.132 | — | 0.740 | 0.95 |
| (384, 128) | 1.323 | 1.324 | — | 0.847 | 1.56 |
| (512, 128) | 2.163 | 1.728 | — | 0.887 | 1.95 |
| (640, 128) | 3.437 | 2.659 | — | 1.034 | 2.57 |
| (768, 128) | 5.147 | 3.653 | — | 1.421 | 2.57 |
| (128, 256) | 0.612 | 0.598 | — | 0.877 | 0.68 |
| (256, 256) | 1.167 | 0.910 | — | 0.866 | 1.05 |
| (384, 256) | 2.350 | 1.748 | — | 1.125 | 1.55 |
| (512, 256) | 3.977 | 2.906 | — | 1.815 | 1.60 |
| (640, 256) | 6.344 | 4.563 | — | 2.895 | 1.58 |
| (768, 256) | 10.245 | 6.958 | — | 3.875 | 1.80 |
| (128, 384) | 0.603 | 1.102 | — | 0.835 | 0.72 |
| (256, 384) | 1.719 | 1.971 | — | 1.007 | 1.71 |
| (384, 384) | 3.560 | 4.286 | — | 1.797 | 1.98 |
| (512, 384) | 6.233 | 7.386 | — | 3.069 | 2.03 |
| (640, 384) | 9.977 | 12.010 | — | 4.848 | 2.06 |
| (768, 384) | 16.378 | 16.555 | — | 6.851 | 2.39 |
| (128, 512) | 0.734 | 1.135 | — | 0.826 | 0.89 |
| (256, 512) | 2.215 | 2.850 | — | 1.220 | 1.82 |
| (384, 512) | 4.609 | 6.391 | — | 2.572 | 1.79 |
| (512, 512) | 8.283 | 11.480 | — | 4.045 | 2.05 |
| (640, 512) | 14.046 | 18.193 | — | 7.052 | 1.99 |
| (768, 512) | 22.818 | 25.956 | — | 9.548 | 2.39 |

![Single direction (outgoing) · Training, dropout 0.25 (no CUDA graph), length sweep at D128](figures/trimul_single_direction_outgoing_training_dropout_0_25_no_cuda_graph_length.png) ![Single direction (outgoing) · Training, dropout 0.25 (no CUDA graph), dimension sweep at L384](figures/trimul_single_direction_outgoing_training_dropout_0_25_no_cuda_graph_dimension.png) <!-- measure_bars -->

Without a graph, ours costs about 0.6-0.9 ms per training step at L128 (and D64 / D128 one
direction at L256) whatever the width: host-side time (tensor-map encoding, custom-op dispatch,
weight packs, small launches) above the GPU work. Those are the rows where ours loses (× < 1, to
PyTorch compiled or cuEquivariance); from L256-L384 on the GPU work dominates. Reducing the host
path is open.

### Hardware limit (SoL) per kernel

2026-09-30, time roofline (no power readings). Ceilings: HBM 6.75 TB/s (the highest of an elementwise read + write kernel
measured next to every run; 5.8-6.75 run to run) and tensor 2.23 PF/s (resident-operand bf16 MMA; spec 2.25). The sustained
cuBLAS 8192^3 GEMM reaches only 1.42-1.47 PF/s under the card's 1000 W cap and is not the limit: k1w runs above it. Floor =
max(minimum HBM bytes / BW, FLOPs / tensor rate) from each kernel's inputs, outputs and GEMM work; SoL = floor / measured, the
median over 40 CUDA-graph replays of the module step (bf16, masked, dropout 0.25 in training), both modules, every width of the
path. The weight-side kernels (k1w_prep, wide_fold_prep: 2-3 µs, fixed) have no data floor. Computed by a script that is not shipped.

| path | kernel | bound | L128 | L256 | L384-L768 |
|---|---|---|---|---|---|
| D64 / D128 | k1w | HBM | 18-33 % | 38-55 % | 53-81 % |
| D64 / D128 | k3g (inference / saving) | HBM | 15-23 % | 30-35 % | 39-53 % |
| D64 / D128 | b1s (D64) | HBM | 14-16 % | 29-29 % | 33-40 % |
| D64 / D128 | b1g (D128) | HBM | 17-17 % | 29-32 % | 35-42 % |
| D64 / D128 | b7m (D64) | HBM | 15-17 % | 33-39 % | 44-57 % |
| D64 / D128 | b7g (D128) | HBM | 18-21 % | 26-32 % | 28-43 % |
| D256-D512 | k1w_stats | HBM | 36-59 % | 44-68 % | 48-89 % |
| D256-D512 | k1w | tensor | 34-66 % | 53-70 % | 65-80 % |
| D256-D512 | wide_ln_stats | HBM | 38-56 % | 71-82 % | 73-92 % |
| D256-D512 | k3w (inference / saving) | HBM / tensor | 26-37 % | 32-47 % | 34-50 % |
| D256-D512 | wide_gate_bwd | HBM | 13-24 % | 33-45 % | 45-65 % |
| D256-D512 | wide_lnout_bwd | HBM | 44-55 % | 53-61 % | 56-65 % |
| D256-D512 | k1wb | HBM | 46-53 % | 44-56 % | 52-64 % |
| D256-D512 | wide_ln_apply | HBM | 46-57 % | 68-79 % | 81-93 % |
| D256-D512 | wide_lnin_bwd | HBM | 39-53 % | 51-68 % | 56-87 % |

성능 확인 (kernel tables above, 2026-09-30): ✓ = the fastest measured and SoL >= 70 % (the weight-side prep kernels: ✓);
△ = the fastest measured, SoL below 70 %; ✗ = slower than another implementation (none in the current tables; one direction,
D64, L128 inference ties Anthropic v5 at 22.46 µs, 1.00×). A module cell in [../b200.md](../b200.md) takes the lowest of its
kernels.
