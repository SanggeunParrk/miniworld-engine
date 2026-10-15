# TriangleMultiplication on H100 (sm90)

Kernel-level status of the TriMul modules on H100; the module-level summary is in
[h100.md](../h100.md). bf16 only; columns are (Length, Dimension). H100 = CUDA; a shape
without a CUDA path is 미구현. Figures: one box per kernel, left to right, HBM reads (blue,
left) and writes (red, right); generated from `figures/trimul_bidir.json` by
`python -m miniworld_engine.viz.kernel_flow`, then `rsvg-convert -z 2 -b white` to PNG.
Dispatch: `integrations/trimul_h100.py`.

## Bidirectional (`BidirectionalTriangleMultiplication`)

Outgoing + incoming in one module (h = D per direction), one output LayerNorm over 2h.

### Inference

#### K1/K3 path · D64 / D128, every L

![TriMul bidirectional inference, K1/K3 path](figures/trimul_bidir_inference_k1k3.png)

##### F1 · input LN + gated proj

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

#### Wide path · D256 / D384 / D512, L384 / L768

![TriMul bidirectional inference, wide path](figures/trimul_bidir_inference_wide.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | Triton | Triton | Triton | Triton | Triton | Triton |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · LN-affine fold

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | Triton | Triton | Triton | Triton | Triton | Triton |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · output LN + proj + gate

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### D64 path · L384 / L768

![TriMul bidirectional training, D64](figures/trimul_bidir_training_d64.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | Triton | Triton |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B1 · output-side backward

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B3 · input-side backward

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B4 · weight-grad reduce

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

#### D128 path · L384 / L768

![TriMul bidirectional training, D128](figures/trimul_bidir_training_d128.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | Triton | Triton |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B1 · output-side backward

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B3 · input-side backward

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, L384 / L768

![TriMul bidirectional training, wide path](figures/trimul_bidir_training_wide.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | Triton | Triton | Triton | Triton | Triton | Triton |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · gate + residual

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate backward

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · output-LN backward

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · contraction grads

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | cuBLAS + CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B7 · input-LN backward + reduce

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


## Single direction (`TriangleMultiplication`, outgoing / incoming)

One contraction, hidden = D, bf16 only. Both directions run the same kernels (`GP_DIR` / the transposed contraction operand selects the
direction). Kernel level, per width:

| Width | Inference | Training |
|---|---|---|
| D64 | K1 → cuBLAS contraction → K3 (`h100_inference`, packaged tile table) | `h100_uni_d64_b200gate` / `h100_uni_d64_training` |
| D128 | K1 → cuBLAS → K3 (tile table) | fused B1, two cuBLAS gradient contractions, streamed producer-consumer B7 (`h100_single*`) |
| D256 | tmn K1 / K3 tiles at every L except 384; B200-style port at L384 | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D384 | B200-style port (`mw_b200style_k1_direct`, cuBLAS, `mw_b200style_k3_static`) | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D512 | B200-style port; D512 L384 / L768 may use `h100_uni_wide_inference` | `h100_uni_wide_training`: `ln_in_d512` → `front_pre_ss` (wgmma A / B from shared memory) → cuBLAS contraction → output LN → projection / gate GEMMs → output epilogue; backward gate → dn GEMM → output-LN backward → `contract_gp` → weight GEMMs → input-LN backward |

D512 training at L768 (outgoing, ms per kernel): front 2.74 + LN 0.43, `contract_gp` 2.50 (78 % of DRAM), two weight / dgrad GEMMs
1.74 + 1.76 (cuBLAS, about 70 % of the tensor peak), output-LN backward 1.24, gate 1.09, input-LN backward 0.84, output epilogue 0.82,
forward LN-out 0.47: 17.1 ms in total. Kernel flow figures for the single direction are not generated yet.

Weight pack, LN-affine fold and input LN use the CUDA aux kernels whenever the call runs eagerly (every caller sits inside an opaque
custom op); the Triton versions only serve compiled regions. No Triton kernel remains in any D256 / D384 / D512 cell at L384 / L768.

## Measurements (2026-10-13)

H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, B=1, bf16 activations, FP32 master parameters, `benchmarks/runners/bench.py` (compiled,
manual CUDA graph), default dispatch (`trimul_h100_b200style` on, per-cell fallback to the preserved path at L384 / 768; see
[the record](../../../records/trimul_h100_dispatch_selection_20261013.md)). Latency in ms, median; training = forward + backward
with dropout 0.25, inference dropout 0, mask probability 0.1. × = fastest other / ours. PyTorch compiled is far behind and omitted;
the Anthropic payload (inference, D64–D384 single direction, D64 / D128 bidirectional) is compared in the
[inference records](../../../records/trimul_h100_inference_master_matched_20261012.md) (1.01–1.27x) and, for L128 – L1024, in
[the 2026-10-15 matrix](../../../records/trimul_h100_comparison_20261015.md) (1.006–1.41x).

### Bidirectional · Inference

| (Length, Dimension) | Triton | DTv1 | cuEquivariance | ours | × (fastest other / ours) |
|---|---|---|---|---|---|
| (128, 64) | 0.039 | 0.053 | 0.049 | **0.029** | 1.31 |
| (256, 64) | 0.090 | 0.135 | 0.118 | **0.061** | 1.48 |
| (384, 64) | 0.182 | 0.296 | 0.268 | **0.120** | 1.52 |
| (512, 64) | 0.310 | 0.511 | 0.485 | **0.210** | 1.47 |
| (640, 64) | 0.469 | 0.793 | 0.745 | **0.326** | 1.44 |
| (768, 64) | 0.663 | 1.115 | 1.064 | **0.466** | 1.42 |
| (128, 128) | 0.064 | 0.082 | 0.080 | **0.042** | 1.54 |
| (256, 128) | 0.204 | 0.289 | 0.326 | **0.126** | 1.61 |
| (384, 128) | 0.432 | 0.627 | 0.523 | **0.244** | 1.77 |
| (512, 128) | 0.733 | 1.080 | 0.891 | **0.417** | 1.76 |
| (640, 128) | 1.150 | 1.694 | 1.380 | **0.662** | 1.74 |
| (768, 128) | 1.649 | 2.421 | 1.985 | **0.956** | 1.72 |
| (128, 256) | 0.142 | 0.164 | 0.169 | **0.083** | 1.72 |
| (256, 256) | 0.498 | 0.605 | 0.680 | **0.281** | 1.77 |
| (384, 256) | 1.104 | 1.289 | 1.678 | **0.595** | 1.86 |
| (512, 256) | 1.898 | 2.261 | 2.954 | **1.051** | 1.81 |
| (640, 256) | 3.030 | 3.776 | 4.636 | **1.681** | 1.80 |
| (768, 256) | 4.394 | 5.320 | 6.686 | **2.466** | 1.78 |
| (128, 384) | 0.266 | 0.277 | 0.307 | **0.150** | 1.77 |
| (256, 384) | 0.909 | 1.001 | 1.557 | **0.516** | 1.76 |
| (384, 384) | 2.092 | 2.201 | 3.492 | **1.100** | 1.90 |
| (512, 384) | 3.650 | 3.847 | 6.169 | **2.019** | 1.81 |
| (640, 384) | 5.811 | 6.247 | 9.721 | **3.319** | 1.75 |
| (768, 384) | 8.403 | 8.989 | 14.014 | **4.791** | 1.75 |
| (128, 512) | 0.434 | 0.398 | 0.472 | **0.298** | 1.34 |
| (256, 512) | 1.591 | 1.456 | 2.632 | **0.896** | 1.62 |
| (384, 512) | 3.665 | 3.211 | 5.895 | **1.873** | 1.71 |
| (512, 512) | 6.553 | 5.839 | 10.467 | **3.462** | 1.69 |
| (640, 512) | 10.584 | 9.161 | 16.422 | **5.514** | 1.66 |
| (768, 512) | 15.350 | 13.405 | 23.744 | **8.065** | 1.66 |

### Bidirectional · Training

| (Length, Dimension) | Triton | DTv1 | cuEquivariance | ours | × (fastest other / ours) |
|---|---|---|---|---|---|
| (128, 64) | 0.152 | 0.177 | 0.195 | **0.103** | 1.47 |
| (256, 64) | 0.393 | 0.499 | 0.525 | **0.204** | 1.93 |
| (384, 64) | 0.817 | 1.066 | 1.139 | **0.392** | 2.09 |
| (512, 64) | 1.368 | 1.823 | 1.954 | **0.661** | 2.07 |
| (640, 64) | 2.132 | 2.804 | 2.990 | **1.009** | 2.11 |
| (768, 64) | 3.018 | 3.977 | 4.249 | **1.482** | 2.04 |
| (128, 128) | 0.245 | 0.286 | 0.318 | **0.207** | 1.18 |
| (256, 128) | 0.789 | 0.992 | 1.107 | **0.524** | 1.51 |
| (384, 128) | 1.629 | 2.119 | 2.311 | **1.006** | 1.62 |
| (512, 128) | 2.797 | 3.632 | 3.933 | **1.737** | 1.61 |
| (640, 128) | 4.562 | 5.646 | 6.173 | **2.828** | 1.61 |
| (768, 128) | 6.549 | 8.117 | 8.740 | **3.972** | 1.65 |
| (128, 256) | 0.485 | 0.536 | 0.626 | **0.432** | 1.12 |
| (256, 256) | 1.698 | 1.954 | 2.444 | **1.367** | 1.24 |
| (384, 256) | 3.722 | 4.194 | 5.579 | **2.675** | 1.39 |
| (512, 256) | 6.552 | 7.379 | 9.657 | **4.948** | 1.32 |
| (640, 256) | 11.287 | 11.649 | 15.102 | **7.729** | 1.46 |
| (768, 256) | 16.329 | 16.646 | 21.746 | **11.070** | 1.48 |
| (128, 384) | 0.843 | 0.869 | 1.066 | **0.706** | 1.19 |
| (256, 384) | 2.953 | 3.066 | 4.577 | **2.284** | 1.29 |
| (384, 384) | 6.488 | 6.768 | 10.171 | **4.590** | 1.41 |
| (512, 384) | 11.536 | 11.886 | 17.881 | **8.723** | 1.32 |
| (640, 384) | 19.496 | 18.798 | 28.127 | **13.753** | 1.37 |
| (768, 384) | 28.187 | 27.077 | 40.850 | **19.007** | 1.42 |
| (128, 512) | 1.232 | 1.207 | 1.528 | **1.100** | 1.10 |
| (256, 512) | 4.421 | 4.425 | 7.328 | **3.632** | 1.22 |
| (384, 512) | 9.928 | 9.778 | 16.277 | **7.248** | 1.35 |
| (512, 512) | 17.619 | 17.303 | 28.750 | **14.121** | 1.23 |
| (640, 512) | 30.197 | 27.012 | 45.269 | **22.117** | 1.22 |
| (768, 512) | 43.707 | 40.160 | 66.108 | **29.384** | 1.37 |

## Measurements (2026-10-15): single direction, incoming and bidirectional refreshed

Same setup as above (B=1, bf16, FP32 masters, compiled, manual CUDA graph, L = 128 ... 1024 in steps of 128, D = 64 ... 512). Cells show
× = fastest other implementation / ours, with that implementation in parentheses; the others are PyTorch, cuEquivariance and DTv1. Absolute
latencies, the Anthropic payload (same-process alternating A/B, D64 ... D384 single direction and D64 / D128 bidirectional, ours faster in
all 48 cells, thinnest 1.006x at D256 L768) and the Triton rows are in
[the record](../../../records/trimul_h100_comparison_20261015.md). **Ours is the fastest in all 240 cells** (smallest 1.10x).
The D256 single-direction inference cells were measured before the 2026-10-15 selection change (0.7 - 2.5 % faster now).

### Single direction, outgoing · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.89 (DTv1) | 1.21 (DTv1) | 1.18 (DTv1) | 1.23 (DTv1) | 1.13 (DTv1) |
| 256 | 2.26 (DTv1) | 1.66 (DTv1) | 1.43 (DTv1) | 1.38 (DTv1) | 1.27 (DTv1) |
| 384 | 2.60 (DTv1) | 1.81 (DTv1) | 1.55 (DTv1) | 1.52 (DTv1) | 1.41 (DTv1) |
| 512 | 2.69 (DTv1) | 1.84 (DTv1) | 1.47 (DTv1) | 1.41 (DTv1) | 1.27 (DTv1) |
| 640 | 2.68 (DTv1) | 1.85 (DTv1) | 1.49 (DTv1) | 1.41 (DTv1) | 1.27 (DTv1) |
| 768 | 2.65 (DTv1) | 1.85 (DTv1) | 1.54 (DTv1) | 1.48 (DTv1) | 1.36 (DTv1) |
| 896 | 2.64 (DTv1) | 1.78 (DTv1) | 1.49 (DTv1) | 1.39 (DTv1) | 1.28 (DTv1) |
| 1024 | 2.47 (DTv1) | 1.80 (DTv1) | 1.49 (DTv1) | 1.40 (DTv1) | 1.27 (DTv1) |

### Single direction, outgoing · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.61 (cuEquivariance) | 1.64 (DTv1) | 1.77 (cuEquivariance) | 2.02 (DTv1) | 1.49 (DTv1) |
| 256 | 1.66 (cuEquivariance) | 1.85 (cuEquivariance) | 1.96 (cuEquivariance) | 2.11 (DTv1) | 1.75 (DTv1) |
| 384 | 1.94 (cuEquivariance) | 1.92 (cuEquivariance) | 2.06 (cuEquivariance) | 2.16 (DTv1) | 1.80 (DTv1) |
| 512 | 2.05 (cuEquivariance) | 1.98 (cuEquivariance) | 2.01 (cuEquivariance) | 2.07 (DTv1) | 1.84 (DTv1) |
| 640 | 2.03 (cuEquivariance) | 1.99 (cuEquivariance) | 2.05 (cuEquivariance) | 2.19 (DTv1) | 1.78 (DTv1) |
| 768 | 2.06 (cuEquivariance) | 1.97 (cuEquivariance) | 2.05 (cuEquivariance) | 2.25 (DTv1) | 1.81 (DTv1) |
| 896 | 2.09 (cuEquivariance) | 1.96 (cuEquivariance) | 1.97 (cuEquivariance) | 2.01 (DTv1) | 1.75 (DTv1) |
| 1024 | 1.98 (cuEquivariance) | 1.92 (cuEquivariance) | 1.96 (cuEquivariance) | 1.97 (DTv1) | 1.74 (DTv1) |

### Single direction, incoming · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.88 (DTv1) | 1.21 (DTv1) | 1.18 (DTv1) | 1.23 (DTv1) | 1.13 (DTv1) |
| 256 | 2.28 (DTv1) | 1.66 (DTv1) | 1.42 (DTv1) | 1.38 (DTv1) | 1.26 (DTv1) |
| 384 | 2.59 (DTv1) | 1.80 (DTv1) | 1.55 (DTv1) | 1.51 (DTv1) | 1.42 (DTv1) |
| 512 | 2.69 (DTv1) | 1.85 (DTv1) | 1.47 (DTv1) | 1.39 (DTv1) | 1.28 (DTv1) |
| 640 | 2.68 (DTv1) | 1.84 (DTv1) | 1.50 (DTv1) | 1.41 (DTv1) | 1.28 (DTv1) |
| 768 | 2.65 (DTv1) | 1.85 (DTv1) | 1.53 (DTv1) | 1.46 (DTv1) | 1.35 (DTv1) |
| 896 | 2.61 (DTv1) | 1.82 (DTv1) | 1.48 (DTv1) | 1.40 (DTv1) | 1.26 (DTv1) |
| 1024 | 2.51 (DTv1) | 1.80 (DTv1) | 1.50 (DTv1) | 1.40 (DTv1) | 1.27 (DTv1) |

### Single direction, incoming · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.63 (cuEquivariance) | 1.61 (DTv1) | 1.79 (cuEquivariance) | 2.00 (DTv1) | 1.49 (DTv1) |
| 256 | 1.67 (cuEquivariance) | 1.83 (cuEquivariance) | 1.97 (cuEquivariance) | 2.09 (DTv1) | 1.76 (DTv1) |
| 384 | 1.90 (cuEquivariance) | 1.92 (cuEquivariance) | 2.05 (cuEquivariance) | 2.16 (DTv1) | 1.78 (DTv1) |
| 512 | 2.04 (cuEquivariance) | 1.99 (cuEquivariance) | 2.03 (cuEquivariance) | 2.08 (DTv1) | 1.78 (DTv1) |
| 640 | 2.03 (cuEquivariance) | 2.00 (cuEquivariance) | 2.01 (cuEquivariance) | 2.11 (DTv1) | 1.83 (DTv1) |
| 768 | 2.03 (cuEquivariance) | 1.95 (cuEquivariance) | 1.98 (cuEquivariance) | 2.04 (DTv1) | 1.83 (DTv1) |
| 896 | 2.07 (cuEquivariance) | 1.92 (cuEquivariance) | 1.96 (cuEquivariance) | 2.01 (DTv1) | 1.75 (DTv1) |
| 1024 | 2.00 (cuEquivariance) | 1.91 (cuEquivariance) | 2.04 (cuEquivariance) | 2.01 (DTv1) | 1.76 (DTv1) |

### Bidirectional · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.72 (DTv1) | 1.45 (DTv1) | 1.24 (DTv1) | 1.23 (DTv1) | 1.10 (DTv1) |
| 256 | 2.46 (DTv1) | 2.05 (DTv1) | 1.43 (DTv1) | 1.34 (DTv1) | 1.22 (DTv1) |
| 384 | 2.73 (DTv1) | 2.24 (DTv1) | 1.59 (DTv1) | 1.47 (DTv1) | 1.39 (DTv1) |
| 512 | 2.85 (DTv1) | 2.19 (DTv1) | 1.49 (DTv1) | 1.37 (DTv1) | 1.23 (DTv1) |
| 640 | 2.82 (DTv1) | 2.17 (DTv1) | 1.49 (DTv1) | 1.36 (DTv1) | 1.22 (DTv1) |
| 768 | 2.73 (DTv1) | 2.10 (DTv1) | 1.52 (DTv1) | 1.44 (DTv1) | 1.38 (DTv1) |
| 896 | 2.77 (DTv1) | 1.95 (DTv1) | 1.48 (DTv1) | 1.36 (DTv1) | 1.21 (DTv1) |
| 1024 | 2.69 (DTv1) | 1.94 (DTv1) | 1.49 (DTv1) | 1.23 (DTv1) | 1.24 (DTv1) |

### Bidirectional · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.68 (cuEquivariance) | 1.90 (cuEquivariance) | 2.01 (DTv1) | 1.87 (DTv1) | 1.34 (DTv1) |
| 256 | 1.94 (cuEquivariance) | 2.38 (DTv1) | 2.16 (DTv1) | 1.95 (DTv1) | 1.67 (DTv1) |
| 384 | 2.23 (cuEquivariance) | 2.19 (cuEquivariance) | 2.21 (DTv1) | 1.98 (DTv1) | 1.72 (DTv1) |
| 512 | 2.37 (cuEquivariance) | 2.16 (cuEquivariance) | 2.21 (DTv1) | 1.88 (DTv1) | 1.66 (DTv1) |
| 640 | 2.32 (cuEquivariance) | 2.13 (cuEquivariance) | 2.31 (DTv1) | 1.96 (DTv1) | 1.67 (DTv1) |
| 768 | 2.31 (cuEquivariance) | 2.00 (cuEquivariance) | 2.20 (DTv1) | 1.96 (DTv1) | 1.61 (DTv1) |
| 896 | 2.33 (cuEquivariance) | 1.96 (cuEquivariance) | 2.02 (DTv1) | 1.81 (DTv1) | 1.58 (DTv1) |
| 1024 | 2.19 (cuEquivariance) | 1.93 (cuEquivariance) | 2.06 (DTv1) | 1.86 (DTv1) | 1.64 (DTv1) |
