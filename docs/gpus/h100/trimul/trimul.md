# TriangleMultiplication on H100 (sm90)

Kernel-level status of the TriMul modules on H100; the module-level summary is in
[h100.md](../h100.md). bf16 only; columns are (Length, Dimension). H100 = CUDA; a shape
without a CUDA path is 미구현 (it runs the PyTorch reference: no Triton fallback on H100, see [dispatch.md](../dispatch.md)). Figures: one box per kernel, left to right, HBM reads (blue,
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
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
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
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
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
| implementation | CUDA | CUDA |
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
| implementation | CUDA | CUDA |
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
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
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
| D64 | K1 → cuBLAS contraction → K3 (`h100_inference`, packaged tile table) | `h100_uni_d64_b200gate` |
| D128 | K1 → cuBLAS → K3 (tile table) | fused B1, two cuBLAS gradient contractions, streamed producer-consumer B7 (`h100_single*`) |
| D256 | tmn K1 / K3 tiles at every L except 384; B200-style port at L384 | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D384 | B200-style port (`mw_b200style_k1_direct`, cuBLAS, `mw_b200style_k3_static`) | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D512 | B200-style port; D512 L384 / L768 may use `h100_uni_wide_inference` | `h100_uni_wide_training`: `ln_in_d512` → `front_pre_ss` (wgmma A / B from shared memory) → cuBLAS contraction → output LN → projection / gate GEMMs → output epilogue; backward gate → dn GEMM → output-LN backward → `contract_gp` → weight GEMMs → input-LN backward |

D512 training at L768 (outgoing, ms per kernel): front 2.74 + LN 0.43, `contract_gp` 2.50 (78 % of DRAM), two weight / dgrad GEMMs
1.74 + 1.76 (cuBLAS, about 70 % of the tensor peak), output-LN backward 1.24, gate 1.09, input-LN backward 0.84, output epilogue 0.82,
forward LN-out 0.47: 17.1 ms in total. Kernel flow figures for the single direction are not generated yet.

Weight pack, LN-affine fold and input LN use the CUDA aux kernels whenever the call runs eagerly, and every caller sits inside an
opaque custom op, so the P1 / F3-fold rows above are CUDA; their Triton versions only serve a `torch.compile` region that traces the
caller. No Triton kernel remains in any D256 / D384 / D512 cell at L384 / L768.

## Open

- Single direction in fp32 (bf16 only today); kernel flow figures for the single direction.
- D512 single-direction training at L768 (17.1 ms): front 2.7 ms (latency bound; multicast, N-split and BI=2 variants measured and
  not faster), `contract_gp` 2.5 ms (at its DRAM traffic), output-LN backward 1.2 ms (reads 0.7 GB more than needed). Each remaining
  item is 1 - 3 % of the step.
- Wide single-direction training towards 90 % of speed of light: at D384 L768 the step is at 86.5 % of the design SoL; the worst
  kernels are the front (69.6 %), the output-LN backward (65.3 %) and `contract_gp` (88 %).
- Bidirectional D512 L128 training is the thinnest cell against the other implementations (1.10x).

(Carried over from the 2026-09-30 status snapshot, [docs/records/h100_status_20260930.md](../../../records/h100_status_20260930.md).)

## Measurements (2026-10-15)

H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, `benchmarks/runners/bench.py` (compiled, manual CUDA graph), default dispatch; B=1, bf16
activations, FP32 master parameters, training = forward + backward with dropout 0.25, inference dropout 0, mask probability 0.1;
L = 128 ... 1024 in steps of 128, D = 64 ... 512. Cells show
× = fastest other implementation / ours, with that implementation in parentheses; the others are PyTorch, cuEquivariance and DTv1. Absolute
latencies, the Anthropic payload (same-process alternating A/B, D64 ... D384 single direction and D64 / D128 bidirectional, ours faster in
all 48 cells, thinnest 1.006x at D256 L768) and the Triton rows are in
[the record](../../../records/trimul_h100_comparison_20261015.md). **Ours is the fastest in all 240 cells** (smallest 1.10x).
The D256 single-direction inference cells were measured before the 2026-10-15 selection change (0.7 - 2.5 % faster now).
The 2026-10-13 tables (absolute latencies of the bidirectional module against Triton, DTv1 and cuEquivariance):
[docs/records/h100_trimul_20261013.md](../../../records/h100_trimul_20261013.md).

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
