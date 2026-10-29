# OuterProductMean on B200 (sm100)

Kernel-level status of `OuterProductMean` on B200; the module-level summary is in [b200.md](../b200.md). bf16; the
module's shape is fixed (d_msa 64, d_hidden 32, d_pair 128), so the columns are (Length, MSA depth). B200 = CUDA where a
hand-written sm_100a kernel runs a step; the outer-product and gradient GEMMs are cuBLAS (figures only, no kernel tables).
Figures: one box per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/opm.json` by `python -m miniworld_engine.viz.kernel_flow`, then converted to PNG (`cairosvg -s 2 -b white`).
Dispatch: `integrations/opm_train.py` (the H100 integration; compute capability (10, 0) selects the sm_100a extension);
kernels: `integrations/csrc/sm100/opm_sm100.cu` (tcgen05 / TMEM / TMA), one extension.

Served when `implementation=miniworld` (or `anthropic`), bf16, batch 1, d_msa 64 / d_pair 128, either normalisation order
(`normalize_before_proj=False`, the count dividing the projection and its bias, since 2026-10-29: the epilogue takes the order as a flag,
the training backward takes dbias = sum dz / n off the dgrad's dz / n), no interchain masking, inference and training alike; a length that
is not a multiple of 64 / a depth that is not a multiple of 256 is padded with masked rows. The pair residual is added in the epilogue
(its gradient is `dz` itself). Since 2026-10-29 (second pass) the other FoldForge widths, RoseTTAFold3's projection biases and Chai-1's
groups are sm_100a kernels too, inference and training (`csrc/sm100/opm_wide_sm100.cu`, below). B200 inference those kernels do not take
(a batch of MSA stacks, the interchain mask in training, d_msa 256 / d_pair 512) runs the composed CUDA / cuBLAS path
(`kernels/outer_product_mean/composed.py`, as H100).

## FoldForge widths, RoseTTAFold3, Chai-1 (`opm_wide_sm100.cu`, 2026-10-29)

One extension (`opm_train._ext_wide`), the stages of `opm_sm100.cu`; MiniWorld's (64, 128) without biases keeps `opm_sm100.cu`.

- **d_msa 64 / 128 x d_pair 128 / 256 / 384, projection biases** (Protenix-v2 / ESMFold2 (128, 256), OpenDDE (128, 384),
  RoseTTAFold3 (64, 128) + biases), `integrations/opm_train.py`, either order:
  - prologue `opm_wprologue_sm100<CM>`: d_msa 128 is two 128B-swizzled 64-channel k blocks per x tile (one 4-D box), K = 128 in eight
    k steps, one CTA an SM (four x stages); the projection biases are added in the drain to the MSA rows whose mask bit is set (the
    LayerNorm warps leave the tile's bits in a shared ring), so a masked row keeps its zero as `(proj + b) * mask` does. The biases go
    in through `_forward_sm100(pbias=...)`, inference included: `composed.biased_on_fused` (which added bias terms after a bias-less
    fused call) is gone.
  - epilogue `opm_wepilogue_sm100<NW>`: c_z 256 is one N = 256 accumulator, c_z 384 two N = 192 slices; O is read once / twice.
  - dgrad `opm_wdgrad_sm100<CZ>`: the dz tile stays in a ring of 128-z units (one more than a tile, so the next tile's first unit
    lands under this tile's last chunk), Wo streams as [64 z x 128 n] sub-chunks, dbias per unit on the tensor core.
  - dWo `opm_wdwo_sm100<CZ>`: every 128-wide c_z slice of a column part accumulates in one CTA's TMEM (2 x 256 / 3 x 128 columns),
    so O is read once; split-K partials summed in split order.
  - prologue backward at d_msa 128 `opm_wprologue_bwd_sm100`: one group of eight compute warps (two threads a row, 64 channels each,
    their row sums combined through shared memory), dy double-buffered; the masked column sums of dA / dB (dbeta, dW's beta term, the
    projection-bias gradients) by two cuBLAS GEMVs. At d_msa 64 the existing kernel's row 64 of R is exactly those sums.
- **Chai-1** (8 groups of 8, d_hidden 64, product LayerNorm, c_z 128 / 256), `kernels/outer_product_mean/chai.py`; any length / depth
  (padded to 32 / 128 with masked rows):
  - `opm_chai_prologue_sm100`: tile = 8 tokens x 16 MSA rows, D[row][channel] on the tensor core, group-major a / b by 128-byte boxes.
  - products: one cuBLAS batched GEMM (O [8, 8N, 8N]).
  - `opm_chai_epilogue_sm100`: the pair tile of O (4 i x 32 j) lands by 4-D TMA boxes straight into the no-swizzle K-major core-matrix
    layout (8 consecutive j of one row are 128 contiguous bytes of O: the gather costs no thread work), statistics in one shifted pass as
    the eight group blocks land, normalisation in place, one GEMM against W' = W_out . gamma (N = c_z, K = 512); per-group barriers let the
    next tile's blocks load under this tile's GEMM. Training writes y and the statistics.
  - backward: dy = dz W', dW' = dz^T y, dbias (cuBLAS); `opm_chai_lnbwd_sm100` (CUDA cores, shared-memory staged, memory-bound) into the
    grouped layout; dA^T / dB^T by two batched GEMMs; `opm_chai_prologue_bwd_sm100` (the d_msa 64 prologue backward with 64 + 64 channels:
    dA^T tiles are the dy GEMM's MN-major A and the dW GEMM's K-major B).
- ptxas (nvcc 13.1): no spills in any of these kernels.
- Tests: `tests/gpu/sm100/test_opm_wide_b200_gpu.py`, `test_opm_chai_b200_gpu.py` (fp64 statements, bf16 / fp32 master, masks, ragged
  shapes, determinism, CUDA graph), `test_opm_variants_b200_gpu.py`.

Measured 2026-10-29 on one B200 (GPU 4 of node 2, nvcc 13.1 kernels), S1024, bf16 activations:

Training (`benchmarks/runners/bench.py target=outer_product level=module mode=training +n_msa=1024`, bf16-mixed with fp32 master
parameters, CUDA graph, ms; before = the PyTorch statements B200 ran, after = these kernels; a cell is noisy by about +-4 % run to run):

| width / option | L384 before | L384 after | L768 before | L768 after |
|---|---|---|---|---|
| (128, 256) Protenix-v2 / ESMFold2 | 1.490 | 1.16-1.21 | 5.405 | 4.28-4.34 |
| (128, 384) OpenDDE | 1.600 | 1.27-1.37 | 5.657 | 4.81-4.97 |
| (64, 128) RoseTTAFold3 biases | 1.351 | 0.89-0.96 | 4.563 | 3.48-3.56 |
| (64, 128) RoseTTAFold3, after-order | 1.309 | 0.93-1.00 | 4.637 | 3.44-3.78 |
| (128, 256) with the biases | 1.511 | 1.15-1.21 | 4.938 | 3.89-4.15 |
| (64, 256) Chai-1, 8 groups | 1.172 | 0.823 | 3.844 | 2.894 |
| (64, 128) MiniWorld (unchanged kernels) | 0.894 | 0.93-0.96 | 3.659 | 3.32-3.35 |

Inference (module call with the pair residual, bf16 module, CUDA graph, us; the composed CUDA / cuBLAS path B200 ran before against
today's path, same script, no torch.compile): (128, 256) L384 1479 -> 338, L768 5401 -> 1267; (128, 384) L384 1566 -> 390,
L768 5576 -> 1502; RoseTTAFold3 (64, 128) L384 1361 -> 285, (128, 256) 1476 -> 340; Chai-1 L384 443 -> 242, L768 1477 -> 800.

Where the time goes (L768 S1024, training, torch profiler): the three O / dA / dB GEMMs are ~2.3 ms at every width; (128, 256): epilogue
0.45 ms, dgrad 0.49, dWo 0.24, prologue backward 0.09, prologue 0.06 (MiniWorld's (64, 128): 0.24 / 0.31 / 0.20 / 0.06 / 0.04). The
epilogue and dgrad stream the whole Wo once per 128-pair tile (the L2 -> SM intake bounds them, as at c_z 128); a cta_group::2 version
(M = 256 pairs sharing each Wo slice) is what is left there. Chai-1: epilogue 0.55 ms (single 128 KiB tile buffer: the next tile's
statistics wait for its last group block), product-norm backward 0.39 ms (HBM: dy, y in, dO out), prologue 0.06, prologue backward 0.11.

#### Fused inference kernel (`opm_fused_sm100`, `opm_fused_infer`)

The grouped outer product and the c_hidden^2 -> c_z projection in one persistent kernel on CTA pairs, so the [N*32, N*32] product
(1.2 GB at L768) is never written: per pair of CTAs an item of 32 i x 8 j, four passes over c (8 values each) as cta_group::2
M 256 x N 256 products into two of three rotating 128-column TMEM slots (the next pass starts once the shared slot is drained), eight
warps drain the slot to a bf16 Opair tile in shared memory (the H100 kernel's k permutation and weight), the projection is a
cta_group::2 M 256 x N 128 product into TMEM columns 384-511 (each CTA holds the weight rows of its 64 outputs), four warps finish
(count, bias, either order, residual). Measured against cuBLAS + the epilogue kernel (CUDA graph, steady state under the 1000 W cap,
2026-10-29, us): S1024 L128 41 / 38, L256 140 / 137, L384 288 / 293, L512 505 / 513, L640 788 / 800, L768 1141 / 1144; S2048 L384
521 / 503, L768 2062 / 1959; S4096 L384 1001 / 917, L768 4014 / 3632; through the module harness (ms, today's path -> fused,
two runs each) S1024 L384 0.309 / 0.295 -> 0.303 / 0.294, L512 0.496 / 0.494 -> 0.503 / 0.514, L640 0.729 / 0.731 -> 0.808 / 0.795,
L768 1.043 / 1.058 -> 1.153 / 1.051, S4096 L768 3.528 / 3.687 -> 3.472 / 3.651. It wins at no cell beyond noise, so the default
dispatch does not serve it (`_FUSED_SM100_CELLS` is empty; the tests route it explicitly); it does avoid the product buffer.
Its mainloop alone measures 909 us at L768 S1024 and 3773 us at S4096 against cuBLAS's 894 / 3331 for the same GEMM with the product
written: per unit of energy it is ~13 % behind cuBLAS's mainloop at long K, which eats the saved HBM traffic. What is left: that
mainloop gap (cuBLAS runs nvjet 128x256_64x6_2x1_2cta, the same tile; neither 6 stages, the raster nor a 2-D A box closed it), and
the pass-to-pass gap (0.5-1 us of a 7-8 us pass at L768 S1024; it was 1.5-2 us while the drain warps released the slots with a
release-scope cluster arrive).

## OuterProductMean (`OuterProductMean`)

### Inference

#### Fused path · N a multiple of 64, S a multiple of 256

![OPM inference](figures/opm_inference.png)

##### F1 · LN + projections + mask (opm_prologue)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · count + projection + residual (opm_epilogue)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### Fused path · N a multiple of 64, S a multiple of 256

The forward keeps the LayerNorm statistics, A2 / BT and O (`settings.opm_train_save_o=0` recomputes O in the backward
instead of keeping 302 MB at L384).

![OPM training](figures/opm_training.png)

##### F1 · LN + projections + mask (opm_prologue)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · count + projection + residual (opm_epilogue)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · projection dgrad (opm_dgrad)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · dW_o (opm_dwo)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · prologue backward (opm_prologue_bwd)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements (2026-09-30; CUDA-graph tables re-measured 2026-10-11 with fp32 master parameters)

- Setup: B200 (148 SMs), torch 2.13.0+cu129, triton 3.7.1, sm_100a kernels built by nvcc 13.1, B = 1, bf16-mixed (the module's LayerNorm parameters stay fp32).
- Harness: `benchmarks/runners/bench.py target=outer_product level=module mode=<inference|training> min_seq_len=128
  max_seq_len=768 seq_len_step=128 +n_msa=<S>`; compiled, inference in a CUDA graph, training with and without one.
- Tables: latency in ms (median); × = ours against the fastest of the others. cuEquivariance ships no OuterProductMean
  kernel (—).
- Anthropic: the harness's `anthropic` row -- `opt_core.ops.msa_opm.forward_mask_norm` as shipped (its fused LN / projection
  / mask prologue and one Triton outer-product / projection kernel with the release's default tile config), on this module's
  weights; the engine adds the residual. Triton, so it runs on sm_100 unmodified; forward-only, so the training tables have no
  Anthropic column (—).
- Accuracy against the harness's fp32 reference, max over every row: ours out 1.7e-3, grad 2.4e-4; PyTorch compiled 1.7e-3 / 2.5e-4; Anthropic 1.7e-3 (inference).
- Where the time goes: the O GEMM and the two gradient GEMMs (dA, dB) are cuBLAS at its sustained rate on this power-capped
  card; in the inference sweep the path runs at 68-88 % of its energy floor. Folding the epilogue into the O GEMM does not fit
  B200: re-streaming W_o per tile exceeds the SM's TMA intake; the dgrad kernel is bound by the same intake (a 2-CTA version
  is what is left).
- Re-verification (2026-10-10, engine v2.2.0 plus unreleased changes, the setup above with the kernels built by nvcc 13.1, driver 580.95.05, one GPU with no other process on it): every cell of the inference table and the CUDA-graph training table (66 cells) was re-measured through the harness command above, one run per width and mode; a cell that moved by more than 8 % against the table was repeated (once, twice if still beyond 8 %) and the median of the runs is reported. Median change 0.9 % (2.7 % for ours), 90th percentile 5.1 %; two cells were replaced ((640, 4096) and (768, 4096) ours inference: 2.364 -> 2.621 and 3.803 -> 3.352 ms; the S4096 inference rows vary by +-12 % between runs). The no-CUDA-graph training tables are host-bound (their cells move by 20-60 % between runs with the load on the host) and keep their 2026-09-30 values. **2026-10-11:** every CUDA-graph cell of these tables was re-measured with fp32 master parameters (`precision=bf16-mixed`, the harness default since that date: fp32 parameters over bf16 activations under `torch.autocast`; one run per cell, `parameter_dtype` = float32 in the CSV, the engine environment's python called directly); the cells above are those values. The no-CUDA-graph training tables are host-bound and keep their earlier values (bf16 parameters).
- Charts: under each table, a length sweep at S1024 and an MSA-depth sweep at L384
  (`python -m miniworld_engine.viz.measure_bars docs/gpus/b200/opm/opm.md --length-d 1024`).

### OPM · Inference (CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.082 | — | 0.137 | 0.055 | 1.49 |
| (256, 1024) | 0.223 | — | 0.401 | 0.143 | 1.56 |
| (384, 1024) | 0.428 | — | 0.861 | 0.295 | 1.45 |
| (512, 1024) | 0.747 | — | 1.474 | 0.471 | 1.59 |
| (640, 1024) | 1.108 | — | 2.278 | 0.732 | 1.51 |
| (768, 1024) | 1.600 | — | 3.357 | 1.032 | 1.55 |
| (128, 2048) | 0.123 | — | 0.182 | 0.082 | 1.50 |
| (256, 2048) | 0.338 | — | 0.520 | 0.235 | 1.44 |
| (384, 2048) | 0.694 | — | 1.133 | 0.512 | 1.36 |
| (512, 2048) | 1.155 | — | 2.102 | 0.816 | 1.42 |
| (640, 2048) | 1.730 | — | 3.231 | 1.298 | 1.33 |
| (768, 2048) | 2.475 | — | 4.593 | 1.800 | 1.38 |
| (128, 4096) | 0.197 | — | 0.274 | 0.139 | 1.42 |
| (256, 4096) | 0.594 | — | 0.878 | 0.469 | 1.27 |
| (384, 4096) | 1.218 | — | 1.832 | 1.010 | 1.21 |
| (512, 4096) | 2.087 | — | 3.137 | 1.613 | 1.29 |
| (640, 4096) | 3.022 | — | 4.850 | 2.560 | 1.18 |
| (768, 4096) | 4.312 | — | 6.901 | 3.636 | 1.19 |

![OPM · Inference (CUDA graph), msa depth sweep at L384](figures/opm_opm_inference_cuda_graph_msa_depth.png) <!-- measure_bars -->

### OPM · Training (CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.262 | — | — | 0.170 | 1.54 |
| (256, 1024) | 0.651 | — | — | 0.451 | 1.44 |
| (384, 1024) | 1.234 | — | — | 0.946 | 1.30 |
| (512, 1024) | 2.138 | — | — | 1.630 | 1.31 |
| (640, 1024) | 3.114 | — | — | 2.456 | 1.27 |
| (768, 1024) | 4.390 | — | — | 3.442 | 1.28 |

![OPM · Training (CUDA graph), length sweep at S1024](figures/opm_opm_training_cuda_graph_length.png) <!-- measure_bars -->

### OPM · Training (no CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.436 | — | — | 0.432 | 1.01 |
| (256, 1024) | 0.697 | — | — | 0.472 | 1.48 |
| (384, 1024) | 1.349 | — | — | 0.962 | 1.40 |
| (512, 1024) | 2.148 | — | — | 1.575 | 1.36 |
| (640, 1024) | 3.258 | — | — | 2.258 | 1.44 |
| (768, 1024) | 4.602 | — | — | 3.598 | 1.28 |

![OPM · Training (no CUDA graph), length sweep at S1024](figures/opm_opm_training_no_cuda_graph_length.png) <!-- measure_bars -->
