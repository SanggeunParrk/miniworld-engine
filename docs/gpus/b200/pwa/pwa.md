# MSAPairWeightedAveraging on B200 (sm100)

Kernel-level status of `MSAPairWeightedAveraging` on B200; the module-level summary is in [b200.md](../b200.md). bf16; the
module's shape is fixed (d_msa 64, d_pair 128, 8 heads x 32), so the columns are (Length, MSA depth). B200 = CUDA where a
hand-written sm_100a kernel runs a step; the two contractions and the weight gradient are cuBLAS `bmm` (figures only).
Figures: one box per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/pwa.json` by `python -m miniworld_engine.viz.kernel_flow`, then converted to PNG (`cairosvg -s 2 -b white`).
Dispatch: `integrations/pwa_train.py` (the H100 integration; compute capability (10, 0) selects the sm_100a extension);
kernels: `integrations/csrc/sm100/pwa_sm100.cu` (tcgen05 / TMEM / TMA), one extension.

Served when `implementation=miniworld` (or `anthropic`), bf16, batch 1, a token key mask (or none), N (length) a multiple of 128 up
to 1024 and S (MSA depth) a multiple of 128, inference and training alike. The MSA residual and the module's row-broadcast dropout
(`drop_msa`, training) are applied in the gate / out-projection pass. Every other call runs the portable portable path.

Design notes:
- The contractions o = w · v and dv = wᵀ · do run on cuBLAS `bmm`: 5-15 % faster than a tcgen05 kernel of ours at L256-1024,
  same bits. The gate / out pass reads o back; that round trip is the structural gap to the energy floor (the inference sweep
  runs at 44-65 % of it). A token-major gate / out tile (o read in 8 KB runs) scattered the output rows and was 5-12 % slower.
- The pair-side forward folds LN_z into proj_z (the tensor core reads raw z): same error against fp32 as rounding
  bf16(LN_z(z)) first, 1.5x faster than that form.
- The backward glue keeps dgp on chip: it forms dW_g = yᵀ dgp and the gate branch of dy (dgp · W_g) itself, and the MSA-side
  tail adds dv · W_v; dgamma / dbeta and the pair side's parameter gradients come from tensor-core sums (per-head
  M = dbᵀ x̂, 1ᵀ [dy x̂ | dy]) instead of per-tile warp reduce-scatters.

## MSAPairWeightedAveraging (`MSAPairWeightedAveraging`)

### Inference

#### Fused path · N a multiple of 128 up to 1024, S a multiple of 128

![PWA inference](figures/pwa_inference.png)

##### F1 · pair LN + bias + softmax (pair_fwd)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · LN + value projection (ln_vg)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · gate + out projection + residual (pwa_gate_out)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### Fused path · N a multiple of 128 up to 1024, S a multiple of 128

![PWA training](figures/pwa_training.png)

##### F1 · pair LN + bias + softmax (pair_fwd)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F2 · LN + value projection (ln_vg)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · gate + out projection + dropout + residual (pwa_gate_out)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · weight layouts (pwa_wprep)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B2 · gate glue (pwa_glue2)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · MSA-side tail (dv_bwd)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B5 · pair-side backward (pair_sdot + pair_bwd)

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements (2026-09-30; CUDA-graph tables re-measured 2026-10-11 with fp32 master parameters)

- Setup: B200 (148 SMs), torch 2.13.0+cu129, triton 3.7.1, sm_100a kernels built by nvcc 13.1, B = 1, bf16-mixed (the module's LayerNorm parameters stay fp32).
- Harness: `benchmarks/runners/bench.py target=msa_pair_weighted_averaging level=module mode=<inference|training>
  min_seq_len=128 max_seq_len=768 seq_len_step=128 +n_msa=<S>`; compiled, inference in a CUDA graph, training with and
  without one, training with the consumer model's p_drop_msa = 0.15.
- Tables: latency in ms (median); × = ours against the fastest of the others. cuEquivariance ships no pair-weighted-averaging
  kernel (—).
- Anthropic: the harness's `anthropic` row -- `opt_core.ops.msa_pwa.forward_masked` as shipped (its default fo kernel with the
  fused LN_m / projection prologue), stock LayerNorms on this module's parameters; the engine adds the residual. Triton, so it
  runs on sm_100 unmodified; forward-only, so the training tables have no Anthropic column (—).
- Accuracy against the harness's fp32 reference, max over every row: ours, PyTorch compiled and Anthropic out 1.8e-3 (inference; training with dropout
  leaves the harness's accuracy columns blank). `tests/gpu/any/test_pwa_train_gpu.py` checks the training output and every
  gradient, and the inference update, within 1.15x of the module's own bf16 statements' error against an fp32 copy.
- Re-verification (2026-10-10, engine v2.2.0 plus unreleased changes, the setup above with the kernels built by nvcc 13.1, driver 580.95.05, one GPU with no other process on it): every cell of the inference table and the CUDA-graph training table (66 cells) was re-measured through the harness command above, one run per width and mode; a cell that moved by more than 8 % against the table was repeated (once, twice if still beyond 8 %) and the median of the runs is reported. Median change 0.5 % (0.6 % for ours), 90th percentile 1.6 %; no cell moved by more than 8 %. The no-CUDA-graph training tables are host-bound (their cells move by 20-60 % between runs with the load on the host) and keep their 2026-09-30 values. **2026-10-11:** every CUDA-graph cell of these tables was re-measured with fp32 master parameters (`precision=bf16-mixed`, the harness default since that date: fp32 parameters over bf16 activations under `torch.autocast`; one run per cell, `parameter_dtype` = float32 in the CSV, the engine environment's python called directly); the cells above are those values. The no-CUDA-graph training tables are host-bound and keep their earlier values (bf16 parameters).
- Charts: under each table, a length sweep at S1024 and an MSA-depth sweep at L384
  (`python -m miniworld_engine.viz.measure_bars docs/gpus/b200/pwa/pwa.md --length-d 1024`).

### PWA · Inference (CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.162 | — | 0.219 | 0.082 | 1.98 |
| (256, 1024) | 0.311 | — | 0.440 | 0.152 | 2.05 |
| (384, 1024) | 0.483 | — | 0.686 | 0.219 | 2.21 |
| (512, 1024) | 0.661 | — | 0.916 | 0.295 | 2.24 |
| (640, 1024) | 0.875 | — | 1.210 | 0.403 | 2.17 |
| (768, 1024) | 1.063 | — | 1.534 | 0.476 | 2.23 |
| (128, 2048) | 0.303 | — | 0.409 | 0.147 | 2.06 |
| (256, 2048) | 0.617 | — | 0.801 | 0.265 | 2.33 |
| (384, 2048) | 0.934 | — | 1.272 | 0.401 | 2.33 |
| (512, 2048) | 1.272 | — | 1.758 | 0.555 | 2.29 |
| (640, 2048) | 1.740 | — | 2.321 | 0.774 | 2.25 |
| (768, 2048) | 2.057 | — | 2.878 | 0.954 | 2.16 |
| (128, 4096) | 0.604 | — | 0.750 | 0.262 | 2.31 |
| (256, 4096) | 1.199 | — | 1.555 | 0.491 | 2.44 |
| (384, 4096) | 1.815 | — | 2.452 | 0.754 | 2.41 |
| (512, 4096) | 2.462 | — | 3.455 | 1.044 | 2.36 |
| (640, 4096) | 3.249 | — | 4.492 | 1.448 | 2.24 |
| (768, 4096) | 3.950 | — | 5.627 | 1.748 | 2.26 |

![PWA · Inference (CUDA graph), msa depth sweep at L384](figures/pwa_pwa_inference_cuda_graph_msa_depth.png) <!-- measure_bars -->

### PWA · Training, dropout 0.15 (CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.509 | — | — | 0.304 | 1.67 |
| (256, 1024) | 0.914 | — | — | 0.494 | 1.85 |
| (384, 1024) | 1.366 | — | — | 0.724 | 1.89 |
| (512, 1024) | 1.805 | — | — | 0.968 | 1.86 |
| (640, 1024) | 2.439 | — | — | 1.275 | 1.91 |
| (768, 1024) | 2.892 | — | — | 1.580 | 1.83 |

![PWA · Training, dropout 0.15 (CUDA graph), length sweep at S1024](figures/pwa_pwa_training_dropout_0_15_cuda_graph_length.png) <!-- measure_bars -->

### PWA · Training, dropout 0.15 (no CUDA graph)

| (Length, MSA depth) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 1024) | 0.576 | — | — | 0.391 | 1.47 |
| (256, 1024) | 0.980 | — | — | 0.519 | 1.89 |
| (384, 1024) | 1.427 | — | — | 0.743 | 1.92 |
| (512, 1024) | 1.875 | — | — | 0.990 | 1.89 |
| (640, 1024) | 2.523 | — | — | 1.359 | 1.86 |
| (768, 1024) | 2.985 | — | — | 1.578 | 1.89 |

![PWA · Training, dropout 0.15 (no CUDA graph), length sweep at S1024](figures/pwa_pwa_training_dropout_0_15_no_cuda_graph_length.png) <!-- measure_bars -->
