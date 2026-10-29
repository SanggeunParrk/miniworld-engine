# OuterProductMean on B200 (sm100)

Kernel-level status of `OuterProductMean` on B200; the module-level summary is in [b200.md](../b200.md). bf16; the
module's shape is fixed (d_msa 64, d_hidden 32, d_pair 128), so the columns are (Length, MSA depth). B200 = CUDA where a
hand-written sm_100a kernel runs a step; the outer-product and gradient GEMMs are cuBLAS (figures only, no kernel tables).
Figures: one box per kernel, left to right, HBM reads (blue, left) and writes (red, right); generated from
`figures/opm.json` by `python -m miniworld_engine.viz.kernel_flow`, then converted to PNG (`cairosvg -s 2 -b white`).
Dispatch: `integrations/opm_train.py` (the H100 integration; compute capability (10, 0) selects the sm_100a extension);
kernels: `integrations/csrc/sm100/opm_sm100.cu` (tcgen05 / TMEM / TMA), one extension.

Served when `implementation=miniworld` (or `anthropic`), bf16, d_msa 64 / d_pair 128, either normalisation order
(`normalize_before_proj=False`, the count dividing the projection and its bias, since 2026-10-29: the epilogue takes the order as a flag,
the training backward takes dbias = sum dz / n off the dgrad's dz / n), inference and training alike; a length that
is not a multiple of 64 / a depth that is not a multiple of 256 is padded with masked rows. The pair residual is added in the epilogue
(its gradient is `dz` itself). Since 2026-10-29 (second pass) the other FoldForge widths, RoseTTAFold3's projection biases and Chai-1's
groups are sm_100a kernels too, inference and training (`csrc/sm100/opm_wide_sm100.cu`, below). Since 2026-10-29 (third pass) also
IntelliFold-v2's (d_msa 256, d_pair 512), the interchain mask (`mask_interchain` with chain ids: the epilogue zeroes the update of a pair
of two chains, the dgrad its gradient) and a batch of MSA stacks (one stack at a time), training and inference, every width and Chai-1;
nothing of the module falls to the PyTorch statements on B200 any more (`kernels/outer_product_mean/composed.py` serves only H100).

## FoldForge widths, RoseTTAFold3, Chai-1 (`opm_wide_sm100.cu`, 2026-10-29)

One extension (`opm_train._ext_wide`), the stages of `opm_sm100.cu`; MiniWorld's (64, 128) without biases keeps `opm_sm100.cu`.

- **d_msa 64 / 128 / 256 x d_pair 128 / 256 / 384 / 512, projection biases** (Protenix-v2 / ESMFold2 (128, 256), OpenDDE (128, 384),
  IntelliFold-v2 (256, 512), RoseTTAFold3 (64, 128) + biases), `integrations/opm_train.py`, either order:
  - prologue `opm_wprologue_sm100<CM>`: d_msa 128 is two 128B-swizzled 64-channel k blocks per x tile (one 4-D box), K = 128 in eight
    k steps, one CTA an SM (four x stages); the projection biases are added in the drain to the MSA rows whose mask bit is set (the
    LayerNorm warps leave the tile's bits in a shared ring), so a masked row keeps its zero as `(proj + b) * mask` does. The biases go
    in through `_forward_sm100(pbias=...)`, inference included: `composed.biased_on_fused` (which added bias terms after a bias-less
    fused call) is gone. d_msa 256: a 64 KiB x tile, two x stages and one accumulator; the LayerNorm (two threads a row, 128 channels
    each) makes three passes over the swizzled row in shared memory (sum, centred squares, normalise).
  - mask counts: one table [N, N] fp32 (`opm_norm`, off the prologue's bit rows; made once in the forward and kept for the dgrad); every
    epilogue / dgrad thread loads its pair's count one tile ahead. (Counting per tile from the bit rows -- 16-byte loads 128 bytes
    apart across a warp -- put its load latency on the drain's critical path.)
  - epilogue `opm_wepilogue_sm100<NW>` on CTA pairs (cluster of two, `cta_group::2`): two neighbouring 128-pair tiles, one M 256 x N NW
    product per k step, each CTA landing its own pairs' O and half of the Wo slice; c_z 256 / 512 are N = 256 slices, c_z 384 two
    N = 192 slices (O read once / twice, the second an L2 hit). The drain stages the bias slice in shared memory and writes z through
    one staging block per 64 columns, each stored by its own TMA box, its residual loaded one item ahead as soon as the block's store
    has left it. (Ablations at c_z 256, L768: drain compiled out 247 of 401 us; the per-element bias `__ldg` 90 us, the residual load at
    the start of each drain 64 us; the CTA pair alone 424 -> 401 us.) The interchain mask zeroes a pair's update (`asym`).
  - dgrad `opm_wdgrad_sm100<CZ>` on CTA pairs: per 128-column chunk of dO one M 256 x N 128 product per K = 128 stage, each CTA landing
    half of the chunk's Wo columns; each CTA keeps its tile's dz in a ring of 128-z units (the A operand); dbias per unit by one
    M 256 x N 16 product (the ones in the leader's half of B). A forwarding warp tells the leader's MMA that its CTA's unit landed.
    The interchain mask: inv 0 for the pair (dO, dz / n), its dz rows zeroed before the dbias product.
  - dWo `opm_wdwo_sm100<CZ>`: every 128-wide c_z slice of a column part accumulates in one CTA's TMEM (2 x 256 / 3 x 128 / 4 x 128
    columns), so O is read once; split-K partials summed in split order.
  - prologue backward at d_msa 128 `opm_wprologue_bwd_sm100<WT, 1>`: one group of eight compute warps (two threads a row, 64 channels
    each, their row sums combined through shared memory), dy double-buffered, the gradient row read from TMEM 32 columns at a time
    (twice: row sums, then dm); the masked column sums of dA / dB (dbeta, dW's beta term, the projection-bias gradients) by two
    cuBLAS GEMVs. At d_msa 64 the existing kernel's row 64 of R is exactly those sums. d_msa 256 `<WT, 2>`: a CTA pair (cluster of
    two) per tile, each CTA 128 of the channels (its x half, Wf columns, dm half and rows of R: the d_msa 128 kernel's buffers); the four
    partial row sums are exchanged through distributed shared memory (`st.shared::cluster` + a release arrive on the peer's barrier)
    and added in one fixed order in both CTAs.
- **Interchain mask, batch** (`mask_interchain` with chain ids; B > 1): `opm_train.outer_product_mean(token_asym_id=...)` passes the
  chain ids (int32 [N]) to the epilogue and dgrad of either extension; a batch runs one MSA stack at a time (each its own autograd node).
- **Chai-1** (8 groups of 8, d_hidden 64, product LayerNorm, c_z 128 / 256), `kernels/outer_product_mean/chai.py`; any length / depth
  (padded to 32 / 128 with masked rows):
  - `opm_chai_prologue_sm100`: tile = 8 tokens x 16 MSA rows, D[row][channel] on the tensor core, group-major a / b by 128-byte boxes.
  - products: one cuBLAS batched GEMM (O [8, 8N, 8N]).
  - `opm_chai_epilogue_sm100`: the pair tile of O (4 i x 32 j) lands by 4-D TMA boxes straight into the no-swizzle K-major core-matrix
    layout (8 consecutive j of one row are 128 contiguous bytes of O: the gather costs no thread work), statistics in one shifted pass as
    the eight group blocks land, normalisation in place, one GEMM against W' = W_out . gamma (N = c_z, K = 512); per-group barriers let the
    next tile's blocks load under this tile's GEMM. Training writes y (by TMA straight from the GEMM operand: one 5-D box per group,
    y [N N, 512] viewed as (l, j % 8, (g, k), j / 8, i); a dedicated warp stores each group once normalised, and the group's next O
    block waits for the store to leave) and the statistics. z leaves through a ring of two 64-column staging blocks by TMA, the
    residual loaded two blocks ahead; b' in shared memory. (A second 128 KiB tile buffer, the earlier proposal, does not fit next to W'
    and was not the bound: training at L768 the kernel took 544 us, 337 without the y stores, 350 without the z stores / residual
    loads, 195 without both -- the threads wrote y and z as rows of their own, 32 lines a warp store.) The interchain mask in training
    zeroes the pair's normalised products in the operand (y and its GEMM row), rstd 0 and its dbias weight 0.
  - backward: dy = dz W', dW' = dz^T y, dbias (cuBLAS); `opm_chai_lnbwd_sm100` (CUDA cores, shared-memory staged, memory-bound) into the
    grouped layout; dA^T / dB^T by two batched GEMMs; `opm_chai_prologue_bwd_sm100` (the d_msa 64 prologue backward with 64 + 64 channels:
    dA^T tiles are the dy GEMM's MN-major A and the dW GEMM's K-major B).
- ptxas (nvcc 13.1): no spills in any of these kernels (the MiniWorld-width `opm_prologue_bwd_sm100` of `opm_sm100.cu` spills 44 bytes,
  as before this pass).
- Tests: `tests/gpu/sm100/test_opm_wide_b200_gpu.py` (IntelliFold-v2, the interchain mask, batches), `test_opm_chai_b200_gpu.py`
  (fp64 statements, bf16 / fp32 master, masks, ragged shapes, interchain, batch, determinism, CUDA graph), `test_opm_variants_b200_gpu.py`.

Measured 2026-10-29 on one B200 (GPU 4 of node 2, nvcc 13.1 kernels), S1024, bf16 activations:

Training (`benchmarks/runners/bench.py target=outer_product level=module mode=training cudagraph=manual +n_msa=1024`, bf16-mixed with
fp32 master parameters, CUDA graph, ms; PyTorch = the module's statements (`implementations=[pytorch]`, the same day); second pass = the
kernels as of `6fb200f9` and third pass = this page's kernels, run back to back per row from a copy of each tree; a cell is noisy by
about +-4 % run to run):

| width / option | L384 PyTorch | L384 second pass | L384 third pass | L768 PyTorch | L768 second pass | L768 third pass |
|---|---|---|---|---|---|---|
| (128, 256) Protenix-v2 / ESMFold2 | 1.413 | 1.222 | 1.098 | 4.938 | 4.343 | 3.868 |
| (128, 384) OpenDDE | 1.508 | 1.280 | 1.253 | 5.238 | 5.047 | 4.452 |
| (256, 512) IntelliFold-v2 | 1.854 | (PyTorch) 1.889 | 1.510 | 6.052 | (PyTorch) 6.460 | 5.238 |
| (64, 128) RoseTTAFold3 biases | 1.357 | 1.018 | 1.007 | 4.592 | 3.579 | 3.539 |
| (128, 256) RoseTTAFold3 biases, after-order | 1.533 | 1.230 | 1.156 | 5.158 | 4.184 | 4.093 |
| (64, 256) Chai-1, 8 groups | 1.162 | 0.833 | 0.789 | 3.994 | 2.828 | 2.742 |
| (64, 128) MiniWorld | 1.263 | 0.906 | 0.968 | 4.635 | 3.571 | 3.467 |

The interchain mask and batches > 1 ran the PyTorch statements in training before this pass; they run the same kernels now (the mask is
two loads and a select per thread; a batch is one call per MSA stack).

Inference (the same harness, `mode=inference`, CUDA graph, ms, second -> third pass): IntelliFold-v2 (256, 512) L384 0.930 -> 0.397,
L768 3.236 -> 1.559 (it ran the composed CUDA / cuBLAS path); (128, 256) 0.338 -> 0.309, 1.366 -> 1.280; Chai-1 0.221 -> 0.197,
0.803 -> 0.747.  Earlier (second pass, module call with the pair residual, us, the composed path -> the kernels): (128, 256) L384
1479 -> 338, L768 5401 -> 1267; (128, 384) L384 1566 -> 390, L768 5576 -> 1502; RoseTTAFold3 (64, 128) L384 1361 -> 285, (128, 256)
1476 -> 340; Chai-1 L384 443 -> 242, L768 1477 -> 800.

Kernels alone (L768 S1024, CUDA graph of 10 calls after a 4 s warm-up, us, second -> third pass): wide epilogue c_z 256 424 -> 284,
c_z 384 645 -> 433, c_z 512 818 -> 499; wide dgrad c_z 256 511 -> 464-470, c_z 384 772 -> 691-790, c_z 512 1093 -> 1015-1060 (the
dgrad moves +-8 % run to run); Chai-1 epilogue training 560 -> 390, inference 360 -> 276 (L384: 141 -> 104, 96 -> 72).

Where the time goes (L768 S1024, training, torch profiler, ms): the three O / dA / dB GEMMs are ~2.27 at every single-group width (the
floor of this composition); (128, 256): dgrad 0.475, epilogue 0.290, dWo 0.239, prologue backward 0.094; (128, 384): 0.688 / 0.425 /
0.368 / 0.092; (256, 512): 0.859 / 0.509 / 0.553 / 0.302, prologue 0.147; MiniWorld's (64, 128): 0.294 / 0.236 / 0.204 / 0.056. The
epilogue is at its HBM floor (O read, z written, the residual read: 1.8 GB at c_z 256, 284 us = 6.3 TB/s). The dgrad has no single bound
left (each of the MMAs, the dO stores, the dz / n stores, the chunk drains compiled out alone moves it by <= 6 %, drains and dz / n
stores together 22 %; a deeper Wo ring did not help): what is left there is the dz / n stores (rows of their own per thread; through
the dO staging ring by TMA instead) and the coupling of the drain to the MMA. Chai-1: the product-norm backward 0.391 (HBM: dy, y in,
dO out), the epilogue 0.387, prologue backward 0.109; past that only cuBLAS. Beyond these kernels the GEMMs over the materialised
grouped product dominate every width (the fused inference kernel below shows the fused mainloop does not yet beat cuBLAS').

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
