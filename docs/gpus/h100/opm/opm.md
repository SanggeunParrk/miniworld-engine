# OuterProductMean on H100 (sm90)

Kernel-level status of `OuterProductMean` on H100; the module-level summary is in [h100.md](../h100.md) and the dispatch contract
in [dispatch.md](../dispatch.md). bf16; d_hidden 32, the widths are listed below; the columns are
(Length, MSA depth). The B200 page of the same module is [../../b200/opm/opm.md](../../b200/opm/opm.md); the fusion algorithm is
the same, the instructions are not (see [Against B200](#against-b200)).

Dispatch: `integrations/opm_train.py` (`_h100_path`, `_forward_wide`); kernels: `integrations/csrc/opm_epilogue.cu` (one extension per
(d_msa, d_pair), built by JIT for sm_90a).
Served when `implementation=miniworld` (or `anthropic`), bf16, batch 1, no interchain masking, inference and training alike; the pair
residual is added in the epilogue (its gradient is `dz`). The `opm_train` setting (default on) turns the path off. Every other call runs the
module's PyTorch statements on H100 (no portable-kernel fallback, see [dispatch.md](../dispatch.md); other cards run the portable portable path).
Widths: (d_msa, d_pair) = (64, 128) MiniWorld, (128, 256) Protenix-v2 / ESMFold2, (128, 384) OpenDDE,
d_hidden 32. Either normalisation order (`normalize_before_proj=False` is ESMFold2's: divide after the projection and its bias). A length
that is not a multiple of 64 or an MSA depth that is not a multiple of 256 is padded with masked rows and the output cut back.

## Fusion

Per call (training adds the right-hand column), the same three kernel boundaries as B200:

| stage | forward | backward |
|---|---|---|
| F1 / B4 | prologue: LayerNorm + both projections + mask straight into the grouped layouts A2 `[(i,c), s]` and BT `[(j,e), s]` (+ LN statistics, bit mask) | prologue backward: mask, both projections and the LayerNorm backward in one pass -> dm, dWa, dWb, dgamma, dbeta |
| GEMM | O = A2 . BT^T, cuBLAS | dA, dB: cuBLAS on the grouped dO (plus O again when the `opm_train_save_o` setting is off) |
| F3 / B1, B3 | epilogue: permute `[i,j,c,e] -> [(i,j),(c,e)]`, `/ n`, `W_o`, bias, residual, bf16 | dgrad: `dz / n . W_o` into the grouped layout (+ dbias); dWo off the grouped O |
| F2+F3 inference | **one kernel** `opm_fused_kernel` (`opm_fused_infer`): O = A2 . BT^T and the projection together, see [Fused inference kernel](#fused-inference-kernel); O is never written | n/a |
| count | `n[i,j] = max(1, sum_s mask[s,i] mask[s,j])` from the bit mask (`opm_count_tiled`, exact) | reused |

Kernels (`opm_epilogue.cu`): `opm_prologue_tma_hopper` (persistent, TMA + WGMMA), `opm_prologue_wgmma_native` (non-TMA),
`opm_count_tiled`, `opm_pack_native`, `opm_finish_native`, and `opm_prologue_tma_wide<CM>` (persistent TMA prologue at d_msa 64 / 128, the wide path and the
former Triton-prologue inference cells). The Triton prologue of MiniWorld's `runs/msa_bench_20260921` stays in `opm_train.py` as the reference
(`_CUDA_PROLOGUE_FOR_LEGACY`); the legacy-composition switch (`MINIWORLD_OPM_H100_CUDA_ONLY`) was removed, so a native CUDA path is the only
route on H100.

## Fused inference kernel

`opm_fused_infer` (`opm_epilogue.cu`, extension flavour `_fused`, `settings.opm_fused`, default on) serves inference at d_msa 64 / d_pair 128 (MiniWorld, RF3, ESMFold2 order; N % 16, S % 64) and replaces the cuBLAS product + `opm_epilogue_kernel` pair, so the [N*32, N*32] bf16 product (134 MB at L256, 1.2 GB at L768) is neither written nor read. Persistent, 384 threads: a producer warpgroup (warp 0 TMA A / B, warp 1 the weight) and two consumer warpgroups. Work item = 16 i x 8 j = 128 pairs, four passes over c (8 values each): a pass is a 128 x 256 x S wgmma m64n256k16 GEMM (A through a 3-D TMA map [i][c][s]), the fp32 tile is rounded to bf16 and written to shared memory as Opair (k permuted so the fragment stores are conflict free), then z^T += W_q Opair^T (wgmma m64n128k16, the 64 KiB weight slice of the pass, k permuted on the host and cached) with the 128 x 128 fp32 result kept in registers across the passes; the epilogue divides by the popcount of the two tokens' mask bit rows, adds the bias (or divides after it), stages a [pair][o] tile in shared memory and writes coalesced rows (+ residual). Shared memory 227 KiB (ring 4 x 24 KiB, Opair 64 KiB, weight 4 x 16 KiB). Splitting the projection over four passes is what makes the weight stream cheap: with 32 pairs per work item the 256 KiB weight was re-read per tile and the projection cost 5 us per tile (latency and L2 bound), now it costs 12 % of the GEMM (the tensor work it adds).

Measured (CUDA graph, one H100, L256 S1024, kernel alone): cuBLAS 190 us + epilogue 59 us -> 212 us fused (178 mainloop, 14 conversion, 20 projection); the module row 0.308 -> 0.265 ms. Training still uses the unfused forward (the backward needs the product). Test: `tests/gpu/sm90/test_opm_fused_gpu.py`. Record: [opm_fused_ladder_20261026.md](../../../records/opm_fused_ladder_20261026.md).

## Dispatch (`_h100_path`)

| mode | (S, N) | prologue |
|---|---|---|
| training | S in {1024, 2048, 4096} | persistent TMA |
| training | other S | non-TMA WGMMA (native) |
| inference | (1024, 128 / 256 / 384), (2048, 128 / 256 / 384), (4096, 128) | persistent TMA; the gathered-TMA epilogue also at (1024, 384) and (2048, 384) |
| inference | other S in {1024, 2048, 4096}, N in 128..768 | CUDA prologue `opm_prologue_wide` (the former legacy cells; Triton prologue kept compiled behind `_CUDA_PROLOGUE_FOR_LEGACY`) |
| inference | any other shape | native |
| both | d_msa 128 (pair 256 / 384) or `normalize_before_proj=False` | `_forward_wide`: `opm_prologue_wide` (persistent TMA, d_msa 64 / 128) + cuBLAS + the c_z-sliced epilogue |

The choice per cell follows paired measurements (the dispatch is a measured table, not a rule). Training: the non-TMA prologue was
5-6x slower than the legacy Triton prologue, the TMA one matches or beats it at every measured cell
([record](../../../records/h100_opm_sol_20261007.md)). The former legacy inference cells (Triton prologue) now run the CUDA prologue,
1.6-5.5 % faster at all 11 cells ([record](../../../records/h100_opm_pwa_wide_20261013.md)); training and inference there were not
re-checked for bitwise equality.

## Completion

Rows: **implementation** (what the default dispatch runs), **성능 확인**
(maintainer only). The cuBLAS contractions appear in the flow of the kernels above, not in the rows.

### Inference

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

### Training

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

The tables have no cache-build row: the CUDA path is built by JIT (cache root: `settings.engine_jit_root()`, field `engine_jit_root`); one
extension per (d_msa, d_pair) is built on first use. Figures are not generated yet (B200's come from `viz.kernel_flow`).

## Speed of light

Measured 2026-10-07 on one H100 (CUDA graph, ours = default dispatch, the code of that date with the tiled count kernel); method,
ceilings and raw numbers: [record](../../../records/h100_opm_sol_20261007.md). Ceilings measured on this card: HBM 3.04 TB/s, cuBLAS bf16
703 TFLOP/s (also measured on the module's own grouped GEMM per shape). Floor = sum over the kernels of `max(bytes / HBM, FLOPs / TC)`.
The floor takes cuBLAS's measured rate as the tensor ceiling, so a step that beats it exceeds 100 %. Spread between jobs 2-3 %.

| (Length, MSA depth) | inference ms | inference SoL | training ms | training SoL |
|---|---|---|---|---|
| (128, 1024) | 0.096 | 87.6 % | 0.309 | 79.9 % |
| (256, 1024) | 0.313 | 86.8 % | 0.948 | 84.4 % |
| (384, 1024) | 0.645 | 90.3 % | 1.945 | 88.4 % |
| (512, 1024) | 1.173 | 93.1 % | 3.463 | 93.4 % |
| (640, 1024) | 1.886 | 89.7 % | 5.272 | 95.2 % |
| (768, 1024) | 2.562 | 95.6 % | 7.155 | 101.6 % |
| (384, 2048) | 1.135 | 93.5 % | 3.372 | 93.1 % |
| (768, 2048) | 4.326 | 99.7 % | 12.823 | 100.0 % |
| (384, 4096) | 2.204 | 92.0 % | 6.481 | 92.8 % |
| (768, 4096) | 7.990 | 99.3 % | 24.008 | 98.4 % |

SoL here is against the module-TC floor; against the 8192^3 cuBLAS rate it is 2-9 points lower on the large cells. The earlier
"27-36 %" figure used B200 ceilings and is void. Where the rest goes (L384 S1024, profiler): the cuBLAS GEMMs run at or above the
floor, the hand-written epilogue / dgrad / dwo / prologue kernels sit 10-40 % over their own floors (about 0.1 ms of 1.8 ms in
training), and the CUDA-graph step is ~10 % longer than the sum of the eager kernels there (cause not determined).

## Against other implementations

cuEquivariance ships no OuterProductMean kernel. Anthropic's `opt_core.ops.msa_opm` is forward only (no training column).

### Inference (CUDA graph), H100 protocol run of 2026-10-05

[anthropic_protocol_20261005.md](../../../records/anthropic_protocol_20261005.md): MiniWorld ms, Anthropic ms (upstream, compiled), PyTorch ms;
ratio = Anthropic / MiniWorld. Geometric mean 1.50x over the 18 shapes (1.26-1.97x); every shape is faster. The protocol's
MiniWorld column predates the tiled count kernel, so a re-run is a few percent faster at the TMA cells.

| (Length, MSA depth) | MiniWorld | Anthropic | PyTorch | Anthropic / MiniWorld |
|---|---|---|---|---|
| (128, 1024) | 0.113 | 0.169 | 0.135 | 1.49x |
| (256, 1024) | 0.319 | 0.566 | 0.445 | 1.77x |
| (384, 1024) | 0.658 | 1.219 | 0.923 | 1.85x |
| (512, 1024) | 1.120 | 2.106 | 1.582 | 1.88x |
| (640, 1024) | 1.639 | 3.235 | 2.403 | 1.97x |
| (768, 1024) | 2.366 | 4.625 | 3.367 | 1.96x |
| (128, 2048) | 0.172 | 0.231 | 0.218 | 1.34x |
| (256, 2048) | 0.525 | 0.767 | 0.709 | 1.46x |
| (384, 2048) | 1.135 | 1.622 | 1.423 | 1.43x |
| (512, 2048) | 1.803 | 2.792 | 2.422 | 1.55x |
| (640, 2048) | 3.036 | 4.416 | 3.626 | 1.46x |
| (768, 2048) | 4.305 | 6.400 | 5.153 | 1.49x |
| (128, 4096) | 0.280 | 0.359 | 0.395 | 1.28x |
| (256, 4096) | 0.952 | 1.200 | 1.218 | 1.26x |
| (384, 4096) | 1.938 | 2.569 | 2.441 | 1.33x |
| (512, 4096) | 3.593 | 4.516 | 4.215 | 1.26x |
| (640, 4096) | 5.685 | 7.199 | 6.609 | 1.27x |
| (768, 4096) | 8.022 | 10.166 | 9.294 | 1.27x |

### Training

Earlier runs against PyTorch compiled (their `native_final_*` artifact files are no longer in the tree; before the training
prologue change): ours / PyTorch compiled, ms: (384, 1024) 2.090 / 2.467, (768, 1024) 7.867 / 9.003, (384, 2048) 3.722 / 3.945,
(768, 2048) 13.987 / 14.809, (384, 4096) 7.030 / 7.333, (768, 4096) 24.997 / 25.591 (1.02-1.18x). The current code is 4-9 % faster than
those "ours" values at the six cells above ([record](../../../records/h100_opm_sol_20261007.md)); a protocol-style
re-run of the training table has not been done, so no current ratio is stated.

## Against B200

| | H100 (sm90) | B200 (sm100) |
|---|---|---|
| tensor core | WGMMA (register accumulators) | tcgen05, TMEM accumulators, one issuing thread |
| operands | TMA in the persistent prologue and the TMA epilogue variants; other kernels load through threads | every operand by TMA, warp-specialized roles |
| epilogue permute | gathered-TMA / half-row variants, selected per cell | one 4-D TMA box absorbs the permute |
| prologue | persistent TMA prologue at d_msa 64 and 128 in every cell (the Triton prologue is kept only as the reference) | CUDA prologue in every cell (projection run transposed so a TMEM lane is an output channel) |
| mask count | bit mask from the prologue, `opm_count_tiled` (separate launch; the fp32 cuBLAS count was removed) | `opm_norm_sm100` from the prologue's bit mask |
| dgrad rounding | `dz / n` rounded to bf16 first | `1 / n` in fp32, one bf16 rounding |
| dWo | split-K partials + `opm_reduce_partials_chunked` | one CTA owns the K range in TMEM, partials folded by a column-sum GEMV in split order |
| dispatch | measured table per cell (above) | one CUDA path |

The fusion boundaries and tensors are the same.

## Wide widths (d_msa 128: Protenix-v2 / ESMFold2 d_pair 256, OpenDDE d_pair 384)

Same fusion boundaries; the epilogue and dWo cut c_z into 128-wide CTA slices (one CTA per slice), dgrad contracts over the whole
c_z, the prologue backward widens its LayerNorm phase to 128 channels, and the CUDA prologue `opm_prologue_wide` runs the 128-row
MSA tile with the projection K = d_msa. Every stage is CUDA or cuBLAS. Measured 2026-10-13 ([record](../../../records/h100_opm_pwa_wide_20261013.md)),
CUDA graph, S 1024, ms, against the portable portable path (Anthropic's `msa_opm` cell has no d_pair 256 / 384 entry; cuEquivariance
ships no OPM):

| (d_msa, d_pair), L | inference ours / Triton | training ours / Triton | SoL inference / training |
|---|---|---|---|
| 128 / 256, 128 | 0.104 / 0.125 | 0.394 / 0.593 | 79 % / 61 % |
| 128 / 256, 384 | 0.794 / 0.865 | 2.429 / 3.284 | 80 % / 77 % |
| 128 / 256, 768 | 2.912 / 3.278 | 8.623 / 11.585 | 84 % / 84 % |
| 128 / 384, 128 | 0.121 / 0.149 | 0.430 / 0.666 | 70 % / 58 % |
| 128 / 384, 384 | 0.902 / 1.007 | 2.736 / 3.829 | 74 % / 72 % |
| 128 / 384, 768 | 3.324 / 3.727 | 9.973 / 13.722 | 77 % / 76 % |

Tests: `tests/gpu/sm90/test_opm_wide_sm90a_gpu.py`. Per-cell implementation rows are CUDA at every length and depth (padded).

## Open

- L128-384 at S1024: 80-90 % SoL at the MiniWorld width, 58-77 % at the wide widths; `opm_dgrad` at c_z 384 is about 2x its floor
  (serial stage / chunk loop, one CTA per SM) and needs a warp-specialised TMA-pipelined rewrite.
- The CUDA-graph step is about 10 % longer than the kernel sum at the large cells (cause not determined).
- Training and inference bitwise parity at the former legacy-inference cells (they now use the CUDA prologue; not re-checked bitwise).
- The gathered-TMA epilogue variants and the TMA prologue at d_msa 64 are written for the MiniWorld width only (`static_assert`).
