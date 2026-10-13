# OuterProductMean on H100 (sm90)

Kernel-level status of `OuterProductMean` on H100; the module-level summary is in [h100.md](../h100.md) and the dispatch contract
in [dispatch.md](../dispatch.md). bf16; the module's width is fixed (d_msa 64, d_hidden 32, d_pair 128), so the columns are
(Length, MSA depth). The B200 page of the same module is [../../b200/opm/opm.md](../../b200/opm/opm.md); the fusion algorithm is
the same, the instructions are not (see [Against B200](#against-b200)).

Dispatch: `integrations/opm_train.py` (`_h100_path`); kernels: `integrations/csrc/opm_epilogue.cu` (one extension, built by JIT for sm_90a).
Served when `implementation=miniworld` (or `anthropic`), bf16, batch 1, `normalize_before_proj` (the AF3 order), no interchain
masking, N (length) a multiple of 64 and S (MSA depth) a multiple of 256, inference and training alike; the pair residual is added in
the epilogue (its gradient is `dz`). The `opm_train` setting (default on) turns the path off. Every other call runs the portable Triton path.
Not served on H100: other widths (MSA 128 with pair 256 / 384 in Protenix-v2, ESMFold2 and OpenDDE) and `normalize_before_proj=False`
([todo.md](../todo.md) item 5).

## Fusion

Per call (training adds the right-hand column), the same three kernel boundaries as B200:

| stage | forward | backward |
|---|---|---|
| F1 / B4 | prologue: LayerNorm + both projections + mask straight into the grouped layouts A2 `[(i,c), s]` and BT `[(j,e), s]` (+ LN statistics, bit mask) | prologue backward: mask, both projections and the LayerNorm backward in one pass -> dm, dWa, dWb, dgamma, dbeta |
| GEMM | O = A2 . BT^T, cuBLAS | dA, dB: cuBLAS on the grouped dO (plus O again when the `opm_train_save_o` setting is off) |
| F3 / B1, B3 | epilogue: permute `[i,j,c,e] -> [(i,j),(c,e)]`, `/ n`, `W_o`, bias, residual, bf16 | dgrad: `dz / n . W_o` into the grouped layout (+ dbias); dWo off the grouped O |
| count | `n[i,j] = max(1, sum_s mask[s,i] mask[s,j])` from the bit mask (`opm_count_tiled`, exact) | reused |

Kernels (`opm_epilogue.cu`): `opm_prologue_tma_hopper` (persistent, TMA + WGMMA), `opm_prologue_wgmma_native` (non-TMA),
`opm_epilogue_kernel`, `opm_dgrad_kernel`, `opm_dwo_kernel` + `opm_reduce_partials_chunked`, `opm_prologue_bwd_kernel`,
`opm_count_tiled`, `opm_pack_native`, `opm_finish_native`. The Triton prologue of MiniWorld's `runs/msa_bench_20260921` still serves the inference cells above; the legacy-composition switch
(`MINIWORLD_OPM_H100_CUDA_ONLY`) was removed, so the native CUDA path is the only training route on H100.

## Dispatch (`_h100_path`)

| mode | (S, N) | prologue |
|---|---|---|
| training | S in {1024, 2048, 4096} | persistent TMA |
| training | other S | non-TMA WGMMA (native) |
| inference | (1024, 128 / 256 / 384), (2048, 128 / 256 / 384), (4096, 128) | persistent TMA; the gathered-TMA epilogue also at (1024, 384) and (2048, 384) |
| inference | other S in {1024, 2048, 4096}, N in 128..768 | legacy composition (Triton prologue) |
| inference | any other shape | native |

The choice per cell follows paired measurements (the dispatch is a measured table, not a rule). Training: the non-TMA prologue was
5-6x slower than the legacy Triton prologue, the TMA one matches or beats it at every measured cell
([record](../../../records/h100_opm_sol_20261007.md)). Training and inference are not bitwise equal at the cells whose inference
path is the legacy composition.

## Completion

Rows: **implementation** (what the default dispatch runs; "CUDA + Triton" = the inference prologue is Triton), **성능 확인**
(maintainer only). The cuBLAS contractions appear in the flow of the kernels above, not in the rows.

### Inference

| (Length, MSA depth) | (128, 1024) | (256, 1024) | (384, 1024) | (512, 1024) | (640, 1024) | (768, 1024) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA + Triton | CUDA + Triton | CUDA + Triton |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 2048) | (256, 2048) | (384, 2048) | (512, 2048) | (640, 2048) | (768, 2048) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA + Triton | CUDA + Triton | CUDA + Triton |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

| (Length, MSA depth) | (128, 4096) | (256, 4096) | (384, 4096) | (512, 4096) | (640, 4096) | (768, 4096) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA + Triton | CUDA + Triton | CUDA + Triton | CUDA + Triton | CUDA + Triton |
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

The tables have no cache-build row: the CUDA path is built by JIT (cache root: `settings.engine_jit_root()`, field `engine_jit_root`), and the Triton prologue of the
inference cells was not checked with `dev cache-status`. Figures are not generated yet (B200's come from `viz.kernel_flow`).

## Speed of light

Measured 2026-10-07 on one H100 (CUDA graph, ours = default dispatch, code of this branch with the tiled count kernel); method,
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

Earlier runs against PyTorch compiled (`benchmarks/modules/outer_product/artifacts/`, `native_final_*` files, before the training
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
| prologue | TMA prologue; the legacy Triton prologue still serves inference cells above | CUDA prologue in every cell (projection run transposed so a TMEM lane is an output channel) |
| mask count | bit mask from the prologue, `opm_count_tiled` (separate launch; the fp32 cuBLAS count was removed) | `opm_norm_sm100` from the prologue's bit mask |
| dgrad rounding | `dz / n` rounded to bf16 first | `1 / n` in fp32, one bf16 rounding |
| dWo | split-K partials + `opm_reduce_partials_chunked` | one CTA owns the K range in TMEM, partials folded by a column-sum GEMV in split order |
| dispatch | measured table per cell (above) | one CUDA path |

The fusion boundaries and tensors are the same.

## Open

- Inference cells above marked "CUDA + Triton": the TMA prologue was slower than the legacy one there when qualified; a CUDA
  prologue that wins at those cells would complete the table.
- L128-384 at S1024: 80-90 % SoL; hand-written epilogue / dgrad / dwo kernels are 10-40 % over their floors.
- Widths other than 64 / 32 / 128 and `normalize_before_proj=False` ([todo.md](../todo.md) item 5).
- Training and inference bitwise parity at the legacy-inference cells.
