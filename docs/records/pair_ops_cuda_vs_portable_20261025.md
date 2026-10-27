# H100 CUDA pair-op kernels against the portable kernels: what was slow and what changed (2026-10-25)

Branch `perf/h100-pair-ops-cuda`. [portable_pair_ops_20261025.md](portable_pair_ops_20261025.md) measured the portable (Triton) kernels of `pair_embed_geom` / `pair_embed_ca`,
`pair_embed_adaln` and `token_pair_init` against the hand CUDA kernels on H100 and found the portable ones faster for the encoder forward (CUDA 1.03-1.19x of the portable time),
the encoder + LayerNorm training step (1.15-1.35x) and the decoder + LayerNorm training step (1.3-1.75x), and level for `pair_embed_adaln` training. On H100 the CUDA kernels are
the product, so this change finds why and fixes the CUDA kernels. `token_pair_init` and the `pair_embed_adaln` forward were already faster in CUDA and are not touched.

The tables below are the final state of the branch (after the warpgroup-uniformity and work-item fixes of the last bullet of Reading). Measured on one H100 80GB HBM3 (node01), torch 2.13.0+cu129, CUDA-graph replay (min over 12 rounds of 10 replays after one second of warm-up; the earlier record used the median of 15
single replays, so "before" below is that record's CUDA column, the portable column was re-measured and agrees with the record within 2 %). `scripts` and raw rows: the
harness is `bench_pair.py` of the parity work (portable = `force_portable_kernels`, CUDA = default).

## Root causes (Nsight Compute, SASS, nsys-free kernel timelines)

| kernel | evidence | cause | change |
|---|---|---|---|
| `pg_fwd`, `pg_fwd_ca` (gather + optional LayerNorm) | 8 warps / SM (11.8 % warps active), issue slots 47 % busy, 72 registers, one CTA per SM because the table fills shared memory; SASS: odd bf16 elements unpacked with PRMT + SHL (2 instructions), the 20 bin limits re-read from global per bucket (100 LDG) and compared one by one | latency-bound: too few warps for the long dependent chain (bin geometry, 10 table rows, two shuffle reductions); 1.5 instructions per element for the bf16 unpack; linear bucket search | 16 (encoder) / 32 (decoder) warps per CTA; unpack with one SHL / one AND; bucket by binary search over the limits in shared memory (bit-identical, the limits are ascending as `torch.bucketize` requires). Forward output and codes are bit-identical to the previous kernel |
| `pg_ln_bwd` | memory-level parallelism: one pair in flight per warp, dh load dependent on the code load; 4 serial shuffle reductions per pair | latency-bound (long-scoreboard stalls); decoder at 60 % of the DRAM roofline | `UL` pairs per step (3 encoder, 3 decoder), codes two steps and dh one step ahead (dh of dead pairs still never read), the reductions of the pairs interleaved (`ln_bwd_group`). Bit-identical |
| `pg_bwd_mma` (table gradient) | 548 M instructions for 54 M mma, 46 % issue, 126 registers, SASS shows a run-time `switch` on the feature group per element and `F2F` for the chain value | the one-hot operand was rebuilt per element in registers by every warp and every channel quarter | one-hot operand written once per tile into a swizzled shared-memory tile by two extra warps (about 11 entries per pair; entries cleared again when the buffer is reused) and consumed by `wgmma.m64n64k16` (A and B from shared memory through descriptors, one warpgroup per 64 table rows). Same fixed pair slices and tile order, bit-identical to the previous kernel on every shape tested |
| `pe_bwd_tbl` (adaln table gradient) | 2.4 ms of the 3.7 ms training step at B = 16, L = 256: shared-memory read-modify-write per channel | serialised shared-memory atomics-free accumulation | the same one-hot + wgmma kernel (`pe_bwd_tbl4`), fp32 slice partials as before; `MINIWORLD_PAIR_EMBED_BWD_IMPL=0` selects the old kernel |

What the portable kernels do differently, and why it does not need to be copied: the bins kernel + gather kernel split buys occupancy (here obtained by more warps per CTA and the cheap bucket
search), and its table gradient is the same one-hot product on the tensor cores that the CUDA kernel now uses (with wgmma instead of Triton's generated code).

## Parity and determinism

* `pair_embed_geom` / `pair_embed_ca` forward (with and without codes, with and without `ln_eps`), the 16-byte codes, the table gradient (impl 1 and the previous kernel), `ln_backward`
  and a rerun of each, compared bit for bit with the previous (origin/main) CUDA extension on B x n = 2x128, 3x100, 4x61, 2x300, 1x33, 5x97, 16x256: **equal** everywhere.
* `pair_embed_adaln` table gradient against the previous kernel (same inputs): relative L2 2e-9 .. 5e-8 (fp32 summation order only), all other gradients identical; bitwise
  identical on a rerun at four shapes. Existing tests: `tests/gpu/sm90/test_pair_embed_geom_gpu.py`, `test_pair_embed_geom_train_gpu.py`, `test_pair_embed_adaln_gpu.py`,
  `test_pair_ops_portable_vs_cuda_gpu.py`, `test_token_dit_batched_pairs_gpu.py`: 51 passed.

## Speed (CUDA graph, ms; ratio > 1: CUDA faster)

#### pair_embed_geom, forward (frozen encoder)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.141 | 0.17 | 0.120 | 1.42x | **1.17x** |
| 16 | 256 | 0.457 | 0.54 | 0.376 | 1.44x | **1.22x** |
| 16 | 384 | 0.993 | 1.17 | 0.815 | 1.44x | **1.22x** |
| 16 | 512 | 1.715 | 1.97 | 1.333 | 1.48x | **1.29x** |
| 32 | 128 | 0.249 | 0.29 | 0.206 | 1.41x | **1.21x** |
| 32 | 256 | 0.873 | 1.00 | 0.690 | 1.45x | **1.27x** |
| 32 | 384 | 1.928 | 2.08 | 1.502 | 1.38x | **1.28x** |
| 32 | 512 | 3.392 | 3.57 | 2.630 | 1.36x | **1.29x** |
| 48 | 128 | 0.352 | 0.38 | 0.277 | 1.37x | **1.27x** |
| 48 | 256 | 1.292 | 1.44 | 0.979 | 1.47x | **1.32x** |
| 48 | 384 | 2.850 | 2.94 | 2.063 | 1.43x | **1.38x** |
| 48 | 512 | 5.062 | 5.22 | 3.852 | 1.36x | **1.31x** |

#### pair_embed_geom_train + ln_eps, forward + backward (w grad)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.658 | 0.76 | 0.458 | 1.66x | **1.44x** |
| 16 | 256 | 2.132 | 2.80 | 1.611 | 1.74x | **1.32x** |
| 16 | 384 | 4.565 | 6.22 | 3.589 | 1.73x | **1.27x** |
| 16 | 512 | 8.038 | 10.66 | 6.083 | 1.75x | **1.32x** |
| 32 | 128 | 1.139 | 1.44 | 0.833 | 1.73x | **1.37x** |
| 32 | 256 | 4.054 | 5.41 | 3.122 | 1.73x | **1.30x** |
| 32 | 384 | 8.923 | 11.92 | 6.960 | 1.71x | **1.28x** |
| 32 | 512 | 15.597 | 20.89 | 12.008 | 1.74x | **1.30x** |
| 48 | 128 | 1.599 | 2.09 | 1.188 | 1.76x | **1.35x** |
| 48 | 256 | 6.041 | 7.98 | 4.533 | 1.76x | **1.33x** |
| 48 | 384 | 13.221 | 17.59 | 10.067 | 1.75x | **1.31x** |
| 48 | 512 | 23.239 | 30.93 | 17.926 | 1.73x | **1.30x** |

#### pair_embed_ca + ln_eps, forward + backward (w grad)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.338 | 0.44 | 0.248 | 1.77x | **1.36x** |
| 16 | 256 | 1.022 | 1.65 | 0.854 | 1.93x | **1.20x** |
| 16 | 384 | 2.129 | 3.65 | 1.863 | 1.96x | **1.14x** |
| 16 | 512 | 3.811 | 6.33 | 3.286 | 1.93x | **1.16x** |
| 32 | 128 | 0.563 | 0.84 | 0.449 | 1.87x | **1.25x** |
| 32 | 256 | 1.901 | 3.18 | 1.627 | 1.95x | **1.17x** |
| 32 | 384 | 4.175 | 7.23 | 3.732 | 1.94x | **1.12x** |
| 32 | 512 | 7.477 | 12.63 | 6.547 | 1.93x | **1.14x** |
| 48 | 128 | 0.778 | 1.24 | 0.661 | 1.88x | **1.18x** |
| 48 | 256 | 2.810 | 4.72 | 2.471 | 1.91x | **1.14x** |
| 48 | 384 | 6.546 | 10.68 | 5.550 | 1.92x | **1.18x** |
| 48 | 512 | 10.903 | 18.67 | 9.800 | 1.91x | **1.11x** |

#### pair_embed_adaln, forward

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.212 | 0.14 | 0.141 | 0.99x | **1.50x** |
| 16 | 256 | 0.791 | 0.41 | 0.416 | 0.99x | **1.90x** |
| 16 | 384 | 1.752 | 0.99 | 0.994 | 1.00x | **1.76x** |
| 16 | 512 | 3.086 | 1.24 | 1.246 | 1.00x | **2.48x** |
| 32 | 128 | 0.405 | 0.21 | 0.207 | 1.01x | **1.96x** |
| 32 | 256 | 1.559 | 0.73 | 0.730 | 1.00x | **2.14x** |
| 32 | 384 | 3.480 | 1.46 | 1.475 | 0.99x | **2.36x** |
| 32 | 512 | 6.171 | 2.36 | 2.380 | 0.99x | **2.59x** |
| 48 | 128 | 0.596 | 0.32 | 0.330 | 0.97x | **1.81x** |
| 48 | 256 | 2.327 | 0.99 | 0.998 | 0.99x | **2.33x** |
| 48 | 384 | 5.213 | 2.08 | 2.099 | 0.99x | **2.48x** |
| 48 | 512 | 9.236 | 3.37 | 3.395 | 0.99x | **2.72x** |

#### pair_embed_adaln, forward + backward (w, ln_w, ln_b, gamma, beta grads)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 1.032 | 0.98 | 0.530 | 1.85x | **1.95x** |
| 16 | 256 | 3.278 | 3.71 | 1.841 | 2.02x | **1.78x** |
| 16 | 384 | 8.809 | 8.74 | 4.669 | 1.87x | **1.89x** |
| 16 | 512 | 14.691 | 13.99 | 6.706 | 2.09x | **2.19x** |
| 32 | 128 | 1.692 | 1.87 | 0.927 | 2.02x | **1.83x** |
| 32 | 256 | 6.420 | 7.28 | 3.539 | 2.06x | **1.81x** |
| 32 | 384 | 16.339 | 16.13 | 8.034 | 2.01x | **2.03x** |
| 32 | 512 | 29.262 | 27.98 | 13.436 | 2.08x | **2.18x** |
| 48 | 128 | 2.760 | 2.65 | 1.370 | 1.93x | **2.01x** |
| 48 | 256 | 9.512 | 10.69 | 5.154 | 2.07x | **1.85x** |
| 48 | 384 | 24.108 | 24.14 | 11.926 | 2.02x | **2.02x** |
| 48 | 512 | 43.798 | 41.67 | 19.950 | 2.09x | **2.20x** |

#### token_pair_init, forward + backward (left, right, w_rel, w_bond grads)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.500 | 0.31 | 0.311 | 1.00x | **1.61x** |
| 16 | 256 | 1.466 | 1.01 | 1.010 | 1.00x | **1.45x** |
| 16 | 384 | 3.054 | 2.15 | 2.149 | 1.00x | **1.42x** |
| 16 | 512 | 5.292 | 3.69 | 3.639 | 1.01x | **1.45x** |
| 32 | 128 | 0.812 | 0.58 | 0.576 | 1.01x | **1.41x** |
| 32 | 256 | 2.725 | 1.95 | 1.976 | 0.99x | **1.38x** |
| 32 | 384 | 5.908 | 4.24 | 4.184 | 1.01x | **1.41x** |
| 32 | 512 | 10.441 | 7.39 | 7.314 | 1.01x | **1.43x** |
| 48 | 128 | 1.170 | 0.85 | 0.851 | 1.00x | **1.37x** |
| 48 | 256 | 4.003 | 2.93 | 2.894 | 1.01x | **1.38x** |
| 48 | 384 | 8.769 | 6.21 | 6.149 | 1.01x | **1.43x** |
| 48 | 512 | 15.483 | 11.04 | 10.924 | 1.01x | **1.42x** |

## Reading

* Encoder forward 1.17-1.38x faster than the portable kernel (median 1.28x; before: 0.84-0.97x), encoder training step with LayerNorm 1.27-1.44x (before 0.74-0.87x), adaln training
  step 1.78-2.20x (before 0.88-1.06x), adaln forward 1.5-2.7x (unchanged), token_pair_init 1.4-1.6x (unchanged).
* Decoder (`pair_embed_ca` + LayerNorm) forward + backward: 1.11-1.36x (median 1.15x; before 0.57-0.76x), >= 1.11x at every L >= 384. Two later fixes took it over 1.1x:
  (1) ptxas warned "wgmma serialized due to program dependence on compiler-inserted WG.DP in divergent path": the warpgroup role (scatter warps vs wgmma warpgroups) was derived from
  `threadIdx.x`; taking the warp index through `__shfl_sync` (provably warp-uniform) removed the serialisation, table-gradient kernel and `pair_embed_adaln` alike (decoder step 0.93 -> 0.85 ms
  at B16 L256, 3.58 -> 3.28 ms at L512); (2) `pg_fwd_ca` work items are one 32-pair chunk instead of a whole row: at B16 L384 a row per warp left 1.45 waves (0.57 -> 0.42 ms).
  A fused LayerNorm backward + table gradient (mma.sync, built earlier) was bit-exact but slower (0.81 vs 0.73 ms) and a wgmma variant is therefore not needed; block-live skipping was slower.

## La-Proteina end to end (4 x H100, node01, 2026-10-25)

`~/la-proteina` `scripts/ofd` recipe (batch 16 per GPU, cap 512, bucketed, 128-padded, warm cache, `+graph_step=true +opt.fused_adam=true`, fused pair builders, engine blocks),
engine source on `PYTHONPATH` (JIT fallback on, kernels built on first call), median s/step over steps 600-1200 (LD) / 300-1000 (AE). `base` = origin/main `b088341b` (previous CUDA
kernels), `new` = this branch. The wheel's reference numbers (LD 0.0538, AE 0.0671, steps and cache state differ) are not comparable one to one.

| stage | base median / mean | new median / mean | change (median) |
|---|---|---|---|
| LD | 0.0479 / 0.0576 | 0.0464 / 0.0537 | -3.1 % |
| AE | 0.0626 / 0.0689 | 0.0610 / 0.0664 | -2.6 % |

No regression; the step is dominated by the trunk, the pair builders are a few percent of it.
