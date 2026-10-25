# H100 CUDA pair-op kernels against the portable kernels: what was slow and what changed (2026-10-25)

Branch `perf/h100-pair-ops-cuda`. [portable_pair_ops_20261025.md](portable_pair_ops_20261025.md) measured the portable (Triton) kernels of `pair_embed_geom` / `pair_embed_ca`,
`pair_embed_adaln` and `token_pair_init` against the hand CUDA kernels on H100 and found the portable ones faster for the encoder forward (CUDA 1.03-1.19x of the portable time),
the encoder + LayerNorm training step (1.15-1.35x) and the decoder + LayerNorm training step (1.3-1.75x), and level for `pair_embed_adaln` training. On H100 the CUDA kernels are
the product, so this change finds why and fixes the CUDA kernels. `token_pair_init` and the `pair_embed_adaln` forward were already faster in CUDA and are not touched.

Measured on one H100 80GB HBM3 (node01), torch 2.13.0+cu129, CUDA-graph replay (min over 12 rounds of 10 replays after one second of warm-up; the earlier record used the median of 15
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
| 16 | 128 | 0.142 | 0.17 | 0.121 | 1.40x | **1.17x** |
| 16 | 256 | 0.459 | 0.54 | 0.377 | 1.43x | **1.22x** |
| 16 | 384 | 0.993 | 1.17 | 0.815 | 1.44x | **1.22x** |
| 16 | 512 | 1.715 | 1.97 | 1.332 | 1.48x | **1.29x** |
| 32 | 128 | 0.249 | 0.29 | 0.207 | 1.40x | **1.20x** |
| 32 | 256 | 0.874 | 1.00 | 0.691 | 1.45x | **1.26x** |
| 32 | 384 | 1.928 | 2.08 | 1.501 | 1.39x | **1.28x** |
| 32 | 512 | 3.393 | 3.57 | 2.635 | 1.35x | **1.29x** |
| 48 | 128 | 0.352 | 0.38 | 0.277 | 1.37x | **1.27x** |
| 48 | 256 | 1.292 | 1.44 | 0.980 | 1.47x | **1.32x** |
| 48 | 384 | 2.850 | 2.94 | 2.056 | 1.43x | **1.39x** |
| 48 | 512 | 5.063 | 5.22 | 3.821 | 1.37x | **1.33x** |

#### pair_embed_geom_train + ln_eps, forward + backward (w grad)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.653 | 0.76 | 0.472 | 1.61x | **1.38x** |
| 16 | 256 | 2.135 | 2.80 | 1.674 | 1.67x | **1.28x** |
| 16 | 384 | 4.586 | 6.22 | 3.743 | 1.66x | **1.23x** |
| 16 | 512 | 8.013 | 10.66 | 6.338 | 1.68x | **1.26x** |
| 32 | 128 | 1.137 | 1.44 | 0.863 | 1.67x | **1.32x** |
| 32 | 256 | 4.136 | 5.41 | 3.212 | 1.68x | **1.29x** |
| 32 | 384 | 8.959 | 11.92 | 7.206 | 1.65x | **1.24x** |
| 32 | 512 | 15.695 | 20.89 | 12.586 | 1.66x | **1.25x** |
| 48 | 128 | 1.606 | 2.09 | 1.240 | 1.69x | **1.30x** |
| 48 | 256 | 5.982 | 7.98 | 4.696 | 1.70x | **1.27x** |
| 48 | 384 | 13.153 | 17.59 | 10.484 | 1.68x | **1.25x** |
| 48 | 512 | 23.260 | 30.93 | 18.697 | 1.65x | **1.24x** |

#### pair_embed_ca + ln_eps, forward + backward (w grad)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.341 | 0.44 | 0.313 | 1.41x | **1.09x** |
| 16 | 256 | 1.025 | 1.65 | 0.933 | 1.77x | **1.10x** |
| 16 | 384 | 2.128 | 3.65 | 2.191 | 1.67x | **0.97x** |
| 16 | 512 | 3.710 | 6.33 | 3.606 | 1.76x | **1.03x** |
| 32 | 128 | 0.567 | 0.84 | 0.490 | 1.71x | **1.16x** |
| 32 | 256 | 1.916 | 3.18 | 1.796 | 1.77x | **1.07x** |
| 32 | 384 | 4.197 | 7.23 | 4.088 | 1.77x | **1.03x** |
| 32 | 512 | 7.473 | 12.63 | 7.186 | 1.76x | **1.04x** |
| 48 | 128 | 0.778 | 1.24 | 0.750 | 1.65x | **1.04x** |
| 48 | 256 | 2.797 | 4.72 | 2.713 | 1.74x | **1.03x** |
| 48 | 384 | 6.169 | 10.68 | 6.217 | 1.72x | **0.99x** |
| 48 | 512 | 10.898 | 18.67 | 10.768 | 1.73x | **1.01x** |

#### pair_embed_adaln, forward

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.213 | 0.14 | 0.141 | 0.99x | **1.51x** |
| 16 | 256 | 0.792 | 0.41 | 0.415 | 0.99x | **1.91x** |
| 16 | 384 | 1.754 | 0.99 | 0.995 | 0.99x | **1.76x** |
| 16 | 512 | 3.090 | 1.24 | 1.243 | 1.00x | **2.49x** |
| 32 | 128 | 0.405 | 0.21 | 0.206 | 1.02x | **1.97x** |
| 32 | 256 | 1.561 | 0.73 | 0.739 | 0.99x | **2.11x** |
| 32 | 384 | 3.480 | 1.46 | 1.474 | 0.99x | **2.36x** |
| 32 | 512 | 6.170 | 2.36 | 2.381 | 0.99x | **2.59x** |
| 48 | 128 | 0.595 | 0.32 | 0.326 | 0.98x | **1.83x** |
| 48 | 256 | 2.328 | 0.99 | 1.015 | 0.98x | **2.29x** |
| 48 | 384 | 5.215 | 2.08 | 2.091 | 0.99x | **2.49x** |
| 48 | 512 | 9.240 | 3.37 | 3.392 | 0.99x | **2.72x** |

#### pair_embed_adaln, forward + backward (w, ln_w, ln_b, gamma, beta grads)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 1.046 | 0.98 | 0.530 | 1.85x | **1.97x** |
| 16 | 256 | 3.269 | 3.71 | 1.827 | 2.03x | **1.79x** |
| 16 | 384 | 8.825 | 8.74 | 4.625 | 1.89x | **1.91x** |
| 16 | 512 | 14.767 | 13.99 | 6.630 | 2.11x | **2.23x** |
| 32 | 128 | 1.693 | 1.87 | 0.918 | 2.04x | **1.84x** |
| 32 | 256 | 6.419 | 7.28 | 3.501 | 2.08x | **1.83x** |
| 32 | 384 | 16.349 | 16.13 | 7.962 | 2.03x | **2.05x** |
| 32 | 512 | 29.281 | 27.98 | 13.157 | 2.13x | **2.23x** |
| 48 | 128 | 2.778 | 2.65 | 1.356 | 1.95x | **2.05x** |
| 48 | 256 | 9.524 | 10.69 | 5.090 | 2.10x | **1.87x** |
| 48 | 384 | 24.131 | 24.14 | 11.831 | 2.04x | **2.04x** |
| 48 | 512 | 43.778 | 41.67 | 19.749 | 2.11x | **2.22x** |

#### token_pair_init, forward + backward (left, right, w_rel, w_bond grads)

| B | L | portable ms | CUDA before ms | CUDA after ms | CUDA after vs before | portable / CUDA after |
|---|---|---|---|---|---|---|
| 16 | 128 | 0.502 | 0.31 | 0.309 | 1.00x | **1.62x** |
| 16 | 256 | 1.464 | 1.01 | 1.007 | 1.00x | **1.45x** |
| 16 | 384 | 3.056 | 2.15 | 2.144 | 1.00x | **1.43x** |
| 16 | 512 | 5.291 | 3.69 | 3.673 | 1.00x | **1.44x** |
| 32 | 128 | 0.810 | 0.58 | 0.584 | 0.99x | **1.39x** |
| 32 | 256 | 2.746 | 1.95 | 1.977 | 0.99x | **1.39x** |
| 32 | 384 | 5.957 | 4.24 | 4.242 | 1.00x | **1.40x** |
| 32 | 512 | 10.422 | 7.39 | 7.468 | 0.99x | **1.40x** |
| 48 | 128 | 1.168 | 0.85 | 0.843 | 1.01x | **1.39x** |
| 48 | 256 | 3.998 | 2.93 | 2.913 | 1.01x | **1.37x** |
| 48 | 384 | 8.806 | 6.21 | 6.291 | 0.99x | **1.40x** |
| 48 | 512 | 15.508 | 11.04 | 10.927 | 1.01x | **1.42x** |

## Reading

* Encoder forward 1.17-1.39x faster than the portable kernel (median 1.28x; before: 0.84-0.97x), encoder training step with LayerNorm 1.23-1.38x (before 0.74-0.87x), adaln training
  step 1.8-2.2x (before 0.88-1.06x), adaln forward 1.5-2.7x (unchanged), token_pair_init 1.4-1.6x (unchanged).
* Decoder (`pair_embed_ca` + LayerNorm) forward + backward: CUDA is now 0.97-1.16x of the portable time (median 1.03x; before 0.57-0.76x). **Not at the 1.1x target for L >= 384.** Every
  kernel of this path is at the DRAM roofline (forward writes 0.54 GB per 16x256x256 pair, LayerNorm backward reads 0.35 and writes 0.54 GB, the table gradient reads 0.54 GB; 65 % to
  70 % of peak in Nsight Compute), so the only remaining lever is the d-pair round trip between the last two kernels. A fused LayerNorm backward + table gradient kernel (all 157 x 256
  accumulators in registers, LayerNorm of the tile in a first phase, mma.sync in a second) was built and verified bit-exact against the unfused pair (1 element of 40192 one bf16 ulp
  apart), but ran 0.81 ms against 0.73 ms for the two kernels because the two phases do not overlap; a wgmma variant (asynchronous products overlapping the next tile's LayerNorm) did not fit in
  shared memory at 80 pairs per tile and was not built. Skipping the traffic of blocks without a live pair (35 % of the pairs in this benchmark) was also tried (block-live flags, aligned
  slices) and was slower (extra dependent loads), so it is not part of the change.

## La-Proteina end to end (4 x H100, node01, 2026-10-25)

`~/la-proteina` `scripts/ofd` recipe (batch 16 per GPU, cap 512, bucketed, 128-padded, warm cache, `+graph_step=true +opt.fused_adam=true`, fused pair builders, engine blocks),
engine source on `PYTHONPATH` (JIT fallback on, kernels built on first call), median s/step over steps 600-1200 (LD) / 300-1000 (AE). `base` = origin/main `b088341b` (previous CUDA
kernels), `new` = this branch. The wheel's reference numbers (LD 0.0538, AE 0.0671, steps and cache state differ) are not comparable one to one.

| stage | base median / mean | new median / mean | change (median) |
|---|---|---|---|
| LD | 0.0479 / 0.0576 | 0.0464 / 0.0537 | -3.1 % |
| AE | 0.0626 / 0.0689 | 0.0610 / 0.0664 | -2.6 % |

No regression; the step is dominated by the trunk, the pair builders are a few percent of it.
