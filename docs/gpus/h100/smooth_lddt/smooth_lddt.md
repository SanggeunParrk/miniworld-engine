# Smooth lDDT on H100 (sm90)

Kernel-level status of `kernels/smooth_lddt` on H100. The op, its masks and kernels are described in the
[B200 page](../../b200/smooth_lddt/smooth_lddt.md); the H100 path is **the same CUDA source** (`cuda/sm100/smooth_lddt.cu`, plain CUDA-core
code, no tensor cores, so no cuBLAS), built with `-arch=sm_90a` (`cuda/sm100/__init__.py::_arch`) and launched through the same driver loader.
`refusal` serves (10, 0) and (9, 0). Shapes and dtypes are unchanged: `x_pred` [A, B, N, 3] fp32 or bf16 (fp32 compute), `gt` [B, N, 3],
`is_nuc` / `mask` [B, N] bool, any N and A. **성능 확인** is the maintainer's; no registry row yet.

## Correctness

`tests/gpu/sm100/test_smooth_lddt_gpu.py` (skip condition widened to sm90) passes 12/12 on H100: value and gradient vs the dense fp32
reference (rel. gradient error < 5e-5), whole-structure and tile kernels, empty masks, bf16, bitwise determinism, CUDA-graph replay.

## Measurements (2026-10-15, node H100 80GB, forward + backward, A = 48, B = 1, CUDA graph)

Synthetic chain structures padded with unresolved atoms (pair counts are those of this generator, not of the B200 table's real structures).
Baseline = the fastest PyTorch form: dense `[a, N, N]` expression fused by `torch.compile` (chunked over augments), fp32.

| N | resolved | nucleotide | pairs in radius | ours (ms) | compiled PyTorch (ms) | speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 2500 | 0 | 0.52 M | 0.37 | 21.8 | 59x |
| 4096 | 3276 | 352 | 1.04 M | 0.58 | 21.7 | 37x |
| 4096 | 3995 | 0 | 1.04 M | 0.51 | 21.7 | 43x |
| 8192 | 5501 | 0 | 1.44 M | 0.76 | 135.7 | 180x |
| 8192 | 7106 | 0 | 1.84 M | 0.82 | 135.5 | 166x |
| 8192 | 7959 | 0 | 2.15 M | 0.98 | 135.5 | 139x |

Eager dense PyTorch (exact `cdist`, the reference) is 1.8 s (N = 4096) / 8.7 s (N = 8192).

## Profile (`ncu --set full`, N = 8192, 7959 resolved)

| kernel | us | issue slots busy | active threads / warp | regs | DRAM |
|---|---:|---:|---:|---:|---:|
| `slddt_prep` | 88 | 81 % | 31.9 | 40 | 3 % |
| `slddt_fwd` (tile) | 470 | 70 % | 19.9 | 40 | 0.4 % |
| `slddt_bwd` (tile) | 609 | 77 % | 19.9 | 32 | 0.3 % |

Compute bound (MUFU: 2 sqrt + exp + rcp per pair, plus ~40 FMA/ALU); memory is idle. Instruction-issue floor of the pair loops is about
0.35-0.45 ms at the largest case, so the kernels sit at ~45 % of it. The loss is lane imbalance (20 of 32 active lanes from per-row
neighbour-count differences), not latency or bandwidth.

## Decision

Not pushed to SoL90. Reasons: 37-180x over the fastest other implementation, accuracy untouched, and the remaining lever (warp-cooperative
flattening of the neighbour lists to recover the idle lanes, <= ~1.4x) is a non-trivial rewrite for a one-per-step auxiliary loss.
Measured and rejected on H100: raising the whole-structure threshold (`FULL_SMEM` 110 -> 224 KB, one CTA/SM) at N = 4096: 0.37 / 0.72 / 0.69 ms
vs tile 0.38 / 0.58 / 0.51 ms, so the B200 switch point (N <= 3520) stays.

## Portable kernels (other cards; A/B on H100)

`kernels/smooth_lddt/triton/main.py` serves the op on the cards that have no CUDA kernel, and on H100 / B200 when `settings.force_portable_kernels=True` (the A/B switch;
the default H100 / B200 path above is untouched). Same contract: `x_pred` [A, B, N, 3] fp32 or bf16 (fp32 compute, gradient in the input dtype), `gt`, `is_nuc`, `mask`,
any N and A. Routing (`smooth_lddt.interface._backend`): CUDA on H100 / B200; the portable kernels on a card whose manifest has an `ok` row for `smooth_lddt_prep_triton`,
`smooth_lddt_fwd_triton` (and `smooth_lddt_bwd_triton` when the call needs a gradient) at the dtype in use; otherwise the PyTorch reference and one `UnrecordedCardWarning`.
No card has such a record yet: `python -m miniworld_engine.autotune.run_all` on the card writes it.

Design. No N x N x A tensor and no per-augment pair tensor. `smooth_lddt_prep_triton` summarises the ground-truth pair set once per call (it depends on `gt` only) as
32 x 32 tile flags (bit 0: a pair of the loss lies in the tile, bit 1: a pair carrying gradient to either atom does) and per-tile pair counts (the denominator). The forward
and backward programs own BLOCK_I rows of one (augment, batch), loop over the BLOCK_J-column tiles, skip those whose flags are empty, recompute the ground-truth distance of
each pair of a visited tile, and apply the same rational form of the four sigmoids as the CUDA kernel (one `exp`, one division). The backward gives every atom the gradient of
both of its roles in a pair, so there are no atomics; all reductions are fixed-order, hence bitwise determinism run to run. Tile sizes and warps are autotuned
(`autotune/configs/{default,grid}/smooth_lddt_*_triton.csv`); no tuned cache is committed.

Parity (H100, 13 cases: A = 1..48, B = 1..3, N = 64..3000, nucleotide fractions 0 / 0.2 / 1, 30-100 % resolved, structure scale 0.4-6, noise 0.2-10 A; bf16 input; nothing resolved):

| compared with | loss difference | gradient, relative L2 error |
|---|---:|---:|
| PyTorch reference (fp32) | <= 6e-8 | 9e-8 - 2.6e-7 (2e-4 in one case, see below) |
| CUDA kernels | <= 6e-8 | 1e-7 - 4.1e-7 |
| definition in float64 | <= 5e-7 | <= 3e-4 |

The pair score has a kink at `d_pred = d_gt`; an fp32 distance that falls on the other side of it flips one pair's term. Against float64 that limits the gradient to ~1e-4 in the
cases with small noise and many pairs (the CUDA kernels and the fp32 reference show the same, and agree with the portable kernels to 3e-7). Checker errors (registry bands are 4x):
tile flags and counts exact, forward sums 8.8e-8, gradient 1.7e-7 (fp32) / 2.7e-3 (bf16 input, rounding of the bf16 gradient). Two runs and five runs of the same call are
bitwise equal (value and gradient), CUDA-graph capture and replay match.

Speed and memory (H100, A = 48, B = 1, forward + backward, CUDA graph; synthetic structures: `chain` = random walk of 1.5 A steps in chain order, `globule` = atoms in a ball of
0.05 / A^3 in random order, the worst case for tile skipping; 10 % nucleotide, 80 % resolved; compiled PyTorch = the dense `[a, N, N]` expression, `torch.compile`, 4 augments
per chunk):

| structure | N | pairs in radius | portable (ms) | CUDA (ms) | compiled PyTorch (ms) | portable vs compiled | extra memory portable / CUDA / compiled (MiB) |
|---|---:|---:|---:|---:|---:|---:|---|
| chain | 1024 | 0.10 M | 0.30 | 0.18 | 1.59 | 5.3x | 1 / 1 / 108 |
| chain | 2048 | 0.18 M | 0.59 | 0.27 | 11.4 | 19x | 1 / 2 / 432 |
| chain | 4096 | 0.56 M | 1.69 | 0.57 | 45.7 | 27x | 2 / 4 / 1728 |
| chain | 8192 | 1.19 M | 4.16 | 1.31 | 184 | 44x | 5 / 16 / 6912 |
| globule | 1024 | 0.28 M | 0.42 | 0.28 | 3.2 | 7.7x | 1 / 1 / 108 |
| globule | 2048 | 0.72 M | 1.56 | 0.83 | 11.4 | 7.3x | 1 / 2 / 432 |
| globule | 4096 | 1.73 M | 5.84 | 2.10 | 45.7 | 7.8x | 2 / 4 / 1728 |
| globule | 8192 | 4.13 M | 22.8 | 5.50 | 184 | 8.1x | 5 / 16 / 6912 |

The PyTorch reference itself (`smooth_lddt_reference`, exact `cdist`, eager; chain structure, A = 48): N = 1024 forward 66 ms / forward + backward 107 ms (1.7 GiB); N = 2048 264 ms
/ 431 ms (6.9 GiB); N = 4096 forward 1.05 s (15 GiB), forward + backward 1.8 s only in chunks of 8 augments (4.7 GiB) -- in one call it fails with an illegal memory access (the
`cdist` backward indexes more than 2^31 elements); N = 8192 fails already in the forward ("invalid configuration argument"). The portable kernels run all of these shapes in 2-5 MiB.

Where it is slower than CUDA (2-4x): the portable programs compute dense 32-row tiles of pairs, the CUDA kernel walks the set bits of the pair mask, so the portable cost grows with
the number of non-empty tiles (random atom order) rather than the number of pairs. bf16 `x_pred` costs nothing extra (the kernels load either dtype).

Tests: `tests/gpu/any/test_smooth_lddt_portable_gpu.py` (any card; the A/B switch is named in one helper, `portable_kernels()`), `tests/cpu/compile/test_smooth_lddt_routing.py`.
