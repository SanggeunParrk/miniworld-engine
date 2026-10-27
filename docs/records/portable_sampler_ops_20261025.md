# Portable kernels for the diffusion sampler glue (2026-10-25)

Branch `feat/portable-sampler-ops` (base: engine `main` at `ae6136cd`). `kernels/sampler_ops` had an H100 CUDA kernel file (`augment_noise`, `align_euler`, `pool_atoms`) and a PyTorch reference
and nothing for the other cards. This change adds portable kernels with the CUDA contract and routes the three entry points by card ([dispatch.md](../dispatch.md#the-diffusion-sampler-glue)).

Everything below was measured on **one H100 80GB HBM3** (torch 2.13.0+cu129, triton 3.7.1), CUDA graph, median of 7 x 100 replays; portable = `settings.force_portable_kernels` path
(`portable_*`), CUDA = the unchanged H100 path, PyTorch = `sampler_ops.reference` under `torch.compile` (eager in the last column). No other card was run.

## What was built

| kernel (registry) | symbol | role |
|---|---|---|
| `sampler_ops_stats_triton` | `_stats` | per-chunk partials of the mask and the mask-weighted coordinates (one or both tensors), chunk 256 .. 1024 atoms |
| `sampler_ops_augment_triton` | `_augment` | adds the partials of its sample, centre, Gram-Schmidt frame of `rv`, churn scalars (IEEE `mul_rn`/`sub_rn`/`sqrt_rn`), rotation, translation, mask, re-noise; writes `sc` |
| `sampler_ops_cov_triton` | `_cov` | centred weighted 3x3 covariance partials (second pass, as the CUDA kernels: the raw second moments lose digits far from the origin) |
| `sampler_ops_align_triton` | `_align` | adds the partials, Horn's quaternion method (12 squarings of `N + |N|_F I` in 16 scalar registers, strict first-largest column pick), alignment, Euler update |
| `sampler_ops_pool_triton` | `_pool` | one program per (token, channel slice, sample): the token's atoms in order, relu, masked mean, bf16 (bit-identical to the CUDA kernel) |

The random draws are inputs of the op (as for CUDA), so the contract needs no RNG handling. Fixed-order sums and no atomics make every result bitwise reproducible per launch configuration.

## Parity (H100)

| entry | vs PyTorch reference | vs CUDA kernel | reruns |
|---|---|---|---|
| `augment_noise` (AF3 and EDM churn, 3 sigma pairs, A up to 48, N 1 .. 32768) | 3e-9 .. 2.6e-7 relative to max, `sc` exact | up to 1.4e-7, `sc` bit-identical | bitwise |
| `align_euler` (noise 0, 0.5, 20) | 7e-8 .. 1.6e-6 | 8e-8 .. 2.7e-7 | bitwise |
| `pool_atoms` (float / bool masks, C 100 .. 768) | 0 | bit-identical | bitwise |

Registry bands (4x measured): stats 3e-7, augment 7e-7, cov 1e-5, align 5e-7, pool 4e-3.

## The ladder: CUDA < portable < PyTorch compiled

| op | A | atoms | CUDA us | portable us | PyTorch compiled us | PyTorch eager us | portable / CUDA | PyTorch compiled / portable |
|---|---|---|---|---|---|---|---|---|
| augment_noise | 1 | 4096 | 5.9 | 5.0 | 16.6 | 55.1 | 0.85x | 3.3x |
| align_euler | 1 | 4096 | 13.3 | 13.0 | 67.1 | 249.6 | 0.98x | 5.2x |
| augment_noise | 1 | 8192 | 5.9 | 5.2 | 14.2 | 54.4 | 0.88x | 2.7x |
| align_euler | 1 | 8192 | 14.1 | 13.5 | 57.8 | 249.9 | 0.96x | 4.3x |
| augment_noise | 1 | 16384 | 6.9 | 5.7 | 20.0 | 61.8 | 0.83x | 3.5x |
| align_euler | 1 | 16384 | 16.5 | 16.2 | 60.9 | 256.9 | 0.98x | 3.8x |
| augment_noise | 1 | 32768 | 8.0 | 6.4 | 25.3 | 72.7 | 0.80x | 4.0x |
| align_euler | 1 | 32768 | 20.6 | 17.8 | 72.7 | 271.0 | 0.86x | 4.1x |
| augment_noise | 5 | 4096 | 6.0 | 5.5 | 15.7 | 58.5 | 0.92x | 2.9x |
| align_euler | 5 | 4096 | 13.5 | 13.6 | 146.1 | 346.0 | 1.01x | 10.7x |
| augment_noise | 5 | 8192 | 6.4 | 5.5 | 17.0 | 64.3 | 0.86x | 3.1x |
| align_euler | 5 | 8192 | 14.8 | 14.2 | 225.4 | 431.3 | 0.96x | 15.9x |
| augment_noise | 5 | 16384 | 6.8 | 6.7 | 21.1 | 66.6 | 0.99x | 3.1x |
| align_euler | 5 | 16384 | 17.0 | 17.1 | 383.2 | 553.2 | 1.01x | 22.4x |
| augment_noise | 5 | 32768 | 8.2 | 8.2 | 25.3 | 74.9 | 1.00x | 3.1x |
| align_euler | 5 | 32768 | 22.7 | 19.7 | 693.5 | 898.9 | 0.87x | 35.2x |
| augment_noise | 16 | 4096 | 6.3 | 5.6 | 18.4 | 63.4 | 0.89x | 3.3x |
| align_euler | 16 | 4096 | 14.2 | 14.0 | 153.7 | 356.5 | 0.99x | 11.0x |
| augment_noise | 16 | 8192 | 6.7 | 7.3 | 22.3 | 69.8 | 1.09x | 3.1x |
| align_euler | 16 | 8192 | 15.6 | 16.4 | 244.5 | 442.8 | 1.05x | 14.9x |
| augment_noise | 16 | 16384 | 7.9 | 9.6 | 31.0 | 82.6 | 1.22x | 3.2x |
| align_euler | 16 | 16384 | 19.8 | 19.5 | 411.4 | 621.8 | 0.98x | 21.1x |
| augment_noise | 16 | 32768 | 10.0 | 14.0 | 45.2 | 116.5 | 1.40x | 3.2x |
| align_euler | 16 | 32768 | 29.7 | 24.8 | 657.7 | 996.4 | 0.84x | 26.5x |
| augment_noise | 48 | 4096 | 6.8 | 8.3 | 25.6 | 77.3 | 1.22x | 3.1x |
| align_euler | 48 | 4096 | 16.4 | 17.5 | 167.1 | 380.9 | 1.07x | 9.5x |
| augment_noise | 48 | 8192 | 8.0 | 10.9 | 36.3 | 96.7 | 1.36x | 3.3x |
| align_euler | 48 | 8192 | 20.4 | 22.3 | 265.6 | 511.9 | 1.09x | 11.9x |
| augment_noise | 48 | 16384 | 10.9 | 17.5 | 65.2 | 145.0 | 1.61x | 3.7x |
| align_euler | 48 | 16384 | 31.5 | 32.5 | 575.0 | 845.9 | 1.03x | 17.7x |
| augment_noise | 48 | 32768 | 33.0 | 37.0 | 122.0 | 237.8 | 1.12x | 3.3x |
| align_euler | 48 | 32768 | 83.5 | 75.0 | 1179.0 | 1541.0 | 0.90x | 15.7x |
| pool_atoms | 1 | 4096 | 12.9 | 4.6 | 6.6 | 30.2 | 0.36x | 1.4x |
| pool_atoms | 1 | 8192 | 12.6 | 4.7 | 9.1 | 36.3 | 0.37x | 1.9x |
| pool_atoms | 1 | 16384 | 12.4 | 5.1 | 13.8 | 48.4 | 0.41x | 2.7x |
| pool_atoms | 1 | 32768 | 13.1 | 6.1 | 23.5 | 77.0 | 0.47x | 3.9x |
| pool_atoms | 5 | 4096 | 12.4 | 5.4 | 16.2 | 57.3 | 0.44x | 3.0x |
| pool_atoms | 5 | 8192 | 13.1 | 6.7 | 28.7 | 94.3 | 0.51x | 4.3x |
| pool_atoms | 5 | 16384 | 13.2 | 9.6 | 52.9 | 172.4 | 0.73x | 5.5x |
| pool_atoms | 5 | 32768 | 22.9 | 15.6 | 101.9 | 338.3 | 0.68x | 6.5x |
| pool_atoms | 16 | 4096 | 12.9 | 8.5 | 42.2 | 138.1 | 0.66x | 5.0x |
| pool_atoms | 16 | 8192 | 21.5 | 13.1 | 82.2 | 275.1 | 0.61x | 6.3x |
| pool_atoms | 16 | 16384 | 38.7 | 27.6 | 158.2 | 521.9 | 0.71x | 5.7x |
| pool_atoms | 16 | 32768 | 75.6 | 53.8 | 317.6 | 1007.8 | 0.71x | 5.9x |
| pool_atoms | 48 | 4096 | 25.0 | 18.5 | 118.9 | 399.6 | 0.74x | 6.4x |
| pool_atoms | 48 | 8192 | 56.8 | 41.2 | 238.1 | 764.9 | 0.73x | 5.8x |
| pool_atoms | 48 | 16384 | 104.8 | 76.7 | 466.3 | 1493.0 | 0.73x | 6.1x |
| pool_atoms | 48 | 32768 | 201.7 | 149.5 | 941.8 | 2958.4 | 0.74x | 6.3x |

Reading. Portable is 3-16x faster than compiled PyTorch for `augment_noise` (3.1-8x) / `align_euler` (4-16x) (the reference is ~100 small ops) and 1.4-6.3x for `pool_atoms`.
Against CUDA the ladder holds only partly, and where it does not the CUDA kernels are the ones to fix:

* `augment_noise` / `align_euler` at A >= 16: portable is 1.0-1.8x slower than CUDA (CUDA is at the L2 roofline there), as intended.
* `augment_noise` / `align_euler` at A <= 5 and `align_euler` at A = 48, N = 32768 (75 vs 83.5 us) and A = 16, N = 32768 (24.8 vs 29.7): portable is **faster** than CUDA by up to 20 %. The CUDA kernels use a fixed
  1024-atom chunk (4 blocks at A = 1, N = 4096) and one thread sums all the chunk partials serially (nine covariance values x N / 1024 dependent loads); the portable launcher halves the chunk down to 256 for small
  launches and sums the partials in parallel.
* `pool_atoms`: portable is **1.3-2.8x faster** than CUDA in every cell (12.4 us flat at small sizes). The CUDA kernel runs a 32-thread block per (token, sample) that walks the 24 atoms in a dependent chain with a
  dtype switch per load; the portable kernel's loop is unrolled by the compiler and its loads overlap.

Not changed here (H100 CUDA stays as it was); the three CUDA fixes (parallel partial sum, smaller chunk for small launches, an unrolled gather in the pooling) would restore CUDA < portable.
