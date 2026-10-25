# Sampler glue (`sampler_ops`) on H100 (sm90)

Kernel-level status of `kernels/sampler_ops`: the rigid-motion glue of an EDM / AF3 diffusion step (`augment_noise`: centre, random rotation and translation, re-noise; `align_euler`: Kabsch alignment +
Euler update) and the atom encoder's `pool_atoms`. Module-level summary: [h100.md](../h100.md). Not in the shape registry and no bench target, so there is no **성능 확인** to give (the maintainer's call;
the cells stay empty). **cache build ✓** for the CUDA kernels (one extension, no shape keys, prebuilt bundle case `sampler_ops`).

## Scope and dispatch

| op | dtype | served by |
|---|---|---|
| `augment_noise`, `align_euler` | fp32 coordinates `[A, N, 3]`, fp32 mask | CUDA sm_90a; portable kernels under the A/B switch and on a recorded card |
| `pool_atoms` | bf16 features `[S, Q, C]`, `C % 4 == 0` on CUDA | same |

H100 runs `cuda/sampler_ops.cu`. Any other card runs the portable kernels once the engine holds a record for it and otherwise the PyTorch reference with one warning
([dispatch.md](../../../dispatch.md#the-diffusion-sampler-glue)); the A/B switch puts every card on the portable kernels.

## Portable kernels

| kernel | role |
|---|---|
| `_stats` | chunk partials of the mask / weighted coordinates |
| `_augment` | partial sum, frame, churn scalars, rotate + translate + mask + noise |
| `_cov` | centred covariance partials |
| `_align` | partial sum, Horn quaternion rotation, Euler update |
| `_pool` | gather + relu + masked mean per (token, channel slice, sample) |

## CUDA kernels

| kernel | role |
|---|---|
| `moments_k<K>` / `moments2_k<K>` | per-chunk partials of the mask and the weighted coordinates (one / both tensors); `K` atoms per thread, chunk 256 / 512 / 1024 atoms |
| `augment_noise_k<K>` | rotate + translate + mask + re-noise; the frame and churn scalars are computed, and the block's inputs are read, before the partials arrive |
| `cov_k<K>` | centred covariance partials |
| `align_euler_k<K>` | partial sums, Horn quaternion rotation (warp 0), Euler update |
| `pool_atoms_fast_k<idx, gather, mask dtype>` | one warp-batch of up to 32 atoms per round trip (a lane loads one atom's mask / row, the others get it by shuffle); `pool_atoms_k` stays as the generic dtype fallback |

The kernels of one op are chained with programmatic dependent launch (the next kernel's input loads and scalar prologue run while the previous one drains; `griddepcontrol.wait` before the
first read of its output), and the chunk partials of a sample are staged in shared memory and added in chunk order by one thread per value, so the sum is the serial one without the chain of
dependent loads. A launch smaller than two waves of blocks uses 512 / 256-atom chunks (as the portable kernels). With the chunk of 1024 atoms the results are bit-identical to the previous CUDA
kernels; with a smaller chunk the fp32 sums associate differently (relative to the largest coordinate: at most 1.8e-7 for `augment_noise`, 8.7e-8 for `align_euler`, the same order as the
distance of the old kernels to the PyTorch reference). `pool_atoms` is bit-identical in every case. Record: [records/h100_sampler_ops_cuda_20261025.md](../../../records/h100_sampler_ops_cuda_20261025.md).

## Measurements (2026-10-25)

H100, CUDA graph, microseconds per call, after the CUDA change; `CUDA before us` is the previous CUDA kernel. The PyTorch columns are those of
[records/portable_sampler_ops_20261025.md](../../../records/portable_sampler_ops_20261025.md) (portable / CUDA is above 1 where CUDA is the faster one, which is the order the engine keeps).

| op | A | atoms | CUDA before us | CUDA us | portable us | PyTorch compiled us | PyTorch eager us | portable / CUDA | PyTorch compiled / portable |
|---|---|---|---|---|---|---|---|---|---|
| augment_noise | 1 | 4096 | 5.9 | 4.1 | 4.6 | 16.6 | 55.1 | 1.12x | 3.6x |
| align_euler | 1 | 4096 | 13.3 | 7.8 | 13.1 | 67.1 | 249.6 | 1.68x | 5.1x |
| augment_noise | 1 | 8192 | 5.9 | 4.5 | 5.0 | 14.2 | 54.4 | 1.11x | 2.8x |
| align_euler | 1 | 8192 | 14.1 | 8.4 | 14.0 | 57.8 | 249.9 | 1.67x | 4.1x |
| augment_noise | 1 | 16384 | 6.9 | 4.6 | 6.0 | 20.0 | 61.8 | 1.30x | 3.3x |
| align_euler | 1 | 16384 | 16.5 | 9.2 | 15.8 | 60.9 | 256.9 | 1.72x | 3.9x |
| augment_noise | 1 | 32768 | 8.0 | 5.5 | 6.2 | 25.3 | 72.7 | 1.13x | 4.1x |
| align_euler | 1 | 32768 | 20.6 | 11.5 | 17.5 | 72.7 | 271.0 | 1.52x | 4.2x |
| augment_noise | 5 | 4096 | 6.0 | 4.8 | 5.3 | 15.7 | 58.5 | 1.10x | 3.0x |
| align_euler | 5 | 4096 | 13.5 | 8.1 | 13.7 | 146.1 | 346.0 | 1.69x | 10.7x |
| augment_noise | 5 | 8192 | 6.4 | 4.9 | 5.7 | 17.0 | 64.3 | 1.16x | 3.0x |
| align_euler | 5 | 8192 | 14.8 | 9.4 | 13.9 | 225.4 | 431.3 | 1.48x | 16.2x |
| augment_noise | 5 | 16384 | 6.8 | 5.6 | 6.8 | 21.1 | 66.6 | 1.21x | 3.1x |
| align_euler | 5 | 16384 | 17.0 | 11.4 | 17.2 | 383.2 | 553.2 | 1.51x | 22.3x |
| augment_noise | 5 | 32768 | 8.2 | 6.1 | 8.5 | 25.3 | 74.9 | 1.39x | 3.0x |
| align_euler | 5 | 32768 | 22.7 | 13.1 | 19.9 | 693.5 | 898.9 | 1.52x | 34.8x |
| augment_noise | 16 | 4096 | 6.3 | 5.0 | 5.6 | 18.4 | 63.4 | 1.12x | 3.3x |
| align_euler | 16 | 4096 | 14.2 | 9.5 | 14.4 | 153.7 | 356.5 | 1.52x | 10.7x |
| augment_noise | 16 | 8192 | 6.7 | 5.5 | 7.0 | 22.3 | 69.8 | 1.27x | 3.2x |
| align_euler | 16 | 8192 | 15.6 | 11.2 | 16.5 | 244.5 | 442.8 | 1.47x | 14.8x |
| augment_noise | 16 | 16384 | 7.9 | 6.5 | 9.4 | 31.0 | 82.6 | 1.45x | 3.3x |
| align_euler | 16 | 16384 | 19.8 | 14.4 | 19.8 | 411.4 | 621.8 | 1.38x | 20.8x |
| augment_noise | 16 | 32768 | 10.0 | 8.4 | 14.1 | 45.2 | 116.5 | 1.68x | 3.2x |
| align_euler | 16 | 32768 | 29.7 | 23.0 | 25.2 | 657.7 | 996.4 | 1.10x | 26.1x |
| augment_noise | 48 | 4096 | 6.8 | 6.1 | 8.4 | 25.6 | 77.3 | 1.38x | 3.0x |
| align_euler | 48 | 4096 | 16.4 | 12.9 | 17.6 | 167.1 | 380.9 | 1.36x | 9.5x |
| augment_noise | 48 | 8192 | 8.0 | 7.0 | 11.5 | 36.3 | 96.7 | 1.64x | 3.2x |
| align_euler | 48 | 8192 | 20.4 | 17.1 | 22.6 | 265.6 | 511.9 | 1.32x | 11.8x |
| augment_noise | 48 | 16384 | 10.9 | 10.4 | 17.4 | 65.2 | 145.0 | 1.67x | 3.7x |
| align_euler | 48 | 16384 | 31.5 | 27.4 | 32.4 | 575.0 | 845.9 | 1.18x | 17.7x |
| augment_noise | 48 | 32768 | 33.0 | 27.0 | 37.0 | 122.0 | 237.8 | 1.37x | 3.3x |
| align_euler | 48 | 32768 | 83.5 | 64.6 | 74.2 | 1179.0 | 1541.0 | 1.15x | 15.9x |
| pool_atoms | 1 | 4096 | 12.9 | 3.9 | 4.6 | 6.6 | 30.2 | 1.18x | 1.4x |
| pool_atoms | 1 | 8192 | 12.6 | 3.8 | 4.8 | 9.1 | 36.3 | 1.26x | 1.9x |
| pool_atoms | 1 | 16384 | 12.4 | 4.1 | 5.3 | 13.8 | 48.4 | 1.29x | 2.6x |
| pool_atoms | 1 | 32768 | 13.1 | 4.9 | 6.1 | 23.5 | 77.0 | 1.24x | 3.9x |
| pool_atoms | 5 | 4096 | 12.4 | 4.2 | 5.5 | 16.2 | 57.3 | 1.31x | 2.9x |
| pool_atoms | 5 | 8192 | 13.1 | 5.4 | 6.6 | 28.7 | 94.3 | 1.22x | 4.3x |
| pool_atoms | 5 | 16384 | 13.2 | 8.2 | 9.5 | 52.9 | 172.4 | 1.16x | 5.6x |
| pool_atoms | 5 | 32768 | 22.9 | 12.3 | 15.6 | 101.9 | 338.3 | 1.27x | 6.5x |
| pool_atoms | 16 | 4096 | 12.9 | 7.1 | 8.4 | 42.2 | 138.1 | 1.18x | 5.0x |
| pool_atoms | 16 | 8192 | 21.5 | 10.5 | 13.1 | 82.2 | 275.1 | 1.25x | 6.3x |
| pool_atoms | 16 | 16384 | 38.7 | 18.7 | 27.1 | 158.2 | 521.9 | 1.45x | 5.8x |
| pool_atoms | 16 | 32768 | 75.6 | 35.6 | 53.7 | 317.6 | 1007.8 | 1.51x | 5.9x |
| pool_atoms | 48 | 4096 | 25.0 | 14.2 | 18.9 | 118.9 | 399.6 | 1.33x | 6.3x |
| pool_atoms | 48 | 8192 | 56.8 | 27.4 | 40.8 | 238.1 | 764.9 | 1.49x | 5.8x |
| pool_atoms | 48 | 16384 | 104.8 | 50.3 | 77.1 | 466.3 | 1493.0 | 1.53x | 6.0x |
| pool_atoms | 48 | 32768 | 201.7 | 96.4 | 149.1 | 941.8 | 2958.4 | 1.55x | 6.3x |
