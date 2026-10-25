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

## Measurements (2026-10-25)

H100, CUDA graph, microseconds per call. Full table, parity and reading: [records/portable_sampler_ops_20261025.md](../../../records/portable_sampler_ops_20261025.md).

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
