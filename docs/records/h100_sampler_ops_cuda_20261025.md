# H100 sampler glue: the CUDA kernels made faster than the portable ones (2026-10-25)

Branch `perf/h100-sampler-ops-cuda` (base: engine `main` at `b088341b`). The portable kernels of `kernels/sampler_ops` ([portable_sampler_ops_20261025.md](portable_sampler_ops_20261025.md)) were
faster than the H100 CUDA kernels in 13/13 `pool_atoms` cells and in about a third of the `augment_noise` / `align_euler` cells. The engine's order is CUDA < portable < PyTorch compiled, so the CUDA
kernels were the ones to fix. Measured on **one H100 80GB HBM3** (torch 2.13.0+cu129, triton 3.7.1), CUDA graph, median of 7 x 100 replays, the shapes of the earlier record.

## Why the CUDA kernels were slow

* `pool_atoms`: a 32-thread block per (token, sample) walked its 24 atoms in a loop with a `continue` and a dtype `switch` per load: per atom a dependent chain mask -> gather mask -> row index -> row,
  nothing in flight across iterations, 12.4-13.1 us flat. The portable kernel's loop is unrolled and its loads overlap.
* `augment_noise` / `align_euler`: (1) every kernel of the chain was launched after the previous one had fully drained (~1-2 us of launch gap and prologue per kernel, 2-3 kernels per op);
  (2) thread 0 of every block summed the chunk partials serially (nine covariance values x N / 1024 dependent loads), and the coordinates were read only after that;
  (3) a fixed 1024-atom chunk gave 4 blocks at A = 1, N = 4096 (32 at N = 32768); (4) in Horn's iteration the max-abs reduction was five dependent shuffle rounds per squaring.

## What changed (`cuda/sampler_ops.cu`)

* Programmatic dependent launch between the kernels of an op (`cudaLaunchKernelEx`, `griddepcontrol.launch_dependents` at the start of the producers, `griddepcontrol.wait` before the first read
  of a producer's output). The consumers read their coordinates / noise / mask into registers, and `augment_noise_k` computes its rotation frame and churn scalars, before the wait.
* The chunk partials of a sample are staged in shared memory (one round trip of independent loads per 128 chunks) and added in chunk order, one thread per value: the serial sum, without the
  dependent load chain. The centres are formed per thread from the shared sums (no thread-0 broadcast).
* Chunk 1024 / 512 / 256 atoms (`K` = 4 / 2 / 1 atoms per thread) with the portable rule: halve while the launch is smaller than two waves (264 blocks).
* Horn: the max over the 16 entries is a local reduction of 16 gathered values (overlapping shuffles) instead of the xor tree.
* `pool_atoms`: a lane loads one atom's mask / gather mask / row index; the rows of up to 32 atoms are requested at once (shuffle hand-over, predicated row loads, accumulation in atom order);
  templates on the index (int32 / int64) and mask (float / bool) dtypes, the old kernel remains as the fallback for any other dtype.

## Parity

* `pool_atoms`: bit-identical to the previous CUDA kernel in 8 test configurations (float / bool / bf16 masks, int32 / int64 indices, C 36 .. 768, A 5 .. 40, S 1 .. 48, a token without atoms).
* `augment_noise` / `align_euler`: **bit-identical** to the previous CUDA kernels wherever the chunk stays 1024 atoms (B = 16 N = 32768 / 50000, B = 48 N >= 8192; AF3 and EDM churn, three sigma
  pairs, noise 0 / 0.5 / 20). Where the chunk shrinks (small launches) the fp32 sums associate differently: against the previous kernel at most 1.8e-7 (`augment_noise`) and 8.7e-8
  (`align_euler`) relative to the largest value over the 16 benchmark cells; against the PyTorch reference 4.5e-7 / 1.7e-7 (the old kernels: 4.5e-7 / 1.5e-7); against the portable kernels 1.8e-7 /
  8.7e-8. `sc` (sigma_hat, noise magnitude) is bit-identical in every case. This is the one deliberate departure from bitwise equality with the old CUDA kernels.
* Tests: `tests/gpu/sm90/test_sampler_ops_h100_gpu.py`, `test_sampler_ops_portable_vs_cuda_gpu.py`, `tests/gpu/any/test_sampler_ops_portable_gpu.py`, `tests/cpu/compile/test_sampler_ops_routing.py`: 163 passed.
  Reruns of the new kernels are bitwise (fixed-order sums, no atomics). `dev derive --arch sm86` and `viz.sweep_page` leave the repository unchanged (no registry row changed).

## Result

microseconds per call; `CUDA before` the previous CUDA kernel (earlier record), the rest from this run; portable / CUDA > 1 means CUDA is faster.

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

All 48 cells keep CUDA < portable < PyTorch compiled; the smallest margin is 1.09x (`align_euler`, A = 16, 32768 atoms). The earlier record's CUDA times are repeated here for comparison, not
re-measured, and run-to-run noise of this harness is about 0.2-0.3 us at the small sizes.

Not done: the kernels were not tried on another card (the portable kernels serve those); `align_euler` at A = 16, N = 32768 (23.0 us) is the cell with the least headroom over portable (25.2 us), a
three-kernel chain at 1024-atom chunks that is bound by the partial passes, not by the Horn step.
