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
