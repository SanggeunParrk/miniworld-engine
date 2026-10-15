# H100 local atom DiT — shared-conditioning backward restructure (2026-10-14)

Worktree `h100/local-dit-opt` (uncommitted). Hardware: H100 80 GB (node01 / node02), CUDA graph, 4096 atoms, A48 unless stated.
Kernel sums are `torch.profiler` totals of one training step (`prof_sh.py`, shared conditioning); step times are `bench.py target=local_dit
mode=training cudagraph=manual`.

| change | bf16 kernel sum | fp32 kernel sum | note |
|---|---|---|---|
| per-sample dmod rows + `sum(0)` (before) | 3.27 ms | 5.21 ms | torch glue: 2 bf16 / 4 fp32 reduce kernels, 13 small fp32 GEMMs |
| four samples per CTA group, per-pass slabs (2 barriers) | 3.47 | 4.74 | measured on a different node state (3.52 / 5.08 baseline) |
| 128-column slabs, one barrier per flush (bf16 two buffers) | 3.15 | 4.83 | |
| + `cond_bwd_sh90` (partial sums + products + LN backward in one kernel) | 3.05 | 4.72 | 107 us bf16 / 143 us fp32; P-loop unroll 2 |

Step times, harness, shared / per-sample conditioning:

| A | bf16 shared | bf16 per-sample | fp32 shared | fp32 per-sample |
|---|---|---|---|---|
| 48 | 3.09 ms | 3.66 | 4.79 | 6.05 |
| 12 | 1.26 | 1.34 | 1.71 | 2.04 |
| 5 | 0.89 (no grouping: A % 4 != 0) | 0.83 | 1.19 | 1.26 |

Checks: 29 tests pass (shared vs materialised at A = 4, 5, 8; fp64 accuracy; bitwise repeatability; CUDA graph; compile).

Rejected in the same session (numbers in `docs/gpus/h100/local_dit/local_dit.md`, "measured and rejected"): conditioning tables fused
into pre / post (inference hoists the tables, training needs them in the backward), dQ fused into the dK / dV kernel, in-kernel fp32
accumulation over a sample loop, a fused weight-gradient kernel (cuBLAS at 435 vs 490 us HBM floor).

Open: A not a multiple of four (A5 shared is slower than per-sample rows: ~0.05 ms); the P-sum kernel is latency-bound at 64 CTAs.
