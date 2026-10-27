# AdaLN portable Triton training rows under PyTorch (2026-10-27)

Branch `perf/final-adaln-triton` (from `perf/final-triton-notma`, TMA already removed). One H100 80GB HBM3 (torch 2.13 + cu129, triton 3.7),
`benchmarks/runners/ladder.py` (compile + CUDA graph, median of 3; rows are +-1-3 % noisy), `settings.force_portable_kernels` for the portable
rung. The ladder groups `adaln_fp32_c768`, `adaln_fp32_atom` come from `perf/final-fp32-rows`; two groups were added here
(`adaln_bf16_c768`, `adaln_bf16_atom`) so every AdaLN registry row (bf16 and fp32, d768 c384 / d768 c768 / atom d128 c128) has a ladder row.

Starting point: portable ÷ PyTorch 1.03 / 1.05 / 1.09 for fp32 atom training (1024 / 4096 / 8192 atoms) and 1.004 for fp32 d768 c768 training at
L768. No TMA anywhere (`test_no_tma_in_triton_trees` green).

## Result (ms; PyTorch = best of compiled / eager-in-graph / eager)

Before = `perf/final-triton-notma` (ladder run 37033 for the token rows; the atom training rows from the fp32-rows ladder: PyTorch
0.292 / 1.048 / 1.980, portable 0.302 / 1.100 / 2.165). After = this branch (jobs 37107-37109).

| row | L | PyTorch | portable before -> after | portable ÷ PyTorch before -> after |
|---|---|---|---|---|
| training fp32 atom d128 c128 | 1024 | 0.293 | 0.302 -> 0.241 | 1.03 -> 0.82 |
| training fp32 atom d128 c128 | 4096 | 1.047 | 1.100 -> 0.843 | 1.05 -> 0.81 |
| training fp32 atom d128 c128 | 8192 | 1.981 | 2.165 -> 1.623 | 1.09 -> 0.82 |
| training fp32 d768 c768 | 128 | 0.372 | 0.366 -> 0.327 | 0.985 -> 0.88 |
| training fp32 d768 c768 | 384 | 0.935 | 0.884 -> 0.779 | 0.942 -> 0.83 |
| training fp32 d768 c768 | 768 | 1.609 | 1.619 -> 1.434 | 1.004 -> 0.89 |
| training fp32 d768 c384 | 128 | 0.260 | 0.259 -> 0.230 | 0.995 -> 0.88 |
| training fp32 d768 c384 | 384 | 0.640 | 0.594 -> 0.549 | 0.928 -> 0.86 |
| training fp32 d768 c384 | 768 | 1.169 | 1.132 -> 1.028 | 0.968 -> 0.88 |
| training bf16 d768 c384 | 384 / 768 / 1024 | 0.373 / 0.692 / 0.888 | 0.293 / 0.533 / 0.691 -> 0.293 / 0.535 / 0.693 | 0.79 / 0.77 / 0.78 -> 0.79 / 0.77 / 0.78 (unchanged: this path is not touched) |
| training bf16 d768 c768 | 128 / 384 / 768 | 0.182 / 0.509 / 0.941 | not in the old ladder -> 0.157 / 0.409 / 0.757 | -> 0.86 / 0.80 / 0.81 |
| training bf16 atom d128 c128 | 1024 / 4096 / 8192 | 0.188 / 0.694 / 1.309 | not in the old ladder -> 0.135 / 0.479 / 0.907 | -> 0.72 / 0.69 / 0.69 |
| inference fp32 d768 c384 | 128 / 384 / 768 | 0.022 / 0.034 / 0.051 | 0.016 / 0.025 / 0.044 -> 0.016 / 0.025 / 0.044 | 0.73 / 0.73 / 0.86 -> 0.71 / 0.73 / 0.87 |
| inference fp32 d768 c768 | 128 / 384 / 768 | 0.026 / 0.044 / 0.070 | 0.021 / 0.037 / 0.066 -> 0.022 / 0.037 / 0.066 | 0.82 / 0.85 / 0.95 -> 0.82 / 0.84 / 0.94 |
| inference fp32 atom d128 c128 | 1024 / 4096 / 8192 | 0.022 / 0.045 / 0.082 | 0.017 / 0.035 / 0.063 -> same | 0.78 / 0.78 / 0.77 -> 0.79 / 0.79 / 0.77 |
| inference bf16 d768 c384 | 384 / 768 / 1024 | 0.027 / 0.036 / 0.041 | 0.018 / 0.027 / 0.032 -> same | 0.67 / 0.77 / 0.78 -> 0.67 / 0.77 / 0.79 |
| inference bf16 d768 c768 | 128 / 384 / 768 | 0.024 / 0.032 / 0.044 | not in the old ladder -> 0.017 / 0.023 / 0.037 | -> 0.68 / 0.71 / 0.85 |
| inference bf16 atom d128 c128 | 1024 / 4096 / 8192 | 0.021 / 0.031 / 0.046 | not in the old ladder -> 0.013 / 0.025 / 0.038 | -> 0.64 / 0.80 / 0.83 |

Every AdaLN portable row is now <= 0.95 of PyTorch (worst: fp32 inference d768 c768 L768, 0.94, unchanged code).

Accuracy (fp32, module output / dx / dcond / parameter gradients, relative L2 error vs PyTorch with TF32): 4e-6 .. 9e-6 on all rows, the same
order as before (`test_portable_fp32_accuracy_gpu`: every AdaLN case within the 1.5x yardstick).

## What changed (kernels/adaln/triton/training.py)

1. **K-major weight copy for the fp32 dgrad GEMM.** `dcond_aff = Dᵀ @ [Ws; Wb]`: the right operand `[Ws; Wb]` is N-contiguous, which makes
   cuBLAS pick its slower "nn" TF32 kernel. A `w_cat.t().contiguous()` copy (4.7 MB at d768) handed over as `.t()` selects the "nt" kernel:
   dgrad 263 -> ~150 us at d768 c768 L768. Worth 6-9 % of the whole row. fp32 only (suggested by the ConditionedTransition fp32 agent, record
   `portable_ct_fp32_training_20261027.md`; the bf16 rows are left alone).
2. **Atom width on sm90+: cuBLAS dgrad + LayerNorm backward instead of the in-kernel GEMM, and the GEMM+gate forward instead of
   cuBLAS-materialise + epilogue.** At 4096 atoms (fp32) `_dgrad_condln_kernel` took 389 us against 103 (cuBLAS "nt") + 103 (`_ln_bwd`) = 206;
   the fused forward replaces cutlass GEMM 113 + epilogue 173 by 181 (plus the stats launch it shares with the cond LayerNorm). Routing is
   `_cublas_class(device)` (compute capability >= 9, a plain capability test, no Hopper feature in the kernels). The older cards keep the choice
   that was measured on them. Individually: dgrad alone 0.92, fused forward alone 1.00, both 0.84 (fp32 probe, 4096 atoms); bf16 atom 0.94 -> 0.85.
3. **`dsb` folded into `_bwd_x_kernel`.** The bias-of-scale gradient (column sums of `dscale`) was a separate reduction over D (48 us at 4096
   atoms and at d768 L768). It is now one fp32 `atomic_add` per column per program in the dx kernel (`HAS_DSB`, `reset_to_zero=["DSB"]` for the
   autotuner). Not for the row-major bf16 token D, which keeps its skinny GEMM. fp32 atomics: the sum order is not fixed, as for `dlnw`.

## Tried and rejected

* More `adaln_fwd_gate_triton` tiles in the default space (128x128x32/64 with 4/8 warps, 64x256, 2-stage 64x128 so that two CTAs share an SM and
  one's epilogue hides under the other's mainloop): the tuner kept 64x128x32 w4 s3 every time. The Triton TF32 GEMM sits at ~190-230 TFLOP/s
  against cuBLAS 330. Removed again.
* `pre_a=True` (round the `cond_aff` operand once where it is written instead of in the kernel): slower (489 vs 460 us) and 100x worse error.
* cuBLAS-materialised forward at d768: 0.91 / 0.93 / 0.99 against fused 0.95 / 0.92 / 0.97 (c768 L128 / 384 / 768): a wash, kept fused.
* `dsb` through the wgrad GEMM with a ones column in `cond_aff`: TF32 rounding of D makes the sum inexact (cancellation), rejected before building it.

## Tests

`tests/cpu/layout/test_no_tma_in_triton_trees.py`, `tests/gpu/any/test_engine_adaln_alignment.py`, `test_portable_ct_adaln_residual_gpu.py`
(jobs 37124, 37112): pass. `test_portable_fp32_accuracy_gpu.py::test_portable_fp32_is_within_1p5x_of_the_pytorch_tf32_error[train-bodit_768]`
FAILS identically (2.07x on `d_attention.to_bias.weight`, `d_arg2`) on the untouched `perf/final-triton-notma` (job 37146): not caused by this
change (an attention-bias gradient of the bias-only DiT block). `tests/cpu/autotune/test_compact_search_space.py::
test_inventory_does_not_add_an_unreached_b2b_driver` also fails on this branch (unrelated, not looked at).

## Consequence for the ladder order CUDA < portable

The portable training rows are now faster than the H100 hand-CUDA rows (CUDA ÷ portable):

| row | L | CUDA ÷ portable |
|---|---|---|
| training fp32 atom | 1024 / 4096 / 8192 | 1.08 / 1.04 / 1.04 |
| training fp32 d768 c768 | 128 / 384 / 768 | 1.07 / 1.07 / 1.05 |
| training fp32 d768 c384 | 128 / 384 | 1.06 / 1.01 |
| training bf16 atom | 1024 / 4096 / 8192 | 1.20 / 1.15 / 1.15 |
| training bf16 d768 c768 | 128 | 1.06 |
| inference fp32 atom (not changed here) | 4096 / 8192 | 1.08 / 1.10 |
| inference bf16 atom (not changed here) | 1024 / 4096 / 8192 | 1.03 / 1.27 / 1.44 |

The CUDA side (`kernels/adaln/cuda`, owned elsewhere) can take the same levers: the K-major dgrad operand, the `dsb` fold, cuBLAS dgrad +
LayerNorm backward at the atom width.

## Left

* wgrad `D @ cond_aff` (382 us at d768 c768 L768, the largest single kernel) has the same N-contiguous right operand; a K-major `cond_aff` would
  need the LayerNorm kernel to write a second, transposed copy (+38 us of writes for an estimated -90 us): not tried.
* The atom routing is gated on sm90+ because that is the card measured; A100 / A6000 were not run (see `docs/gpus/a100/a100.md`).
