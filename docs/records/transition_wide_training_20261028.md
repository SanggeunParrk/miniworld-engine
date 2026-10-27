# Portable Transition, wide bf16 training, after the TMA removal (2026-10-28)

Branch `perf/final-tr-wide` (from `perf/final-triton-notma` at `538e3393`). Goal: portable / PyTorch (compiled) <= 0.95 on the bf16 Transition training rows
(D128 ... D768), without TMA, fp32 and inference rows staying under 1, module error unchanged. Outcome: **D128 and D256 are under 0.95, D384 and wider are
not** (0.97 / 1.02 / 1.02 / 1.04); this record lists what was measured, what moved, and why the rest cannot be recovered by editing the pointer-based
kernels. Nothing below uses TMA.

## Ladder (H100 80GB HBM3, torch 2.13 + cu129, triton 3.7; `ladder.py run --group '^tr_' --kinds rungs --repeats 3`, CUDA-graph / compiled, bf16-mixed)

Before = `538e3393` run today on the same nodes (`out_base`), after = `011096f8` (`out_fin3`). Medians of 3; the run-to-run noise of one row is 1-3 %.

| row | mode | L | PyTorch ms | portable ms before -> after | portable / PyTorch before -> after |
|---|---|---|---|---|---|
| tr_D128 | training | 128 | 0.160 | 0.140 -> 0.141 | 0.877 -> 0.878 |
| tr_D128 | training | 384 | 1.215 -> 1.212 | 0.998 -> 0.998 | 0.821 -> 0.823 |
| tr_D128 | training | 768 | 4.516 -> 4.513 | 3.756 -> 3.757 | 0.832 -> 0.832 |
| tr_D128 | training | 1024 | 7.955 -> 7.957 | 6.621 -> 6.626 | 0.832 -> 0.833 |
| tr_D256 | training | 384 | 2.344 -> 2.348 | 2.151 -> 2.148 | 0.918 -> 0.915 |
| tr_D256 | training | 768 | 9.046 -> 9.036 | 8.355 -> 8.337 | 0.924 -> 0.923 |
| tr_D384 | training | 384 | 4.088 -> 4.098 | 4.067 -> 3.979 | 0.995 -> **0.971** |
| tr_D384 | training | 768 | 15.808 -> 15.879 | 16.505 -> 16.232 | 1.044 -> **1.022** |
| tr_D512 | training | 384 | 6.147 -> 6.220 | 6.264 -> 6.319 | 1.019 -> **1.016** |
| tr_D768 | training | 384 | 12.031 -> 12.267 | 12.598 -> 12.764 | 1.047 -> **1.041** |
| tr_D256 | inference | 384 | 0.879 | 0.616 -> 0.617 | 0.701 -> 0.702 |
| tr_D384 | inference | 384 | 1.502 -> 1.509 | 1.250 -> 1.255 | 0.832 -> 0.832 |
| tr_D768 | inference | 384 | 4.298 -> 4.349 | 3.978 -> 4.062 | 0.926 -> 0.934 |
| tr_fp32_D128 | training | 384 | 2.367 -> 2.370 | 1.751 -> 1.751 | 0.739 -> 0.739 |
| tr_fp32_D256 | training | 384 | 4.909 -> 4.966 | 4.694 -> 4.677 | 0.956 -> 0.942 |

(D128 and D256 training and every fp32 / inference row were at or under the bar before this branch and stay there; the rest of the `^tr_` rows are in
`out_fin3/ladder.csv`, not copied here. The D128 / D256 / D512 / D768 inference rows move only inside the noise.)

### What the ladder cannot show: the steady state is power-capped

The H100 runs these steps at the 700 W cap (`nvidia-smi`: throttle reason 0x4, SM clock 1335-1785 MHz while a step loops), and the ladder's weights are
non-zero (`bench_module_transition` initialises `squeeze` with a normal). A probe with the Transition's default zero `squeeze` weight runs at near-boost clocks
and reads 5-10 % better for the portable rung than the ladder does; do not use it for a ratio. `probe/gpclk.py` (scratch, not in the repo) loops one graph per
rung for 3 s each, alternating, with the ladder's weights; portable / PyTorch there, base -> final:

| row (L384 unless noted) | base | final |
|---|---|---|
| D128 | 0.846 | 0.832 |
| D256 | 0.944 | 0.924 |
| D384 | 0.998 | 0.991 |
| D512 | 1.022 | 0.998 |
| D768 | 1.065 | 1.035 |
| D256 L768 | 0.960 | 0.938 |
| D384 L768 | 1.009 | 0.996 |

## Where the step goes (D384 L384, bf16-mixed, ms of a graph replay, base)

fused expand + SwiGLU (saves h, a, b) 1.02, fused dh + gate backward 1.05, squeeze + residual 0.34, cuBLAS dx / dWab / dWs 1.21, LN backward 0.22, LN forward 0.08:
4.0. PyTorch: three cuBLAS GEMM groups 2.4, gate forward 0.44, gate backward 0.74, add 0.11, LN 0.28: 4.1. The Triton kernels move fewer bytes than the PyTorch
pointwise kernels (h, a, b are written once, dh never reaches HBM) but run their GEMMs through cp.async + wgmma, which is slower than cuBLAS's TMA-multicast
mainloop for these shapes (K = D, 6 to 12 K iterations per tile): expand mainloop without any epilogue 0.71 ms against cuBLAS `a|b = x @ Wab^T` 0.54 ms at D384,
2.13 against 1.91 ms at D768. The epilogue (SwiGLU, three stores) then runs after the MMAs of the same tile, so the kernel time is about the sum of the two:

| D768 L384 expand kernel (probe, one process) | ms |
|---|---|
| GEMM + `h = A+B` store only | 2.23 |
| + tl.sigmoid SwiGLU, no store of a, b | 2.53 |
| + a, b stored (h, a, b) | 3.01 |
| same with the sigmoid through one `tanh.approx` | 2.68 |
| a, b as one interleaved wide store, h separate | 2.84 (sigmoid) |
| cuBLAS `a|b` GEMM alone | 1.91 |

## What changed (kept)

1. **One `tanh.approx.f32` for the gate** (`kernels/_fastmath.py`, `sigmoid_fast = 0.5 + 0.5 tanh(x/2)`, inline PTX; `libdevice.tanh` is the accurate routine and is
   slower than `tl.sigmoid`). Used in the bf16 expand kernel (`transition_fwd_kernel`) and the bf16 saved-halves backward kernel (`_squeeze_swiglu_bwd_kernel`); fp32
   and fp16 keep `tl.sigmoid` (the error, 2.5e-4 absolute, is below the half-ulp of a bf16 result and not of fp32). Expand kernel at D768 3.07 -> 2.81 ms, D384 0.98 ->
   0.92 ms; backward kernel -0.04 ms at D384. h differs from the exact-sigmoid h in 0.13 % of the elements by one bf16 ulp (0.08 % without the change, against the fp32
   reference); the module-level gradient error against PyTorch is unchanged (`test_transition_residual_gpu.py`, `test_transition_layout_gpu.py`).
2. **Default candidate sets**: `transition_expand_swiglu_triton` + `64,128,128,16,8,4` (the D768 winner of the in-process sweep); `trimul_input_ln_residual_bwd_triton`
   `128,32,2,3`, `128,16,2,3`, `1024,8,4,1` (D256 / D384 / D768 winners: 0.145 -> 0.123, 0.231 -> 0.197, 0.497 -> 0.414 ms standalone) replacing three extremes
   (`64,64,32,2`, `64,64,16,2`, `64,64,4,6`) to stay inside the 32-config default limit. The online autotuner of the ladder does not always pick them (its quick benchmark is noisy),
   so the ladder shows only part of the standalone gain; a built (measured) cache would.

## Tried and rejected (all H100, bf16, one process, kernel time per step unless said)

- **Persistent expand kernel** (pointer loads, `tl.range(..., flatten=True)`, with and without `warp_specialize`): 3.11-3.2 against 3.0 ms at D768; at D384 1.07 against 0.98.
- **Saving less**: (a) store `a` and `h` only and form `b = h / silu(a)` in the backward (saves 0.9 GB at D384 / 1.8 GB at D768): backward error against the exact
  gradient 2.0e-3 relative (8e-5 with the stored b), one bf16 ulp, and the divide makes the kernel 2.95 against 2.48 ms; rejected on numerics and speed. (b) a, b as one interleaved
  `[a0 b0 a1 b1 ...]` wide store (acc stored without the split): forward 2.65 against 2.68 ms at D768, 0.92 against 0.89 at D384, backward reading it 2.55 against 2.47; no gain.
  (c) the expand kernel without h (a, b only, 2.78) plus the gate folded into the squeeze kernel's A operand: the squeeze kernel goes from 0.96 to 1.99 ms (register-operand
  wgmma), and 3.44 if it also writes h for dWs; the MUFU work is also repeated once per N tile (6x at D768).
- **Recompute instead of save** (h only): backward would add the expand GEMM (1.39 TFLOP at D768, +2.3 ms) to save 0.5 ms of stores.
- **Overlapping the epilogue loads**: loading a, b before the K loop of the backward kernel (so the loads are in flight during the MMAs): 2.507 against 2.501 ms at D768, 0.594 against 0.600 at D256.
- **cuBLAS for the GEMM and a Triton pointwise gate** (what PyTorch does, with our gate): expand 1.91 + gate 0.88 = 2.8 ms against the fused 2.68; backward dh 1.0-1.17 + gate 1.5 = 2.5-2.7
  against the fused 2.45-2.5. Equal, not better: the HBM bytes of the pointwise passes sit at the roofline already.
- **Row-chunking so a, b, h stay in L2** (2048-32768 rows per cuBLAS + gate + squeeze call): worse at every chunk size (D768 forward 3.87 -> 4.4 ms at 2048 rows, D384 1.33 -> 1.78).
- **Tile / schedule**: 16-warp and 256-wide tiles, `num_stages` 5-6, BLOCK_K 32 with 6 stages, `maxnreg` 128 / 168: no config beats the existing pick by more than 1 %; splitting the backward
  accumulator into two column halves (permuted weight columns, so the epilogue holds half the registers and larger tiles fit): 3.69 against 2.50 ms.
- **EVEN_M / EVEN_K (no masks)** in the expand kernel: 0.874 against 0.893 ms at D384, nothing in the backward kernel; not wired (needs a heuristics layer for 0.5 % of the step).
- cuBLAS orientation of the three weight-gradient / dx GEMMs (`dAB.T @ xn` against `xn.T @ dAB`, `dy.T @ h` against `h.T @ dy`): identical.
- squeeze + residual: the existing pick is within 1-2 % of a 140-config sweep at D256 and D384.

## Why 0.95 is out of reach for D384 and wider with these kernels

The bytes are already fewer than PyTorch's; what is left is GEMM mainloop efficiency (cp.async against cuBLAS's TMA multicast: 11-30 % at K = D) plus an epilogue that runs after the
MMAs of its tile. On H100 the only pointer-only ways to hide the epilogue (persistent loop, warp specialisation, store/MMA interleaving by the compiler) were measured and do nothing
here. D384 L384 ends at 0.97-0.99, D768 at 1.03-1.04: the remaining step is either TMA (excluded), or giving the three big GEMMs to cuBLAS and keeping the gate / residual / LN kernels
Triton, which measures equal to the current split (see above) and is not worth the extra launches.

## Tests

CPU (`tests/cpu`, whole directory): the four failures of the base branch only (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, two `test_a_ladder_offers_what_wins` errors);
`test_no_tma_in_triton_trees` green; `dev derive` + `viz.sweep_page` re-run (`registry_kernel.units.json`). GPU (H100): `test_transition_residual_gpu`, `test_transition_layout_gpu`,
`test_transition_fp32_portable_gpu`, `test_transition_fp16_halves_gpu`, `test_force_portable_option`, `test_conditioned_transition_precision_gpu`, `test_portable_fp32_accuracy_gpu`: all green except
`test_portable_fp32_is_within_1p5x_of_the_pytorch_tf32_error[train-bodit_768]`, which fails on the base branch as well (two other train cases of that test failed once on a run that shared its
Triton cache with other jobs and passed on a rerun).
