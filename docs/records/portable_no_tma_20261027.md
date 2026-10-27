# Portable Triton family without TMA (2026-10-27)

Branch `perf/final-triton-notma` (from `integ/ladder-all2`). The maintainer's rule: the Triton family is card-agnostic (H100 is served by hand CUDA), so it
uses no Hopper-only data movement. This record lists what was removed, what the H100 ladder says afterwards, and the rows that did not come back under 1.0.

Method: `benchmarks/runners/ladder.py run --group '^(tr_|adaln_|ct_)' --kinds rungs --repeats 3` (CUDA-graph / compiled, portable rung = `settings.force_portable_kernels`),
one H100 80GB HBM3 (node01), torch 2.13 + cu129, triton 3.7. Before = commit `23009c44` (TMA on), after = this branch. PyTorch = the compiled + graph rung (`pt_compiled`,
the fastest of the three PyTorch forms in every row below). Medians of 3; the run-to-run noise of one row is 1-3 %.

## Ladder, portable ms and portable / PyTorch (bold: >= 1.0)

| group | mode | L | PyTorch ms | portable ms before | portable ms after | portable / PyTorch before | after |
|---|---|---|---|---|---|---|---|
| tr_D128 | inference | 128 | 0.057 | 0.045 | 0.040 | 0.787 | 0.709 |
| tr_D128 | inference | 384 | 0.452 | 0.251 | 0.275 | 0.556 | 0.608 |
| tr_D128 | inference | 768 | 1.711 | 0.900 | 1.023 | 0.526 | 0.598 |
| tr_D128 | inference | 1024 | 3.033 | 1.597 | 1.787 | 0.527 | 0.589 |
| tr_D128 | training | 128 | 0.161 | 0.153 | 0.145 | 0.956 | 0.866 |
| tr_D128 | training | 384 | 1.213 | 1.015 | 1.015 | 0.837 | 0.836 |
| tr_D128 | training | 768 | 4.513 | 3.866 | 3.834 | 0.857 | 0.849 |
| tr_D128 | training | 1024 | 7.954 | 6.884 | 6.764 | 0.865 | 0.850 |
| tr_D256 | inference | 384 | 0.880 | 0.589 | 0.655 | 0.670 | 0.743 |
| tr_D256 | inference | 768 | 3.404 | 2.301 | 2.598 | 0.676 | 0.761 |
| tr_D256 | training | 384 | 2.345 | 2.154 | 2.216 | 0.919 | 0.949 |
| tr_D256 | training | 768 | 8.982 | 8.326 | 8.633 | 0.927 | 0.958 |
| tr_D384 | inference | 384 | 1.506 | 1.124 | 1.322 | 0.746 | 0.880 |
| tr_D384 | inference | 768 | 5.942 | 4.334 | 5.112 | 0.729 | 0.857 |
| tr_D384 | training | 384 | 4.086 | 3.863 | 4.124 | 0.945 | **1.004** |
| tr_D384 | training | 768 | 15.831 | 15.453 | 16.355 | 0.976 | **1.022** |
| tr_D512 | inference | 384 | 2.296 | 1.795 | 2.057 | 0.782 | 0.889 |
| tr_D512 | training | 384 | 6.181 | 5.990 | 6.581 | 0.969 | **1.057** |
| tr_D768 | inference | 384 | 4.347 | 3.655 | 4.158 | 0.841 | 0.946 |
| tr_D768 | training | 384 | 12.278 | 11.802 | 12.911 | 0.961 | **1.053** |
| tr_fp32_D128 | inference | 384 | 0.845 | 0.473 | 0.477 | 0.560 | 0.564 |
| tr_fp32_D128 | inference | 768 | 3.286 | 1.762 | 1.833 | 0.536 | 0.558 |
| tr_fp32_D128 | training | 384 | 2.367 | 1.872 | 1.751 | 0.791 | 0.739 |
| tr_fp32_D128 | training | 768 | 9.137 | 7.158 | 6.743 | 0.783 | 0.738 |
| tr_fp32_D256 | inference | 384 | 1.690 | 1.195 | 1.226 | 0.707 | 0.725 |
| tr_fp32_D256 | training | 384 | 4.932 | 5.060 | 4.762 | **1.026** | 0.962 |
| adaln_bf16 | inference | 384 | 0.027 | 0.018 | 0.018 | 0.664 | 0.665 |
| adaln_bf16 | inference | 768 | 0.036 | 0.027 | 0.027 | 0.768 | 0.767 |
| adaln_bf16 | inference | 1024 | 0.041 | 0.032 | 0.032 | 0.778 | 0.782 |
| adaln_bf16 | training | 384 | 0.373 | 0.282 | 0.292 | 0.756 | 0.785 |
| adaln_bf16 | training | 768 | 0.692 | 0.503 | 0.535 | 0.728 | 0.775 |
| adaln_bf16 | training | 1024 | 0.886 | 0.649 | 0.692 | 0.733 | 0.782 |
| adaln_fp32 | inference | 384 | 0.034 | 0.025 | 0.028 | 0.739 | 0.819 |
| adaln_fp32 | training | 384 | 0.641 | 0.596 | 0.597 | 0.930 | 0.930 |
| ct_bf16 | inference | 384 | 0.083 | 0.059 | 0.059 | 0.706 | 0.708 |
| ct_bf16 | inference | 768 | 0.123 | 0.101 | 0.102 | 0.816 | 0.829 |
| ct_bf16 | inference | 1024 | 0.148 | 0.126 | 0.127 | 0.854 | 0.855 |
| ct_bf16 | training | 384 | 1.246 | 1.165 | 1.189 | 0.935 | 0.956 |
| ct_bf16 | training | 768 | 2.306 | 2.165 | 2.243 | 0.939 | 0.974 |
| ct_bf16 | training | 1024 | 2.987 | 2.810 | 2.916 | 0.941 | 0.977 |
| ct_fp32 | inference | 384 | 0.137 | 0.112 | 0.115 | 0.817 | 0.846 |
| ct_fp32 | training | 384 | 2.897 | 2.793 | 2.800 | 0.964 | 0.967 |

(`ct_*` / `adaln_*` rows: the CUDA rung is within noise of the portable rung in `adaln_bf16` / `ct_bf16` training - that is the CUDA kernels' ladder, not changed here.)

## What was removed

TMA (`tl.make_tensor_descriptor`, `TensorDescriptor`, the in-kernel persistent tile loops built on them, the `TMA` / `USE_TMA` / `NUM_SMS` axes) from:

| file | what |
|---|---|
| `kernels/transition/triton/main.py`, `training.py`, `residual.py` | expand + SwiGLU, squeeze + SwiGLU backward, squeeze + residual: the pointer branch is the only one; `USE_TMA`, `NUM_SMS` and the tensor-map host code are gone; `kernels/transition/triton/_tma.py` deleted |
| `kernels/conditioned_transition/triton/fwd_saveact.py`, `training.py` | the two persistent forward kernels (they were switched off, `_TMA_FWD = False`: measured slower) and the TMA form of `_dh_swiglu_bwd_kernel` |
| `kernels/adaln/triton/training.py` | the TMA form of `_adaln_fwd_gate_kernel` |
| `kernels/conditioned_transition/triton/token_dit_attn.py` | the v2 attention core read its bias through a host `TensorDescriptor` (`Bdesc.load`); now `tl.load` of the same tile (`bias_descriptor` keeps its name for the runner, and returns the `[nb*H*L, L]` view) |
| `kernels/_tma.py` | deleted (`ensure_allocator`, `tma_launchable`, `prune_tma`); `num_sms` moved into `bias_only_attention/triton/pair_bias.py` (its only other user, a persistent pointer grid) |
| `autotune/configs/{default,grid}/{adaln_fwd_gate,cond_transition_bwd_gemm_swiglu,cond_transition_expand_swiglu_saveact,cond_transition_squeeze_gate_saveact}_triton.csv` | the `TMA` column / axis; the `TMA > 0` rows of the default sets |
| `build/key_gaps_allowed.csv` | the `NUM_SMS` rows (and `ROUND_X` added) |
| `tests/gpu/any/test_portable_ct_adaln_residual_gpu.py` | the TMA-vs-pointer parity test; `tests/cpu/layout/test_no_tma_in_triton_trees.py` is new (AST-free text check: no TMA name in any `kernels/*/triton/` file or shared Triton helper) |
| `tests/cpu/data/autotune-space-20260916.json`, registry, sweep page | the grid sizes of the two kernels whose grid had grown a TMA axis go back to 2880 / 2160; `dev derive` and `viz.sweep_page` re-run |

The TMA sources stay reachable in git (`git show 23009c44:<path>`; the introducing commits are `7a5aabb1`, `3580829c`, `12be0d1b`).

## Other Hopper-only branches in the Triton trees

| where | what | decision |
|---|---|---|
| `kernels/transition/triton/fused.py` (`_is_sm90`, `_gatebwd_wgmma_enabled`, the hand-CUDA b2b gate) | CUDA stage choices inside a Triton file | not touched: the concurrent branch `perf/final-pure-triton` moves them to the callers |
| `kernels/conditioned_transition/triton/token_dit_runner.py` (`_row_kernels`, `_cuda_core`, `_gemm_swiglu`, capability checks `(9, 0)` / `(10, 0)`) | hand-CUDA routing inside the runner | not touched, same reason; its only Triton-side Hopper feature (the bias descriptor) is removed above |
| `adaln/triton/training.py`, `conditioned_transition/triton/training.py`: `is_h100(...)` -> `ctx.master` | bf16 activations with fp32 master weights: a bf16 cast pack + fp32 weight gradients | stays: a numerics policy that uses a Triton cast and PyTorch GEMMs, not a Hopper feature; it only switches on the card that runs bf16-mixed with fp32 parameters |
| `adaln/triton/inference.py`: `tl.inline_asm_elementwise("mov.b32 ...")` | an identity that breaks a Triton 3.6 / ptxas 12.8 epilogue pattern on sm_90 (misaligned smem vector store of one 128x256, 4-warp, 1-stage tile) | stays: bit-identical no-op on every card, and the portable rung still runs on H100 in the ladder |
| `augmented_attention/triton/main.py`: dQ formed query-major | avoids an invalid WGMMA smem descriptor for bf16 padded heads (Triton 3.6 / ptxas 12.8) | stays: a layout choice that is correct on every card |
| `kernels/_tf32.py` comment on the B operand | describes cp.async / TMA | wording only |

## A5000 / A6000 workarounds (kept)

- `kernels/layernorm_linear/triton/fused.py`, `_prefer_covering_lnl`: for K = 128, N = 520, bf16 on an RTX A5000 / A6000 (sm_86) it drops the one schedule (BLOCK_K 128, BLOCK_M1 64, BLOCK_N 128, 1 warp, 1 stage) that makes `tl.dot` write outside shared memory (reproduced with compute-sanitizer on both cards; adjacent N = 512, stages = 2, warps >= 2 and BLOCK_N = 256 schedules pass). The comment at the branch states the evidence and its scope; the concurrent branch `perf/final-pure-triton` also lists it in the allow-list of `tests/cpu/layout/test_triton_trees_are_triton_only.py`.
- `kernels/mpnn_edge_mlp/interface.py` routes compute capability (8, 6) cards at the A5000 crop shapes to the one-kernel edge MLP: a performance route measured on that card, documented in the module docstring.

## Rows that stay at or above 1.0, and why

Wide bf16 Transition training (D384 and up, the first row of each pair at the 384 pair, "after" column above): the persistent tensor-map kernels were 457 us (expand + SwiGLU) and 209 us (squeeze + SwiGLU backward) faster per step at D768 than any pointer-based form measured (`top_kernels` of the ladder rows: 2622 -> 3079 us and 2405 -> 2614 us; the squeeze + residual kernel is 94 us faster without TMA, 1192 -> 1098 us). The step is GEMM-bound: the pointer expand kernel (1.39 TFLOP in 3.07 ms, 450 TFLOPS overall) does its MMAs and then stores `h`, `a`, `b` (2.7 GB) without overlapping the next tile's MMAs; the tensor-map kernel overlapped them.

Tried and rejected (all H100, bf16, D768 unless stated; kernel time per step):

- a tile sweep of the three kernels over BLOCK_K 32/64, BLOCK_M1 64/128/256, BLOCK_N 64/128/256, GROUP_M 4/16, warps 4/8, stages 2-4 (180 configs per kernel; plus 3-warps/1-warp small tiles, 144 per kernel): the autotuner's pick is within 0-3 % of the best on every wide row;
- the expand kernel as a persistent loop of pointer-loaded tiles (`tl.range(..., flatten=True)`, one program per SM), with and without `warp_specialize`: 3.1-3.2 ms against 3.07 ms;
- `eviction_policy="evict_first"` on the output stores: no change (3.05 / 2.54 / 0.96 ms);
- the expand halves written once as an interleaved `[a|b]` buffer and the fp16-scaled `dab`: not built (the backward consumers need the layout work back).

A note on measuring: the autotuner's own benchmark and a steady-state profile of the same pinned config differ by up to 1.5x between processes on this node (power / clock state of the shared 8-GPU node), so kernel A/Bs here compare configs inside one process only.

fp32: `tr_fp32_D256` training 1.026 -> 0.962: the activation tile is rounded to the TF32 grid inside the expand kernel instead of by a separate pass over `x` (`ROUND_X`, taken when K is a multiple of 64; the autotuner prunes BLOCK_K that do not divide K, which was the ragged-tile failure that made the pass necessary), and the backward kernel got the small-tile configs a tile sweep found (BLOCK_M1 32-64, 2-4 warps: 1311 -> 1049 us), plus a GROUP_M 16 expand config. The module error against PyTorch TF32 is unchanged (`test_portable_fp32_accuracy_gpu.py`, `test_transition_fp32_portable_gpu.py`).

`ct_fp32` training stays at 0.967: the fused dh + SwiGLU backward kernel in fp32 (pointer-based, already supported: PRE rounds `ws`) takes 407 us at M = 18432 against about 300 us for the cuBLAS dh (115 us) and the flat SwiGLU pass (185 us, at its HBM roofline), so the split stays (comment at `wide_fp32`). The other fp32 elementwise kernels of the step (`_swiglu_fwd`, `_sigmul_*`, `_adaln_fwd_gate`) are within 5 % of their HBM roofline; what is left is fusion, not tuning.
