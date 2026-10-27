# ConditionedTransition fp32 (TF32) training, portable Triton family (2026-10-27)

Branch `perf/final-ct-fp32` (from `perf/final-triton-notma`, 538e3393). Starting point: `portable_no_tma_20261027.md` left `ct_fp32` training at 0.967 of PyTorch (compiled + CUDA graph) at L384, and
named the fused fp32 dh + SwiGLU backward kernel (407 us against about 300 us for cuBLAS + the flat pass) as the gap. H100 80GB HBM3, torch 2.13 + cu129, triton 3.7.

## Where the time was

Per-kernel (CUDA-graph replay, `torch.profiler`), fp32, B = 48, d = 768, d_cond = 384, L = 384, PyTorch compiled against portable: both run the same cuBLAS TF32 GEMMs. The CT elementwise passes
(SwiGLU fwd 113 us, gate fwd 77, gate bwd 92, SwiGLU bwd 185) are the same bytes as PyTorch's fused pointwise kernels and sit at the HBM roofline (565 MB for the SwiGLU backward, 3.05 TB/s), so
the gap was not in the elementwise code. It was in how the GEMMs were fed.

TF32 `wgmma` reads both operands K-major. A GEMM whose right operand is stored N-contiguous (`a @ w` with `w` a (K, N) row-major weight: cuBLAS "nn") runs a different, slower kernel than the same GEMM
with a K-major right operand ("nt"). Measured in one CUDA graph (10 replays), H100, M = 18432, d = 768:

| GEMM (backward) | "nn" (before) | "nt" on a K-major copy of the weight (after) |
|---|---|---|
| dx = dab @ wcat (87 GFLOP) | 327 us, 266 TFLOP/s | 250 us, 348 |
| dh = dout @ ws (43.5 GFLOP) | 164 us, 266 | 131 us, 332 |
| dcond = dscale @ wsc (10.9 GFLOP) | 70 us, 156 | 48 us, 225 |

The weight transposes are 12 MB in total (a few us each inside the compiled graph).

## What changed

All in `kernels/conditioned_transition/triton/training.py` unless stated.

1. fp32 backward multiplies from the right with `wcat.t().contiguous()`, `ws.t().contiguous()`, `wsc.t().contiguous()` (as `a @ wT.t()`): dx, dh, dcond become "nt" GEMMs. bf16 is not touched (its "nn" kernels are not penalised the same way).
2. `_dh_swiglu_bwd_kernel` (fp32, used at the atom width): the squeeze weight arrives transposed (ND, D) so the B tile is K-major (`tl.trans`), the `dout` operand arrives on the TF32 grid
   (no in-kernel rounding pass over the A operand), and `a` / `b` are loaded before the K loop. Kernel alone, CUDA events with L2 flushed: M = 196608, d = 128: 426 -> 315 us; M = 18432, d = 768: 420 -> 373 us
   (still behind cuBLAS dh 131 us + flat pass 185 us = 316 us, so the token width keeps the split). bf16 path unchanged (`PRE` is the fp32 switch).
3. `kernels/gated_projection/triton/main.py::_sigmul_bwd`: new `ROUND_DO` constexpr (default off, no other caller changes): the fp32 `dout` is stored on the TF32 grid (round to nearest, the value a cuBLAS
   TF32 GEMM rounds to anyway). Row added to `build/key_gaps_allowed.csv`.
4. `autotune/configs/default/cond_transition_bwd_gemm_swiglu_triton.csv`: two rows for the K-major kernel (64x64x32 with 4 warps / 3 stages and 8 warps / 4 stages). `registry_kernel.units.json` re-derived (`dev derive --arch sm86`).

Numerics: unchanged by construction (the same TF32 products, the same rounding points; `dout` rounding is idempotent with cuBLAS's own). `test_portable_fp32_accuracy_gpu.py`: module error vs PyTorch TF32 within 1.5x on every CT row.

## Result

Ladder (`benchmarks/runners/ladder.py run --group '^ct_' --kinds rungs --repeats 3`, CUDA-graph / compiled, portable rung = `force_portable_kernels`), ms:

| row | PyTorch | portable before | portable after | portable / PyTorch before -> after |
|---|---|---|---|---|
| ct_fp32 training L384 | 2.895 | 2.800 | 2.679 | 0.967 -> 0.925 |
| ct_bf16 training L384 | 1.244 | 1.189 | 1.187 | 0.956 -> 0.954 |
| ct_bf16 training L768 | 2.296 | 2.243 | 2.237 | 0.974 -> 0.974 |
| ct_bf16 training L1024 | 2.972 | 2.916 | 2.914 | 0.977 -> 0.980 (noise, 0.007) |
| ct_fp32 inference L384 | 0.137 | 0.115 | 0.115 | 0.846 -> 0.840 |
| ct_bf16 inference L384/768/1024 | 0.083/0.123/0.148 | 0.059/0.102/0.127 | 0.058/0.102/0.126 | 0.708/0.829/0.855 -> 0.703/0.824/0.853 |

Every registry fp32 training row (B = 48, in-process compile + CUDA-graph replay of module forward + backward, medians of 3 interleaved timings; the run-to-run spread of one row is 1-2 %, L512 d_cond 384 measured 0.968 / 0.940 / 0.957 in three runs):

| rows | L | PyTorch ms | portable ms before | after | ratio before | after |
|---|---|---|---|---|---|---|
| token d768 / d_cond 384 | 128 | 1.269 | 1.224 | 1.171 | 0.962 | 0.923 |
| | 256 | 2.152 | 2.078 | 1.993 | 0.958 | 0.926 |
| | 384 | 3.110 | 2.979 | 2.871 | 0.951 | 0.923 |
| | 512 | 3.950 | 3.951 | 3.824 | 0.995 | 0.968 |
| | 640 | 4.966 | 4.869 | 4.627 | 0.988 | 0.932 |
| | 768 | 5.927 | 5.812 | 5.548 | 0.979 | 0.936 |
| token d768 / d_cond 768 | 128 | 1.443 | 1.381 | 1.340 | 0.959 | 0.929 |
| | 256 | 2.504 | 2.384 | 2.306 | 0.954 | 0.921 |
| | 384 | 3.581 | 3.525 | 3.425 | 0.996 | 0.956 |
| | 512 | 4.562 | 4.489 | 4.346 | 0.989 | 0.953 |
| | 640 | 5.538 | 5.472 | 5.351 | 0.981 | 0.966 |
| | 768 | 6.697 | 6.586 | 6.342 | 0.986 | 0.947 |
| atom d128 / d_cond 128 | 1024 | 0.997 | 0.968 | 0.942 | 0.970 | 0.945 |
| | 2048 | 1.851 | 1.851 | 1.790 | 0.998 | 0.967 |
| | 4096 | 3.551 | 3.588 | 3.442 | 1.011 | 0.969 |
| | 8192 | 6.777 | 6.969 | 6.711 | 1.029 | 0.990 |

(Module step includes the AdaLN, which is outside this change; its fp32 training kernels are part of the same time.)

## Tried and rejected

* Two-stream backward (dWs / dcond / dWsc GEMMs on a side stream under the SwiGLU pass, as one opaque op): 762 -> 660 us in a stand-alone graph, but in the full step the overlapped kernels slow down
  (SwiGLU pass 185 -> 228 us, nn GEMMs 214 -> 254 us) and the step time did not move (0.979 vs 0.981); the opaque op also costs the compiler its fusions. Removed.
* Fused dh + SwiGLU at the token width with the K-major tile: 373 us against 316 us for the split; the GEMM alone (no epilogue) with a 128x256 tile matches cuBLAS (143 vs 146 us) but the epilogue
  (a, b in, da, db out, 452 MB) does not overlap with the MMAs at the small tiles the registers allow.
* Forward `mixed` (expand + SwiGLU in one Triton kernel, cuBLAS squeeze): 452 us for the kernel against 343 us for cuBLAS + the flat pass; `fused` composed forward 3.20 ms vs 3.01 ms.
* Merged dWcat (`dab.t() @ x`) or `(x.t() @ dab).t()` for fp32: within noise (0.976-0.978 vs 0.978).
* Interleaved `[a|b]` saved buffer and an fp16-scaled `dab`: not tried. The saved-buffer layout does not change the bytes the pass moves (both halves are already coalesced); fp16 `dab` needs a per-tensor
  scale (an extra reduction pass) and is only worth it if the module error stays within 1.5x of PyTorch TF32.
* bf16 with the same K-major weights (`CT_TWB` experiment): ladder 0.952 / 0.964 / 0.973 against 0.955 / 0.971 / 0.980, but it puts the portable rung at or below the CUDA rung (c/p 1.006, 1.002),
  which is the ordering the maintainer forbids; left off.
* Atom `b2b` forward: a wider tile search (128x128, 8 warps, K tiles 64) finds 696 us against 742 us; the 8-warp row is outside the grid's `num_warps` domain (and the default set is at its 32-row cap), so it is not added.

## What is left

* Atom rows at L >= 2048 are at 0.967 / 0.969 / 0.990 (target 0.95). The step is dominated by the AdaLN kernels (`_dgrad_condln_kernel` 380 us, `_bwd_x_kernel` 240, `_epilogue_train_kernel` 173, a 64 us
  device memcpy of an (M, 128) tensor in the AdaLN backward) and the fused atom forward `_b2b_fwd_train_kernel` (740 us against a bandwidth floor near 330 us).
* Token d_cond 768, L 384 / 512 / 640 (0.953-0.966) and token d_cond 384, L512 (0.94-0.97 across runs) are within the 1-2 % noise of 0.95.
* The "nn" GEMM penalty also applies to the AdaLN's own backward GEMMs (`[18432, 1536] x [1536, 384]` runs at 185 TFLOP/s); that is `kernels/adaln/triton/training.py`.
