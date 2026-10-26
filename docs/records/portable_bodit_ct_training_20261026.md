| op | mode | dtype | shape | L | PyTorch best (base) | portable before | portable after | portable/PT before | portable/PT after | CUDA (base) | CUDA/portable before | CUDA (after) | CUDA/portable after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.247 | 1.225 | 1.161 | 0.982 | 0.934 | 1.291 | 1.054 | 1.299 | 1.119 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 2.312 | 2.233 | 2.173 | 0.966 | 0.943 | 2.369 | 1.061 | 2.376 | 1.094 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 2.982 | 2.889 | 2.822 | 0.969 | 0.948 | 3.079 | 1.066 | 3.081 | 1.092 |
| ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | 2.886 | 2.814 | 2.791 | 0.975 | 0.965 | 2.968 | 1.055 | 2.968 | 1.063 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 0.083 | 0.058 | 0.059 | 0.695 | 0.707 | 0.078 | 1.352 | 0.079 | 1.344 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.123 | 0.102 | 0.102 | 0.827 | 0.824 | 0.121 | 1.188 | 0.121 | 1.183 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | 0.148 | 0.125 | 0.126 | 0.846 | 0.844 | 0.149 | 1.188 | 0.150 | 1.188 |
| ConditionedTransition | inference | fp32 | d768 A5/A48 | 384 | 0.136 | 0.110 | 0.109 | 0.804 | 0.792 | 0.138 | 1.257 | 0.137 | 1.257 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | 1.674 | 1.686 | 1.534 | 1.007 | 0.911 | 1.472 | 0.873 | 1.471 | 0.959 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | 3.339 | 3.311 | 2.974 | 0.992 | 0.889 | 2.859 | 0.864 | 2.855 | 0.960 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | 5.062 | 5.005 | 4.460 | 0.989 | 0.880 | 4.367 | 0.872 | 4.351 | 0.976 |
| bias-only DiT | training | bf16 | 12x64 | 512 | 3.344 | 3.314 | 2.963 | 0.991 | 0.888 | 2.908 | 0.877 | 2.905 | 0.980 |
| bias-only DiT | training | bf16 | 24x32 | 512 | 3.364 | 3.343 | 2.996 | 0.994 | 0.888 | 3.070 | 0.918 | 3.087 | 1.031 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | 3.869 | 3.724 | 3.523 | 0.962 | 0.911 | 3.548 | 0.953 | 3.539 | 1.005 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | 7.310 | 7.186 | 6.727 | 0.983 | 0.922 | 6.752 | 0.939 | 6.757 | 1.004 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | 0.156 | 0.095 | 0.097 | 0.609 | 0.619 | 0.076 | 0.794 | 0.076 | 0.784 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | 0.280 | 0.148 | 0.149 | 0.528 | 0.531 | 0.122 | 0.826 | 0.122 | 0.817 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | 0.429 | 0.217 | 0.218 | 0.505 | 0.502 | 0.173 | 0.796 | 0.173 | 0.796 |
| bias-only DiT | inference | bf16 | 12x64 | 512 | 0.285 | 0.145 | 0.145 | 0.510 | 0.507 | 0.125 | 0.858 | 0.125 | 0.865 |
| bias-only DiT | inference | bf16 | 24x32 | 512 | 0.281 | 0.156 | 0.154 | 0.555 | 0.548 | 0.127 | 0.813 | 0.127 | 0.824 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | 0.216 | 0.140 | 0.140 | 0.650 | 0.653 | 0.136 | 0.969 | 0.136 | 0.970 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | 0.436 | 0.235 | 0.235 | 0.539 | 0.541 | 0.231 | 0.982 | 0.231 | 0.980 |

The "PyTorch best" column is the faster of `pt_compiled` / `pt_eager_graph` of the base run; the ratios are those of each run itself (its own PyTorch time), so they differ a little from
dividing the printed milliseconds. The ladder takes the faster of "compile + graph" and "graph of the eager module" for the portable rung. Run-to-run noise on the shared node is about 1 %.

All 22 training/inference rows of these two ops are now `OK` for `portable_vs_pytorch` (0 violations); every bias-only DiT training row is 0.88 - 0.92 and every
ConditionedTransition bf16 training row 0.934 - 0.948. The one row above 0.95 is ConditionedTransition **fp32** training (0.965, was 0.975), which was not on the list of rows asked for;
it is open (see "Not fixed"). Inference rows did not move (the changes are training-only).

## What changed (per-kernel profile of one training step, bias-only DiT bf16 L = 512, M = 48 x 512 rows; kernel-time sum 3251 us -> 2857 us, PyTorch 3225 us)

1. **The attention core in feature-major layout** (`kernels/bias_only_attention/triton/core_t.py`, ~ -135 us). The core of `BiasOnlyAttention` (value / gate GEMM, softmax(bias) v, sigmoid gate, to_out)
   was a chain of cuBLAS GEMMs joined by layout changes: the value copied to head-major for the batched GEMM, the product permuted back inside the gate pass, and in the backward a permuted
   `dout` and a `cat` of the value and gate gradients (30 + 54 + 40 + 107 us of copy kernels, as much as the GEMMs around them). Kept as the transposes `t[(h, d), (a, l)]` (what `W @ x^T`
   produces; cuBLAS reads the activation as a transposed operand, a flag and not a copy) the product `softmax(bias) v` is one plain batched GEMM over the heads
   (`Vt` is `[H, Dh * A, L]` contiguous, `P^T` a transposed view), `dV^T` and `dP` likewise, and `dvg^T = [dV^T ; dgate^T]` is written into one buffer by the GEMM (`out=`) and the flat gate-backward kernel.
   No GEMM got slower (vg^T 86 us, PV 36 us, to_out 45 us). One autograd Function per call; the softmax of the pair bias stays in autograd and receives `dP`. Only for one pair (`single` `[A, 1, L, d]`), else the old path.
2. **Pair LayerNorm + bias projection in one pass over the pair** (`.../triton/pair_bias.py`, ~ -100 us). `ln_pair` has no offset, so `bias = LN0(pair) @ (W diag(gamma))^T` (the fold `hoist.py`
   already uses): forward 48 us (LayerNorm) + 31 us (GEMM) -> 40 us, backward 75 + 33 + 30 us (LayerNorm backward, skinny weight-gradient GEMM, `dy` GEMM) -> 73 us. A row block is held in registers; the logits come
   out head-major `[H, B L L]`; the backward is a persistent kernel that keeps `dW'` in registers and adds it once per program. Fixed configurations, swept once on the H100 (forward 40 us at 64 rows per program, any warps / stages; backward 73 us at 64 rows, 4 warps, 2 stages, 2 programs per SM; L = 768: 72 / 141 us).
3. **One launch for the weight casts** (`kernels/_cast.py`, ~ -50 us). The fp32 master weights were cast with `torch._foreach_copy_` (3 - 5 launches of 16 - 20 us on 11 - 60 CTAs); `cast_pack` casts up to six
   tensors in one launch of 4096-element programs (3 - 10 us) and writes the expand / scale-bias weights straight into the halves of their concatenation (no `cat`, kept for the backward).
4. **AdaLN backward `D` row-major at the token width** (bf16): `_bwd_x_kernel` wrote the stacked `[dscale; dy]` as 16-row column strips of a `(2N, M)` buffer (169 us at M = 36864); storing the same
   values row-major and handing out the transpose view is 137 us (the floor of its 396 MB is 132 us); cuBLAS reads either layout of `D` at the same speed (measured), the bias gradient becomes a skinny GEMM against ones.
5. **ConditionedTransition: the cond gradient joins the cond LayerNorm backward in-kernel.** `AdaptiveLayerNorm.forward_with_stream_cond` returns `cond` as an output of the same autograd node (like the residual stream
   already was), so the gradient of the gate projection into `cond` is added by `_ln_bwd_kernel` (`HAS_DXR`) instead of a separate 26 us add; and for M < 30 000 rows (H100, bf16) the weight gradients of the two expand
   halves are one GEMM (cuBLAS picks the better kernel there: 138 -> 124 us at M = 18432; at M = 36864 it picks a worse one, so the split stays).

## Tried and not kept (measured)

* Atomics for the bias gradients: `dsb = sum_m dscale` accumulated with one atomic per column per program inside `_bwd_x_kernel`: the kernel went 110 -> 143 us (the autotuner picked another tile), net loss.
* A 2-D persistent gate-backward kernel that also reduces `db_sc` (replacing `_sigmul_bwd` + `dscale.sum(0)`): 109 us against 92 + 17 + 1.5 = 111 us of the real pipeline; the flat kernel is already at the HBM wall (5 streams, 283 MB, 94 us).
* Prefetching the `a` / `b` tiles of `_dh_swiglu_bwd` before the K loop (modes 11 / 12 of a copy of the kernel): 540 - 1700 us against 285 us. The kernel is GEMM (134 us, cuBLAS) + the epilogue traffic (150 us), serialised; Triton does not overlap the epilogue stores with the next tile's MMAs.
* Fused expand + SwiGLU forward, single interleaved accumulator: not written; for the same reason (the existing two-accumulator kernel measured 500 vs 472 us at M = 49152).
* Storing `(a, h)` instead of `(a, b)` (b = h / silu(a)): same bytes, no gain. fp16 / fp8 `ab`: the fp32 path has no 16-bit GEMM epilogue to write it, and fp8 is far outside the error budget.
* A `(M, 2N)` layout of the stacked AdaLN gradient for fp32 / the atom width: the fp32 kernel is already at its floor (143.6 us of 132) and the row-sum of a row-major buffer is 2x slower than the column-major one.

## Not fixed

* ConditionedTransition fp32 training L = 384: 0.965 (kernel mix: TF32 cutlass / xmma GEMMs 60 %, flat SwiGLU backward 186 us at the HBM wall because the fused `_dh_swiglu_bwd` is bf16-only (TMA); a TF32 TMA variant would save an estimated 20 - 40 us of 2.8 ms). Not attempted.
* CUDA < portable now fails where the hand-CUDA bias-only DiT is **barely** faster or slower than the improved portable block: training bf16 24x32 L512 (1.031), fp32 L256 / L512 (1.005 / 1.004, ties within noise);
  the other bias-only rows are 0.96 - 0.98 (they were 0.86 - 0.88). The ConditionedTransition CUDA rung on H100 is the PyTorch fallback in this tree (another branch writes it), so its `cuda_vs_portable` rows (1.06 - 1.34) are unchanged in nature.
  The CUDA paths were not touched here.
* The pair-bias fused kernel serves a non-hoisted pair (`pair_bias_all` keeps the PyTorch fold on non-B200 cards) and only in training; the one-pair feature-major core only for `B == 1`.

## Tests

New `tests/gpu/any/test_bodit_ct_training_portable_gpu.py` (22 cases: `cast_pack` against `.to(bf16)` + `cat`, the feature-major core forward and backward against the fp64 einsum statements for 16x48 / 12x64 / 24x32 / 8x96, bf16 with fp32 master weights and fp32,
the pair-bias kernel against LN + projection in fp64 for B = 1, 2, H = 12 .. 32, C = 64 .. 256; the bf16 error is held to the same-dtype PyTorch error x 1.5 as in the other portable tests, floor 3e-3 / 6e-3).
`test_portable_ct_adaln_residual_gpu.py::test_training_residual_fold_matches_the_unfused_add` compares the bf16 conditioning gradient with the fp32 statements instead of with the unfused bf16 add (the fold now adds in fp32 before one rounding; it is the more accurate of the two).
