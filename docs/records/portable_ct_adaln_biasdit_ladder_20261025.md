# Portable ConditionedTransition / AdaLN / bias-only DiT against PyTorch, H100 (2026-10-25)

Goal: the portable (Triton) kernel family at least as fast as the best PyTorch statements (`torch.compile` + CUDA graph, or eager + graph) in every case the
ladder audit (`docs/records/performance_ladder_20261025.md`, commit 12ae4ac8) flagged `portable_vs_pytorch` for these three ops. Reached only through
`settings.force_portable_kernels` on H100; the H100 CUDA paths are untouched. Measured with the ladder harness itself
(`benchmarks/runners/ladder.py run --group '^(adaln_|ct_|bodit_)' --repeats 3`, one H100, commit of this record for "after").

## Result

Of the 30 rows of these three ops (the audit's violations plus the rows that were already fine), 19 are 5 % or more faster than PyTorch, and none of the 11 others is slower by more than the repeat spread (0.4 %): ConditionedTransition bf16 training 0.968 - 0.983, ConditionedTransition fp32 training 0.969, bias-only DiT fp32 training 0.961 / 0.993, bias-only DiT bf16 training 0.995 - 1.004 (three of its five rows are ties within the noise and two are 0.4 % over 1.0). None of these reaches the 0.95 that was asked for; why is under "Not fixed". All inference rows and AdaLN training are below 0.95.

ms; ratio = portable / best PyTorch (< 1 wins). "before" = commit 12ae4ac8 (`out_full_A` / `out_full_B`), "after" = the head of this branch (`out_final3`).

| op | mode | dtype | shape | L | PyTorch best (before) | portable before | portable after | ratio before | ratio after |
|---|---|---|---|---|---|---|---|---|---|
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | 0.041 | 0.039 | 0.032 | 0.958 | 0.778 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | 0.027 | 0.025 | 0.018 | 0.942 | 0.658 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | 0.036 | 0.035 | 0.027 | 0.986 | 0.771 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.034 | 0.040 | 0.024 | 1.176 | 0.711 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 1024 | 0.885 | 0.793 | 0.699 | 0.896 | 0.789 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 384 | 0.372 | 0.337 | 0.305 | 0.905 | 0.822 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 768 | 0.690 | 0.618 | 0.540 | 0.895 | 0.784 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | 0.643 | 0.699 | 0.589 | 1.088 | 0.917 |
| bias-only DiT | inference | bf16 | 12x64 | 512 | 0.284 | 0.301 | 0.146 | 1.062 | 0.514 |
| bias-only DiT | inference | bf16 | 24x32 | 512 | 0.281 | 0.293 | 0.154 | 1.042 | 0.546 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | 0.156 | 0.165 | 0.095 | 1.060 | 0.605 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | 0.280 | 0.296 | 0.149 | 1.058 | 0.534 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | 0.429 | 0.441 | 0.218 | 1.028 | 0.507 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | 0.216 | 0.237 | 0.140 | 1.100 | 0.653 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | 0.435 | 0.453 | 0.235 | 1.042 | 0.539 |
| bias-only DiT | training | bf16 | 12x64 | 512 | 3.314 | 3.515 | 3.300 | 1.061 | 1.000 |
| bias-only DiT | training | bf16 | 24x32 | 512 | 3.336 | 3.578 | 3.327 | 1.073 | 0.995 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | 1.670 | 1.798 | 1.680 | 1.077 | 1.004 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | 3.313 | 3.552 | 3.307 | 1.072 | 1.001 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | 5.044 | 5.356 | 5.007 | 1.062 | 0.997 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | 3.877 | 4.021 | 3.706 | 1.037 | 0.961 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | 7.218 | 7.578 | 7.138 | 1.050 | 0.993 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | 0.148 | 0.171 | 0.125 | 1.153 | 0.839 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 0.083 | 0.087 | 0.059 | 1.051 | 0.705 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.124 | 0.131 | 0.102 | 1.058 | 0.829 |
| ConditionedTransition | inference | fp32 | d768 A5/A48 | 384 | 0.137 | 0.125 | 0.109 | 0.913 | 0.798 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 2.986 | 3.206 | 2.886 | 1.074 | 0.971 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.245 | 1.335 | 1.223 | 1.072 | 0.983 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 2.301 | 2.473 | 2.228 | 1.075 | 0.968 |
| ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | 2.902 | 3.004 | 2.804 | 1.035 | 0.969 |

The "before" PyTorch time of a row differs from the "after" one by run-to-run noise (up to 2 %); the ratio after uses the PyTorch time of the same run.

## What was wrong (per-kernel profiles of the rows, `kernel_probe`)

* **Weight casts and copies per call.** The fp32 master weights were re-cast to bf16 at every call (about 12 us of a 87 us ConditionedTransition at L = 384, 13 of 182 at
  L = 1024), AdaLN copied two transposed weights (2 launches) and ran the x statistics as a launch of its own. Now: the casts are cached per
  (pointer, version) with `kernels._capture.param_forms` (scoped to the CUDA-graph capture, so an in-place weight update is seen), the GEMM+gate kernel takes the
  weights as the strided `(NC, NX)` views of their `(NX, NC)` storage (K contiguous, the layout a TF32 / bf16 MMA takes B in), and `_ln_kernel` computes LN(cond) and the
  statistics of x in one launch (`HAS_STATS`).
* **The stream add.** `ConditionedTransition.forward` is `x + delta`; the add was a launch of its own (and 71 us at M = 49152). Inference folds it into the
  squeeze+gate epilogue (`residual=`), the training forward into the gate pass (`_sigmul_fwd` `HAS_RES`). In training the gradient of the two paths into `x`
  was added by autograd (a 71 us kernel); `AdaptiveLayerNorm.forward_with_stream` now returns the stream as an output of the same autograd node, so
  `_bwd_x` adds the residual gradient into `dx` while it writes it (`HAS_DRES`). `BiasOnlyAttention` and `ConditionedTransition` use it.
* **Tiles.** The default config sets were one-axis perturbations of a 64 x 64 x 32, 4-warp tile. A per-kernel sweep on the H100 (CUDA-graph replay, no L2 flush, M = 5 L and 48 L)
  found 128 x 128 x 32..64, 8 warps, 3-4 stages for the bf16 GEMMs (expand 58 -> 52 us, squeeze+gate 53 -> 38 us at M = 5120), 64 x 64 x 32 with 2 warps for the fp32
  adaln training forward, and BLOCK_K 256..1024 with 8 warps for the memory-bound `_bwd_x` (bf16 M = 49152: 243 -> 188 us, fp32 M = 18432: 179 -> 124 us) and
  1024 x 2 rows for the fused LN (6.5 -> 3.5 us). The defaults (`autotune/configs/default/<op>.csv`) now hold those rows, at most 24 of them: a cache miss searches a set of
  more than `autotune_miss_cap` (24) configs only through `heuristic_subset`, which drops the unusual ones, so a longer list would hide the winners. The grids were widened to contain them
  (`cond_transition_expand_swiglu[_saveact]`, `cond_transition_squeeze_gate`).
* Tried and rejected: the large-M bf16 training forward through the Triton expand+SwiGLU kernel instead of cuBLAS + the flat pass (549 us against about 505 us at M = 49152); the weight gradients and the conditioning gradients of the ConditionedTransition backward on side streams (CT bf16 training 1.000 -> 1.018: the overlapped kernels contend for the SMs); a warp-specialised `_dh_swiglu_bwd` (no gain, some configs fail to compile).

The H100 per-card caches (`autotune/data/*/NVIDIA H100 80GB HBM3 (sm90).json`) were not rebuilt: every one of these kernels' entries was already stale against the
current triton / the kernel source, so the runtime takes the default sets above. A rebuilt cache would hold the same winners and make the first call free.

* **Bias-only attention weights.** In inference (no gradient) `BiasOnlyAttention` keeps the softmax of the pair bias per (pair, mask, weights) through `kernels._capture.lookup_inputs`
  (the engine's contract for per-item inputs; the benchmark sets `static_inputs` for `mode=inference`, a sampler declares it), as the CUDA block path does; the pair LayerNorm (100 us at L = 768), the bias
  GEMM (63 us) and the softmax (32 us) were a quarter to a half of the block. Only the portable family (`ada_ln_in.implementation == MINIWORLD`); the PyTorch reference is unchanged. That is what takes the
  bias-only DiT inference rows to 0.52 - 0.65; training cannot keep it (the gradient flows through it).
* **AdaLN training forward.** LN(cond) with its saved mean / rstd and the statistics of x in one launch (`SAVE_C`), the GEMM+gate kernel reads the weights in place (fp32 AdaLN training 0.976 -> 0.917).

## Persistent TMA GEMMs (the one change that moved the training rows)

A Triton GEMM that loads with pointers and lets one program own a tile runs at 55 - 80 % of cuBLAS on an H100. The same GEMM with on-device tensor descriptors (TMA loads and
stores) in a persistent loop (`tl.range(pid, n_tiles, NUM_SMS, flatten=True)`) matches it: M = 49152, K = 768 / 1536, bf16, 328 vs 323 us, 167 vs 166 us, 183 vs 189 us
(`kernels/_tma.py`, plumbing; `kernels/_tma.py::prune_tma` takes the TMA configs off a card or an operand that cannot use them). The kernels got a `TMA` config axis
(0 = off, 1 = TMA, 2 = epilogue in two column halves, 3 / 4 = the same with a warp-specialised loop), so the registry rows, the autotuner and the per-card caches stay as they were:

* `_dh_swiglu_bwd` (the ConditionedTransition backward GEMM with the SwiGLU backward as epilogue): 431 -> 361 us at M = 49152 (bf16).
* `_adaln_fwd_gate` (AdaLN training forward): 187 -> about 150 us.
* `_fwd_expand_swiglu` and `_fwd_squeeze_gate` (the forward pair) have the TMA branches too, and are correct (tests), but measured SLOWER than cuBLAS + the flat passes
  (500 vs 472 us, 380 vs 315 us), and a single-accumulator squeeze with the gate as epilogue only ties (254 vs 256 us): `training._TMA_FWD = False`.

The limit: a fused epilogue is overlapped with the next tile only partly. `_dh_swiglu_bwd` costs its GEMM (190 us) plus about 170 us, which is its 604 MB of epilogue
traffic at HBM speed; persistent grids of 2 - 3 programs per SM, warp specialisation and BLOCK_N = 256 did not change it. So a fusion wins only where it removes a whole round trip through HBM
(the dh of the backward: yes; the gate pass after the squeeze GEMM: no, the traffic is the same).

## Not fixed

Ratio portable / PyTorch, after, of the rows that are not 0.95 or better:

| row | ratio |
|---|---|
| ConditionedTransition training bf16 L384 / 768 / 1024 | 0.983 / 0.968 / 0.971 |
| ConditionedTransition training fp32 L384 | 0.969 |
| bias-only DiT training bf16 d768 16x48 L256 / 512 / 768 | 1.004 / 1.001 / 0.997 |
| bias-only DiT training bf16 24x32 / 12x64 L512 | 0.995 / 1.000 |
| bias-only DiT training fp32 L256 / 512 | 0.961 / 0.993 |

The training time of both blocks is about 55 % cuBLAS GEMMs that are the same on both sides and about 45 % memory-bound passes; the portable side has fewer kernels (113 against 136 for
the bias-only DiT at L = 512) and the same total kernel time, so the ratio sits at 1.0 +- 1 % and the remaining 5 % would have to come from a new class of kernel (a warp-specialised
epilogue GEMM written for the card), not from tiles or fusion at this level. Rejected with numbers: side streams for the weight gradients (CT 1.000 -> 1.018), a persistent /
prefetching dh kernel with pointer loads (431 -> 472 us), the AdaLN backward without the stacked dy copy (bf16 no change, fp32 AdaLN training 0.917 -> 1.020), a fused pair-LayerNorm + bias kernel
(not written: the pair LayerNorm forward and backward are 138 us of 3300 and the PyTorch side spends the same).

The ladder prints `VIOLATION(cuda_vs_portable)` on AdaLN / ConditionedTransition rows: the H100 "CUDA" rung of these two ops is the PyTorch fallback (`docs/dispatch.md`), which the portable kernels
beat; another agent is writing those CUDA kernels (AdaLN route `adaln_rows.cu` / `settings.adaln_h100_cuda`) and nothing here touches CUDA.

## Notes

* Side effect to know about: a first `dev derive` attempt of this work ran with an unset worktree path and wrote the derived (git-ignored) files
  `src/miniworld_engine/kernels/registry/registry_kernel.csv` and `registry_kernel.units.json` of the MAIN checkout `~/miniworld-engine` (derived from its own sources, arch sm86). No tracked file there changed.
* Tests run on the H100 for the final tree: `test_portable_ct_adaln_residual_gpu.py` (incl. the attention-keep test), `test_bias_only_dit_portable_gpu.py`, `test_conditioned_transition_precision_gpu.py`,
  `test_engine_adaln_alignment.py`, `test_determinism_gpu.py`, `test_op_contracts_gpu.py` (110 passed, 1 skipped, TMA configs included) and `test_numerical.py -k "adaln or cond_transition or gated_projection or layernorm_fwd_strided or layernorm_stats"` (31 passed).
  An earlier full `test_numerical.py` run had 10 failures, all in other families (`transition_*_sm90_cuda`, `mpnn_*`). CPU suite: 2 failures, `test_no_stale_caches` and `test_key_flags_are_built_on_both_sides`, identical on the base commit. The registry fixture `tests/cpu/data/autotune-space-20260916.json` was updated for the two pinned kernels whose grid gained the `TMA` axis (`cond_transition_bwd_gemm_swiglu`, `adaln_fwd_gate`).
