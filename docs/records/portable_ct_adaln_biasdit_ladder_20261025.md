# Portable ConditionedTransition / AdaLN / bias-only DiT against PyTorch, H100 (2026-10-25)

Goal: the portable (Triton) kernel family at least as fast as the best PyTorch statements (`torch.compile` + CUDA graph, or eager + graph) in every case the
ladder audit (`docs/records/performance_ladder_20261025.md`, commit 12ae4ac8) flagged `portable_vs_pytorch` for these three ops. Reached only through
`settings.force_portable_kernels` on H100; the H100 CUDA paths are untouched. Measured with the ladder harness itself
(`benchmarks/runners/ladder.py run --group '^(adaln_|ct_|bodit_)' --repeats 3`, one H100, commit of this record for "after").

## Result

Of the 30 rows of these three ops (the audit's violations plus the rows that were already fine), 19 are 5 % or more faster than PyTorch, 3 are faster by less (bias-only DiT fp32 training 0.979 / 0.991, ConditionedTransition fp32 training 0.969), and 8 are not faster: the bf16 `training` rows of ConditionedTransition (1.006 / 1.007 / 0.999, a tie) and of the bias-only DiT (1.017 - 1.029). All inference rows and AdaLN training are below 0.95 (bias-only DiT inference 0.52 - 0.65). The unfixed rows are listed below.

ms; ratio = portable / best PyTorch (< 1 wins). "before" = commit 12ae4ac8 (`out_full_A` / `out_full_B`), "after" = the commit that adds this record (`out_final2`).

| op | mode | dtype | shape | L | PyTorch best (before) | portable before | portable after | ratio before | ratio after |
|---|---|---|---|---|---|---|---|---|---|
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | 0.041 | 0.039 | 0.032 | 0.958 | 0.782 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | 0.027 | 0.025 | 0.018 | 0.942 | 0.657 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | 0.036 | 0.035 | 0.027 | 0.986 | 0.773 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.034 | 0.040 | 0.024 | 1.176 | 0.707 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 1024 | 0.885 | 0.793 | 0.735 | 0.896 | 0.829 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 384 | 0.372 | 0.337 | 0.316 | 0.905 | 0.852 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 768 | 0.690 | 0.618 | 0.565 | 0.895 | 0.821 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | 0.643 | 0.699 | 0.587 | 1.088 | 0.917 |
| bias-only DiT | inference | bf16 | 12x64 | 512 | 0.284 | 0.301 | 0.148 | 1.062 | 0.520 |
| bias-only DiT | inference | bf16 | 24x32 | 512 | 0.281 | 0.293 | 0.157 | 1.042 | 0.561 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | 0.156 | 0.165 | 0.101 | 1.060 | 0.645 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | 0.280 | 0.296 | 0.151 | 1.058 | 0.538 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | 0.429 | 0.441 | 0.224 | 1.028 | 0.518 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | 0.216 | 0.237 | 0.140 | 1.100 | 0.653 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | 0.435 | 0.453 | 0.238 | 1.042 | 0.545 |
| bias-only DiT | training | bf16 | 12x64 | 512 | 3.314 | 3.515 | 3.364 | 1.061 | 1.017 |
| bias-only DiT | training | bf16 | 24x32 | 512 | 3.336 | 3.578 | 3.419 | 1.073 | 1.023 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | 1.670 | 1.798 | 1.718 | 1.077 | 1.029 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | 3.313 | 3.552 | 3.385 | 1.072 | 1.024 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | 5.044 | 5.356 | 5.125 | 1.062 | 1.027 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | 3.877 | 4.021 | 3.774 | 1.037 | 0.979 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | 7.218 | 7.578 | 7.144 | 1.050 | 0.991 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | 0.148 | 0.171 | 0.126 | 1.153 | 0.845 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 0.083 | 0.087 | 0.058 | 1.051 | 0.705 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.124 | 0.131 | 0.102 | 1.058 | 0.824 |
| ConditionedTransition | inference | fp32 | d768 A5/A48 | 384 | 0.137 | 0.125 | 0.108 | 0.913 | 0.789 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 2.986 | 3.206 | 2.990 | 1.074 | 1.006 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.245 | 1.335 | 1.250 | 1.072 | 1.007 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 2.301 | 2.473 | 2.300 | 1.075 | 0.999 |
| ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | 2.902 | 3.004 | 2.803 | 1.035 | 0.969 |

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

## Not fixed

Ratio portable / PyTorch, after (ladder rows still marked `portable_vs_pytorch`; those with a ratio inside the repeat spread are marked `noise` in the ladder note):

| row | ratio | note |
|---|---|---|
| ConditionedTransition training bf16 L384 / 768 / 1024 | 1.007 / 0.999 / 1.006 | tie (ladder: within noise); the remaining launches are the GEMM-fused `_dh_swiglu_bwd` (431 us at L1024, Triton does not reach cuBLAS + a fused pass) and bandwidth-bound elementwise passes both sides run |
| bias-only DiT training bf16 d768 16x48 L256 / 512 / 768 | 1.029 / 1.024 / 1.027 | 70-150 us of 1.7-5.2 ms. The block's pair LayerNorm forward + backward (`layer_norm_fwd_fused` + `layer_norm_bwd_dx_fused`, 140 us at L512) and the attention-side elementwise ops are opaque ops that inductor cannot fuse into its neighbours, which is what the PyTorch statements get |
| bias-only DiT training bf16 24x32 / 12x64 L512 | 1.023 / 1.017 | same |
| bias-only DiT training fp32 L256 / 512, ConditionedTransition training fp32 | 0.979 / 0.991, 0.969 | faster, but by less than 5 %; TF32 cuBLAS GEMMs are equal on both sides and make most of the time |

AdaLN fp32 training is 0.917 and every inference row is below 0.95. The ladder also prints `VIOLATION(cuda_vs_portable)` on the AdaLN / ConditionedTransition rows
now: the "CUDA" rung of these two ops on an H100 is the PyTorch fallback (`docs/dispatch.md`: no CUDA path yet), so the portable kernels beat it.

## Tests

* `tests/gpu/any/test_portable_ct_adaln_residual_gpu.py` (new): the folded residual is the unfused add, forward (bitwise in bf16) and backward, on both training forwards; the
  cached casts follow an in-place weight update.
* `tests/gpu/any/test_engine_adaln_alignment.py` follows the kernel's two new stride arguments (no tolerance touched).
* Parity (`test_bias_only_dit_portable_gpu.py`, `test_conditioned_transition_precision_gpu.py`, determinism, op contracts, numerical): see the commit message / report.
* CPU suite: the registry key-gap audit (four flags recorded in `build/key_gaps_allowed.csv`), the compile-wrap coverage (a method named `_run` collided with a launcher by name in the audit),
  the sweep page and the derived registry (`dev derive --arch sm86`) were brought up to date; `test_no_stale_caches` and `test_key_flags_are_built_on_both_sides` fail identically on the base commit.

## Method notes

`BENCH_TOP_K` (all kernels in the `kernel_probe` list) was a local, uncommitted change to `bench.py`. Every tuning number above is a CUDA-graph replay of the
kernel alone; the ladder rows are the harness's own.

## Notes

* Side effect to know about: a first `dev derive` attempt of this work ran with an unset worktree path and wrote the derived (git-ignored) files
  `src/miniworld_engine/kernels/registry/registry_kernel.csv` and `registry_kernel.units.json` of the MAIN checkout `~/miniworld-engine` (derived from its own sources, arch sm86). No tracked file there changed.
* Tests run on the H100 for the final tree: `test_portable_ct_adaln_residual_gpu.py` (incl. the attention-keep test), `test_bias_only_dit_portable_gpu.py`, `test_conditioned_transition_precision_gpu.py`,
  `test_engine_adaln_alignment.py`, `test_determinism_gpu.py`, `test_op_contracts_gpu.py` (106 passed, 1 skipped) and `test_numerical.py -k "adaln or cond_transition or gated_projection or layernorm_fwd_strided or layernorm_stats"` (31 passed).
  An earlier full `test_numerical.py` run had 10 failures, all in other families (`transition_*_sm90_cuda`, `mpnn_*`). CPU suite: 2 failures, `test_no_stale_caches` and `test_key_flags_are_built_on_both_sides`, identical on the base commit.
