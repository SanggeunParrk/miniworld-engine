# AdaLN / ConditionedTransition: H100 hand-CUDA rung of the performance ladder (2026-10-26)

Goal: `hand-CUDA < portable (Triton) < PyTorch` in every row of the two ops. Before this change `docs/dispatch.md` said "no CUDA path yet": the CUDA rung was the PyTorch
fallback and lost to the portable kernels on every AdaLN and ConditionedTransition inference row. Measured with the ladder harness itself
(`benchmarks/runners/ladder.py run --group '^(adaln_|ct_)' --repeats 3`, one H100, d768 cond384, A5 inference / A48 training, compiled + graph and eager + graph, the
faster regime per rung). "before" = commit 32ad4658 (base), "after" = the commit that adds this record.

## Result (ms; ratios: < 1 wins)

| op | mode | dtype | L | PyTorch | portable | CUDA before | CUDA after | CUDA / portable before | CUDA / portable after | portable / PyTorch |
|---|---|---|---|---|---|---|---|---|---|---|
| AdaLN | inference | bf16 | 384 | 0.027 | 0.018 | 0.026 | 0.017 | 1.465 | 0.945 | 0.666 |
| AdaLN | inference | bf16 | 768 | 0.036 | 0.027 | 0.034 | 0.026 | 1.254 | 0.962 | 0.770 |
| AdaLN | inference | bf16 | 1024 | 0.041 | 0.032 | 0.039 | 0.030 | 1.218 | 0.931 | 0.782 |
| AdaLN | training | bf16 | 384 | 0.373 | 0.314 | 0.302 | 0.287 | 0.957 | 0.914 | 0.843 |
| AdaLN | training | bf16 | 768 | 0.689 | 0.567 | 0.558 | 0.533 | 0.985 | 0.940 | 0.823 |
| AdaLN | training | bf16 | 1024 | 0.886 | 0.734 | 0.710 | 0.684 | 0.973 | 0.932 | 0.828 |
| AdaLN | inference | fp32 | 384 | 0.035 | 0.025 | 0.035 | 0.024 | 1.411 | 0.970 | 0.711 |
| AdaLN | training | fp32 | 384 | 0.642 | 0.588 | 0.640 | 0.559 | 1.090 | 0.950 | 0.915 |
| ConditionedTransition | inference | bf16 | 384 | 0.083 | 0.059 | 0.082 | 0.053 | 1.417 | 0.913 | 0.705 |
| ConditionedTransition | inference | bf16 | 768 | 0.123 | 0.101 | 0.120 | 0.095 | 1.173 | 0.938 | 0.822 |
| ConditionedTransition | inference | bf16 | 1024 | 0.149 | 0.126 | 0.144 | 0.113 | 1.139 | 0.898 | 0.849 |
| ConditionedTransition | training | bf16 | 384 | 1.243 | 1.247 | 1.236 | 1.188 | 0.987 | 0.953 | 1.004 |
| ConditionedTransition | training | bf16 | 768 | 2.301 | 2.302 | 2.293 | 2.231 | 0.995 | 0.969 | 1.000 |
| ConditionedTransition | training | bf16 | 1024 | 2.976 | 2.987 | 2.964 | 2.902 | 0.990 | 0.972 | 1.004 |
| ConditionedTransition | inference | fp32 | 384 | 0.137 | 0.109 | 0.137 | 0.090 | 1.268 | 0.819 | 0.796 |
| ConditionedTransition | training | fp32 | 384 | 2.895 | 2.804 | 3.769 | 2.626 | 1.325 | 0.937 | 0.969 |

CUDA < portable in all 16 rows (fp32 training rows fixed in the second round: 1.090 -> 0.950 and 1.325 -> 0.937). CUDA < PyTorch in all 16. The full chain CUDA < portable < PyTorch holds in 13 of 16; the three
ConditionedTransition bf16 training rows have portable / PyTorch = 1.000 - 1.004 (the portable side, owned by another change; CUDA is 3 - 5 % below both there).

Rows still above 0.95 of the portable rung (below 1, but not a "clear" win): AdaLN bf16 inference L768 0.963 and L1024 0.931 / fp32 L384 0.970 (two-launch latency-bound calls of 17 - 30 us:
`ln_rows` + the GEMM; the floor of the second launch is ~9 us), AdaLN fp32 training 0.950, ConditionedTransition bf16 training 0.953 / 0.969 / 0.972 (the remaining time is cuBLAS GEMMs, which are at their
optimum, and bandwidth-bound row passes at 86 % of HBM: `swiglu_fwd/bwd`, `sigmul_fwd/bwd`; fusing them into the GEMM epilogues measured slower, see below), AdaLN bf16 training L768 0.940.
Regimes: the ladder takes, per rung, the faster of compiled+graph and eager+graph; the new GPU test pins that the CUDA kernels launch eager, under `torch.compile` and captured in a graph.

## Second round (what moved the fp32 training rows and the bf16 CT training rows)

* TF32 GEMM operand layouts (measured per GEMM at 18432 rows): the AdaLN weight gradient `(cn^T dsb)^T` 148 -> 119 us, the `dcn` GEMM with a column-major weight 127 -> 91 us; the CT backward `dh`, `dx`, `dcond` with
  column-major weights -21 / -72 / -23 us. bf16 GEMMs are at cuBLAS' optimum in every layout (no change there).
* fp32 AdaLN training stays on the row kernels + cuBLAS forward (171 us) -- the wgmma forward with the pre-activation store is 208 us at fp32 -- and takes the backward GEMM changes: 686 -> 565 us per step.
* Stream gradients folded into the row kernels: `adaln()` returns `(y, x, cond)` with `streams=True`; the gradient of the residual stream joins `dx` inside `adaln_bwd`, the gradient the tail sends to cond (gate projection)
  joins `dcond` inside `ln_cond_bwd` (an optional `add` pointer of both kernels, fp32 add, one rounding): -55 and -31 us of PyTorch add kernels per step at 18432 rows.
* Fixed-order column-sum kernel for the bias / LN-weight gradients (was two PyTorch reductions of 20 us).

## Single entry, shared pieces

* `kernels.adaln.cuda.adaln()` is the only AdaLN entry used by `AdaptiveLayerNorm._cuda_forward` and `ConditionedTransition._cuda_update` / `_cuda_train_update`; the route (wgmma inference, wgmma bf16 training forward, row kernels
  + cuBLAS for fp32 training) is chosen inside it by mode, dtype and shape. One setting (`adaln_h100_cuda`). `adaln_rows.cu` / `__init__.py` carry our additions on top of the attention branch's tip e2a02879
  (optional `add` pointers, `streams`, column sums, TF32 layouts, fake docstrings): take ours on the add/add conflict.
* `kernels/adaln/cuda/row_math.cuh` is the one definition of `warp_sum`, `sigmoid` (`1/(1+__expf)`), `sigmoid_fast` (ex2/rcp, the GEMM epilogues) and `modulate` (`gate * ((x - mean) * rstd) + shift`). Users: `adaln_rows.cu`,
  `adaln_sm90.cu`, `ct_sm90.cu` (through `sm90_common.cuh`) and `token_dit_common.cuh` (so the token DiT, bias-only DiT, DiT-variant and AugmentedAttention row kernels). All uses compile to the expressions they had.
  The token DiT's own `adaln_a` pass computes `gate * d * rstd + shift` (the products in another order, not bit-identical to `modulate`) and keeps that expression; the block-per-row reductions (`block_sum`) stay in
  `token_dit_common.cuh` (a different thread layout from the warp-per-row kernels).

## What was built

* `kernels/adaln/cuda/adaln_sm90.cu` + `infer.py`: sm_90a wgmma AdaLN. `ln_rows` (one warp per row: LN(cond)*lnw to bf16 / TF32 grid and the statistics of x, optionally those of cond) then
  `adaln_gemm`: one warpgroup-specialised GEMM over [Wscale ; Wbias] (two TMA boxes per slab, no per-tile weight packing) with the gate and the LayerNorm of x in its epilogue
  (x is TMA-staged during the main loop, updated in place in the accumulator layout, TMA-stored). The second launch runs under PDL: the weight slabs are requested before `griddepcontrol.wait`.
  The training forward additionally stores the scale pre-activation in the layout of the row-kernel backward. 128-column tiles chosen by measurement (64-column and an A-resident-A variant were slower; kept in the source behind `ADALN_MODE`).
* `kernels/adaln/cuda/__init__.py` + `adaln_rows.cu` are agent D's files, byte-identical to branch perf/ladder-fix-attn tip e020790b (backward, fp32, `settings.adaln_h100_cuda`).
* `kernels/conditioned_transition/cuda/ct_sm90.cu` + `ct_inference.py`: tail inference as two wgmma GEMMs: expand + SwiGLU in the epilogue; squeeze + gate + residual (two accumulators over the h and cond K ranges, bias in registers, residual TMA-staged and added after the bf16 rounding of the gated update, as the unfused ops). Second launch under PDL.
* `ct_rows.cu` + `ct_train.py`: training tail = cuBLAS GEMMs + vectorised row kernels (`swiglu_fwd/bwd`, `sigmul_fwd/bwd` with the gate-bias gradient as block partials) + a fixed-order column-sum kernel. The Transition family's wide kernels are wired too (`_USE_WIDE`, off: 10 us slower on this shape).
* Routing: `dispatch._cuda_on_h100` (H100 -> `KernelBackend.CUDA`; B200 PyTorch; `force_portable_kernels` -> portable, A/B intact). One AdaLN entry (`kernels.adaln.cuda`), one setting (`adaln_h100_cuda`), one module wiring; the wgmma route is chosen **inside** the module wiring by mode / dtype (inference and bf16 training forward: wgmma; fp32 training and unsupported widths: D's row kernels).
* All launches are `@opaque` custom ops with fakes (they run, and are the kernels that launch, under `torch.compile` and in a CUDA graph; pinned by the new GPU test).

### Rejected / measured and kept off
* TMA multicast over a 2-CTA cluster for the expand weights: slower at every M (M=5120 bf16 50 -> 53-68 us), source kept (`CL` template argument).
* A-resident row-block AdaLN GEMM: 112 vs 68 us at M=18432, 183 vs 153 at 36864, 199 vs 194 at 49152 (kept behind `ADALN_MODE=ar`).
* Dual-worker expand (two consumer warpgroups each with own tiles): 187 vs 172 us at M=18432; not committed.
* Wide Transition expand/gate kernels for the CT training tail: net loss ~10 us at L384.
* Fused squeeze + gate + residual wgmma kernel (saving out and sp) for CT bf16 training: 425 us against 108 + 43 + 79 us (a 128 x 128 tile is L2-bound at ~280 TFLOPS); kept, off for training (it is the inference tail).

## Duplicate-implementation table

| stage | same computation as | rung of the ladder that uses it |
|---|---|---|
| `kernels/adaln/cuda/adaln_rows.cu` (D) | AdaLN LN rows + gate, backward | AdaLN fp32 training, unsupported widths |
| `kernels/adaln/cuda/adaln_sm90.cu` (this) | fused AdaLN GEMM + gate + LN | AdaLN inference bf16/fp32, bf16 training forward; the CT module's AdaLN |
| `kernels/bias_only_dit/cuda` `resgate_adaln_rows` / `adaln_in_rows`, `conditioned_transition/triton/token_dit_kernels` `adaln_rows` / `resgate_adaln_rows`, token DiT `token_dit_rows.cu` `adaln_a_k` / `res_adaln_b_k`, local DiT `pre90` / `post90` | the LayerNorm statistics (two-pass, centred, fp32) and the gate/modulate of AdaLN, with the *residual add* (`resgate`), the QKV / pair-bias hoists or the next GEMM's operand rounding folded into the same pass (different fusion boundaries); the standalone AdaLN has no residual stream and its gate comes from a GEMM epilogue | bias-only / token / local DiT rows only. **Shared now:** `warp_sum` and `sigmoid` (one definition, `row_math.cuh`, reached through `token_dit_common.cuh`). **Same arithmetic, kept separate:** `modulate` (adaln_rows.cu, adaln_sm90.cu share it; `adaln_a_k` orders the product differently, so it is not bit-identical). **Separate for a reason:** block-per-row `block_sum` kernels (other thread layout), the residual/gate-residual passes (different operands), `pre90`/`post90` (wgmma row kernels with conditioning tables). Nothing deleted |
| `ct_sm90.cu` expand | `gemm_swiglu_sm90.cu` (token DiT transition expand) | CT inference. Same structure; adds PDL, the 64-column tile and a fp32 rounding option; `gemm_swiglu_sm90` left in place for the DiT block |
| `ct_rows.cu` swiglu/sigmul | `token_dit_rows.cu` `swiglu_rows` / gate rows (same arithmetic, other tensor layouts and fp32/bf16 mixes: they read the DiT block's fused gate buffers) | CT training; not merged with the DiT rows: their layouts differ |

## Tests
* new `tests/gpu/sm90/test_adaln_ct_cuda_gpu.py` (24; passes after the second round): route, kernels launched eager / compiled / graph-captured, fp64 parity within 1.5x the same-dtype PyTorch error for inference and every training gradient (bf16, fp32, eager and compiled), bitwise repeatability, A/B switch launches Triton.
* H100 run, all green: test_conditioned_transition_precision (18), test_engine_adaln_alignment (21), test_portable_ct_adaln_residual (9), test_bias_only_dit_portable (41), test_op_contracts (5), test_determinism (12+1 skipped), test_local_dit_triton (9), test_numerical -k adaln/cond_transition (20), sm90 test_h100_bias_only_dit (84+36 skipped), test_h100_module_wiring (34), test_bias_only_dit_psb (75).
* `test_pair_embed_adaln_portable_gpu.py`: 13 failed, identical failure on the base commit (custom-op aliasing in `pair_embed_adaln_bwd`), untouched.
* CPU suite: 3 expected routing tests updated (H100 now resolves these ops to CUDA); `test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins` fail identically on base. Derived registry units regenerated (`dev derive --arch sm86`, run in this worktree only).
