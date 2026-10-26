# AdaLN / ConditionedTransition: H100 hand-CUDA rung of the performance ladder (2026-10-26)

Goal: `hand-CUDA < portable (Triton) < PyTorch` in every row of the two ops. Before this change `docs/dispatch.md` said "no CUDA path yet": the CUDA rung was the PyTorch
fallback and lost to the portable kernels on every AdaLN and ConditionedTransition inference row. Measured with the ladder harness itself
(`benchmarks/runners/ladder.py run --group '^(adaln_|ct_)' --repeats 3`, one H100, d768 cond384, A5 inference / A48 training, compiled + graph and eager + graph, the
faster regime per rung). "before" = commit 32ad4658 (base), "after" = the commit that adds this record.

## Result (ms; ratios: < 1 wins)

| op | mode | dtype | L | PyTorch | portable | CUDA before | CUDA after | CUDA / portable before | CUDA / portable after | portable / PyTorch |
|---|---|---|---|---|---|---|---|---|---|---|
| AdaLN | inference | bf16 | 384 | 0.027 | 0.018 | 0.026 | 0.017 | 1.465 | 0.952 | 0.666 |
| AdaLN | inference | bf16 | 768 | 0.036 | 0.027 | 0.034 | 0.027 | 1.254 | 0.968 | 0.766 |
| AdaLN | inference | bf16 | 1024 | 0.041 | 0.032 | 0.039 | 0.030 | 1.218 | 0.932 | 0.778 |
| AdaLN | training | bf16 | 384 | 0.373 | 0.314 | 0.302 | 0.285 | 0.957 | 0.909 | 0.842 |
| AdaLN | training | bf16 | 768 | 0.689 | 0.567 | 0.558 | 0.531 | 0.985 | 0.937 | 0.823 |
| AdaLN | training | bf16 | 1024 | 0.884 | 0.733 | 0.710 | 0.680 | 0.973 | 0.927 | 0.830 |
| AdaLN | inference | fp32 | 384 | 0.034 | 0.024 | 0.035 | 0.024 | 1.411 | 0.977 | 0.711 |
| AdaLN | training | fp32 | 384 | 0.640 | 0.587 | 0.640 | 0.643 | 1.090 | 1.094 | 0.918 |
| ConditionedTransition | inference | bf16 | 384 | 0.083 | 0.059 | 0.082 | 0.054 | 1.417 | 0.915 | 0.707 |
| ConditionedTransition | inference | bf16 | 768 | 0.123 | 0.101 | 0.120 | 0.095 | 1.173 | 0.943 | 0.816 |
| ConditionedTransition | inference | bf16 | 1024 | 0.148 | 0.126 | 0.144 | 0.113 | 1.139 | 0.897 | 0.855 |
| ConditionedTransition | training | bf16 | 384 | 1.243 | 1.245 | 1.236 | 1.204 | 0.987 | 0.967 | 1.001 |
| ConditionedTransition | training | bf16 | 768 | 2.298 | 2.300 | 2.293 | 2.274 | 0.995 | 0.989 | 1.001 |
| ConditionedTransition | training | bf16 | 1024 | 2.974 | 2.987 | 2.964 | 2.960 | 0.990 | 0.991 | 1.004 |
| ConditionedTransition | inference | fp32 | 384 | 0.137 | 0.109 | 0.137 | 0.090 | 1.268 | 0.824 | 0.794 |
| ConditionedTransition | training | fp32 | 384 | 2.891 | 2.806 | 3.769 | 2.875 | 1.325 | 1.024 | 0.971 |

14 of 16 rows now have CUDA < portable (the other two: below), and 13 of 16 the full chain CUDA < portable < PyTorch (the three ConditionedTransition bf16 training rows have portable / PyTorch = 1.001 - 1.004, a tie on the portable side that this change does not touch) (inference: all 8 rows, 0.915 - 0.978; AdaLN bf16 training 0.909 - 0.937; ConditionedTransition bf16 training 0.967 - 0.991, ConditionedTransition fp32 inference 0.824).
Several are inside the repeat spread: CT bf16 training L768 / L1024 (0.989 / 0.991), AdaLN fp32 inference (0.978). Ratios are not the clear wins (<= 0.95) the task asked for in those rows.

## Not fixed (be truthful)

| row | CUDA / portable | why |
|---|---|---|
| AdaLN fp32 training L384 | 1.094 | fp32 training stays on agent D's row kernels + cuBLAS (`adaln_rows.cu`); its TF32 GEMMs / `adaln_bwd` (120 us) lose to the portable fused `_bwd_x` + dgrad kernels. A TF32 wgmma backward is the missing piece |
| ConditionedTransition fp32 training L384 | 1.024 | same AdaLN part (it is 40 % of the step) plus cuBLAS TF32 GEMMs both sides; the tail's row kernels do not recover it |
| ConditionedTransition bf16 training: portable / PyTorch 1.001 - 1.004 | - | the portable side, not this change (ties; see portable_ct_adaln_biasdit_ladder_20261025.md). CUDA is below both (0.967 - 0.991 of portable) |

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

## Duplicate-implementation table

| stage | same computation as | rung of the ladder that uses it |
|---|---|---|
| `kernels/adaln/cuda/adaln_rows.cu` (D) | AdaLN LN rows + gate, backward | AdaLN fp32 training, unsupported widths |
| `kernels/adaln/cuda/adaln_sm90.cu` (this) | fused AdaLN GEMM + gate + LN | AdaLN inference bf16/fp32, bf16 training forward; the CT module's AdaLN |
| `kernels/bias_only_dit/cuda` `resgate_adaln_rows`, `conditioned_transition/triton/token_dit_kernels` `adaln_rows`, token / local DiT `pre90`/`post90` stages | AdaLN-like row passes inside the fused DiT blocks (different fusion boundaries: they fold residuals / QKV) | bias-only / token / local DiT rows only; not the standalone AdaLN / CT rows. Not changed (not the same op boundary), nothing deleted |
| `ct_sm90.cu` expand | `gemm_swiglu_sm90.cu` (token DiT transition expand) | CT inference. Same structure; adds PDL, the 64-column tile and a fp32 rounding option; `gemm_swiglu_sm90` left in place for the DiT block |
| `ct_rows.cu` swiglu/sigmul | `token_dit_rows.cu` `swiglu_rows` / gate rows | CT training |

## Tests
* new `tests/gpu/sm90/test_adaln_ct_cuda_gpu.py` (24): route, kernels launched eager / compiled / graph-captured, fp64 parity within 1.5x the same-dtype PyTorch error for inference and every training gradient (bf16, fp32, eager and compiled), bitwise repeatability, A/B switch launches Triton.
* H100 run, all green: test_conditioned_transition_precision (18), test_engine_adaln_alignment (21), test_portable_ct_adaln_residual (9), test_bias_only_dit_portable (41), test_op_contracts (5), test_determinism (12+1 skipped), test_local_dit_triton (9), test_numerical -k adaln/cond_transition (20), sm90 test_h100_bias_only_dit (84+36 skipped), test_h100_module_wiring (34), test_bias_only_dit_psb (75).
* `test_pair_embed_adaln_portable_gpu.py`: 13 failed, identical failure on the base commit (custom-op aliasing in `pair_embed_adaln_bwd`), untouched.
* CPU suite: 3 expected routing tests updated (H100 now resolves these ops to CUDA); `test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins` fail identically on base. Derived registry units regenerated (`dev derive --arch sm86`, run in this worktree only).
