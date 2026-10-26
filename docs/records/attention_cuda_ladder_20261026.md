# AugmentedAttention / AttentionPairBias on H100: the CUDA rung beats the portable rung (2026-10-26)

Branch `perf/ladder-fix-attn` (base `perf/ladder-audit` d9be45ad). Source of the defect: `docs/records/performance_ladder_20261025.md`
(`cuda_vs_portable` violations of `AugmentedAttention` token / atom, fp32 token, `AttentionPairBias` at d_pair 256; `portable_vs_pytorch` near-misses of APB).

## What was wrong

The H100 hand-CUDA attention core of `kernels/augmented_attention/cuda` existed (wgmma / TMA, forward + backward, bf16 and TF32, 16 x 48 heads) but the
module never reached it: `AugmentedAttentionPairBias` took it only for `compute_dtype=torch.bfloat16` (the bench passes none), it served 16 x 48 only
(the atom stack is 4 x 32), fp32 had no route, and everything around the core -- pair LayerNorm, the to_bias GEMM, a permute copy, a scale pass, the key
mask, fp32 outputs of the core, the PyTorch adaLN leg (ten fp32 kernels) -- was PyTorch. So the "CUDA" rung ran the PyTorch statements (its time equals
the PyTorch rung's in the ladder) and the portable Triton rung, whose adaLN and attention are fused, won. `AttentionPairBias` at d_pair 256 fell to the
PyTorch leg because the native path is d_pair 128 only. The APB portable rung itself ran the pair LayerNorm and the head GEMM as two passes.

## What changed

* `AugmentedAttentionPairBias` routes a native-bf16 call (and fp32, below) to the hand-CUDA core; `kernels/augmented_attention/cuda` serves 16 x 48 **and 4 x 32**
  (`BF16_LAYOUTS`), writes O and dQ / dK / dV as bf16 (`FWD_BF16` / `BWD_BF16`, `attn_dqb.cu` gained the bf16 store).
* `cuda/fused_pair.py`: pair LayerNorm + head projection (one read of the pair tensor, head-major bias in log2 units, key mask folded in as -1e30) + the wgmma core +
  the pair-side backward (dpair, dWf, dlnw). Pair kernels: the existing tensor-core `apb_rows.cu` `pair_bias` (token, d_pair 128), `atom_pair.cu` (d_pair 16),
  the 256-wide `apb_rows.cu` forward (d_pair 256 forward on mma.sync) and `pair_rows.cu` (warp per row: the d_pair 256 backward, the fp32 pair backward, the TF32 core's glue).
* `cuda/fused_pair_tf32.py`: the fp32 token stack on the TF32 core (`attn_tf32.cu`) with the fp32 pair bias (`pair_bias_tok_cuda`).
  **Numerics:** the operands (q scaled, k, v, dO) are rounded to the TF32 grid to nearest in the glue kernels (`prep_fwd32`, `dd_perm`), as cuBLAS does; the tensor core
  alone truncates them. Core-level error against fp64: 2.4-3.0x the PyTorch TF32 statements before, 0.83-1.01x after. Module level: every tensor 0.84-1.00x (the earlier 1.75x on the to_query / to_key weight gradients is gone; the
  test bar is back to the 1.5x yardstick).
* **All routes are `torch.compile`-compatible**: each is an `autograd.Function` around two `opaque` custom ops (`augmented_attention_pair_fused_fwd/bwd`, `augmented_attention_tf32_fwd/bwd`,
  `adaln_h100_fwd/bwd`), the pattern of the other hand-CUDA blocks. Under tracing the routes are taken (before they were skipped and the compiled regime ran PyTorch); extensions build when an op first runs; `dev derive`
  (fake tensors outside a trace) keeps the portable kernels.
* `AttentionPairBias`: d_pair != 128 (256) goes through `fused_pair`; d_pair 128 stays on the native APB path. The **portable** APB inference path computes LayerNorm + head projection in one Triton pass
  (`layernorm_linear_triton_fwd_t`, head-major) and leaves the key mask to the attention kernel (0.1053 -> 0.077 ms at 8 x 48 pair 128, L 384).
* `AdaptiveLayerNorm` -> `kernels/adaln/cuda` (below).
* `registry_kernel.units.json` re-derived (`dev derive --arch sm86`, `MINIWORLD_COMPILE_WRAP=disable`); no registry row changed.

## Deployment regimes (what FoldForge / MiniWorld run)

* **FoldForge** (`~/FoldForge/src/foldforge/models/execution.py`): `torch.compile(model, dynamic=False, options={"triton.cudagraphs": False})` when `config.compile`, plus a CUDA graph of the compiled callable
  (`config.cuda_graph`); the sampler step is one CUDA graph (`sampler_graph.py`). Both are switchable: compile + graph is the default, graph without compile is the fallback / A-B.
* **MiniWorld**: `torch.compile` of the trunk (`models/diffusion/model.py`: `self._trunk_compiled`) and of selected helpers, training with `bf16-mixed` (fp32 master parameters, bf16 autocast) in every config under `configs/miniworld`;
  full-model CUDA-graph training is a separate A/B (`docs/miniworld-training-cudagraph-ab.md`).

So the compiled regime is the deployed one, and the table below holds for **both** regimes (compile + graph, graph only).

## Before / after (performance ladder, H100 SXM, CUDA graph, best regime per rung, median of 3 (APB 5); ms; ratio = CUDA / portable, must be < 1)

Before: `ladder/out_full_A` + `out_full_B` (12ae4ac8). After: `lf-attn/out5` (token, atom) and `out6` (APB), same harness, this branch. All rows of `aat_*`, `aaa_*`, `apb_*` are OK.

| op | mode | dtype | L | PyTorch best | portable | CUDA before | ratio before | CUDA after | ratio after |
|---|---|---|---|---|---|---|---|---|---|
| AugmentedAttention token | inference | bf16 | 384 | 0.212 | 0.162 | 0.212 | 1.309 | 0.120 | 0.743 |
| AugmentedAttention token | inference | bf16 | 768 | 0.548 | 0.389 | 0.553 | 1.429 | 0.227 | 0.582 |
| AugmentedAttention token | training | bf16 | 384 | 4.303 | 2.582 | 4.328 | 1.673 | 2.008 | 0.778 |
| AugmentedAttention token | training | bf16 | 768 | 14.455 | 7.276 | 14.424 | 1.981 | 4.646 | 0.639 |
| AugmentedAttention token | inference | fp32 | 384 | 0.339 | 0.290 | 0.343 | 1.196 | 0.203 | 0.698 |
| AugmentedAttention token | training | fp32 | 384 | 6.181 | 5.236 | 6.161 | 1.179 | 4.795 | 0.916 |
| AugmentedAttention atom | inference | bf16 | 128 | 0.239 | 0.183 | 0.242 | 1.336 | 0.096 | 0.527 |
| AugmentedAttention atom | inference | bf16 | 384 | 1.797 | 1.127 | 1.843 | 1.622 | 0.355 | 0.315 |
| AugmentedAttention atom | training | bf16 | 128 | 4.053 | 2.221 | 4.028 | 1.810 | 1.383 | 0.623 |
| AugmentedAttention atom | training | bf16 | 384 | 37.596 | 15.139 | 37.574 | 2.472 | 8.384 | 0.554 |
| AttentionPairBias 8x48 pair256 | inference | bf16 | 384 | 0.152 | 0.114 (was 0.147) | 0.158 | 1.073 | 0.088 | 0.771 |
| AttentionPairBias 8x48 pair256 | training | bf16 | 384 | 0.425 | 0.406 | 0.419 | 1.033 | 0.334 | 0.825 |

The `portable_vs_pytorch` near-misses (portable / PyTorch best) before -> after: APB 8x48 pair128 inference L384 0.961 -> 0.768, 16x24 inference 0.995 -> 0.794, 8x48 pair256 inference 0.968 -> 0.748;
pair256 training 0.959 -> **0.955** (see open problems).

### Both regimes (`compiled` = torch.compile + CUDA graph, `graph_only` = CUDA graph of the eager module; PyTorch = compiled / eager + graph of the same regime)

Source: the ladder's per-regime CSVs (`lf-attn/regimes.py` over `out5`, `out6`). Only one cell is above 0.95 (`apb_p256` training compiled, portable / PyTorch 0.955); every CUDA / portable and CUDA / PyTorch ratio is <= 0.92.

| group | mode | L | regime | PyTorch | portable | CUDA | portable/PyTorch | CUDA/portable | CUDA/PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| aaa_bf16 | inference | 128 | graph_only | 2.227 | 0.2451 | 0.1027 | 0.110 | 0.419 | 0.046 |
| aaa_bf16 | inference | 128 | compiled | 0.2388 | 0.1826 | 0.09616 | 0.765 | 0.527 | 0.403 |
| aaa_bf16 | inference | 384 | graph_only | 19.15 | 1.624 | 0.3644 | 0.085 | 0.224 | 0.019 |
| aaa_bf16 | inference | 384 | compiled | 1.797 | 1.127 | 0.3552 | 0.628 | 0.315 | 0.198 |
| aaa_bf16 | training | 128 | graph_only | 12.03 | 2.424 | 1.567 | 0.201 | 0.647 | 0.130 |
| aaa_bf16 | training | 128 | compiled | 4.053 | 2.221 | 1.383 | 0.548 | 0.623 | 0.341 |
| aaa_bf16 | training | 384 | graph_only | 99.27 | 16.68 | 8.872 | 0.168 | 0.532 | 0.089 |
| aaa_bf16 | training | 384 | compiled | 37.6 | 15.14 | 8.384 | 0.403 | 0.554 | 0.223 |
| aat_bf16 | inference | 384 | graph_only | 0.716 | 0.1825 | 0.1282 | 0.255 | 0.703 | 0.179 |
| aat_bf16 | inference | 384 | compiled | 0.2117 | 0.1619 | 0.1203 | 0.765 | 0.743 | 0.568 |
| aat_bf16 | inference | 768 | graph_only | 2.419 | 0.4284 | 0.2397 | 0.177 | 0.560 | 0.099 |
| aat_bf16 | inference | 768 | compiled | 0.5482 | 0.389 | 0.2265 | 0.709 | 0.582 | 0.413 |
| aat_bf16 | training | 384 | graph_only | 8.01 | 2.802 | 2.255 | 0.350 | 0.805 | 0.282 |
| aat_bf16 | training | 384 | compiled | 4.303 | 2.582 | 2.008 | 0.600 | 0.778 | 0.467 |
| aat_bf16 | training | 768 | graph_only | 25.22 | 7.766 | 5.076 | 0.308 | 0.654 | 0.201 |
| aat_bf16 | training | 768 | compiled | 14.45 | 7.276 | 4.646 | 0.503 | 0.639 | 0.321 |
| aat_fp32 | inference | 384 | graph_only | 0.6568 | 0.3064 | 0.2127 | 0.467 | 0.694 | 0.324 |
| aat_fp32 | inference | 384 | compiled | 0.3385 | 0.2903 | 0.2026 | 0.858 | 0.698 | 0.599 |
| aat_fp32 | training | 384 | graph_only | 9.114 | 5.594 | 5.201 | 0.614 | 0.930 | 0.571 |
| aat_fp32 | training | 384 | compiled | 6.181 | 5.236 | 4.795 | 0.847 | 0.916 | 0.776 |
| apb_h16 | inference | 384 | graph_only | 0.4243 | 0.08666 | 0.04202 | 0.204 | 0.485 | 0.099 |
| apb_h16 | inference | 384 | compiled | 0.1071 | 0.08506 | 0.04173 | 0.794 | 0.491 | 0.390 |
| apb_h16 | training | 384 | graph_only | 0.9123 | 0.3232 | 0.13 | 0.354 | 0.402 | 0.142 |
| apb_h16 | training | 384 | compiled | 0.3093 | 0.282 | 0.1292 | 0.912 | 0.458 | 0.418 |
| apb_h8 | inference | 384 | graph_only | 0.4161 | 0.08304 | 0.04128 | 0.200 | 0.497 | 0.099 |
| apb_h8 | inference | 384 | compiled | 0.1055 | 0.08109 | 0.04109 | 0.768 | 0.507 | 0.389 |
| apb_h8 | inference | 768 | graph_only | 1.433 | 0.1489 | 0.08954 | 0.104 | 0.601 | 0.062 |
| apb_h8 | inference | 768 | compiled | 0.2586 | 0.1494 | 0.08976 | 0.578 | 0.601 | 0.347 |
| apb_h8 | training | 384 | graph_only | 0.8986 | 0.3021 | 0.1222 | 0.336 | 0.405 | 0.136 |
| apb_h8 | training | 384 | compiled | 0.3099 | 0.2731 | 0.1226 | 0.881 | 0.449 | 0.396 |
| apb_h8 | training | 768 | graph_only | 2.933 | 0.7363 | 0.2718 | 0.251 | 0.369 | 0.093 |
| apb_h8 | training | 768 | compiled | 0.743 | 0.6706 | 0.2716 | 0.903 | 0.405 | 0.366 |
| apb_p256 | inference | 384 | graph_only | 0.5595 | 0.116 | 0.08938 | 0.207 | 0.770 | 0.160 |
| apb_p256 | inference | 384 | compiled | 0.1518 | 0.1136 | 0.08762 | 0.748 | 0.771 | 0.577 |
| apb_p256 | training | 384 | graph_only | 1.252 | 0.4325 | 0.3483 | 0.345 | 0.805 | 0.278 |
| apb_p256 | training | 384 | compiled | 0.4247 | 0.4056 | 0.3345 | 0.955 | 0.825 | 0.788 |

## Parity (module level, vs an fp64 PyTorch run of the same module; `tests/gpu/sm90/test_attention_pair_fused_gpu.py`)

Relative Frobenius error of the output, d(single), d(cond), d(pair) and the to_bias / ln_pair / to_query / to_key weight gradients, engine vs the PyTorch module in the same precision, bar 1.5x:
token L384, atom L1024, APB d_pair 256 (bf16 autocast): 0.76-1.00x. fp32 token (TF32 matmuls): 0.84-1.00x. A bf16 training step is bitwise repeatable (test). The compiled module launches the CUDA cores and matches the eager route (test).

## Tests

New `test_attention_pair_fused_gpu.py` (routes taken, yardstick, repeatability, adaLN rows, compiled routes). Existing `test_augmented_attention_bf16_sm90_gpu`, `_psb_`, `test_h100_apb_training_gpu`, `test_engine_adaln_alignment`,
`test_augmented_attention_small_head_gpu`, `test_engine_attention_compute_memory`, `test_h100_module_wiring_gpu`, `test_op_contracts_gpu` green (199 passed); `test_h100_attention_master_gpu[swa]` fails identically on the base commit.
CPU suite: the four failures of the base (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, two in `test_a_ladder_offers_what_wins`) remain, nothing new.

## The CUDA adaLN (`kernels/adaln/cuda/adaln_rows.cu`): API and semantics, for the merge with `perf/ladder-fix-adaln-ct-cuda`

Canonical entry points (unchanged names): `kernels.adaln.cuda.adaln(x, cond, ln_cond_weight, w_scale, b_scale, w_bias, eps_x, eps_cond)`, `supported(x, cond)`, `available(x, cond)`; setting `settings.adaln_h100_cuda` (default on);
wiring in `AdaptiveLayerNorm.forward` (the engine leg on a restricted card: H100, `implementation=MINIWORLD`, bf16 or fp32 activations of equal dtype, widths 128 / 256 / 384 / 512 / 768 / 1024, not an fp32 call under autocast).

Semantics: `y = sigmoid(LN(c) Ws^T + bs) * LN(x) + LN(c) Wb^T`, `LN(c) = (c - mean) rstd * ln_cond.weight` (fp32 statistics, eps_cond), `LN(x)` without affine (eps_x). Forward: `ln_cond_fwd` (one warp per row) -> one cuBLAS GEMM over `[Ws | Wb]` (bf16: a
one-launch pack of the fp32 master weights; fp32: `torch.cat`, TF32 per the process flag) -> `adaln_fwd` (gate + LN(x) in one pass; the statistics of x saved). Backward: `adaln_bwd` (dy -> `[dscale_pre | dy]`, dx, block partials of `d bs`) -> GEMMs (`dcn`, `dW`) -> `ln_cond_bwd`
(dcond, block partials of `dwc`). Reductions have one fixed order (bitwise repeatable). Weights are saved for backward so autograd refuses an in-place update. Both passes are `opaque` ops (`adaln_h100_fwd` / `adaln_h100_bwd`), compile-compatible.
**Changes since the 5ae2eda6 snapshot agent I copied** (the entry points and the setting are unchanged): `adaln_rows.cu` is templated on the activation dtype (bf16 and fp32; `ld_row` / `st_row` overloads, `chk_act`, `ACT_SWITCH`), and `__init__.py` is rebuilt on two opaque ops
instead of a bare `autograd.Function` calling the extension. Agent I's inference kernel (`adaln_sm90.cu`: one wgmma GEMM with the gate and LN(x) in its epilogue) has no backward; the two are complementary (theirs for no-grad calls, this one for training) if both are kept.

## New kernels vs existing equivalents (no parallel stage kept without a reason)

| new | existing equivalent | kept / why |
|---|---|---|
| `kernels/adaln/cuda/adaln_rows.cu` | agent I's `adaln_sm90.cu` (inference only); PyTorch leg; Triton `adaln_train` (portable) | canonical, byte-identical origin; has the training backward, bf16 + fp32 |
| the 256-wide `apb_rows.cu` forward (d_pair 256 forward) | `apb_rows.cu` `pair_bias_fwd_k` (d_pair 128 hard-wired; the same code) | **a duplicate of the forward with the width a template parameter**; kept separate so the shared APB extension and its prebuilt key do not change; fold back into `apb_rows.cu` as a follow-up |
| `pair_rows.cu` bwd (bf16, d_pair 256) | `apb_rows.cu` `pair_bias_bwd_k` (128 only, TMA) | no 256 variant exists; CUDA-core warp kernel (2x its read bound) |
| `pair_rows.cu` `pair_rows_bwd32` (fp32 pair backward) | `token_dit_train_rows.cu` `pair_ln_bwd` / `pair_bias_bwd` (need the token DiT block's weight pack / ext object) | standalone entry point for the module; same math |
| `pair_rows.cu` `transpose32` | `token_dit_train_rows.cu` `transpose` (same tiled kernel, in the token DiT ext) | duplicate of ~10 lines, adds the scale; unify with the DiT ext later |
| `atom_pair.cu` (d_pair 16 LN + projection) | `layernorm_linear_triton_fwd_t` (portable Triton, inference, head-major); local DiT `pre90` (windowed pair) | CUDA with backward; nothing CUDA for the dense 16-wide pair existed |
| `fused_pair*.py` attention + pair backward | `attention_pair_bias_b200.py` (B = 1, d_pair 128, with the whole block) | different contract (A samples, module-level bias leaves) |

## fp32 atom

Not added. The dense 4 x 32 `AugmentedAttention` atom stack runs in bf16 in every deployed configuration found: MiniWorld trains and infers with `bf16-mixed` (`configs/miniworld/*`), and FoldForge's fp32 / `af3_default` diffusion policy
(`execution.py`: "the engine's FP32 kernels are TF32 tensor core kernels ... in the diffusion head") serves the **windowed** atom DiT, whose TF32 CUDA kernels already exist (`local_dit_h100`, `docs/gpus/h100/dispatch.md`). A ladder row for a configuration nothing runs would
only measure the PyTorch leg against Triton; not added.

## Not fixed / open

* **AttentionPairBias 8x48 pair256 training, compiled regime: portable / PyTorch = 0.955** (needs <= 0.95). Portable is the Triton rung; the pair LayerNorm + projection + their backward are ~60% of the 0.405 ms and the
  compiled PyTorch statements are equally fused. Tried: fused LN + projection forward for training (`layernorm_linear_triton_fn`: graph-only -1%, compiled +6%, so not used), `triton_layer_norm_linear` (+60%, not used), dropping the redundant
  `masked_fill` (no change). A fused Triton pair backward (dpair from dbias without the 75 MB dxn) would take portable to ~0.33 ms but the CUDA rung (0.334) would then fail its own 5% margin; both need to move together.
* The `pair_rows` d_pair 256 backward (about 140 us at L384) and the 4 x 32 core at large atom lengths have headroom; the fp32 training step is bound by TF32 GEMMs and PyTorch elementwise work shared with the portable rung (CUDA / portable 0.92-0.93).
* The new kernels JIT-build on first use (about 80 s each here); the prebuilt bundle picks them up at the next `dev bundle-kernels`.
* Only H100 was run.
