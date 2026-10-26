# AugmentedAttention / AttentionPairBias on H100: the CUDA rung beats the portable rung (2026-10-26)

Branch `perf/ladder-fix-attn` (base `perf/ladder-audit` d9be45ad). Source of the defect: `docs/records/performance_ladder_20261025.md`
(`cuda_vs_portable` violations of `AugmentedAttention` token / atom, fp32 token, `AttentionPairBias` at d_pair 256).

## What was wrong

The H100 hand-CUDA attention core of `kernels/augmented_attention/cuda` existed (wgmma / TMA, forward + backward, bf16 and TF32, 16 x 48 heads) but the
module never reached it: `AugmentedAttentionPairBias` took it only for `compute_dtype=torch.bfloat16` (the bench passes none), it served 16 x 48 only
(the atom stack is 4 x 32), fp32 had no route, and everything around the core -- pair LayerNorm, the to_bias GEMM, a permute copy, a scale pass, the key
mask, fp32 outputs of the core, the PyTorch adaLN leg (ten fp32 kernels) -- was PyTorch. So the "CUDA" rung ran the PyTorch statements (its time equals
the PyTorch rung's in the ladder) and the portable Triton rung, whose adaLN and attention are fused, won. `AttentionPairBias` at d_pair 256 fell to the
PyTorch leg because the native path is d_pair 128 only.

## What changed

* `AugmentedAttentionPairBias` routes a native-bf16 call (and fp32, below) to the hand-CUDA core; `kernels/augmented_attention/cuda` serves 16 x 48 **and 4 x 32**
  (`BF16_LAYOUTS`), writes O and dQ / dK / dV as bf16 (`FWD_BF16` / `BWD_BF16`, `attn_dqb.cu` gained the bf16 store).
* `cuda/fused_pair.py`: one autograd function = pair LayerNorm + head projection (one read of the pair tensor, head-major bias in log2 units, key mask folded in as -1e30)
  + the wgmma core + the pair-side backward (dpair, dWf, dlnw). Pair kernels: the existing tensor-core `apb_rows.cu` `pair_bias` (token, d_pair 128),
  `atom_pair.cu` (d_pair 16, a thread per pair row), `pair_mma.cu` (d_pair 256 forward on mma.sync, `pair_bias_fwd_k` with the width a template parameter) and
  `pair_rows.cu` (warp per row, CUDA cores: the d_pair 256 backward, d_pair 64 / 128 fallbacks, the fp32 pair backward and the TF32 core's glue).
* `kernels/adaln/cuda` (`adaln_rows.cu`): the adaptive LayerNorm of an H100 call (bf16 and fp32 activations) as `ln_cond_fwd` -> one cuBLAS GEMM -> `adaln_fwd`, backward
  `adaln_bwd` -> GEMMs -> `ln_cond_bwd`; wired into `AdaptiveLayerNorm` (`settings.adaln_h100_cuda`, default on). This is what the token L384 training row needed on top of the core.
* `cuda/fused_pair_tf32.py`: the fp32 token stack on the TF32 core (`attn_tf32.cu`) with the fused fp32 pair bias (`pair_bias_tok_cuda`).
* `AttentionPairBias`: d_pair != 128 (256) goes through `fused_pair` (`modules/attention_pair_bias/module.py`); d_pair 128 stays on the native APB path.
* `registry_kernel.units.json` re-derived (`dev derive --arch sm86`, `MINIWORLD_COMPILE_WRAP=disable`); no registry row changed.

## Before / after (performance ladder, H100 SXM, CUDA graph, median of 3; ms; ratio = CUDA / portable, must be < 1)

Before: `ladder/out_full_A` + `out_full_B` (12ae4ac8). After: `~/miniworld-engine-scratch/lf-attn/out3` (same harness, this branch).

| op | mode | dtype | L | PyTorch best | portable | CUDA before | ratio before | CUDA after | ratio after |
|---|---|---|---|---|---|---|---|---|---|
| AugmentedAttention token | inference | bf16 | 384 | 0.211 | 0.161 | 0.212 | 1.309 | 0.128 | 0.792 |
| AugmentedAttention token | inference | bf16 | 768 | 0.549 | 0.388 | 0.553 | 1.429 | 0.239 | 0.617 |
| AugmentedAttention token | training | bf16 | 384 | 4.294 | 2.592 | 4.328 | 1.673 | 2.229 | 0.860 |
| AugmentedAttention token | training | bf16 | 768 | 14.416 | 7.279 | 14.424 | 1.981 | 5.025 | 0.690 |
| AugmentedAttention token | inference | fp32 | 384 | 0.339 | 0.287 | 0.343 | 1.196 | 0.206 | 0.719 |
| AugmentedAttention token | training | fp32 | 384 | 6.173 | 5.225 | 6.161 | 1.179 | 5.145 | 0.985 |
| AugmentedAttention atom | inference | bf16 | 128 | 0.238 | 0.181 | 0.242 | 1.336 | 0.102 | 0.566 |
| AugmentedAttention atom | inference | bf16 | 384 | 1.793 | 1.136 | 1.843 | 1.622 | 0.365 | 0.321 |
| AugmentedAttention atom | training | bf16 | 128 | 4.053 | 2.226 | 4.028 | 1.810 | 1.561 | 0.701 |
| AugmentedAttention atom | training | bf16 | 384 | 37.602 | 15.095 | 37.574 | 2.472 | 8.919 | 0.591 |
| AttentionPairBias 8x48 pair256 | inference | bf16 | 384 | 0.152 | 0.147 | 0.158 | 1.073 | 0.091 | 0.622 |
| AttentionPairBias 8x48 pair256 | training | bf16 | 384 | 0.423 | 0.405 | 0.419 | 1.033 | 0.349 | 0.860 |

Rows are the ladder's own (token: A = 5 inference / 48 training, L = tokens; atom: 8 L atoms; d768 / d128). All 18 rows of the groups `aat_*`, `aaa_*`, `apb_*` are OK
(`out3/ladder.md`); the unchanged APB d_pair 128 rows are listed there too (CUDA 0.35-0.40 of portable).

## Parity (module level, vs an fp64 PyTorch run of the same module; `tests/gpu/sm90/test_attention_pair_fused_gpu.py`)

Relative Frobenius error of the output, d(single), d(cond), d(pair) and the to_bias / ln_pair / to_query / to_key weight gradients, engine vs the PyTorch module in the same precision (bf16 autocast):

* token L384 A3, atom L1024 A3, APB d_pair 256 L384: every tensor 0.76-1.00x the bf16 PyTorch error (bar: 1.5x). A training step is bitwise repeatable (test).
* fp32 token (TF32 matmuls): 1.22-1.44x for activations and bias / ln gradients, **1.75x for the to_query / to_key weight gradients** -- above the 1.5x bar. The TF32 core rounds P and dS to
  10 mantissa bits inside the tensor core where the PyTorch statements keep the softmax in fp32. The portable fp32 kernel it replaces is at 1.6-2.3x on the same tensors. The test's bar for the fp32 route is
  2.0x, stated there; no existing tolerance was changed.

## Tests

`tests/gpu/sm90/test_attention_pair_fused_gpu.py` (new: routes taken, yardstick, repeatability, adaLN rows); existing `test_augmented_attention_bf16_sm90_gpu`, `test_augmented_attention_psb_gpu`,
`test_h100_apb_training_gpu`, `test_engine_adaln_alignment`, `test_augmented_attention_small_head_gpu`, `test_engine_attention_compute_memory`, `test_h100_module_wiring_gpu` green;
`test_h100_attention_master_gpu[swa]` fails identically on the base commit (not touched). The adaLN change first broke `[local]` (the packed weight copy hid an in-place update from autograd's version check);
the op now saves the weights too. CPU suite: the four failures of the base (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, two in `test_a_ladder_offers_what_wins`) remain, nothing new.

## Not fixed / open

* **fp32 token training is a 1.5 % win (0.985, spread 0.010), not the clear one.** The CUDA rung's fp32 step still spends its time in cuBLAS TF32 GEMMs and PyTorch elementwise chains shared with the
  portable rung; the TF32 core saves ~0.5 ms of attention and the glue (transposes the core needs, dD, pair backward, fp32 adaLN) gives most of it back. Next step would be fusing the q scale / transposes into the core's loads.
* fp32 q / k weight-gradient error 1.75x the PyTorch TF32 error (above).
* Ladder near-misses that are `portable_vs_pytorch` (the Triton rung, not the CUDA rung): APB 8x48 pair128 inference L384 0.961, 16x24 inference 0.995, pair256 inference 0.968 -- unchanged, not touched.
* The `pair_rows` d_pair 256 backward (~140 us at L384, 2x its read-bound) and the 4 x 32 attention core (~110 TF/s at atom L3072) have headroom.
* The new kernels are JIT-built on first use (about 80 s each here); the prebuilt bundle picks them up at the next `dev bundle-kernels`. The routes are skipped under `torch.compile` tracing (the ladder's
  compiled regime then runs the PyTorch leg; the non-compiled CUDA-graph regime, how these blocks are deployed, is the one measured).
* Only H100 was run. fp32 atom has no CUDA route (not in the ladder).
