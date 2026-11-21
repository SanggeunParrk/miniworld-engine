# miniworld-engine 3.0.7 (DRAFT, local candidate `rc-3.0.7`; wheels built and verified, not released)

Base: 3.0.6 (tag `v3.0.6`, `main` = `477d48073` + the work merged since). Candidate branch `rc-3.0.7` = `main` at `ba5994b03` + four commits (version bump, one bug fix, one test fix, these notes / README). Not pushed, not tagged, not uploaded.

**What this release is.** A feature and repair step on top of 3.0.6: the H100 fp32 Transition now runs on CUDA when TF32 is off (training and inference), the bias-only token DiT gets an opt-in bf16 pair-bias-logits inference mode (fp32 stays the default and the only training mode), the B200 (sm_100a) kernel set is in the tree (source / JIT only), and the paper benchmark harness is extended. No module `version` changed; one `backend_versions` entry for a shipped card changed (H100 Transition).

## Changes

1. **Transition, H100 (`cuda:h100_sm90` 7 -> 8): fp32 with TF32 off runs on CUDA.** Until 3.0.6 the H100 `cuda` backend refused every fp32 Transition call when torch denies TF32 (the diffusion-head conditioning transitions of af3_default training), and the ladder ran Triton. The H100 now uses the same IEEE fp32 CUDA chain as the A100 / RTX 5090 (3.0.4): FFMA GEMMs with fused SwiGLU / residual epilogues, LayerNorm row kernels, forward and backward (`kernels/ieee_f32`, built for sm_90a, artifact `ieee_f32_sm90`, new in the sm90 kernels wheel). Only the Transition uses the sm_90a build; the other fp32 callers keep cuBLAS on the H100. Judge tokens of `h100_sm90` Transition were re-issued for version 8 (accuracy and speed measured, owed 0).
2. **Bias-only token DiT: opt-in bf16 pair-bias logits (inference).** `miniworld_engine.bias_logits` (docs/bias_logits.md): `"fp32"` (default, and the only mode under autograd / fp32 activations) is exactly the 3.0.6 behaviour (outputs bit-identical on every judge eval cell on H100 / A100 / 5090, from the branch measurement); `"bf16"` computes and stores the logits in bf16 (bf16 GEMM, fp32 accumulate), fp32 softmax over them, P rounded to bf16 (the engine's hoist before 2026-11-19). Switch: `block.set_pair_bias_logits("fp32"|"bf16"|None)`, `bias_logits.set_default` / `scope`, `MINIWORLD_BIAS_LOGITS`. No `backend_versions` change (default outputs unchanged) and no kernel source change for this item.
3. **B200 (sm_100a) kernel set and FP8 routes (commit `63825572b`), source / JIT only.** The B200 work (about 800 sm100 artifacts, TriAttn e4m3 q/k opt-in default OFF, FP8 routes, bounded gated-core caches, BiasOnlyTokenDiT TF32-split / tcgen05 logits kernels) is in the tree with `cuda:b200_sm100` entries in `backend_versions`. **There is no B200 kernels wheel**: the engine pins only the sm90 / sm80 / sm120 kernels wheels, the 3.0.7 prebuild plans exist for those three cards only, and the README now says so. On a B200 the kernels compile on first use (CUDA 12.9+ toolkit) or ahead of time with `python -m miniworld_engine.kernels._prebuild --arch sm100`. `judge owed` for `b200_sm100` is 5119 (tokens not re-issued after the merge); B200 is not a supported release target. See docs/b200/README.md and HANDOFF.md section 21.
4. **Fixed before release (found by the installed-wheel verification of this candidate): `BiasOnlyAttention` Triton backend raised `AttributeError: 'NoneType' object has no attribute 'requires_grad'`** with `bias=None` (4 cases of `test_restored_triton_backends.py::test_bias_only_attention`, H100 and A100). Cause: the bias-logits switch (item 2) calls `norm._wants_grad(module, single, cond, pair, bias)`, which did not accept `None`. The bug exists on `main` since the switch was added and was never in a release; `_wants_grad` now skips `None` tensors (`src/miniworld_engine/modules/norm.py`).
5. **Test stand-in fixed:** `tests/gpu/test_bias_only_fp32_logits_sm80.py` replaced `_attention_weights` with a five-argument function; the method takes the logits mode as a sixth argument since item 2 (6 failures on the A100, test only).
6. **Prebuilt artifacts rebuilt (same sources, shared `.cu` edited by the B200 merge / macros):** `parallel_residual_sm90 / _sm80 / _sm120` and `bias_only_dit_f32_rows_cuda` have new content keys on all three cards (and `ieee_f32_sm90` is new on sm90). No `backend_versions` entry changed for these. Their outputs were checked by the accuracy measurer (ParallelResidual and TokenDiTMiniWorldBiasOnly rows, 0 RED) but were **not compared bit for bit** against 3.0.6.
7. **Benchmarks / paper harness (not in the wheel):** `benchmarks/paper` E1-E6 for H100 / A100 / RTX 5090 (3.0.6) and B200, E5 unified (third-party baselines), `benchmarks/bias_logits_modes.py`, `third_party/` fetch scripts.
8. **Tests (not in the wheel):** H100 / A100 / RTX 5090 large-magnitude inference nets (|x| 1e3, spikes 6.5e4 / 1e5 / 1e6), `test_ieee_f32.py`, `test_bias_only_logit_modes.py`, B200 tests (skip off-card).

## Versions (`backend_versions`)

| class | card | 3.0.6 | 3.0.7 | cause |
|---|---|---|---|---|
| Transition | h100_sm90 | 7 | 8 | fp32 TF32-off IEEE CUDA chain |
| (many classes) | b200_sm100 | absent | new keys | B200 kernel set (source / JIT only) |

All other `h100_sm90`, `a100_sm80`, `5090_sm120` entries are unchanged (the diff of the modules shows only reformatting there). Judge: `owed 0` / `0` (accuracy / speed) on `h100_sm90`, `a100_sm80`, `5090_sm120`; `b200_sm100` owed 5119 / 5119.

## Measured results (from the branch agents, not re-measured in the integration step)

**H100 fp32 Transition, decisive check (TF32 off, real AF3 Transition shapes, cuda vs triton, error vs float64):**

| shape | mode | cuda rms / max | triton rms / max | cuda us | triton us |
|---|---|---|---|---|---|
| pair_transition d=128, 147456 rows | train | 1.20e-6 / 1.14e-6 | 2.51e-4 / 2.86e-4 | 3184 | 3636 |
| single_transition d=384, 384 rows | train | 6.57e-7 / 8.08e-7 | 6.49e-7 / 8.93e-7 | 544 | 1263 |
| pair rollout d=128, 147456 rows | eval | 3.87e-8 / 7.26e-8 | 3.88e-8 / 6.97e-8 | 1193 | 1700 |
| single rollout [20,384,384] d=384, 7680 rows | eval | 5.59e-8 / 8.76e-8 | 5.60e-8 / 9.32e-8 | 467 | 1755 |

In all four the cuda backend ran with `refused=None` and no Triton kernel in the profile.

**F6 (H100, AF3 / Protenix-v1 training step with the engine, 8 micro-steps):** 0.8121 s per example (AF3) and 0.7870 s per example (Protenix v1) with the Transition served on `cuda` (1716 Transition calls, refused: none).

**Accuracy standard** (engine standard, rms / max vs float64): 0 RED everywhere (installed-wheel run below). The H100 `AtomDiTRoseTTAFold3` `norm_key.bias` RED rows in the training-gradient audit are benign: the gradient of that bias is analytically zero (softmax over keys is shift-invariant), so any nonzero reference is noise.

## Release build (`release307`, wheels built from `rc-3.0.7` `86859840e`; later commits change docs only)

- Prebuild (incremental, warm 3.0.6 cache, GCC 12.4 host): sm90 3 new artifacts (`parallel_residual_sm90`, `ieee_f32_sm90`, `bias_only_dit_f32_rows_cuda`), sm80 2, sm120 2; 0 failed. Plans: sm90 580, sm80 75, sm120 67 artifacts.
- Manifest names against the 3.0.6 wheels: sm90 364 -> 365 (+ `ieee_f32_sm90`), sm80 75 -> 75, sm120 67 -> 67; dropped 0. Carried-from-older-wheel names: sm90 8 files, unchanged list, byte-identical (check passed).
- `ldd` of every staged and every installed `.so`: 0 link a CUDA 13 library, 0 unresolved, no RPATH. The sm120 `pwa_gemm_sm120.so` is still the nvcc 12.8 override (unchanged from 3.0.6).
- `twine check`: 5 of 5 PASSED; the engine METADATA pins the three kernels wheels `==3.0.7`, no git URLs. No B200 wheel.
- sha256 (`release307/dist/SHA256SUMS`): see the file; the three kernels wheels have the same sha256 on every build; the engine wheel / sdist are rebuilt whenever source or tests change.

## Installed-wheel verification (extracted final wheels, empty JIT cache, no nvcc, no repo on the path)

| item | H100 | A100 (cssb3) | RTX 5090 (cssb-rtx) |
|---|---|---|---|
| import, versions | 3.0.7 | 3.0.7 / 3.0.7 | 3.0.7 / 3.0.7 |
| every indexed artifact loads, 0 builds | sm90 580 of 580 (164 + 190 + 226) | sm80 75 of 75 | sm120 67 of 67 |
| accuracy per module (pytorch / cuda / triton) | 12 modules, 36 rows, 620 G / 190 Y / 0 RED / 0 error; builds 0 | 12 modules, 36 rows, 608 G / 202 Y / 0 RED / 0 error; builds 0 | 13 modules (incl. TokenDiTChai1), 39 rows, 0 RED / 0 error; builds 0 |
| `test_ieee_f32` | 5 passed, 49 skipped | 49 passed, 5 skipped | 49 passed, 5 skipped |
| `test_bias_only_logit_modes` | 24 passed, 4 skipped | 20 passed, 8 skipped | 20 passed, 8 skipped |
| `test_bias_only_fp32_logits` (_sm80 on A100) | 82 passed, 8 skipped | 6 passed | skipped on this card (as in 3.0.6; the file is not run there) |
| `test_large_magnitude_inference_sm90` / `_sm80` | 98 passed, 1 xfailed (documented TriAttn limit) | 282 passed, 7 skipped | 282 passed, 7 skipped |
| `test_pwa_long_inference_sm90` / `_sm80` | 26 passed | 27 passed, 3 skipped | 25 passed, 5 skipped |
| `test_bias_only_long_inference_sm90` | 21 passed | | |
| `test_bias_only_mid_inference` | 25 passed, 6 skipped | 18 passed, 13 skipped | 18 passed, 13 skipped |
| `test_transition_sm80_resid_overflow` | | 4 passed | 4 passed |
| `test_restored_triton_backends` | 63 passed | 63 passed | 59 passed, 4 skipped |
| `test_gated_core_cache` | 19 skipped (B200 only) | 19 skipped | 19 skipped |
| kernel-build lines / `KernelBuildError` in all test logs | 0 | 0 | 0 |

The first run on the first engine wheel found the two failures of items 4 and 5 (H100: 4, A100: 4 + 6); all of them pass on the final wheel. B200: not verified in this release (no B200 wheel; the B200 branch's own tests ran on the card, see docs/b200/HANDOFF.md).

CPU gate (`rc-3.0.7`, before the last test-only commit): `tests/cpu` 535 + 1062 + 1516 + 409 = 3522 passed, 2 skipped. **ruff and pyright are not clean any more** (3.0.6: clean): ruff 141 findings and pyright 55 errors, all in the merged B200 code (FP8 b200 tests and integrations `pwa_fp8_b200`, `opm_fp8_b200`, `pwa_sm100`, `opm_sm100`, `embedder.py`, `fp8.py`); style / typing only, not fixed in this candidate.

## Known limits (unchanged from 3.0.6 unless stated)

- TriAttn fp16 score margin (real-weight survey: IntelliFold2 2.5x, Chai1 8.7x) and Transition fp16 `h` margin (ESMFold2 `msa_stack.2.msa_transition` 1.8x): guards not implemented; the H100 large-magnitude net carries the documented TriAttn xfail.
- Judge requirements rows still cover PWA L <= 2048 and token DiT L <= 768; long L is GPU-test-verified only.
- B200: source / JIT only (above); `judge owed` 5119.
- 5090 AF3 n >= 25 OOM and H100 n >= 37 / A100 n >= 38 limits as in 3.0.6 (not re-measured).
- `ruff` / `pyright` findings in the B200 code (above).

## Release steps

NOT DONE: tag, PyPI upload, `pixi.lock`, push (waiting for the user's explicit go for 3.0.7). Runbook: `~/miniworld-engine-scratch/release307/UPLOAD_RUNBOOK.md`.
