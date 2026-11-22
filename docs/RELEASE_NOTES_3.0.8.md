# miniworld-engine 3.0.8

Base: 3.0.7 (tag `v3.0.7` = `09615f07e`, `main` = `ffc4204cd` after the `pixi.lock` commit). Release branch `rc-3.0.8` = `main` at `621b0729f` + two commits (version bump, these notes); tag `v3.0.8` sits on the notes commit.

**What this release is.** A repair step on top of 3.0.7 for the A100 and the RTX 5090: `TokenDiTProtenix` / `TokenDiTMiniWorld` with `extra_pair_bias` (OpenDDE's [L, L] logit bias) now run on the `cuda` backend of the sm_80 route instead of falling back to Triton. The H100 is unchanged. No prebuilt artifact changed (the fix is Python only); the three kernels wheels are republished at 3.0.8 because the engine pins them `==3.0.8`.

## Changes

1. **A100 and RTX 5090: `extra_pair_bias` on the `cuda` backend of the token DiT (inference).** Cause: a single gate in `token_dit.TokenDiTProtenix._a100_route` refused every call with `extra_pair_bias` ("not an input of the sm_80 attention kernels"), so the ladder ran Triton: 9,600 Triton calls per OpenDDE target in FoldForge F4. Now the sm_80 attention (`kernels/integrations/augattn_sm80.py`, `_fwd` / `_fused_update` / `delta` / `forward` take `extra`) adds the bias in one broadcast pass into the pair-bias planes (`[1, 1, L, L]` over B and H; no `[B, H, L, L]` fp32 temporaries), in the core's units, summed in fp32 and rounded once; masked keys stay masked (a large positive extra on a padded key does not unmask it; tested). The route accepts a floating `[L, L]` bias on 4-D activations at inference; with autograd (or any other shape) it still refuses and the ladder runs Triton, as before. Applies to `TokenDiTProtenix` and, by inheritance, `TokenDiTMiniWorld`; `TokenDiTBoltz2` and `TokenDiTRoseTTAFold3` still refuse `extra_pair_bias` and are pinned to their old versions.
2. **Speed (from the branch agent report, not re-measured here), cuda vs the Triton route it replaces:** A100 bf16, L = 2048, S = 1: 2410 us vs 17835 us; RTX 5090 bf16, L = 768, S = 5: 928 us vs 2222 us.
3. **Accuracy** (engine standard, rms / max vs float64): GREEN vs fp64 in the branch tests (`tests/gpu/test_extra_bias_and_mixed_dtypes.py`: bf16 / master and fp32, S = 1 and 5, plus a padded-key test; on the installed wheels below). The fp32 core is TF32 (1e-4 level), the same YELLOW class as the module's fp32 cuda path before this change.

## Versions (`backend_versions`)

Only these entries changed:

| class | card | 3.0.7 | 3.0.8 |
|---|---|---|---|
| TokenDiTProtenix | cuda:a100_sm80 | 5 | 6 |
| TokenDiTProtenix | cuda:5090_sm120 | 2 | 3 |
| TokenDiTMiniWorld | cuda:a100_sm80 | 6 | 7 |
| TokenDiTMiniWorld | cuda:5090_sm120 | 2 | 3 |

`TokenDiTBoltz2` / `TokenDiTRoseTTAFold3` inherited the bump and are pinned back (`cuda:a100_sm80` 5, `cuda:5090_sm120` 2). No module `version` changed; no `h100_sm90` or `b200_sm100` entry changed. Judge: tokens for the four re-versioned rows were re-issued on `a100_sm80` and `5090_sm120` (accuracy and speed measured, owed 0); `h100_sm90` owed 0, `b200_sm100` owed 5119 as in 3.0.7.

## Prebuilt artifacts

Unchanged. The prebuild plans (`_prebuild --check`, same tree) produce byte-identical index files to 3.0.7 (sm90 580, sm80 75, sm120 67 artifacts); manifest names against the 3.0.7 wheels: sm90 365 -> 365, sm80 75 -> 75, sm120 67 -> 67, dropped 0, added 0; the carried-from-older-wheel names (sm90, 8 files) are unchanged and byte-identical. The kernels wheels differ from 3.0.7 only in the version in the dist-info. `ldd` of every staged and installed `.so`: 0 link a CUDA 13 library, 0 unresolved, no RPATH. The sm120 `pwa_gemm_sm120.so` is still the nvcc 12.8 override. `twine check`: 5 of 5 PASSED; the engine METADATA pins the three kernels wheels `==3.0.8`, no git URLs. No B200 wheel.

## Installed-wheel verification (extracted final wheels, empty JIT cache, no nvcc, no repo on the path)

| item | H100 | A100 (cssb3) | RTX 5090 (cssb-rtx) |
|---|---|---|---|
| import, versions | 3.0.8 | 3.0.8 / 3.0.8 | 3.0.8 / 3.0.8 |
| every indexed artifact loads, 0 builds | sm90 580 of 580 (164 + 190 + 226) | sm80 75 of 75 | sm120 67 of 67 |
| accuracy per module (pytorch / cuda / triton) | 12 modules, 36 rows, 619 G / 191 Y / 0 RED / 0 error; builds 0 | 12 modules, 36 rows, 610 G / 200 Y / 0 RED / 0 error; builds 0 | 13 modules (incl. TokenDiTChai1), 39 rows, 651 G / 201 Y / 0 RED / 0 error; builds 0 |
| `test_extra_bias_and_mixed_dtypes` (incl. TokenDiTProtenix `extra_pair_bias`, cuda served, masked keys) | 12 passed, 6 skipped (A100 / 5090 tests) | 18 passed | 6 passed, 12 skipped |
| `test_ieee_f32` | 5 passed, 49 skipped | 49 passed, 5 skipped | 49 passed, 5 skipped |
| `test_bias_only_logit_modes` | 24 passed, 4 skipped | 20 passed, 8 skipped | 20 passed, 8 skipped |
| `test_bias_only_fp32_logits` (_sm80 on A100) | 82 passed, 8 skipped | 6 passed | 90 skipped (not an sm_80 card test) |
| `test_large_magnitude_inference_sm90` / `_sm80` | 98 passed, 1 xfailed (documented TriAttn limit) | 282 passed, 7 skipped | 282 passed, 7 skipped |
| `test_pwa_long_inference_sm90` / `_sm80` | 26 passed | 27 passed, 3 skipped | 25 passed, 5 skipped |
| `test_bias_only_long_inference_sm90` | 21 passed | | |
| `test_bias_only_mid_inference` | 25 passed, 6 skipped | 18 passed, 13 skipped | 18 passed, 13 skipped |
| `test_transition_sm80_resid_overflow` | | 4 passed | 4 passed |
| `test_restored_triton_backends` | 63 passed | 63 passed | 59 passed, 4 skipped |
| `test_gated_core_cache` | 19 skipped (B200 only) | 19 skipped | 19 skipped |
| kernel-build lines / `KernelBuildError` in all logs | 0 | 0 | 0 |

**Known failing test (pre-existing, not caused by this release):** `tests/gpu/test_chai1_identity_adaln_sm80.py::test_identity_kernels_vs_fp64[True-dtype1-transition]` fails on `main` before the `extra_pair_bias` change (A100: `_rel(cuda) = 0.01023 > max(_rel(triton) = 0.00876, 0.008)`; 13 of 14 cases pass; same result on the installed 3.0.8 wheel). It is a bf16 Transition tolerance comparison against the Triton backend and does not involve `extra_pair_bias`.

B200: not verified in this release (source / JIT only).

## Known limits (unchanged from 3.0.7)

- TriAttn fp16 score margin and Transition fp16 `h` margin: guards not implemented; the H100 large-magnitude net carries the documented TriAttn xfail.
- Judge requirements rows still cover PWA L <= 2048 and token DiT L <= 768; long L is GPU-test-verified only.
- `extra_pair_bias` on the A100 / 5090 is inference only; `TokenDiTBoltz2` / `TokenDiTRoseTTAFold3` still refuse it (Triton).
- B200: source / JIT only (no kernels wheel); `judge owed` 5119; not a supported release target.
- 5090 AF3 n >= 25 OOM and H100 n >= 37 / A100 n >= 38 limits as in 3.0.6 (not re-measured).
- `ruff` (141) / `pyright` (55) findings in the merged B200 code, not fixed.

## Release steps

NOT DONE: tag, PyPI upload, `pixi.lock`, push. Runbook: `~/miniworld-engine-scratch/release308/UPLOAD_RUNBOOK.md`.
