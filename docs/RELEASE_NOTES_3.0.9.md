# miniworld-engine 3.0.9

Base: 3.0.8 (tag `v3.0.8` = `b1b3a4ca0`, `main` = `219228976` after the `pixi.lock` commit). Release branch `rc-3.0.9` = `main` at `f06154bc7` + two commits (version bump `f14045d3d`, these notes); tag `v3.0.9` sits on the notes commit.

**What this release is.** B200 (sm_100a) fixes on top of 3.0.8: a NaN / dropped-key bug in the B200 TriangleAttention inference cores on real AF3 / Chai-1 targets, the B200 fp32 Transition with TF32 off on an IEEE fp32 CUDA chain, and `extra_pair_bias` of `TokenDiTProtenix` / `TokenDiTChai1` on the B200 composed runner. The B200 is **source / JIT only** (no B200 kernels wheel exists; it is not a supported release target). The H100, A100 and RTX 5090 are unchanged: no prebuilt artifact changed, the three kernels wheels are republished at 3.0.9 because the engine pins them `==3.0.9`.

## Changes (all `cuda:b200_sm100`)

1. **B200 TriangleAttention: clamp `ex2_poly2` at 128 (`3698ba9ac`).** The B200 inference cores fix a row's softmax offset from its first 32 keys and raise it only in a guard that runs after the tile's exponentials. The FMA-pipe exponential `ex2_poly2` (Cody-Waite integer part written into the exponent bits) had no upper clamp: a logit >= 2^129 above the row offset carried out of the exponent field, giving a tiny finite or NaN probability that the guard never saw (NaN / dropped keys). Scores on real AF3 / Chai-1 targets reach 160 log2 units (a 143-unit spike over the first 32 keys); N(0, 1) judge inputs never get there. Fixed in three kernels: `triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`. FoldBench targets with non-finite coordinates on the B200: AF3 21 of 27 and Chai-1 25 of 27 before, 0 after. Speed change below 1%. Regression test `tests/gpu/test_triattn_b200_exp_clamp.py`: 13 of 24 cases failed before, 24 pass.
2. **B200 Transition, fp32 with TF32 off, on the IEEE fp32 chain (`28fb99be9`, `f06154bc7`).** `kernels/ieee_f32` is built for sm_100a (artifact `ieee_f32_sm100`, in the sm100 prebuild plan); the training and inference fp32 TF32-off route no longer falls to Triton. Accuracy at the PyTorch fp32 level; cuda / Triton time ratio 0.27 - 0.45. Test `tests/gpu/test_transition_ieee_b200.py`.
3. **B200 `TokenDiTProtenix` / `TokenDiTChai1` `extra_pair_bias` served by the composed runner (inference).** bf16 rms vs float64 2.9 - 3.2e-3 (PyTorch 3.8 - 4.0e-3); cuda / Triton time ratio 0.22 - 0.31. Test `tests/gpu/test_token_dit_extra_bias_b200.py`.

## Versions (`backend_versions`)

Only these entries changed (all `b200_sm100`):

| class | backend | 3.0.8 | 3.0.9 |
|---|---|---|---|
| `_TriangleAttentionBase` | cuda:b200_sm100 | 74 | 75 |
| `TriangleAttention` | cuda:b200_sm100 | 76 | 77 |
| `Transition` | cuda:b200_sm100 | 66 | 67 |
| `TokenDiTProtenix` / `TokenDiTChai1` (token DiT) | cuda:b200_sm100 | 24 | 25 |

No module `version` changed; no `h100_sm90`, `a100_sm80` or `5090_sm120` entry changed. B200 judge tokens cannot be issued (`on_kit_stack` rejects b200 cuda records); `judge owed` for `b200_sm100` stays, h100 / a100 / 5090 owed 0 as in 3.0.8.

## Prebuilt artifacts

Unchanged for the three released arches. The prebuild plans (`_prebuild --check`, this tree) produce byte-identical index files to 3.0.8 (sm90 580, sm80 75, sm120 67 artifacts); manifest names against the 3.0.8 wheels: sm90 365 -> 365, sm80 75 -> 75, sm120 67 -> 67, dropped 0, added 0; the payload members of every kernels wheel have the same CRC and size as in 3.0.8 (the wheels differ only in the dist-info version). `ldd` of every staged and installed `.so`: 0 link a CUDA 13 library, 0 unresolved, no RPATH. The sm120 `pwa_gemm_sm120.so` is still the nvcc 12.8 override. `twine check`: 5 of 5 PASSED; the engine METADATA pins the three kernels wheels `==3.0.9`, no git URLs. No B200 wheel.

## Installed-wheel verification (extracted final wheels, empty JIT cache, no nvcc, no repo on the path)

| item | H100 | A100 (cssb3) | RTX 5090 (cssb-rtx) |
|---|---|---|---|
| import, versions | 3.0.9 | 3.0.9 / 3.0.9 | 3.0.9 / 3.0.9 |
| every indexed artifact loads, 0 builds | sm90 580 of 580 (164 + 190 + 226) | sm80 75 of 75 | sm120 67 of 67 |
| accuracy per module (pytorch / cuda / triton) | 12 modules, 36 rows, 616 G / 194 Y / 0 RED / 0 error; builds 0 | 12 modules, 36 rows, 610 G / 200 Y / 0 RED / 0 error; builds 0 | 13 modules (incl. TokenDiTChai1), 39 rows, 651 G / 201 Y / 0 RED / 0 error; builds 0 |
| `test_extra_bias_and_mixed_dtypes` | 12 passed, 6 skipped | 18 passed | 6 passed, 12 skipped |
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

The H100 accuracy counts differ from the 3.0.8 table (619 G / 191 Y) in `InputFeatureEmbedder` / `pytorch` only (13 G / 5 Y here, 16 / 2 in 3.0.8): the `pytorch` rows are the reference backend, and repeated runs of that one module on the 3.0.8 and 3.0.9 wheels give 18 / 0, 15 / 3 and so on, i.e. run-to-run noise of the reference rows (its `cuda` and `triton` rows are identical in every run). All other rows are equal to 3.0.8 on all three cards.

**B200 (source-tree check, not a wheel).** From the `rc-3.0.9` source tree (not the PyPI wheels), JIT cache of the sm100 artifacts, B200 GPU 0, torch 2.13.0+cu129: `test_triattn_b200_exp_clamp` 24 passed; `test_transition_ieee_b200` 6 passed; `test_token_dit_extra_bias_b200` 19 passed, 1 xfailed (fp32 S = 1, L = 100, see below).

**Known failing test (pre-existing, not caused by this release):** `tests/gpu/test_chai1_identity_adaln_sm80.py::test_identity_kernels_vs_fp64[True-dtype1-transition]` on the A100 (bf16 Transition tolerance against the Triton backend), as in 3.0.8; not re-run here.

## Known limits

- B200 fp32 inference at S = 1, L = 100 / 104: NaN in the composed runner (also without `extra_pair_bias`; xfail in the new test; bf16 unaffected). Not fixed.
- B200, multi-size CUDA-graph replay: a process that runs several growing sizes with the sampler graph cache produced non-finite output at some sizes (boltz2 n = 12 / 16, chai1 n = 8, rf3 n = 12). Unresolved; workaround: one process per size. Suspected interaction between the engine `capture` `refresh_static_inputs` and FoldForge's `SamplerGraphCache`.
- B200: source / JIT only (no kernels wheel); judge tokens cannot be issued; `judge owed` for b200 stays; not a supported release target.
- TriAttn fp16 score margin and Transition fp16 `h` margin: guards not implemented (H100 large-magnitude net carries the documented TriAttn xfail).
- Judge requirements rows still cover PWA L <= 2048 and token DiT L <= 768; long L is GPU-test-verified only.
- `extra_pair_bias` on the A100 / 5090 is inference only; `TokenDiTBoltz2` / `TokenDiTRoseTTAFold3` still refuse it (Triton).
- 5090 AF3 n >= 25 OOM and H100 n >= 37 / A100 n >= 38 limits as in 3.0.6 (not re-measured).
- `ruff` (141) / `pyright` (55) findings in the merged B200 code, not fixed.

## Release steps

NOT DONE: tag, PyPI upload, `pixi.lock`, push. Runbook: `~/miniworld-engine-scratch/release309/UPLOAD_RUNBOOK.md`.
