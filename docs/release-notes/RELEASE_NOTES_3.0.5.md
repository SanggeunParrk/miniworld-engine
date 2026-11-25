# miniworld-engine 3.0.5

Base: 3.0.4 (`8863ed764` = `pixi.lock` commit; tag `v3.0.4` -> `805b80be7`). Integration branch `rc/3.0.5` = 3.0.4 + `r305/5090-trans-opm` (`2641797da`) + `r305/5090-trans-opm-tokens` (`183e7b0aa`) + the version bump.

**What this release is.** A small RTX 5090 follow-up to 3.0.4. Only two modules change, only on the 5090 (`cuda:5090_sm120`); the H100 sources and every H100 artifact are unchanged (the sm90 plan builds 0 new artifacts), and no module `version` changed.

## Changes

1. **Transition, fp32 inputs (5090).** The squeeze GEMM accumulates in f16 with 32 rows per warp and two half tiles (`tr_fwd_sm80.cuh`, `transition_fwd_sm80.cu`).
2. **OuterProductMean, masked epilogue (5090).** The masked epilogue (`opm_epilogue_mask_sm80.cuh`) runs on CTAs of 64 pairs.

Both changes are in headers shared with the sm_80 sources, so the A100 (sm80) artifact keys changed too (3 sm80 and 3 sm120 artifacts rebuilt); the A100 backend versions are unchanged and A100 outputs are bitwise identical to 3.0.4 (see "Verification").

## Versions (`backend_versions`)

| class | card | 3.0.4 | 3.0.5 | cause |
|---|---|---|---|---|
| Transition | 5090_sm120 | 8 | 9 | f16-accumulation squeeze for fp32 |
| OuterProductMean | 5090_sm120 | 5 | 6 | 64-pair masked epilogue CTAs |

No other entry changed. `judge owed` is 0 on `h100_sm90`, `a100_sm80` and `5090_sm120` (the 5090 Transition and OuterProductMean tokens were re-measured; tokens are not packaged in the wheels).

## Results (RTX 5090 tokens, cuda median / triton median, speed records, 3.0.4 -> 3.0.5)

| module, dtype, mode | cells | cells <= 1.00 | worst cell |
|---|---|---|---|
| Transition fp32 eval | 42 | 42 -> 42 | 0.773 -> 0.774 |
| Transition fp32 train | 6 | 6 -> 6 | 0.717 -> 0.708 |
| Transition master (bf16) eval | 137 | 126 -> **128** | 1.068 -> **1.051** |
| OuterProductMean master eval | 126 | 120 -> **121** | 1.114 -> **1.076** |
| OuterProductMean master train | 36 | 36 -> 36 | 0.996 -> 0.996 |

(Cells that have both a cuda and a triton record. Transition master train: 15 -> 16 of 30 cells <= 1.00 with a pre-existing outlier row in the Triton tokens (row 218, L768) in both versions; not touched by this release.)

## Known limits

- 5090 OpenFold3 L128 fp32 eval is 1.027x and 5090 ParallelResidual L128 is 1.03-1.045x of Triton.
- Every known limit of 3.0.4 remains (exact fp32 logits cost on the H100 bias-only token DiT, 5090 Transition / OPM residual cells, LayerNorm exceptions, FoldForge's `choose_backend` allows cuda only on the H100, the peptide-bond C-N end-to-end check not run, B200 not validated): see `docs/RELEASE_NOTES_3.0.4.md`.

## Prebuild regeneration (release build)

Incremental from the 3.0.4 release cache (GCC 12.4, pixi CUDA 12.9, torch 2.13+cu129): sm90a 579 artifacts (0 built), sm80 74 (3 built), sm120a 67 (3 built). Carried names as in 3.0.4 (8 files).

## Verification

Installed-wheel runs use the release artifacts extracted without the repo on the path, an empty JIT cache, no CUTLASS / MathDx and no `nvcc` on `PATH`.

| item | result |
|---|---|
| sha256 of the five artifacts | `dist/SHA256SUMS`, checked before every run |
| ldd of every `.so` (staged trees; installed wheels 378 files) | 0 link CUDA 13, 0 unresolved (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` only) |
| manifest names vs the 3.0.4 wheels | sm90 364 -> 364, sm80 74 -> 74, sm120 67 -> 67; **0 dropped, 0 added** |
| sm90 kernels wheel size | 101,674,305 B (PyPI limit 104,857,600 B) |
| H100 installed load of every indexed sm90 artifact (579 = 163 + 190 + 226) | 579 resolved, **0 compiled** |
| H100 installed accuracy (12 modules, pytorch / cuda / triton) | RED 0, error 0 (cuda 172 GREEN / 98 YELLOW), 0 builds |
| H100 `test_restored_triton_backends.py` / `test_bias_only_fp32_logits.py` | 63 passed / 66 passed, 8 skipped; 0 kernels built |
| H100 cuda eval outputs, installed 3.0.4 vs 3.0.5 (199 registry cases) | **186 bitwise identical, 0 different**, 5 cases unsupported in both |
| A100 installed (cssb3) | 74/74 resolved, 0 compiled; accuracy RED 0, error 0 (cuda 181 GREEN / 103 YELLOW); restored Triton + sm80 bias-only fp32 tests 69 passed; 0 kernels built |
| A100 cuda eval outputs, installed 3.0.4 vs 3.0.5 (86 cases: Transition, OPM, ParallelResidual, AttentionPairBias, TriangleAttention, TriangleMultiplication) | **86 bitwise identical, 0 different** |
| RTX 5090 installed | 67/67 resolved, 0 compiled; accuracy RED 0, error 0 (cuda 188 GREEN / 96 YELLOW); restored Triton + sm80 tests 65 passed, 4 skipped; 0 kernels built |
| RTX 5090 real Protenix v1 step (48 blocks, chunk 8, step graph, installed wheels) | 2.0 s / step (step 0 loss / gnorm 0.3764 / 1.242), engine modules on Triton 0 (cuda 9021 calls, pytorch 774 = PairformerBlock wrapper), no segfault, 0 kernels built, peak 16.93 / 25.34 GiB |
| RTX 5090 max Pairformer blocks, L 384 | **31** (unchanged from 3.0.4) |
| CPU gate (`rc/3.0.5` = c12c3228d) | ruff check clean, pyright 0 errors, pytest cpu 3414 passed, 2 skipped |
| judge owed | 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120` (carried 17815 per kind) |
