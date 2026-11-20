# miniworld-engine 3.0.6 (DRAFT, local candidate `rc-3.0.6`; not released)

Base: 3.0.5 (`bef4085c2`). Integration branch `rc-3.0.6` = 3.0.5 + `fix-tr-sm80-resid-overflow` + `pwa-long-h100` + `tokendit-long-h100` + `pwa-long-l` (A100 PWA) + `tokendit-bugs-h100` (tests only) + `pwa-long-l-5090` (5090 PWA, cherry-picked `7992df822`). The package version is NOT bumped in this candidate.

**What this release is.** A 3.0.5 repair (the 5090 / A100 Transition overflow that broke 3.0.4 / 3.0.5 AF3 on the 5090) plus long-sequence inference for two modules (MSA pair-weighted averaging, bias-only token DiT). No module `version` changed; only `backend_versions` entries.

## Changes

1. **Transition, bf16 one-kernel forward (sm_80 family: A100 and RTX 5090).** FoldForge AF3 5I28 n=1 on the 5090 (3.0.5) gave `-inf` coordinates ("non-finite AF3 coordinates"): the `msa_transition` input had |x| = 65536. Cause: the D 64 / 128 one-kernel forward (`fused_fwd_sm80`; `fused_sm80` on the A100) starts its f16 accumulator at the residual x, so any |x| >= 65504 (f16 max) overflowed to inf. The accumulator now carries (x + Ws h) / 16 (`TR_RESID_SCALE`; pack flag bit 30 scales the packed Ws) and the bf16 store multiplies back. Result stays finite up to |x| of about 1e6 (the limit of this fix; above it the f16 accumulator overflows again). Outputs are otherwise within 1 bf16 ulp of the previous version. Backward is unchanged. **3.0.4 and 3.0.5 on the 5090 produced broken AF3 results for such inputs.**
2. **MSAPairWeightedAveraging, inference past the old length limits (A100, H100, RTX 5090).** The kernels stage the bf16 logits in the output rows when the shared-memory row does not fit: `pair_fwd` (sm_80 family), `pair_fwd3` (H100: `pair3.cu` fp32-row path, `pair_wide.cu` bf16-row path). A100 and H100: L up to 8192 (A100 `MAX_LEN_INFER` 8192). RTX 5090: the limit is now the card's memory (L^2 (2 d_pair + 16) bytes <= total memory - 2 GiB, L a multiple of 16, cap 8192): d_pair 128 / 256 / 384 = 8192 / 7712 / 6336 on the 32 GB card, was 4080 / 1840 / 2672 (L 2048 at d_pair 256 is newly served). L within the previous limit is unchanged (5090: 15 cells at L <= the old limit are bitwise identical to 3.0.5). A call past what fits is refused cleanly (`BackendUnsupported` / a refusal reason), never an out-of-memory.
3. **Bias-only token DiT, inference past L 3072 (H100).** Two-pass softmax rows (bf16 rows and fp32 `rows32`) past the 3072 keys a warp holds; `pvx` / `pvx32` bound 3072 -> 8192; a call that cannot fit the device memory is refused (`BackendUnsupported`). L <= 3072 outputs are identical; the pair-bias logits and the softmax stay fp32 (user rule 2026-11-19).
4. **Bug fix (H100): `softmax_t` held 1024.** In 3.0.5, bf16 / master-weights bias-only token DiT inference on the H100 with 1024 < L <= 3072 (AF3 n = 9..24) raised `RuntimeError: softmax_t: L ... at most 1024`. The hoist now takes `softmax_rows` past L 1024 and the case is served. (`tokendit-bugs-h100` adds `tests/gpu/test_bias_only_mid_inference.py` for L 1152 .. 3072 on the H100 and 3200 / 4096 on A100 / 5090.)
5. **Token DiT bug investigation (`tokendit-bugs-h100`), results.** (a) The larger error of a "huge" input with master weights is not a defect: it is an artifact of the artificial weights in the AdaLN table fold; the pair-bias logits stay fp32. (b) In 3.0.5, n >= 25 (L > 3072) was unsupported on the H100 for every dtype, and on the A100 / RTX 5090 raised `softmax_rows: L ... at most 3072`; both now serve up to 8192. (c) On the A100 and RTX 5090, n = 9..24 was already fine in 3.0.5.

## Versions (`backend_versions`)

| class | card | 3.0.5 | 3.0.6 | cause |
|---|---|---|---|---|
| Transition | a100_sm80 | 7 | 8 | f16 residual scaling |
| Transition | 5090_sm120 | 9 | 10 | f16 residual scaling |
| MSAPairWeightedAveraging | h100_sm90 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | a100_sm80 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | 5090_sm120 | 3 | 4 | inference up to the card's memory |
| TokenDiTMiniWorldBiasOnly | h100_sm90 | 8 | 9 | inference L > 3072, softmax_t fix |

**Not versioned:** the A100 / RTX 5090 token DiT bias-only support for L > 3072 came from a change in a shared `.cu` source and has no `backend_versions` entry. The judge cells (the requirements stop at L <= 768 for this module) do not cover it; it is verified by GPU tests only (`test_bias_only_mid_inference.py` on A100 and 5090: L 3200 / 4096).

`judge owed` on the merged candidate: 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120` (carried 17815 per kind). The 5090 PWA tokens (392 records) were issued on the 5090 branch; no token had to be re-issued in the merged tree.

## Results

Reported by the branch agents (not re-measured in the integration step):

- PWA cuda vs triton, speed: A100 1.9-2.4x faster, H100 1.4-2.8x faster, RTX 5090 1.5-2x faster.
- RTX 5090 PWA accuracy: rms vs float64 1.72e-3 .. 1.74e-3.
- Bias-only token DiT cuda vs triton at long L (H100): about 5x faster.
- Accuracy: engine standard (rms / max vs float64), 0 RED in every re-issued token set.

Integration-step GPU tests: see "Verification".

## Known limits

- Transition fix: |x| up to about 1e6 (measured limit); beyond that the f16 accumulator overflows.
- PWA lengths are limited by pair-tensor memory (5090: 8192 / 7712 / 6336 at d_pair 128 / 256 / 384). The requirements rows stay at L <= 2048 (token DiT bias-only: L <= 768), so the judge cells do not cover long L; the GPU tests (`test_pwa_long_inference_sm80/sm90`, `test_bias_only_long_inference_sm90`, `test_bias_only_mid_inference`) do.
- Every known limit of 3.0.5 remains (LayerNorm exceptions, bias-only fp32 logits cost on the H100, OpenFold3 L128 fp32 / ParallelResidual L128 vs Triton on the 5090, B200 not validated, peptide-bond C-N end-to-end check not run): see `docs/RELEASE_NOTES_3.0.5.md` and `3.0.4`.

## Verification (integration step, rc-3.0.6)

| item | result |
|---|---|
| judge owed | 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120`, each evaluated in that card's own environment (H100 env, cssb3 A100 env, cssb-rtx 5090 env) |
| A100 (cssb3) `test_transition_sm80_resid_overflow`, `test_pwa_long_inference_sm80` | 31 passed, 3 skipped (5090-only refusal tests); with `test_bias_only_mid_inference` (L 3200 / 4096 and the pair path): 49 passed, 16 skipped in the earlier run |
| RTX 5090 (cssb-rtx) `test_transition_sm80_resid_overflow` | 4 passed |
| RTX 5090 `test_pwa_long_inference_sm80` | 25 passed, 5 skipped (A100-only gate tests): every width at L 2176 / 3072 / 4096, the card maximum ("max": 8192 / 7712 / 6336 at d_pair 128 / 256 / 384), BIG width 2176 / 3072, refusal past the limit |
| RTX 5090 `test_bias_only_mid_inference` | 18 passed (pair path 1152 .. 3072 and 3200 / 4096), H100-only cases skipped |
| H100 `test_bias_only_mid_inference`, `test_bias_only_long_inference_sm90`, `test_pwa_long_inference_sm90`, `test_transition_sm80_resid_overflow` in one process | 72 passed, 10 skipped (the Transition test and the A100 / 5090 cases skip on the H100). Before the test fix the long bias-only cases ran out of memory in one process: the cached bias-only runners kept the hoisted P and the (pair, mask) key tensors alive; the tests now clear `bias_only_dit._RUNNERS` between cases |
| H100 Transition spike check (`chk1.py`: D 64 / 128, n 2 / 4, master and bf16 params, rows 16384 / 65536, spikes 65536 and 1e6, eval and train) | nonfinite 0, Triton launches 0 in all 64 cases, relrms 5e-6 .. 8e-5 vs float64 (D 128 rows seen) |
| CPU gate (`rc-3.0.6`) | pyright 0 errors; ruff check `src` clean; ruff format: no new unformatted file versus 3.0.5 (`ruff format --check .` already reports 175 files on 3.0.5, 174 here); pytest cpu (judge, modules_cpu, requirements, prebuild_plan, supports_signature, imports, init, capture_api) 3231 passed, 2 skipped |
| Prebuild regeneration / installed-wheel verification | NOT DONE (release step) |
