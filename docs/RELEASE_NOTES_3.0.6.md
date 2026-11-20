# miniworld-engine 3.0.6 (DRAFT, local candidate `rc-3.0.6`; not released)

Base: 3.0.5 (`bef4085c2`). Integration branch `rc-3.0.6` = 3.0.5 + `fix-tr-sm80-resid-overflow` + `pwa-long-h100` + `tokendit-long-h100` + `pwa-long-l` (A100 PWA). Pending, not merged yet: `pwa-long-l-5090` (5090 PWA) and `tokendit-bugs-h100` (placeholders below). The package version is NOT bumped in this candidate.

**What this release is.** A 3.0.5 repair (the 5090 / A100 Transition overflow that broke 3.0.4 / 3.0.5 AF3 on the 5090) plus long-sequence inference for two modules (MSA pair-weighted averaging, bias-only token DiT). No module `version` changed; only `backend_versions` entries.

## Changes

1. **Transition, bf16 one-kernel forward (sm_80 family: A100 and RTX 5090).** FoldForge AF3 5I28 n=1 on the 5090 (3.0.5) gave `-inf` coordinates ("non-finite AF3 coordinates"): the `msa_transition` input had |x| = 65536. Cause: the D 64 / 128 one-kernel forward (`fused_fwd_sm80`; `fused_sm80` on the A100) starts its f16 accumulator at the residual x, so any |x| >= 65504 (f16 max) overflowed to inf. The accumulator now carries (x + Ws h) / 16 (`TR_RESID_SCALE`; pack flag bit 30 scales the packed Ws) and the bf16 store multiplies back. Result stays finite up to |x| of about 1e6 (the limit of this fix; above it the f16 accumulator overflows again). Outputs are otherwise within 1 bf16 ulp of the previous version. Backward is unchanged. **3.0.4 and 3.0.5 on the 5090 produced broken AF3 results for such inputs.**
2. **MSAPairWeightedAveraging, inference past L 2048 (A100 and H100; 5090: PENDING).** `pair_fwd` (A100) / `pair_fwd3` (H100: `pair3.cu` fp32-row path, `pair_wide.cu` bf16-row path) stage the bf16 logits in the output rows when the shared-memory row does not fit. L up to 8192 (`MAX_LEN_INFER` 8192); L <= 2048 is unchanged (outputs identical). A call past the memory that fits is refused cleanly (`BackendUnsupported`), never an out-of-memory. Lengths are limited in practice by the pair-tensor memory.
3. **Bias-only token DiT, inference past L 3072 (H100).** Two-pass softmax rows (bf16 rows and fp32 `rows32`) past the 3072 keys a warp holds; `pvx` / `pvx32` bound 3072 -> 8192; a call that cannot fit the device memory is refused (`BackendUnsupported`). L <= 3072 outputs are identical; the pair-bias logits and the softmax stay fp32 (user rule 2026-11-19).
4. **Bug fix: `softmax_t` held 1024 in the bf16 / master hoist.** In 3.0.5, bf16 / master-weights inference with 1024 < L <= 3072 raised; the hoist now takes `softmax_rows` past L 1024.
5. **PENDING:** 5090 PWA long L; `tokendit-bugs-h100` (placeholder: describe when merged).

## Versions (`backend_versions`)

| class | card | 3.0.5 | 3.0.6 | cause |
|---|---|---|---|---|
| Transition | a100_sm80 | 7 | 8 | f16 residual scaling |
| Transition | 5090_sm120 | 9 | 10 | f16 residual scaling |
| MSAPairWeightedAveraging | h100_sm90 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | a100_sm80 | 2 | 3 | inference L > 2048 |
| MSAPairWeightedAveraging | 5090_sm120 | 3 | PENDING | |
| TokenDiTMiniWorldBiasOnly | h100_sm90 | 8 | 9 | inference L > 3072, softmax_t fix |

`judge owed` on the merged candidate: 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120` (carried 17815 per kind). Tokens were re-issued by the branches (Transition 5090 + A100: 307 + 307 cells each, 0 RED; PWA and bias-only H100 and PWA A100 per their branches).

## Results

Reported by the branch agents (not re-measured in the integration step):

- PWA cuda vs triton, speed: A100 1.9-2.4x faster, H100 1.4-2.8x faster.
- Bias-only token DiT cuda vs triton at long L: about 5x faster.
- Accuracy: engine standard (rms / max vs float64), 0 RED in every re-issued token set.

Integration-step GPU tests (all passed): see "Verification".

## Known limits

- Transition fix: |x| up to about 1e6 (measured limit); beyond that the f16 accumulator overflows.
- PWA lengths are limited by pair-tensor memory; the requirements rows stay at L <= 2048, so the judge cells do not cover long L; the GPU tests (`test_pwa_long_inference_sm80/sm90`, `test_bias_only_long_inference_sm90`) do.
- PWA on the 5090: PENDING.
- Every known limit of 3.0.5 remains (LayerNorm exceptions, bias-only fp32 logits cost on the H100, OpenFold3 L128 fp32 / ParallelResidual L128 vs Triton on the 5090, B200 not validated, peptide-bond C-N end-to-end check not run): see `docs/RELEASE_NOTES_3.0.5.md` and `3.0.4`.

## Verification (integration step, rc-3.0.6)

| item | result |
|---|---|
| judge owed | 0 / 0 on `h100_sm90`, `a100_sm80`, `5090_sm120`, evaluated in each card's own environment (H100 env, cssb3 A100 env, cssb-rtx 5090 env) |
| A100 (cssb3) `test_transition_sm80_resid_overflow`, `test_pwa_long_inference_sm80` | 26 passed |
| RTX 5090 (cssb-rtx) `test_transition_sm80_resid_overflow` | 4 passed |
| H100 `test_pwa_long_inference_sm90` | 22 passed; Transition overflow test 4 skipped (sm_80 / sm_120 only) |
| H100 `test_bias_only_long_inference_sm90` | 17 passed and 4 failed in one process with CUDA out-of-memory in the test inputs (device memory accumulated across the parametrized cases); the 4 re-run in smaller selections all passed (8192 case, 4096 x 4, 6144-256 master and float32) |
| H100 Transition spike check (`chk1.py`: D 64 / 128, n 2 / 4, master and bf16 params, rows 16384 / 65536, spikes 65536 and 1e6, eval and train) | nonfinite 0, Triton launches 0 in all 64 cases, relrms 5e-6 .. 8e-5 vs float64 (D 128 rows seen) |
| CPU gate (`rc-3.0.6`) | pyright 0 errors; ruff check `src` clean; ruff format: no new unformatted file versus 3.0.5 (`src` + `tests` 78 vs 79 files; `sm80.py` of the A100 branch is unformatted in 3.0.5 as well); pytest cpu (judge, modules_cpu, requirements, prebuild_plan, supports_signature, imports, init, capture_api) 3231 passed, 2 skipped |
| Prebuild regeneration / installed-wheel verification | NOT DONE (release step) |
