# miniworld-engine 3.0.10 (DRAFT, not released)

Base: 3.0.9 (tag `v3.0.9`). Candidate tree: branch `5090-int` = every RTX 5090 branch of 2026-11-23 / 24 (`boltz2-5090`, `fp8-5090`, `fp8-5090-sw`, `fp8-5090-msa`, `fp8-5090-dit`, `fp8-5090-chai`) merged with `origin/main` `6366afec9` (109 commits after `v3.0.9` by the H100 / A100 / B200 sessions), plus the release-prep fixes listed below. NOT DONE: version bump, tag, `pixi.lock`, push, PyPI. This draft covers the RTX 5090 part; the H100 / A100 / B200 commits of `origin/main` since 3.0.9 are summarised under "Also in this tree" for their owners to complete.

**What this release is (RTX 5090 part).** (1) Two default-path speed fixes on the 5090 token DiT (Boltz-2 up-gated transition on the sm_80 transition kernels; the step-invariant token pair-bias hoist for capability 12.0): FoldForge F3 5I28 every cell 1.06-1.63x of 3.0.6, every comparable cell faster than the kits. (2) FP8 on the RTX 5090: an sm_120a e4m3 route for every H100 FP8 slot (TriMul, TriAttn incl. Chai-1 d64, OPM, PWA, token DiT, atom DiT, Transition), opt-in through the same switch as the H100 (`MINIWORLD_FP8=safe` / `backend="engine-fp8"`); on the 5090 the `safe` preset leaves the e4m3 Transition out (explicit opt-in only). With FP8 off the default route is byte-identical except where listed.

## Changes (RTX 5090, `cuda:5090_sm120`)

1. **Boltz-2 token DiT: the up-gated transition on the sm_80 transition kernels at inference (`a77511b5a`).** Was the module's own torch code on sm_120 (fp32 master -> bf16 weight casts per call + split-K GEMMs, ~0.55 s per FoldForge forward at n = 1). Now `conditioned_transition_sm80` `inference_up` (AdaLN rows, one `[Wa; Wb; Wu]` GEMM, `swiglu_up_fwd`, the squeeze / gate). Test `tests/gpu/test_boltz2_upgate_transition_sm120.py`.
2. **Token DiT pair-bias hoist on capability 12.0 (`aa52c4f1a`).** The A100's step-invariant pair-bias hoist (`augattn_sm80` `TOKEN_BIAS`: one pair-bias pass per (pair, weights, key mask) instead of per block and sampler step) is enabled for (12, 0). Test `tests/gpu/test_token_pair_bias_hoist_sm80.py`.
3. **FP8 routes on sm_120a (opt-in; default bytes unchanged).** `fp8.SM120_ROUTES` puts a 5090 module into every slot; full design, gates, error tables vs fp64 next to the H100 routes and A-B tables in `docs/fp8.md` "RTX 5090":

| slot | 5090 route | gate (same-process A-B, 6 rounds) | speed (module, FP8 / default) | update error vs fp64 (H100 route) |
|---|---|---|---|---|
| TriMul | e4m3 planes (k1w8) + cuBLASLt e4m3 contraction, D64 / 128 / 256 / 384 / 512, RF3 calibration on the parameter | L >= 256 (D64 L >= 512) | 1.08 (L256) .. 1.40 (L2048) | 3.80-3.82 % (about 4 %) |
| TriAttn d_pair-128 | e4m3 P / V + q / k core | L >= 640, L % 128 | 1.03-1.19 | see docs/fp8.md |
| TriAttn Chai-1 d64 | e4m3 P / V core, gate fused | L >= 256 | 1.03-1.14 | 3.0-3.1 % (3.3-3.7 %) |
| Transition (NOT in `safe`) | one-kernel e4m3 D128 / 256; wide chain D384 / 512 | >= 12288 rows (wide >= 4096) | 1.40-2.16 | 6.6 % (6.6 %) |
| OPM | e4m3 grouped product | depth >= 1024 | see docs | see docs |
| PWA | e4m3 key contraction (native, BIG) | S N >= 2^15 (BIG 2^14) | see docs | see docs |
| token DiT | cuBLASLt row-wise e4m3 expand (+ opt-in squeeze / qkvg / out) | 640 rows | see docs | see docs |
| atom DiT | e4m3 pre / post GEMMs (incl. ESMFold2 RMS block) | 4096 rows (ESM 8192) | ESM 1.17-1.49 | ESM 3.6 % (3.8 %) |

   End level (H100 rule: same-seed bf16-vs-FP8 RMSD max <= max(1.1 x bf16 trunk-seed pair max, 0.5 A); FoldForge, 9 predictors x 4 targets x 4 trunk seeds): `safe` without the Transition **33 / 36** inside (with the e4m3 Transition 24 / 36: the H100 F11 effect; owner decision: Transition out of the 5090 `safe`). Chai-1 with the d64 route: PENDING (see "Chai-1 re-check").
   F3 5I28 engine-fp8 / bf16 (official = `safe` without the Transition, 3 reps): every cell faster, **0.79-0.97** of bf16. Chai-1 with d64: PENDING.
4. **Release-prep fixes on `5090-int`:** `tests/cpu/test_capture_api.py` lists the sm120 FP8 route dicts (`2b1ffa630`); the sm_80 (A100) build of `trimul_sm80_wide` / `_wide512` failed (`cvt .e4m3x2` of the 5090-only `k1w8` needs sm_89+): `k1w8` and its binding are now under `MWE_SM120` (PENDING commit / SASS check).

## Versions (`backend_versions`)

From the 5090 branches (only `cuda:5090_sm120`): `TokenDiTProtenix` 3 -> 4, `TokenDiTMiniWorld` 3 -> 4, `TokenDiTBoltz2` 2 -> 4, `TokenDiTRoseTTAFold3` 2 -> 3, `TokenDiTChai1` 4 -> 5. No module `version` changed; the FP8 routes are opt-in and change no entry. Judge (`owed`, this tree): PENDING table.

## Prebuilt artifacts

PENDING (sm120 plan, re-keyed artifacts per card, compiled = 0 check).

## Byte identity of the default path vs 3.0.9 (RTX 5090)

PENDING.

## Known limits

- The e4m3 Transition is not in the 5090 `safe` preset (end level 24 / 36 with it); the H100 `safe` still contains its Transition (same F11 finding; H100 owner decision).
- TriAttn PV-only kinds (opendde, protenix2, chai1 / af3 / openfold3 templates) are off on the 5090 (no gain measured; opt-in `MINIWORLD_TRIATTN_SM120_FP8_PVONLY=1`); Chai-1 d64 e4m3 q / k is opt-in (`MINIWORLD_TRIATTN_SM120_FP8_D64_QK=1`, error 4.2-4.4 % above the H100 d64 level).
- IntelliFold2 TriAttn not whitelisted (as on the H100).
- 5090 FP8 end level has no H100 twin on the same 4 targets with the whole `safe` set.
- 5090 OOM cells (32 GB) as in 3.0.6; peak memory +2-15 % from the cached bf16 weight copies and the hoisted pair bias.
- PWA L2048 D1024 eval unsupported on the 5090 (L <= 1840), as before.
- IntelliFold2 kit refuses sm_120; ESMFold2 F3 re-measure not feasible on the 5090 (16 GB host RAM cap).

## Also in this tree (other sessions, since 3.0.9; for their owners' notes)

B200: FP8 Transition D128 row window, atom-local slid-window tail on the tensor cores, `record_stream` in captures, Chai-1 FP8 blocked, judge `on_kit_stack` fix (b200 owed 0). H100: FP8 token / atom DiT host work and capture fixes, PWA FP8 capture fix, LayerNorm lean forward, PWA BIG, OPM training. A100: token DiT conditioning pack, pair-bias hoist (capability 8.0), ESMFold2 atom block. RTX 5090 from `origin/main`: LayerNorm sm80 / sm120 backward (`cuda:5090_sm120` 4), Transition sm80 training (11), bias-only TriangleAttention CUTLASS contractions (6) / Bidirectional (2), TriangleAttentionAF3 ending-node training on the wide engine (6), ParallelResidual PR_UNROLL (3), SmoothLddtLoss exact kink sign (2); tokens issued by those sessions.
