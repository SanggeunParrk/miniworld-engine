# miniworld-engine 3.0.2

Base: 3.0.1 (`f5be1df46`, tag `v3.0.1` -> `a883805b6`). Integration tree `fp8c/all` (13 FP8 connection branches merged and verified together, source tip `7123ccb7d`),
released as `rel/3.0.2`. Everything new is **off by default**: the default bf16 / fp16 routes are bit-identical to 3.0.1 (see "Verification"). FP8 stays the opt-in backend
`engine-fp8` (`set_backend("engine-fp8")` + `fp8.set_model(...)`, or `MINIWORLD_FP8=safe` with `MINIWORLD_FP8_MODEL=<model>`); it remains outside the judge and the registry.
No module `version` and no `backend_versions` entry changed, so the judge owes nothing (verified below).

## A bug fix that also touches 3.0.0 and 3.0.1: e4m3 NaN byte in the TriAttn 16 / 32-channel FP8 PV core

`triattn_h16_sm90.cu` built the e4m3 probability byte as `round(8 x + 56)` with `x = logit - m + 8` and no upper bound. `x` stays below about 8.5 + a few fp16 roundings,
which is safe while |max logit| is below about 128. At logits in the hundreds (q.k logit std of 300 and up, or pair biases of about 3000 and up; 4 to 32 channels,
L 256 / 512) the byte reached 127 (the e4m3 NaN) or 128+ (a negative probability) and the FP8 PV core returned non-finite outputs. This affected only the opt-in FP8 TriAttn
route (`MINIWORLD_FP8=safe` with TriAttn enabled for the model, or `engine-fp8`) in 3.0.0 and 3.0.1; the default fp16 core was never affected. Normal AF3-style inputs (|offset| < 128)
never reach the edge, which is why the end-level runs of 3.0.0 / 3.0.1 were clean.

Fix (`TA_CLAMPV`, default 3): the same clamp as the 64-channel core (x <= 8.75, byte <= 126), with no extra instruction (the lower clamp is the `.relu` of the first HFMA2, the upper
one a `min`). Same-input comparison with the unclamped core: rms difference 0.55 % of the output (about 5 % of the bytes move by one e4m3 step; **not byte-identical on normal FP8
inputs**, accepted under the accuracy standard); error against fp64 unchanged (3.63 / 3.61 / 3.51 / 3.56 % before and after); module time 1.000 - 1.017x of the unclamped core.
Regression test: `tests/gpu/test_triattn_h16_fp8_extreme.py` (39 cases, all finite and within the error bound).

## What is new (FP8 inference, opt-in)

| area | change |
|---|---|
| `fp8.route_stats()` | per-route call and fallback counters (`why_*` reasons); bumped at Python / capture time (a CUDA-graph replay does not count) |
| TriAttn whitelist (`TRIATTN_MODELS`) | af3, protenix1 (fused + q/k), opendde (unfused), protenix2 (unfused), openfold3 (fused + unfused + q/k), rosettafold3 (fused + q/k), chai1 (d64 + unfused) |
| Chai-1 | token DiT identity block (expand / out / squeeze with the fused `adaln_in_rows_cf_par_q8` pass), TriAttn d64 PV kind (`triattn_h100_d64_f8pv`, gate `FP8_PV_D64_MIN_S` = 1024 tokens), grouped OPM e4m3 kernel `opm_chai_f8` (**built, OFF by default**: net loss at every S) |
| Boltz-2 | up-gate token DiT and atom DiT FP8 routes (`MIN_ROWS_UP`) |
| IntelliFold2 | D512 TriMul planes (`_f8` K1 + K3_CSCALE), PWA BIG 256 x 8 x 32 (`miniworld_pwa_vt8_256_32`) |
| RoseTTAFold3 | TriMul calibration-key fix (the route fell back to bf16 inside graph capture), OPM biased route, token DiT expand, atom DiT KQLN |
| ESMFold2 / OpenDDE | atom DiT RMS route (`post90_f8up`, pre / post f8 KQLN / RMS / UP row kernels) |

## 5I28 warm forward, FP8 / bf16 (H100, FoldForge harness, real MSA + 4 distinct templates, warm median of 6, `MINIWORLD_FP8=safe`)

Seconds per forward at n = 16; the ratio columns are lower-is-faster.

| model | bf16 3.0.1 | FP8 3.0.1 | FP8 3.0.2 | 3.0.2 / bf16 | 3.0.2 / FP8 3.0.1 |
|---|---|---|---|---|---|
| af3 | 16.969 | 13.194 | 13.174 | 0.776 | 0.998 |
| openfold3 | 16.940 | 15.705 | 13.172 | 0.778 | 0.839 |
| rosettafold3 | 17.108 | 17.806 | 13.635 | 0.797 | 0.766 |
| protenix1 | 16.291 | 13.516 | 13.048 | 0.801 | 0.965 |
| protenix2 | 30.965 | 28.763 | 25.854 | 0.835 | 0.899 |
| esmfold2 | 13.342 | 10.755 | 10.748 | 0.806 | 0.999 |
| opendde | 63.193 | 53.570 | 53.284 | 0.843 | 0.995 |
| esmfold2-fast | 8.609 | 7.317 | 7.300 | 0.848 | 0.998 |
| chai1 | 29.322 | 27.873 | 25.851 | 0.882 | 0.927 |
| intellifold2 | 71.742 | 67.039 | 63.783 | 0.889 | 0.951 |
| boltz2 | 21.584 | 20.268 | 19.968 | 0.925 | 0.985 |

(RoseTTAFold3 under FP8 3.0.1 was slower than bf16, 1.041; the calibration-key fix is why it now wins.) n = 1 .. 12 and the per-route counters (`fp8.route_stats()` census) are in
`docs/fp8.md`, section "Combined tree `fp8c/all`".

## Not included in 3.0.2

- FP8 **training** routes (branches `fp8t3/*`, `fp8t4/*`): opt-in experiments with no prebuilt kernels.
- TriAttn whitelist lines for Boltz-2 and IntelliFold2 (owner decision pending; both keep the fp16 TriAttn core under FP8).
- Chai-1 grouped OPM e4m3 kernel: shipped (prebuilt) but OFF by default.
- The bit-identical byte savings `nox` / `hsave` of the token DiT training path.

## Verification (rel/3.0.2, H100, Slurm)

See the "Verification" block at the end of this file (filled from the release jobs).

## Prebuild regeneration (release build)

Kernels wheels (sm90 / sm80 / sm120, cu129, cp312) rebuilt incrementally from the 3.0.1 release cache with GCC 12.4 (CUTLASS 4.2, MathDx): sm90a 29 built, 574 artifacts, 0
absent (3.0.1: 559). The `triattn_h16` source change re-keys the 4 h16 entries (including the default fp16 h16 core, whose output is bitwise unchanged with FP8 off); the FP8
extension sources changed (`token_dit_fp8_sm90`, `triattn_h100_d64`, the f8 pre / post row cubins, the b200style TriMul K1 / K3 cubins). New names (additions only, sm90):
`triattn_h100_d64_f8pv`, `opm_chai_f8`, `atom_sm90/post90_f8up_bf16_w4m1`, `miniworld_pwa_vt8_256_32`. The 4 names carried byte-for-byte in 3.0.1 (`triattn_h100_core_h16_v2`,
`miniworld_pwa_dgv_bwd`, `_glue3`, `_pair3`) are carried again. **No name of 3.0.1 is dropped**: manifest names 0 dropped / 4 added (sm90), 0 / 0 (sm80, sm120); wheel
families 0 dropped / 5 added. The sm80 and sm120 payloads are unchanged (every path and CRC identical to 3.0.1), only the version differs.

## Verification block

All on Slurm (H100 node, 1 GPU, 10-minute jobs); installed-wheel runs use the five release artifacts extracted without the repo on the path, an empty JIT cache, no CUTLASS / MathDx
and no `nvcc` on `PATH`.

| item | result |
|---|---|
| manifest names against the 3.0.1 upload | sm90 0 dropped / 4 added, sm80 0 / 0, sm120 0 / 0; wheel-level families 0 dropped / 5 added; sm80 and sm120 payloads identical to 3.0.1 (path + CRC) |
| installed-wheel load, every indexed sm90 artifact (574 = 158 + 190 + 226 in three parts) | 574 present in the packaged root, 574 resolved by the runtime loaders, **0 compiled**; also with the system libstdc++ preloaded |
| re-keyed routes (`triattn_h100_d64`, `triattn_h100_core_h16_v3` incl. the dh16 core, TriMul single-direction tables) and the restored Triton backends, installed | 11 + 2 + 113 passed; 63 passed; 0 builds |
| FP8 off, default route, installed 3.0.1 vs installed 3.0.2 (separate processes, each twice) | bitwise identical (`torch.equal`): 4 token DiT classes (L 384 / 640), PWA d128, OPM plain / biased, TriMul D128 / 256 / 512 (default / rf3), TriAttn d128 / d64, five atom DiT classes (1024 / 2560 atoms), and the default fp16 h16 TriAttn core (dh 16 / 32, logit spreads 1 / 2.2 / 8): run-to-run and across releases (the PWA 256 x 8 x 32 case is not served by the cuda backend with FP8 off in either release) |
| FP8 CPU tests and every FP8 GPU test file, installed wheels, one pytest call per file | 17 of 23 files pass in full (`test_triattn_h16_fp8_extreme.py` 39 / 39); five GPU files fail only on cases that request JIT builds of non-prebuilt ablation variants (Transition cfg1-3, wide, d256; TriMul K1 direct, k3-pipe) and cannot build without a toolchain: 39 failing cases, **identical to the installed 3.0.1 run of the same files** (no sources of those routes changed); the same files pass in the repo source tree. `tests/cpu/test_capture_api.py` is a repo-layout test, run in the repo tree as part of the CPU gate |
| judge owed (`judge owed`, versions only) | `h100_sm90` 0 / 0; `a100_sm80` and `5090_sm120` 96 / 96 (the new `PairformerBlock` chai1 rows of 3.0.1, no token yet: the same count on the 3.0.1 tree); 3.0.2 adds none |
| CPU gate (`tools/check.sh cpu`) | pyright 0 errors; tests/cpu 3394 passed, 2 skipped; ruff check clean after dropping one unused `noqa`; `ruff format --check` reports 4 files that are not mass-formatted (not enforced by the gate) |
