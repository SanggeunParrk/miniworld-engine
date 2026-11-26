# B200 FP8 (e4m3): consolidated state of the combined tree (n14 integration, 2026-11-19)

This section is the final state after merging agents D (TriAttn), E (atom DiT + token DiT remainder), F (Transition D128 / D256 / wide) and G (OPM / PWA / TriMul remainder) on top of the
3.0.3 + B200 base (branch b200/3.0.3). The per-agent sections below keep the full measurements; where an earlier sentence says "declined" for Transition D128 / D256 it is SUPERSEDED by section 8
of the Transition part (agent F built the fused e4m3 forward). Everything is opt-in, default OFF (bf16 default routes are bit-identical with FP8 off, see Verification).
Switch: `FOLDJET_ENGINE_FP8=safe` (or `fp8.enable("safe", model=...)`, backend `engine-fp8`); every route also has its own variable, an explicitly set variable wins over the preset.

## Served routes (capability (10, 0); ratio = e4m3 / bf16 default route, module level, graph replay; < 1 is faster)

| slot | route (module) | served when (gates) | time ratio (gap / sustained) | energy | accuracy vs default (max rel diff, mixed workload run) | switch / registration |
|---|---|---|---|---|---|---|
| trimul | `trimul_inproj.cuda.b200_fp8` planes e4m3 | D 64-512, L >= MIN_L ((128,out/in) 512, (128,bidir) 384 [G, quiet GPU], (384,out/in) 256, (384,bidir) 128 [G], D256 out/in 384 / bidir 256, D512 128, D64 out/in 512 / bidir 384); RF3 (D <= 128, B 1) L >= `MIN_L_RF3` {64: 1280, 128: 1024} [G]; L/ceil128(L) >= 0.9 | D128 out L2048 0.85 / 0.85; L1024 0.94 / 0.92; bidir L2048 0.84 | 0.80-0.90 | 0.025-0.027 | `FOLDJET_ENGINE_TRIMUL_FP8=calib`; `B200_ROUTES["trimul"]` |
| triattn | `triangle_attention.cuda.b200_fp8` e4m3 P and V (tcgen05 f8f6f4) | head dim 32 sites (fused d_pair 128; unfused 64 / 256 / 384), padded S >= 1024 (`FOLDJET_ENGINE_TRIATTN_B200_FP8_MINS`), HB build, model in `TRIATTN_MODELS`; 64-channel core built, gate closed; e4m3 q / k (front-written rows) from S >= 1024 for the models whose whitelist lists qk (D.7) | L1024 1.013 / 0.973; L2048 0.957 / 0.908 | 0.97 (L1024), 0.90 (L2048) | 0.005 | `enable_triattn` via `FOLDJET_ENGINE_FP8_MODEL` (h100_sm90 switches); `B200_ROUTES["triattn"]` (registered in this integration) |
| transition | `transition.cuda.fp8_sm100a` (chain 3 launches; fused one-kernel forward for D128 and D256 n4 >= 12288 rows) | `ROW_GATE` {(128,4) 147456 (< 1179648), (128,2) 65536 (< 1474560) [2026-11-23 re-measure, see `fp8_sm100a.py`], (256,4) 1024, (256,2) 1024 (< 8192), (512,4) 1024, (512,2) 4096, (768,4) 4096, (768,2) 4096} [F re-measured D512 / D768 on a quiet GPU]; `WIDTHS` default 128,256,512,768 (F); D384 / D64 / D1024 unserved | D512 n4 65536 rows 0.78; D768 n4 0.63; D128 n4 65536 0.97-0.99 gap, 0.85-0.95 sustained | D128 0.79-0.86; D512/768 0.5-0.8 | 0.033-0.040 | `FOLDJET_ENGINE_TRANSITION_FP8=calib` (+ `_WIDTHS`, `_FUSED`, `_PAIRB`); `B200_ROUTES["transition"]` |
| token_dit | `conditioned_transition.cuda.token_dit_fp8(_b200)` e4m3 expand + SwiGLU (plain, Boltz-2 up gate [E], RF3 parallel non-identity [E]) | placement `expand`; rows >= gate (Protenix / Chai-1 plain, Boltz-2 up `MIN_ROWS_B200_UP` 1280, RF3 `MIN_ROWS_B200_RF3` 2560); qkvg / out / squeeze not served (measured 0.99-1.05) | block 0.94-0.97 at M 2560; Boltz-2 0.89-0.96 from 1280; RF3 0.95-0.99 from 2560 | 0.91-0.96 | 0.031-0.047 | `FOLDJET_ENGINE_TOKEN_DIT_FP8=expand`; `_B200_UP`, `_B200_RF3` default on; weight forms are cached only under `static_weights()` (see caveat) |
| opm | `integrations.opm_fp8_b200` e4m3 product (wide prologue, fused S <= 512 [G], Chai-1 grouped [G]) | `_gate`: padded S gate / wide builds from padded S 256; fused padded S >= 256; Chai-1 (N >= 384 and S >= 1024) or (N >= 256 and S >= 2048) or (N >= 768 and S >= 512) | 64/128 L384 S2048 0.635; 128/256 0.66; fused S256-512 0.78-0.95; Chai-1 0.62-0.90 | ~0.6-0.9 | 0.010 | `FOLDJET_ENGINE_OPM_FP8=bound` (H100-named module, B200 branch inside `opm_sm100`) |
| pwa | `integrations.pwa_fp8_b200` e4m3 contraction (narrow, wide + BIG [G], dual [G], `pwa8_forward` one launch [G]) | padded N^2 S >= 2.4e9 (8x8), 1.1e9 (8x16), 6e8 (8x32 / BIG) | 8x32 N1536 S512 0.79 (d_pair 256 N1536: 0.73) | 0.8-0.9 | 0.0055 | `FOLDJET_ENGINE_PWA_FP8=1` |
| atom_dit | `augmented_attention.cuda.b200_atom.f8` rows route (pre + post) | rows <= 10240 (2048 atoms x 5) | 0.93-0.99 at 1024 atoms x 5, 1.00 at 2048 x 5, 1.02-1.06 above | -4 .. -9 % per block | update error 5.5 % rms (6x the bf16 route; outside the judge) | NOT in `safe` (effect on the whole model <= 0.3 %, E's decision): `FOLDJET_ENGINE_ATOMDIT_FP8_B200` (1 / all / pre / post), `fp8.B200_OPTIN["atom_dit"]` |

Framework (`src/foldjet_engine/fp8.py`, this integration): `B200_ROUTES` = trimul, triattn, transition (slot replacement: `_modules()` returns the B200 module on capability (10, 0)); `B200_OPTIN` = atom_dit (opt-in, own
variable, not in a preset; `serves_sm100("atom_dit")` True; `enable()` / `disable()` / `status()` / `route_stats()["atom_dit"]` follow `b200_atom.f8` on a B200: `ENABLED` stages and `{pre, post, pre_declined, post_declined}`);
OPM / PWA / token DiT keep their H100-named modules in the slot (their B200 forms sit behind the same switches). The TriAttn switches stay on `h100_sm90` (`_snapshot` / `_restore` / `enable_triattn` write there; the B200 module delegates reads).
`check()` covers trimul, opm, transition (peak) and pwa / triattn (flag) through captured graphs.

Caveats found in the integration: (1) token DiT e4m3 weight forms are rebuilt per replay when a graph is captured WITHOUT `_capture.static_weights()` / `static_inputs()` (Protenix M2560: 1.74x slower than default instead of 0.97x); the sampler's graphs use both, a hand-rolled capture must too.
(2) `status()["triattn"]["qk"]` is False on a B200 (the e4m3 q / k of the H100 route is not built there) although the whitelist lists qk for some models. (3) the measure harness sets `allow_tf32` process-wide; `tests/conftest.py` now restores the flags after every test (an atom DiT FP8 test file run before `test_misc_b200.py` made the sampler fp32 comparison fail).

## Open items (merged)
* End-level (structure) accuracy of every B200 route has NOT been run (module-level accuracy only); the H100 end-level evidence does not cover e4m3 on tcgen05 (different accumulation / scales). `TRIATTN_MODELS` unchanged.
* Atom DiT rows route: effect on a whole model <= 0.3 %, error 6x the bf16 route: kept out of `safe`; squeeze (Ws) not built; training not served.
* Transition D384 / D64 / D1024 not served; D128 n2 below 16384 rows and D256 n2 >= 8192 rows run the default; `_PAIRB` chain variant not default.
* TriAttn: e4m3 q / k and the 64-channel-head core not served (gate closed); ending-node and small-L cells are neutral to slightly slower (below the gate).
* TriMul: gate-projection-only front for L <= 256 not built; RF3 limited to D <= 128, B 1.
* OPM: O as e4m3 not built (bound 4-12 % of the product); Chai-1 below its (N, S) rule loses. PWA: head dim 8 not served; above per-head N^2 S C 1.5e10 two-stream per-head GEMMs.
* Token DiT: qkvg / out / squeeze e4m3 forms measured 0.99-1.05 (not served); weight forms need `static_weights()` captures.
* backend_versions not bumped (default routes bit-identical with FP8 off).

## End-to-end measurement on engine main (2026-11-23)

5I28, FoldJet F3 protocol (real MSA + 4 templates, 4 trunk passes, 200 steps, 5 samples, bf16, compile + CUDA graphs, steady median), real
weights, engine main 1ac962f49 / a553001e1, `FOLDJET_ENGINE_FP8=safe` + `FOLDJET_ENGINE_FP8_MODEL=<predictor>`; rented B200 (AF3, Protenix v1 n <= 4) and
the shared B200 host (the rest). Ratio = FP8 / bf16 steady time (< 1 is faster). `fp8.check()` was False in every cell.

| predictor | n = 1 | n = 4 | n = 8 | n = 16 |
|---|---|---|---|---|
| AF3 | 0.342 -> 0.345 (1.006) | 0.899 -> 0.853 (0.948) | 2.532 -> 2.334 (0.922) | 11.09 -> 9.44 (**0.852**) |
| Protenix v1 | 0.330 -> 0.328 (0.99) | 0.862 -> 0.819 (0.95) | 2.358 -> 2.190 (0.929) | 10.10 -> 8.78 (**0.869**) |
| Boltz-2 | 0.354 -> 0.354 (1.00) | 1.008 -> 0.965 (0.957) | 2.944 -> 2.801 (0.951) | 13.83 -> 12.78 (0.924) |
| OpenFold3 | 0.340 -> 0.342 (1.006) | 0.906 -> 0.871 (0.961) | 2.466 -> 2.273 (0.922) | 10.78 -> 9.16 (**0.850**) |
| Chai-1 | 0.269 -> 0.265 (0.988) | 0.991 -> 0.953 (0.962) | 3.428 -> **non-finite** | 17.36 -> 16.28 (0.938) |

- Routes that fire from n = 8: TriAttn PV (q / k declined on B200), TriMul, OPM, Transition, token DiT expand (declined at n <= 4 / n = 1),
  PWA from n = 16. Boltz-2 is not in `TRIATTN_MODELS`; Chai-1's D256 n2 pair Transition is not served (36-112 Transition calls only); OPM fires
  for Chai-1 but is not counted by `route_stats()`.
- **Structure (end-level, the first B200 FP8 end-level check)**, n = 1, same seed, all-atom Kabsch RMSD FP8 vs bf16 (bf16 vs bf16 is bit-identical):
  AF3 0.093 A (the model's own sample-to-sample spread 0.44 A), Protenix v1 0.078 A (0.60), Boltz-2 0.057 A (0.68), OpenFold3 0.096 A (0.48),
  Chai-1 0.073 A (0.74); mean |dpLDDT| 0.04-0.12. n >= 4 was not judged here because B200 bf16 itself was wrong under CUDA-graph capture at
  n = 4 / 8 until a553001e1 (`_streams._skips_in_capture` skipped `record_stream` on the B200; see that commit message).
- **Chai-1 n = 8 FP8: non-finite coordinates** (FloatingPointError in decode) with `fp8.check()` False, also with CUDA graphs off; bf16 n = 8 and
  FP8 n = 16 are finite. Re-checked on engine main a553001e1 (after the token DiT e4m3 scale floor a8ddeaac5): still non-finite (bf16 n = 8
  ok, 3.43 s). Not localized (the minus-TriAttn / -PWA / -TriMul / -token-DiT variants were not run: the shared host's GPUs were taken).
  **Blocked in code**: `fp8.B200_BLOCKED_MODELS` = {chai1}: on a B200 `FOLDJET_ENGINE_FP8` reads as off and `fp8.enable()` acts as `disable()` for
  Chai-1 (tests/cpu/test_fp8_b200_blocked_models.py). Remove the entry only after an end-level check on the B200.
  **Unblocked 2026-11-24 (37f357dbf)**: with the PWA e4m3 route off (below) Chai-1 n = 8 FP8 is finite (3.34 s, bf16 3.43 s).

### 2026-11-24: two B200 faults under CUDA graphs, both switched off in engine main

- **RoseTTAFold3 5I28 n = 12 / 16 FP8 non-finite** (F3, compile + CUDA graphs, one size per process). Route isolation at n = 12, one process per
  variant: every route alone finite except **PWA e4m3 alone -> non-finite**; `safe` without PWA finite (4.92 s, bf16 5.48 s); all routes off with
  the FP8 arm finite. A hook on `pwa_fp8_b200` saw the eager calls in range (max |v8| 20 of 448, no NaN in v8 / o): the fault is in the captured
  replay, not localized. Fix (a888a8143): the B200 PWA e4m3 route serves only with `FOLDJET_ENGINE_PWA_FP8_B200=1` (on top of `safe`); its v-scale cache
  is keyed on the weights' versions / shapes (was the wv pointer alone; not the cause: forced on with the new key it is still non-finite).
- **bf16 non-finite with several shapes' sampler graphs in one process** (FoldJet F4 under compile + CUDA graphs: AF3 bf16 9 / 27 targets).
  Execution bisect (12 targets): FoldJet's per-shape sampler-graph cache off -> finite; compile alone, trunk graphs alone -> finite. Engine
  switch bisect: `FOLDJET_ENGINE_TOKEN_B200_STREAMS=0` or `_BF16_FORK=0` -> 12 / 12 finite; PAR / FOLDS / FINAL / GEARLY / the atom DiT early bias
  -> still non-finite; keeping the token packs per capture -> still non-finite; recording the side-stream outputs (bias table, gates) on the caller's
  stream -> still non-finite. Fix (4915ee25a): the bf16 fork of the pair pass off by default (same kernels in program order, outputs unchanged;
  the fork had been 1-8 % faster at L 128-512); TokenDiT* `cuda:b200_sm100` 25 -> 26 (900d8d355). fp32 keeps its fork (not seen failing).
  F4 under graphs on the fixed engine: 21 / 21 runs finite, FP8 / bf16 27-target sums 0.914-0.978.
- AF3 eager profile, n = 16 (11.6 s): Pairformer ~64 % (TriAttn 6.1 s, TriMul 1.4 s), token DiT 17 %, atom DiT 0.5 s, Transition 3 %, OPM 0.11 s,
  PWA 0.014 s: the FP8 ceiling is set by TriAttn; at n = 8 the token DiT (expand-only FP8, block 0.93-0.99) leads.
- Transition D128: the gate now serves only the row windows where e4m3 is faster (commit 6189cf1e0, `fp8_sm100a.py` ROW_GATE / ROW_MAX:
  n 2 65536 <= rows < 1474560, n 4 147456 <= rows < 1179648); the old gates lost up to 9 % at 1.6-2.4 M rows (e.g. AF3 pair transitions at 5I28 n = 12).

## Verification of the combined tree (n14-all)
FP8-off bit identity against n14-base, eager and graph, sha1, separate processes: token DiT + APB 398 / 398 (eager and graph), atom DiT 258 / 258 eager and 258 graph (4 float32 L4096 cases are nondeterministic in the base itself), OPM + PWA 174 / 174 (both), TriMul 24 / 24 (eager + graph, max abs 0),
Transition 23 shapes (all inference shas equal; the last parameter-gradient hash of each training case differs between two runs of the BASE as well), TriAttn 90 cases eager + graph 0 differ.
Tests: tests/gpu 831 passed, tests/cpu 3500 passed (GPU visible). `FOLDJET_ENGINE_FP8=safe` end to end (one process, one captured graph, 14 module workloads): all routes counted, `check()` False, graph == eager bytes, `disable()` returns bf16 bit identity.
Prebuild sm100: 835 artifacts, 0 not built.



# FP8 (e4m3) on the B200 (sm_100a) — opt-in inference routes (engine 3.0.3 + B200 kernel set, agent A, 2026-11-19)

Everything here is **opt-in and off by default**. With FP8 off the B200 default routes are bit-identical to the tree without it (section 6).
Scope of this document (agent A): the foundation facts (tcgen05 e4m3 vs bf16 under the power cap), the framework (`src/foldjet_engine/fp8.py`) and the **Transition**
route (`kernels/transition/cuda/fp8_sm100a.py`, `sm100/fp8/tf8_gemm.cu`, `sm100/fp8/tf8_ln.cu`). TriMul (agent B) and the token / atom DiT (agent C) have their own sections
in their trees; the shared facts are in section 2 and `work/n13-fp8-a/mb_results.txt`.

## 0. Summary

* e4m3 `tcgen05.mma kind::f8f6f4` runs at **1.9x the bf16 rate under the ~1000 W cap** (2.06-2.17 vs 1.08-1.17 PFLOP/s on 4096^3 / 8192^3, same tile and pipeline), energy per GEMM
  0.52x. A GEMM shrinks by that factor only where it is tensor-bound (K >= 512); at K <= 384 the epilogue and the bf16 output bound it (x0.8-0.95).
* The Transition on e4m3 (LayerNorm + quantize, expand + SwiGLU with e4m3 h, squeeze + residual: 3 PDL launches) is **faster than the B200 default route at D 512 and D 768**
  (sustained time and energy x0.47-0.88; D 768 n 4 x0.63, D 512 n 4 x0.77 at large M) and, with the fused one-kernel e4m3 forward of section 8, at **D 128 / 256** too (D 128 n 4 sustained x0.85-0.95 and energy x0.79-0.86 from 65536 rows, x1.00 / energy x0.92 at 4096; D 256 n 4 x0.83-0.92 from 12288 rows and x0.54-0.85 below it on the chain; D 256 n 2 x0.78 below 8192 rows); **D 384 stays unserved** (chain x1.00-1.07 at >= 65536 rows, section 8.5).
* Accuracy = the H100 FP8 band (judge error 0.034-0.039 against fp64 at x ~ N(0,1), update rms 6.6 %), no NaN, saturation handled by the 3.0.3 guard (sync redo / lazy +
  `fp8.check()`, also through captured graphs).
* Framework: `fp8.py` serves capability (10, 0) for the slots in `fp8.B200_ROUTES` (Transition now); `FOLDJET_ENGINE_FP8=safe` / `engine-fp8` turn the route on on a B200.

## 1. H100 design digest (what 3.0.3 does and what transfers)

### 1.1 Framework (`src/foldjet_engine/fp8.py`, 3.0.3)
* `PRESETS["safe"]` maps per-route env names to values (TRIMUL=calib, OPM=bound, PWA=1, TRANSITION=calib, TOKEN_DIT=expand, ATOMDIT=1; TriAttn per validated model).
  `FOLDJET_ENGINE_FP8=safe` selects it; `env(name, default)`: a variable present in `os.environ` (even `""` = off) wins, else the preset value, else `default`. Route modules read it **at import**
  (`POLICY = fp8.env("FOLDJET_ENGINE_X_FP8") or None`); `enable()` / `disable()` set the module attributes at run time.
* `engine-fp8` backend: `resolve_backend` -> (`cuda`, FP8 on); `set_backend` installs a `_forward_cuda` wrapped in `BackendScope(on=True)` which `enable("safe")`s for the call and restores the
  switches on exit (stack `SCOPES`, `_BASE`); training (module.training or autograd enabled) never opens the scope. Every route's `serves()` was H100-only (`capability == (9, 0)`).
* `route_stats()` / `reset_route_stats()`: per-route plain-int counters bumped at Python call time (a graph replay does not recount). Transition keys: `calls retries fallbacks capture_fallbacks calibrations`.
* Guard protocol (static-scale routes): scales calibrated on the first eager call of a module (probe call with the scale 1: the pre-cast peak is exact even when it saturates), headroom 16; `sync`
  (eager: the wrapper redoes an overflowing call, up to 4 times), `lazy` (kernel max-atomics the scaled peak into a per-device buffer; `fp8.check()` once per forward outside any graph: peak > 448 ->
  `rescale_all(peak)` lowers the static scale in place, captured graphs follow, rerun), `off`. A capture of an uncalibrated module takes the default route (`capture_fallbacks`).
* Traps recorded in 3.0.x: scale floors (3.0.3: pow2 scale floored at 2^-30 / `1e-30` on amax: a scale of 2^-99 saturates later input, `1 / scale = inf` = a real-weight NaN); the e4m3 NaN
  byte (a hand-formed byte must be clamped to 126; `cvt.rn.satfinite.e4m3x2` is safe); FP8 is outside the judge (no registry row / `backend_versions` entry).

### 1.2 Transition on the H100 (`fp8_sm90a.py`, `transition_fp8_fwd_sm90a.cu`, `transition_fp8_d256_sm90a.cu`, `wide/fp8_bind.cu`, tests `test_transition_fp8*_h100.py`, docs/fp8.md "Transition forward")
| tensor | e4m3 | scale | where |
|---|---|---|---|
| xn = LayerNorm(x), expand A | yes | ONE pow2 scale from the LayerNorm bound `sx = 2^floor(log2(448 / (sqrt(D-1) max|gamma| + max|beta|)))`: cannot saturate | host per weight version; xn quantized from the fp32 LayerNorm in the LN pass |
| Wa, Wb, Ws | yes | per-tensor pow2, `448 / amax` | host, per parameter version, cached (key: data_ptr + `_version` of the 5 parameters) |
| h = silu(a) b | yes | static pow2 `sh`, calibrated (`HEADROOM` 16 over the calibration amax; measured peaks 15-28 of 448), guarded | expand epilogue, `cvt.rn.satfinite`, max|h sh| -> guard buffer |
| x, y, residual, LayerNorm math | bf16 / fp32 | | |
* silu by ONE `tanh.approx` (`silu(a) = (a/2)(1 + tanh(a/2))`): pays with e4m3 GEMMs (0.68-0.80 vs 0.86-0.94 with ex2 + rcp), not with bf16.
* Gates (rows = L^2, whole 128-row tiles): D 128 16384, D 256 8192, D 384 / 512 65536 (below: the default's generic kernels are faster, 128-row tiles leave SMs idle; at token counts the wide FP8
  chain was 1.5-3.4x SLOWER; cuBLAS e4m3 bounds the gain there at 0-17 %). Speed (cool / sustained, graph, vs 2.5.12): D128 0.66-0.76, D256 0.69-0.77, D384 n4 0.64-0.69, D512 n4 0.56-0.62.
* Accuracy acceptance (fp64 gold, weights N(0, 1/fan_in), gamma 1 +- 0.1, beta +- 0.1, x ~ N(0,1)): judge `max|y - y64| / max|y64|`: bf16 PyTorch 0.0044-0.0068, default 0.0035-0.0042,
  FP8 full 0.035-0.045 (rms 0.034, update rms 0.066), h + squeeze only 0.020-0.029; no NaN, no saturation. All FP8 variants are 3-5x outside the 8e-3 judge tolerance, accepted because FP8 is outside the judge.
* Tests to mirror: fragment permutation (wgmma only); error vs fp64; bf16 twin = default bytes; guard sync retry and lazy `overflow_peak` / `rescale_all`; default untouched when off; graph = eager.
* Rejected / not done on the H100: D 64 (bound 0.72-0.76 on paper), per-channel weight scales, e4m3 x / y storage, e5m2, training (4-11 % step time, gradient error 7-10x bf16).

### 1.3 Transfers unchanged vs redone for tcgen05
| transfers unchanged (used as is) | redone for tcgen05 |
|---|---|
| quantization recipe: LayerNorm-bound xn scale, per-tensor pow2 weight scales shared by Wa / Wb, static calibrated h scale with headroom 16 | MMA `kind::f8f6f4`, instruction descriptor, K = 32 per MMA |
| pack once per weight version, cache keyed by (data_ptr, `_version`), calibrate on the first eager call, capture fallback | smem layout (128 e4m3 per 128-B swizzle row, no K permutation: the H100's `_perm_k` was a wgmma register-fragment trick) |
| guard protocol (`sync` / `lazy`, `overflow_peak`, `rescale_all`, `fp8.check()`), route stats keys, `FOLDJET_ENGINE_TRANSITION_FP8*` switches | epilogue (TMEM -> registers, scales, satfinite cast, st.global), pipelining (TMA ring, TMEM double buffer, PDL chain) |
| tanh silu | tile shapes (128 x BN, BN 128 / 192 / 256), launch structure (3 launches instead of one fused kernel) |
| acceptance: fp64 gold, judge band 0.02-0.045, no NaN | gates: re-measured against the B200 default route (v66), not carried over |
Deviations from the H100 recipe (each with a measured reason): the scale floor `AMAX_FLOOR = 1e-12` instead of `1e-30` (so that `1/(sx s1)` stays finite in fp32 for tiny gamma / weights),
the row gates (re-measured), D 128 / 256 / 384 not served (loses on B200 against v66).

## 2. Foundation facts: tcgen05 e4m3 vs bf16 under the power cap
Kernel `sm100/fp8/tf8_gemm.cu` (`-DE4=1` e4m3 `kind::f8f6f4`, `-DE4=0` bf16 `kind::f16`; MODE 2 = bf16-out GEMM): same 128 x BN tile, 128-B swizzle K-major operands, 4 MMAs per 128-B K block (K = 32 e4m3 /
16 bf16 per MMA, 128 / 64 elements per block), ring of 4-6 stages, TMEM double buffer, 8 epilogue warps, fp32 scale multiply, direct `st.global` of bf16. Method: a CUDA graph of 8 launches
replayed back to back for 2 s (sustained, no L2 flush), `nvidia-smi` power / SM clock every 0.2 s over the window, J = mean W x ms (`work/n13-fp8-a/mb.py`, `mb_raw*.jsonl`, `mb_table2.txt`; clean runs: another
user's job shared GPU 0 for ~15 min and its rows were rerun).
| GEMM (M N K) | e4m3 TFLOP/s | W | MHz | mJ | bf16 TFLOP/s | W | MHz | mJ | time x | energy x |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 4096 4096 | 2156 | 862 | 1602 | 54.9 | 1167 | 887 | 1573 | 104.4 | 0.54 | 0.53 |
| 8192 8192 8192 | 2058 | 941 | 1106 | 502.7 | 1083 | 957 | 1078 | 972.0 | 0.53 | 0.52 |
| expand D 128 (65536, 1024, K 128) | 354 | 791 | 1965 | 38.4 | 324 | 862 | 1950 | 45.7 | 0.92 | 0.84 |
| expand D 256 (65536, 2048, K 256) | 672 | 884 | 1894 | 90.4 | 530 | 892 | 1785 | 115.5 | 0.79 | 0.78 |
| expand D 384 (65536, 3072, K 384) | 770 | 902 | 1949 | 181.0 | 612 | 914 | 1773 | 230.8 | 0.79 | 0.78 |
| expand D 512 (65536, 4096, K 512) | 1061 | 917 | 1746 | 237.6 | 782 | 927 | 1587 | 326.0 | 0.74 | 0.73 |
| expand D 768 (65536, 6144, K 768) | 1235 | 940 | 1719 | 470.9 | 910 | 953 | 1520 | 648.3 | 0.74 | 0.73 |
| squeeze D 128 (65536, 128, K 512) | 733 | 665 | 1965 | 7.8 | 583 | 771 | 1965 | 11.3 | 0.80 | 0.69 |
| squeeze D 256 (65536, 256, K 1024) | 1413 | 852 | 1850 | 20.7 | 774 | 875 | 1604 | 38.8 | 0.55 | 0.53 |
| squeeze D 384 (65536, 384, K 1536) | 1392 | 871 | 1627 | 48.4 | 787 | 888 | 1646 | 87.2 | 0.57 | 0.56 |
| squeeze D 512 (65536, 512, K 2048) | 1793 | 883 | 1447 | 67.7 | 1036 | 885 | 1400 | 117.5 | 0.58 | 0.58 |
| squeeze D 768 (65536, 768, K 3072) | 1968 | 899 | 1455 | 141.3 | 1112 | 915 | 1413 | 254.6 | 0.56 | 0.55 |
(the expand rows are the bf16-output stand-in with N = 2 x 4 D, so the output bytes are the bf16 ones; the whole table: `mb_table2.txt`.)
Facts: (a) compute-bound: **x1.9 the bf16 rate and x0.52 the energy under the cap**; both dtypes run near 860-950 W, the cap lowers the clock (1.1 GHz at 8192^3 for e4m3) and the e4m3 rate stays
2.06 PF/s. (b) K <= 384: the tile is epilogue / output-bound, gain only x0.8-0.95 (D 128 expand 0.92-0.99). (c) low arithmetic intensity (ai(e4) 120 flop/B at D 128): operand bytes halve but a bf16 output does not,
so a route keeps the byte saving only if the NEXT kernel takes e4m3 straight from this kernel's epilogue (the Transition's h) -- an extra quantization pass costs more than it saves. (d) Quantization pass `tf8_rowquant`
(bf16 read + e4m3 write + per-row scale, sustained without L2 flush): [16384,128] 4 us, [65536,384] 15 us (5.2 TB/s, L2 hits), [65536,512] 18 us, [65536,768] 31 us, [262144,128] 56 us (HBM): ~10-20 % of an e4m3 expand GEMM as a standalone pass; the LayerNorm + quantize kernel `tf8_ln` of the Transition takes 25-35 us at 65536 rows (eager, with its launch floor). cuBLAS bf16 on this box: 1319 / 1350 TFLOP/s at 4096^3 / 8192^3, so the e4m3 kernel is x1.55 cuBLAS bf16 (the 1-CTA kernel is 82-87 % of cuBLAS in bf16).
(e) The fp32 scale multiply in the epilogue is in both builds (1 FMUL / element) and not separable. (f) Instruction facts, verified by running: idesc = `(1<<4) | (N>>3)<<17 | (M>>4)<<24` with A / B format 0 = e4m3,
descriptors as bf16 (`desc_k128`, +2 x 16 B per MMA K step = 32 B), TMA maps of an e4m3 tensor = a bf16 view `[rows, K/2]` with a 64-element box (the host extension needs no change). (g) Kernel error against an fp32
reference of the same e4m3 inputs 2.4e-3 (the bf16 output rounding).

## 3. Framework change (`fp8.py`)
* `B200_ROUTES: {slot: module path}` (Transition: `kernels/transition/cuda/fp8_sm100a`); `SLOTS`; `arch()` ("sm100" on a B200); `serves_sm100(slot)`. `_modules()` is cached per arch and returns the B200 module in
  the slot's place on a B200, the H100 module otherwise (so `enable` / `disable` / `status` / `check` / `route_stats` / `BackendScope` need no other change). A slot without an entry keeps the H100 module, whose `serves()` is
  False on a B200. A B200 route module must offer the interface of its slot (`POLICY`, `GUARD`, `GUARD_STATS`, `overflow_peak`, `rescale_all`, ...) and read its switch through `fp8.env` with the preset's variable name.
* Same preset semantics: `FOLDJET_ENGINE_FP8=safe` sets `FOLDJET_ENGINE_TRANSITION_FP8=calib` for both archs; a per-route variable set explicitly (also `""`) wins; default off (`tests/cpu/test_fp8_switch.py`: `test_b200_route_table`,
  `test_b200_transition_route_follows_the_preset`). The token DiT's B200 route (agent C) lives in the H100 module `token_dit_fp8` itself, so it needed no entry.
* Guards: the `and not b200` guard of `conditioned_transition/cuda/__init__.py` is agent C's (removed only for the token-DiT expand route that runs on B200). Nothing else changed in the dispatch except one
  `POLICY is not None` test at the top of `route_sm100a.transition_sm100a` (and its private `_fp8=False` keyword for the guard's fallback).

## 4. Transition FP8 on the B200 (`fp8_sm100a.py`)
### 4.1 Design (the H100 recipe on tcgen05; one switch `FOLDJET_ENGINE_TRANSITION_FP8=calib`, also `FOLDJET_ENGINE_FP8=safe` / `engine-fp8`)
Three launches (PDL chain, `route_sm100a.transition_sm100a` takes the route first when `fp8_sm100a.POLICY is not None`, no grad, `serves()`):
1. `tf8_ln` (`sm100/fp8/tf8_ln.cu`): one warp per row, fp32 two-pass LayerNorm, xn quantized straight to e4m3 with the LayerNorm-bound scale `sx` (never saturates).
2. `tf8_gemm` MODE 0 (expand): A = xn8 [M, D], B = Wa8 / Wb8 [n D, D], two fp32 accumulators in TMEM (`kind::f8f6f4`, BN 128 per accumulator, 2 x 2 x 128 = 512 columns), epilogue
   `h = silu(acc_a ca) * (acc_b ca sh)` with the tanh silu, `cvt.rn.satfinite.e4m3x2`, st.global of the e4m3 h [M, n D] (half the bytes of the bf16 round trip), per-CTA max |h| -> `atomicMax` on the guard buffer.
3. `tf8_gemm` MODE 1 (squeeze): A = h8, B = Ws8 [D, n D], BN 256 when D % 256 == 0 and every SM keeps a tile, 192 when D % 192 == 0 (D 384), else 128; `out = x + acc / (sh s2)` in bf16.
Scales = the H100's: Wa and Wb share one per-tensor pow2 scale `s1`, Ws one `s2`, `sx` from the LayerNorm bound, `sh` static (HEADROOM 16, calibrated on the first eager call, probe with sh = 1), kernel vector
`[sx, 1/(sx s1), sh/(sx s1), 1/(sh s2)]` refreshed in place. Floors: amax >= 1e-12. Weights packed once per (data_ptr, `_version`) of the 5 parameters (fp32 masters or bf16). Any row count (TMA zero-fills the tail
tile, stores / residual loads are row-masked), D a multiple of 128 (`tf8_ln` D <= 1024), n 2 / 4. `fp8.route_stats()["transition"]`: `calls retries fallbacks capture_fallbacks calibrations`.
Cubins (`sm100/fp8/`): `ln`, `swiglu128`, `resid128`, `resid192`, `resid256` (prebuild group `transition_sm100a/fp8/*`, +5 artifacts in `_prebuild --arch sm100`). No host extension change.
Kernel times at 65536 rows (eager, us; a launch floor is in each): D 384 ln 33 / swiglu 96 / resid 62; D 512 34 / 143 / 75; D 768 43 / 282 / 140. Epilogue ablation of the expand kernel (`ABL`, wrong output,
timing only): D 768 full 288 us = no-math 251 (MMA-bound, 2.15 PF/s); D 512 143 vs 123; D 384 93 vs 74; D 256 53 vs 38: below K = 512 the silu + store epilogue and the tile prologue cost 20-30 %.

### 4.2 Gates (rows = batch x L^2 x ..., measured against the B200 default route)
`ROW_GATE = {(512, 4): 1024, (512, 2): 4096, (768, 4): 4096, (768, 2): 4096}`; `WIDTHS = 512, 768`; `FOLDJET_ENGINE_TRANSITION_FP8_MINROWS` / `_WIDTHS` override (experiments only). Reasons: D 768 n 4 x1.07 at 1024
rows, 0.90 at 2048, 0.60-0.65 from 4096; D 768 n 2 x1.30 at 1024, 1.16 at 2048, 0.51-0.76 from 4096; D 512 n 4 x0.50 from 1024 rows (the default is a slow path there: 45 us at 1024 rows); D 512 n 2 x0.63 at 4096, 0.88 at 16384+.
D 128 / 256: served by the fused kernel since section 8 (gates in 8.6; the chain numbers of this paragraph are the chain forced on). Not served: **D 384** (x0.93-1.09 at 4096-262144 rows, 1.50 at 1024 rows, n 2: 1.09-1.21; section 8.5).

### 4.3 Results: B200 default route vs e4m3 route, same process, interleaved, 3 rounds
Graph medians (CUDA graph of one call, 256 MB L2 flush before each replay, medians of 30 replays after a 0.15 s idle gap) and a sustained window (graph replayed back to back for 3 s per round, `nvidia-smi` power
every 0.2 s, J = mean W x ms per step, no L2 flush); master (fp32) weights, bf16 x, lazy guard, calibrated eagerly; no overflow in any case (peak 17 of 448). Script `work/n13-fp8-a/ab.py`, rows `ab_final.jsonl`.
The sustained columns of the 1024- / 4096-row cases are launch-bound (python replay loop, 350-700 W): read them as ratios only.

| D | n | rows | default ms (graph, L2 flushed) | e4m3 ms | x graph | default ms (sustained) | e4m3 ms | x sustained | W default / e4m3 | J default | J e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 4 | 1024 | 0.0451 | 0.0225 | 0.50 | 0.0389 | 0.0184 | 0.47 | 355 / 417 | 13.801 mJ | 7.682 mJ | 0.56 |
| 512 | 4 | 4096 | 0.0553 | 0.0328 | 0.59 | 0.0479 | 0.0281 | 0.59 | 609 / 689 | 29.124 mJ | 19.344 mJ | 0.66 |
| 512 | 4 | 16384 | 0.0903 | 0.0758 | 0.84 | 0.0979 | 0.0751 | 0.77 | 876 / 874 | 85.829 mJ | 65.602 mJ | 0.76 |
| 512 | 4 | 65536 | 0.3113 | 0.2417 | 0.78 | 0.3716 | 0.2854 | 0.77 | 897 / 899 | 333.250 mJ | 256.549 mJ | 0.77 |
| 512 | 4 | 262144 | 1.2524 | 0.9544 | 0.76 | 1.4551 | 1.1173 | 0.77 | 920 / 916 | 1338.833 mJ | 1022.924 mJ | 0.76 |
| 512 | 2 | 4096 | 0.0389 | 0.0246 | 0.63 | 0.0328 | 0.0205 | 0.62 | 534 / 578 | 17.498 mJ | 11.829 mJ | 0.68 |
| 512 | 2 | 16384 | 0.0573 | 0.0532 | 0.93 | 0.0534 | 0.0470 | 0.88 | 870 / 843 | 46.497 mJ | 39.580 mJ | 0.85 |
| 512 | 2 | 65536 | 0.1762 | 0.1618 | 0.92 | 0.2079 | 0.1775 | 0.85 | 890 / 890 | 185.067 mJ | 158.007 mJ | 0.85 |
| 768 | 4 | 4096 | 0.0821 | 0.0492 | 0.60 | 0.0749 | 0.0430 | 0.57 | 867 / 849 | 64.929 mJ | 36.512 mJ | 0.56 |
| 768 | 4 | 16384 | 0.2007 | 0.1311 | 0.65 | 0.2302 | 0.1457 | 0.63 | 890 / 885 | 204.948 mJ | 128.899 mJ | 0.63 |
| 768 | 4 | 65536 | 0.7374 | 0.4629 | 0.63 | 0.8710 | 0.5544 | 0.64 | 908 / 905 | 790.520 mJ | 501.989 mJ | 0.64 |
| 768 | 4 | 131072 | 1.4766 | 0.9259 | 0.63 | 1.7236 | 1.0931 | 0.63 | 920 / 913 | 1586.009 mJ | 998.013 mJ | 0.63 |
| 768 | 2 | 4096 | 0.0655 | 0.0348 | 0.53 | 0.0574 | 0.0290 | 0.51 | 591 / 720 | 33.902 mJ | 20.872 mJ | 0.62 |
| 768 | 2 | 16384 | 0.1126 | 0.0901 | 0.80 | 0.1188 | 0.0884 | 0.74 | 873 / 872 | 103.721 mJ | 77.092 mJ | 0.74 |
| 768 | 2 | 65536 | 0.3789 | 0.2888 | 0.76 | 0.4401 | 0.3345 | 0.76 | 896 / 895 | 394.329 mJ | 299.413 mJ | 0.76 |

(`x` columns = e4m3 route / default route; < 1 = the e4m3 route is faster / cheaper. D 384 and D 256 / 128: section 4.6.)

### 4.4 Accuracy against the fp64 gold (judge form `max|y - y64| / max|y64|`; non-zero random weights N(0, 1/fan_in), gamma 1 +- 0.1, beta +- 0.1, 16384 rows, calibrated on the case)
Next to each case the B200 DEFAULT route's error and the ratio. The default's error is bf16 (0.0035-0.0050 at x ~ N(0,1)). `scale` = x std, `off` = mean offset (x is bf16-rounded before both routes and the gold).
No NaN / inf and no guard retry in any of the 36 cases. (`work/n13-fp8-a/acc.py`, rows `acc_raw.jsonl`; the update rms column is `rms(y - y64) / rms(y64 - x)`.)
```
WARNING: foreign pid 3966857 on GPU 0: .venv/bin/python /home/khidi_hjju/tmp/distill/known/famtest/predict.py --part 0 --parts 8 
   D  n  xscale  off | judge def judge fp8  ratio |  rms def  rms fp8  ratio | upd8  nan retries
 384  4   0.001    0 |    0.0049    0.0591   12.0 |   0.0044   0.0650   14.8 | 0.065 0 0
 384  4   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 384  4       1    0 |    0.0036    0.0383   10.8 |   0.0027   0.0344   12.6 | 0.066 0 0
 384  4       1   25 |    0.0023    0.0085    3.6 |   0.0014   0.0022    1.5 | 0.088 0 0
 384  4      40    0 |    0.0024    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.127 0 0
 384  4      40   25 |    0.0021    0.0026    1.3 |   0.0017   0.0019    1.1 | 0.143 0 0
 384  2   0.001    0 |    0.0050    0.0648   13.0 |   0.0043   0.0652   15.0 | 0.065 0 0
 384  2   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 384  2       1    0 |    0.0036    0.0374   10.3 |   0.0027   0.0341   12.6 | 0.066 0 0
 384  2       1   25 |    0.0024    0.0081    3.4 |   0.0014   0.0021    1.5 | 0.089 0 0
 384  2      40    0 |    0.0025    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.128 0 0
 384  2      40   25 |    0.0021    0.0025    1.2 |   0.0017   0.0019    1.1 | 0.145 0 0
 512  4   0.001    0 |    0.0044    0.0639   14.4 |   0.0043   0.0650   15.0 | 0.065 0 0
 512  4   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 512  4       1    0 |    0.0035    0.0369   10.4 |   0.0027   0.0341   12.7 | 0.066 0 0
 512  4       1   25 |    0.0024    0.0079    3.3 |   0.0014   0.0021    1.5 | 0.089 0 0
 512  4      40    0 |    0.0023    0.0028    1.2 |   0.0017   0.0019    1.2 | 0.127 0 0
 512  4      40   25 |    0.0022    0.0028    1.3 |   0.0017   0.0019    1.1 | 0.144 0 0
 512  2   0.001    0 |    0.0047    0.0636   13.5 |   0.0043   0.0651   15.1 | 0.065 0 0
 512  2   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 512  2       1    0 |    0.0037    0.0353    9.6 |   0.0027   0.0342   12.6 | 0.066 0 0
 512  2       1   25 |    0.0024    0.0086    3.6 |   0.0014   0.0022    1.5 | 0.088 0 0
 512  2      40    0 |    0.0021    0.0025    1.2 |   0.0017   0.0019    1.2 | 0.127 0 0
 512  2      40   25 |    0.0021    0.0027    1.3 |   0.0017   0.0019    1.1 | 0.144 0 0
 768  4   0.001    0 |    0.0046    0.0668   14.6 |   0.0044   0.0651   14.8 | 0.065 0 0
 768  4   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 768  4       1    0 |    0.0036    0.0342    9.5 |   0.0027   0.0343   12.6 | 0.066 0 0
 768  4       1   25 |    0.0024    0.0082    3.5 |   0.0014   0.0022    1.5 | 0.088 0 0
 768  4      40    0 |    0.0024    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.127 0 0
 768  4      40   25 |    0.0022    0.0027    1.2 |   0.0017   0.0019    1.1 | 0.144 0 0
 768  2   0.001    0 |    0.0045    0.0659   14.8 |   0.0044   0.0652   14.8 | 0.065 0 0
 768  2   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 768  2       1    0 |    0.0034    0.0360   10.6 |   0.0027   0.0341   12.6 | 0.066 0 0
 768  2       1   25 |    0.0024    0.0082    3.4 |   0.0014   0.0021    1.5 | 0.089 0 0
 768  2      40    0 |    0.0023    0.0027    1.2 |   0.0017   0.0019    1.2 | 0.128 0 0
 768  2      40   25 |    0.0022    0.0028    1.2 |   0.0017   0.0019    1.1 | 0.144 0 0
```
Reading: at x ~ N(0,1) the e4m3 route is at the H100 FP8 level (judge 0.034-0.039, update rms 6.6 %; the H100 doc: 0.035-0.045 / 6.6 %), 9.5-10.8x the bf16 default's judge error (10x was the H100's ratio too). At
x std 1e-3 the LayerNorm eps dominates (var 1e-6 << eps), y ~ x and the update is tiny, so the judge form (relative to max|y|) reads 0.059-0.067 for the same 6.5 % update error. Large activations (std 40) and offsets (25) are
benign: the error is dominated by the bf16 input rounding of x that both routes share (ratio 1.1-1.3). Offset 25 with std 1 (x quantized to 0.125 steps by bf16): 3.3-3.6x on the judge form (rms 1.5x).
All FP8 judge errors exceed the 8e-3 judge tolerance (as on the H100; FP8 is outside the judge).

### 4.5 Saturation and the guard (tests `test_sync_guard_redoes_a_saturated_call`, `test_lazy_guard_check_and_rerun_through_a_graph`)
`h` is the only static-scale tensor (xn: LayerNorm bound, weights: exact amax). LayerNorm makes h independent of the x scale and offset, so only a direction change (or a different distribution) can raise it. Provoked by calibrating on x of
std 1e-4 (eps dominates: h ~ 1000x smaller) and then running std 1: `sync` guard: the call detects peak > 448, lowers `sh` by 2^ceil(log2(peak / 448 x 2)) and redoes the call (`retries` >= 1, `fallbacks` 0), result inside
the band; `lazy`: the kernel records the peak (also inside a captured graph: the guard buffer is created before capture), `fp8.check()` returns True once, the scale is lowered in place, the captured graph is replayed
again and `check()` is False (error inside the band). Capture of a module that was never calibrated: the default route (`capture_fallbacks`). e4m3 conversion is `cvt.rn.satfinite` everywhere, so no NaN byte exists.

### 4.6 Measured losses and rejected variants
* (SUPERSEDED by section 8: the fused kernel was built and measured) **D 384 (x0.93-1.09; 1.5 at 1024 rows), D 256 (x1.17-1.46), D 128 (x2.0)** of the three-launch chain against the B200 default (v66 tuned fused / wide kernels, 0.0328 ms at D 128 n 4 65536 rows = one fused kernel with LayerNorm in registers): the
  three-launch chain pays ~25-35 us of LayerNorm + launch floor and the expand epilogue at K <= 384 (microbenchmark x0.79-0.95); the default's D 128 / 256 is a one-launch 2-CTA fused kernel. A fused e4m3 forward
  (LayerNorm -> expand -> SwiGLU -> squeeze in one kernel with h in TMEM / smem as e4m3) is the open item for D 128 / 256 / 384 (not attempted: `tr_fwd_sm100.cu` is 2-CTA with 8 roles).
* Squeeze BN 192 for D 384 (`resid192`): 65 -> 62 us only (the squeeze at D 384 is L2-intake-bound: 1536 tiles x 392 KB; the lever would be the CTA pair, not built).
* Expand epilogue in f16x2 (half the MUFU and ALU ops): counted at ~15 % fewer instructions per element, not built (the epilogue ablation bounds the gain at < 6 % of the D 512 route).
* A separate quantization pass for the squeeze operand (instead of the expand epilogue writing e4m3 h): rejected on arithmetic (a pass costs 20-30 % of the GEMM it feeds, section 2d).
* The H100 K permutation (`_perm_k`), the fp32 `-DABL` etc.: not needed (descriptors read the natural order).

## 5. Not done / open
* Fused one-kernel e4m3 Transition: built for D 128 / 256 (section 8); D 384 not fused (sections 8.5, 8.8). TriMul / token DiT / atom DiT: agents B / C.
* Training (out of scope: 3.0.3 measured 4-11 % step time, gradient error 7-10x bf16). Non-power-of-two D (not multiples of 128), D > 1024.
* Squeeze CTA-pair (`cta_group::2`) and a multicast variant for the L2-intake-bound D 384 squeeze; epilogue pipelining for K <= 384.
* End-level (structure) accuracy of the e4m3 Transition on the B200 was not measured (module level only, like the H100 branch before `fp8safe2`).
* No `_prebuild` Transition FP8 cubin is in the default plan unless an `eval` row at D 512 / 768 exists (it does: +5 artifacts).

## 6. Verification
* Bit-identity with FP8 off: `work/n13-fp8-a/sha.py` (12 inference shapes D 64..768 n 2 / 4, eager and captured graph, + a training forward / backward at D 128 / 384 / 512) run in separate processes on the base tree
  (`n13-base`) and this tree: every inference sha1 (eager and graph) and every training y / x-grad / parameter-grad sha1 is identical, except ONE parameter gradient of each training case, which also differs between two runs of
  the BASE tree (atomic accumulation order): `sha_base.json`, `sha_base2.json`, `sha_fp8off.json`. The only default-path change is `POLICY is not None` at the top of `route_sm100a.transition_sm100a`.
* Tests: `tests/gpu/test_transition_fp8_b200.py` (24: route off = default bytes, error vs fp64 for D 128 / 256 / 384 / 512 / 768 x n, large-activation stress, sync guard, lazy guard + `fp8.check()` through a graph,
  graph = eager bytes, uncalibrated capture, gates, ragged rows, route_stats + framework, module `engine-fp8` backend, training stays default, an inf input does not break the guard) 24 passed; `tests/cpu/test_fp8_switch.py` (+2),
  `test_capture_api.py` (+3 listed caches), `test_prebuild_plan.py`: pass; B200 regression (`test_transition_b200_uninit`, `test_trimul_b200`, `test_triattn_b200`, `test_triattn_b200_capture_bits`, `test_misc_b200`,
  ...): 384 passed, 10 skipped (H100-only); the final combined run (this file + `test_fp8_switch` + `test_capture_api` + the B200 transition / trimul / triattn / capture-bits / misc tests): 446 passed. `test_plan_is_reentrant` fails identically on the BASE tree when a GPU test ran earlier in the same pytest process (order-dependent, not this change).
* `_prebuild --arch sm100 --check` (full output `prebuild_check.txt`, base `prebuild_check_base.txt`): 804 -> 809 artifacts (the 5 `transition_sm100a/fp8/*` cubins), native sources compiled by a planned artifact 207 -> 209,
  no other line differs (rc = 1 on both: pre-existing H100-family notes).
* Non-sm100 CUDA sources are byte-identical to `n13-base`; `backend_versions` unchanged (FP8 off changes nothing).

## 7. Fused e4m3 Transition for D 128 (and D 256 / 384): ablation first, decision (2026-11-19, follow-up) -- SUPERSEDED by section 8 (the user order: build first, measure the built kernel; the fused kernel was built and gives x0.85-0.95 sustained at D 128 n 4; the 7 % 'best case' bound below is a timing bound of the bf16 kernel with half the MMA work, not of an e4m3 kernel)
Question: what can an e4m3 MMA (~1.9x tensor rate) save at best in the B200 default D 128 n 4 fused forward `tr_fwd_sm100.cu` (2-CTA, 8 roles, h bf16 in TMEM)? Timing-only variants of a copy
of the kernel (`work/n13-fp8-a/tr_fwd_abl.cu`, kept out of the tree; wrong results; flags `-DABLK=1` halves the MMA instruction count of both GEMMs = what a 2x tensor rate removes at best, `-DABLE=1` removes the SwiGLU
arithmetic, `-DABLE=2` also the LayerNorm statistics), `transition_fused_sm100a` through a CUDA graph, L2 flushed, medians of 50 (5 interleaved rounds, 0.1 s gap), us, `work/n13-fp8-a/trab.py`:
| rows (D 128, n 4) | default | half MMA | no SwiGLU | no SwiGLU + no LN | half MMA + no SwiGLU + no LN |
|---|---|---|---|---|---|
| 4096 | 16.4 | 14.3 (-13 %) | 15.9 | 14.3 | 14.2 |
| 16384 | 16.4 | 16.4 (0 %) | 16.4 | 16.4 | 14.3 |
| 65536 | 34.8 | 32.3 (-7.2 %) | 32.7 | 30.7 | 28.6 |
| 262144 | 88.4 | 82.2 (-7.0 %) | 86.1 | 79.9 | 73.6 |
Reading: the best case of a 2x tensor rate is **7 % of the kernel at 65536-262144 rows** (the AF3 pair-transition range), below the 8 % bar; the 13 % at 4096 rows is on a kernel whose floor is
14.2 us even with the MMA, SwiGLU and LayerNorm all removed (latency / launch bound, not tensor bound). Removing every arithmetic stage together is only 17 % at 262144 rows: the kernel is bound by its
producer / consumer chain and HBM (134 MB of x + y in 88 us = 1.5 TB/s), not by the tensor stage. An e4m3 version would in addition add the xn quantization and h conversion (more epilogue work) and cost the 10x accuracy
of 4.4. **Decision: declined with evidence for D 128** (D 256 is the same kernel family, `tg_fused_h16` / wide, with an even larger epilogue share).
D 384 (chain of `fp8_sm100a`): epilogue share measured in 4.1 (expand: silu + store = 19 us of 93 us; no-math ceiling 74 us vs the 70 us tensor roofline) and squeeze 62 us vs 39 us roofline (L2-intake bound).
Best cases: CTA-pair squeeze -23 us (12 % of the 186 us route), epilogue pipelining -15 us (8 %); at a typical 50-70 % realization 6-8 % each, i.e. a 12-15 % total at best against a default that the chain
only matches (x0.93-1.09). Not built (the break-even is within noise of the default's own variation).


## 8. Fused e4m3 forward for D 128 / D 256 and the other Transition widths (agent F, 2026-11-19; supersedes the "declined" statements of 4.2 / 4.6 / 5 / 7)

The user order of this session: no width is declined on an analytic estimate or an ablation alone: build the best-effort B200 kernel first, measure it at the requirement shapes, decide what is served from the numbers
of the BUILT kernel. The H100 reference (docs/fp8.md "Transition forward", `transition_fp8_fwd_sm90a.cu`, `transition_fp8_d256_sm90a.cu`, `wide/fp8_bind.cu`, tests) is digested in
`docs/b200/fp8_h100_transition_digest.md` (what is e4m3 per width, scales, gates and why, accuracy numbers, traps, transfers-unchanged vs redo-for-tcgen05).

### 8.1 What was built

| file | what |
|---|---|
| `sm100/fp8/tr8_fwd_sm100.cu` (new) | ONE fused 2-CTA kernel, `-DDIM=128|256 -DHID=n D`: the role structure of `tr_fwd_sm100.cu` (leader-issued M = 256 `cta_group::2` MMAs, B split by N across the pair, 128-row tiles per CTA, persistent) with e4m3 operands |
| `sm100/fp8/tf8_gemm.cu` | `PAIRB` build: the chain GEMMs on 2-CTA clusters (two m-tiles at the same n, each CTA loads HALF of the B rows and multicasts them); same results, opt-in (`FOLDJET_ENGINE_TRANSITION_FP8_PAIRB=1`) |
| `fp8_sm100a.py` | fused launch (`_launch_fused`), `FUSED_WIDTHS` / `FUSED_MIN_ROWS` / `ROW_GATE` / `ROW_MAX`, the cubin specs `f128n4 f128n2 f256n4 f256n2` (prebuild plan 813 -> 817 artifacts; 821 with `_PAIRB`) |
| `tests/gpu/test_transition_fp8_b200.py` | 24 -> 48 tests (fused D 128 / 256 error + chain agreement, stress, sync / lazy guard through graphs, ragged rows, gates) |

Fused kernel (per CTA: 16 warps; warp 0 x producer, warp 3 weight producer, warps 1 / 2 the leader's expand / squeeze MMA issue, warps 4-7 + 12-15 SwiGLU, warps 8-11 LayerNorm + epilogue; `setmaxnreg` 56 / 152):
* **LayerNorm**: x (bf16, TMA, 128-B swizzle) is read from shared memory, fp32 two-pass LayerNorm, `xn8 = e4m3(LN(x) sx)` (`sx` = pow2 LayerNorm bound: cannot saturate) written as K-blocks of 128 e4m3 (128 B) in the swizzle the MMA reads;
  `gamma sx / beta sx` are folded into the smem vectors. The packed (f32x2) three-pass LayerNorm is the default (`LNP`); x stays bf16 in smem (D 128) or is re-read from L2 (D 256, `RESG`) for the residual.
* **expand**: `[a|b]_j = xn8 [Wa8_j; Wb8_j]^T`, one 64-unit chunk j at a time, N = 128 (leader: Wa8_j rows, peer: Wb8_j rows, 64 x D bytes each), K = D = 4 / 8 `kind::f8f6f4` K32 steps (+32 B in the descriptor each).
* **SwiGLU** (two warpgroups alternate chunks; D 256: one warpgroup): TMEM -> registers, `h = silu(a ca) (b cb)` with `silu(z) = (z/2)(1 + tanh(z/2))` (one MUFU op per unit), `cvt.rn.satfinite.e4m3x2`, 64 bytes a row and chunk = 16 TMEM columns
  written back to tensor memory, max |h| per thread -> `atomicMax` on the guard buffer (the same `peak` buffer / calibration / sync-lazy guard as the chain).
* **squeeze**: the e4m3 h in tensor memory is the A operand (`kind::f8f6f4`, TS form, 4 K32 steps over a PAIR of chunks = one 128-B row of Ws8 at D 128; one chunk (2 steps) at D 256 on 64 B of the 128-B row), B = Ws8 [D, n D] bytes (128-B-swizzle map, no K permutation: the TMEM A operand is in natural order),
  accumulators in TMEM (128 / 256 columns); epilogue `out = x + acc / (sh s2)` (bf16). Tensor memory: D 128 = AB 2 x 128 + h 64 + OUT 128 (single-buffered, as the default); D 256 = AB 2 x 128 + OUT 256 with h written IN PLACE over the first 16 columns of its own AB buffer
  and expand(c + 2) waiting for the squeeze that read it (as `tfwd_d256.cu`).
* Shared memory (B): D 128: x 4 stages x 32 KB, xn8 2 x 16 KB, Wab ring 4 x 8 KB, Ws8 2 x 8 KB; D 256: x 1 stage x 64 KB (RESG: the residual is read back from L2 and y stored from registers), xn8 2 x 32 KB, Wab 3 x 16 KB, Ws8 2 x 16 KB, 2 KB of LN exchange.
* Scales / calibration / guard / caches: unchanged (`_pack`, `refresh`, `overflow_peak`, `rescale_all`, `fp8.check()`); the kernel reads the 4-element scale vector `[sx, 1/(sx s1), sh/(sx s1), 1/(sh s2)]` and writes `peak`. Rows not a multiple of 128 are zero-padded and cut back.

### 8.2 The journey (what the built kernel taught; same-process interleaved A/B vs the default route, graph replay, L2 flushed, 0.15 s idle gap, D 128 n 4 262144 rows unless noted; x = e4m3 / default)

| step | change | x |
|---|---|---|
| v1 | the structure above, x stages 2, Wab ring 8, pair squeeze, f32x2 SwiGLU | 0.994 (n 2: 1.215) |
| ablations of v1 (wrong results, timing only) | no tanh / no LayerNorm reductions / no residual loads / all three | 0.936 / 0.892 / 0.934 / 0.742 |
| x stages | 3 / 4 (Wab ring 4, Ws 2) / 5 | 0.930 / 0.917 / 0.946 (n 2: 1.055 / 1.015 / 1.053) |
| `LD64` (one x64 TMEM load, AB released before the math) | | 0.986 vs 0.993 (kept off: register spills) |
| f16x2 SwiGLU (`tanh.approx.f16x2`, cvt e4m3x2 from f16x2) | | 1.070 (slower, different bits): rejected |
| per-chunk squeeze instead of the pair squeeze (D 128) | | 1.030 at 2 stages; 0.952 vs 0.928 at 4 stages: pair squeeze kept at D 128 (per chunk kept at D 256: 0.976 vs 1.042) |
| HS = 32 chunks, 4 [a|b] buffers (more steps in flight) | NAB 4 / 3 | 1.152 / 1.128 (n 2: 1.053 / 1.046): rejected, bit-identical output, per-step handshakes cost more than the depth saves |
| packed f32x2 LayerNorm (`LNP`) | D 128 n 4 / n 2 | 0.928 -> 0.913 / 1.003 -> 0.946 |
| LayerNorm + epilogue split over two warpgroups, SwiGLU on one (`LNH`) | D 128 n 4 / n 2 | 1.298 / 1.065: off at D 128 |
| the same at D 256 n 4 / n 2 | | 0.952 -> 0.879 / 1.425 -> 1.052: on at D 256 |
| D 256 residual from L2 instead of smem (`RESG`) + prefetching the next group's residual | n 4 | 1.07 -> 0.98 -> (with `LNP`) 0.94 |
| 2-CTA B multicast for the chain GEMMs (`PAIRB`) | D 384 n 4 262144 / D 512 / D 768 | 1.055 -> 1.032 / 0.759 -> 0.740 / 0.637 -> 0.618 (opt-in) |

Where the time goes (`clock64` stamps of one CTA, tile 3, `-DPROBE`; `work/n14-f/probe.py`): at D 128 the SwiGLU chunk takes 1.4-1.9 k clk from wake-up to the h hand-off (two warpgroups per tile alternate chunks), a tile ~11 k clk, and the
LayerNorm + epilogue warpgroup is busy ~8.8 k of them (one thread a row: LayerNorm ~3.5 k + epilogue ~3.5 k + waits), i.e. the kernel is bound by the CUDA-core stages and their producer / consumer latencies, not by the tensor pipe
(e4m3 halves only the MMA part); at D 256 the LayerNorm alone was 20 k clk of a 42 k clk tile before the packed LayerNorm / `LNH`. This is why the gains are 7-17 %, not the 1.9x of the MMA.


### 8.3 Results: B200 default route vs the e4m3 route, same process, interleaved, 3 rounds (script `work/n14-f/ab.py`, rows `ab_fused_v3.jsonl` / `ab_disp.jsonl`)

Protocol of 4.3: graph medians (CUDA graph of one call, 256 MB L2 flush before every replay, 30 replays, 0.15 s idle gap before every timed block) + a sustained window (3 s back-to-back replays, `nvidia-smi` 0.2 s, J = W x ms per call, no flush); master (fp32) weights, bf16 x, calibrated eagerly, lazy guard; GPU 2 under `dev.sh -x`,
no foreign process on it; no overflow in any case (peak 14-28 of 448). x columns = e4m3 / default (< 1: e4m3 faster / cheaper). The sustained columns of the 4096-16384 row cases are launch-bound (python replay loop): read them as ratios.

**The fused kernel (D 128 / 256), forced on at every row count** (the dispatcher of 8.6 serves only the rows where it wins):

| D | n | rows | default us (graph, flushed) | e4m3 us | x graph | default us (sustained) | e4m3 us | x sustained | W default / e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 4 | 4096 | 16.4 | 16.4 | 0.998 | 12.3 | 12.3 | 1.000 | 335 / 309 | 0.921 |
| 128 | 4 | 16384 | 16.4 | 16.4 | 1.000 | 12.3 | 12.3 | 1.000 | 603 / 499 | 0.828 |
| 128 | 4 | 65536 | 33.7 | 32.8 | 0.971 | 28.1 | 26.6 | 0.946 | 863 / 716 | 0.786 |
| 128 | 4 | 147456 | 55.3 | 53.0 | 0.958 | 54.0 | 46.2 | 0.857 | 872 / 868 | 0.853 |
| 128 | 4 | 262144 | 87.9 | 81.9 | 0.932 | 96.8 | 82.3 | 0.850 | 878 / 883 | 0.855 |
| 128 | 2 | 4096 | 14.3 | 14.3 | 1.000 | 10.2 | 10.2 | 0.999 | 308 / 292 | 0.947 |
| 128 | 2 | 16384 | 14.3 | 14.3 | 1.000 | 10.2 | 10.2 | 1.000 | 487 / 431 | 0.884 |
| 128 | 2 | 65536 | 26.6 | 26.6 | 0.999 | 22.1 | 20.5 | 0.925 | 708 / 607 | 0.793 |
| 128 | 2 | 147456 | 43.0 | 41.0 | 0.952 | 37.2 | 34.8 | 0.936 | 868 / 738 | 0.796 |
| 128 | 2 | 262144 | 66.1 | 63.1 | 0.956 | 67.2 | 58.5 | 0.871 | 881 / 875 | 0.865 |
| 256 | 4 | 4096 | 30.7 | 28.6 | 0.931 | 26.6 | 24.6 | 0.923 | 396 / 357 | 0.833 |
| 256 | 4 | 16384 | 34.7 | 30.7 | 0.883 | 27.2 | 25.0 | 0.922 | 859 / 683 | 0.733 |
| 256 | 4 | 65536 | 92.2 | 85.9 | 0.932 | 92.7 | 77.7 | 0.839 | 875 / 825 | 0.791 |
| 256 | 4 | 147456 | 167.9 | 157.8 | 0.940 | 194.6 | 161.0 | 0.827 | 891 / 884 | 0.820 |
| 256 | 4 | 262144 | 284.7 | 264.1 | 0.928 | 337.4 | 279.8 | 0.829 | 900 / 893 | 0.823 |
| 256 | 2 | 4096 | 22.5 | 22.5 | 1.000 | 18.4 | 18.4 | 1.001 | 363 / 336 | 0.926 |
| 256 | 2 | 16384 | 24.5 | 24.5 | 1.000 | 18.7 | 20.5 | 1.092 | 708 / 564 | 0.869 |
| 256 | 2 | 65536 | 59.4 | 63.1 | 1.062 | 53.6 | 55.3 | 1.031 | 874 / 725 | 0.854 |
| 256 | 2 | 147456 | 101.9 | 110.6 | 1.085 | 112.9 | 107.0 | 0.948 | 885 / 873 | 0.935 |
| 256 | 2 | 262144 | 163.5 | 181.9 | 1.113 | 192.6 | 184.1 | 0.956 | 893 / 888 | 0.950 |

**Dispatcher: the rows actually served** (D 256 below 12288 rows (n 4) / below 8192 rows (n 2): the three-launch chain, whose small-tile latency beats the default's weight-bound 26.6 / 18.4 us floor; `ab_disp.jsonl`):

| D | n | rows | default us (graph, flushed) | e4m3 us | x graph | default us (sustained) | e4m3 us | x sustained | W default / e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 4 | 1024 | 30.7 | 18.4 | 0.600 | 26.6 | 14.3 | 0.539 | 281 / 319 | 0.611 |
| 256 | 4 | 2048 | 30.7 | 20.5 | 0.667 | 26.6 | 14.3 | 0.539 | 320 / 390 | 0.657 |
| 256 | 4 | 4096 | 30.7 | 22.5 | 0.733 | 26.6 | 16.6 | 0.624 | 394 / 490 | 0.776 |
| 256 | 4 | 8192 | 32.8 | 26.6 | 0.812 | 26.6 | 22.5 | 0.846 | 546 / 602 | 0.931 |
| 256 | 4 | 12288 | 32.7 | 30.7 | 0.938 | 26.7 | 24.6 | 0.923 | 700 / 576 | 0.760 |
| 256 | 4 | 16384 | 34.8 | 30.7 | 0.882 | 27.1 | 25.1 | 0.924 | 859 / 681 | 0.733 |
| 256 | 2 | 1024 | 22.5 | 18.4 | 0.818 | 18.4 | 14.3 | 0.778 | 274 / 292 | 0.826 |
| 256 | 2 | 2048 | 22.5 | 18.4 | 0.818 | 18.4 | 14.3 | 0.778 | 303 / 330 | 0.847 |
| 256 | 2 | 4096 | 22.5 | 18.4 | 0.817 | 18.4 | 14.4 | 0.779 | 362 / 411 | 0.885 |
| 256 | 2 | 8192 | 22.8 | 22.5 | 0.987 | 18.4 | 18.4 | 0.999 | 478 / 503 | 1.051 |
| 128 | 4 | 4096 | 16.4 | 16.4 | 1.000 | 12.3 | 12.3 | 1.001 | 334 / 308 | 0.922 |
| 128 | 4 | 8192 | 16.4 | 16.4 | 1.000 | 12.3 | 12.3 | 1.000 | 422 / 370 | 0.877 |
| 128 | 2 | 16384 | 14.3 | 14.3 | 1.000 | 10.2 | 10.2 | 1.000 | 487 / 429 | 0.882 |

### 8.4 Accuracy against the fp64 gold (`work/n14-f/acc.py`, `acc_fused.json`; judge = max|y - y64| / max|y64|; weights N(0, 1/fan_in), gamma 1 +- 0.1, beta +- 0.1; calibrated on the case; the fused kernel; bf16 default route in the same process)

| D | n | rows | x scale / offset | judge e4m3 | judge default | x (ratio) | rms ratio | update rms e4m3 (default) |
|---|---|---|---|---|---|---|---|---|
| 128 | 4 | 16384 | 1 / 0 | 0.0399 | 0.00372 | 10.7 | 12.8 | 6.6 % (0.52 %) |
| 128 | 4 | 65536 | 1 / 0 | 0.0433 | 0.00364 | 11.9 | 12.8 | 6.6 % (0.51 %) |
| 128 | 2 | 16384 / 65536 | 1 / 0 | 0.0436 / 0.0442 | 0.00412 / 0.00406 | 10.6 / 10.9 | 12.8 | 6.6 % (0.52 %) |
| 256 | 4 | 16384 / 65536 | 1 / 0 | 0.0348 / 0.0339 | 0.00352 / 0.00362 | 9.9 / 9.4 | 12.7 | 6.6 % (0.52 %) |
| 256 | 2 | 16384 / 65536 | 1 / 0 | 0.0389 / 0.0382 | 0.00362 / 0.00364 | 10.8 / 10.5 | 12.7 | 6.6 % (0.52 %) |
| 128 | 4 | 16384 | 1e-3 / 0 | 0.0866 | 0.00600 | 14.4 | 14.8 | 6.5 % (0.44 %) |
| 128 | 4 | 16384 | 1e-3 / 25 | 0.00054 | 0.00054 | 1.0 (x rounds to bf16: the same y) | 1.0 | |
| 128 | 4 | 16384 | 1 / 25 | 0.00889 | 0.00238 | 3.7 | 1.5 | 8.6 % (5.6 %) |
| 128 | 4 | 16384 | 40 / 0 | 0.00293 | 0.00245 | 1.20 | 1.19 | 12.2 % (10.3 %) |
| 128 | 4 | 16384 | 40 / 25 | 0.00280 | 0.00235 | 1.19 | 1.14 | 13.8 % (12.1 %) |
| 256 | 4 | 16384 | 1e-3 / 0 | 0.0722 | 0.00440 | 16.4 | 14.8 | 6.5 % (0.44 %) |
| 256 | 4 | 16384 | 1 / 25 | 0.00916 | 0.00235 | 3.9 | 1.5 | 8.8 % (5.9 %) |
| 256 | 4 | 16384 | 40 / 0 | 0.00300 | 0.00246 | 1.22 | 1.17 | 12.6 % (10.7 %) |
| 256 | 4 | 16384 | 40 / 25 | 0.00269 | 0.00222 | 1.21 | 1.13 | 14.2 % (12.6 %) |

The same band as the H100 ("about 10x the bf16 error", 0.034-0.045 judge, update 6.6 %; every full-e4m3 variant is outside the 8e-3 acceptance tolerance by construction: 3 mantissa bits on both GEMM inputs) and as the chain
(the fused kernel and the chain give the same output to within one e4m3 step on 3e-4 of the elements: `test_fused_error_and_chain_agreement`). x scale 1e-3 (LayerNorm eps-dominated, y - x tiny) is the worst ratio (14-16x), as on the H100.
No NaN / inf, no guard retry or fallback in any case; peak |h| 14-28 of 448 (calibration headroom 16). Saturation protocol: `test_fused_sync_guard_redoes_a_saturated_call[128|256]` (calibrate on x * 1e-4, then the real
input: sync redo with a lowered scale), `test_fused_lazy_guard_check_and_graph_replay[128|256]` (the same through a captured CUDA graph: `fp8.check()` lowers the scale in place, rerun, error back in band, graph replay = eager bytes,
no capture fallback), `test_nonfinite_input_does_not_break_the_guard` (inf in the input: no rescale, default route for the call).

### 8.5 Not served: measured with the BUILT kernels at >= 5 row counts (the user's rule)

* **D 384** (chain, the three-launch kernels `tf8_ln` + `tf8_gemm` expand + `tf8_gemm` squeeze; no fused kernel: the 384-column fp32 output accumulator leaves tensor memory for one 128-column [a|b] buffer and no h): n 4 sustained x1.00 (4096), 1.02 (16384), 1.00 (65536), 1.05 (262144) rows,
  cool graph 0.93 / 1.00 / 0.96 / 1.07; n 2 x1.00 / 1.27 / 1.13 / 1.20 (4096 / 16384 / 65536 / 262144), 1.50 / 1.54 at 1024 rows; energy x0.77 at 4096 rows n 4 (latency floor), 1.0-1.2 above. Levers built from the previous agent's list:
  (i) **CTA-pair B multicast** (`PAIRB`, the "CTA-pair squeeze", applied to expand and squeeze): bit-identical, D 384 n 4 65536 / 262144 rows 0.989 -> 0.979 / 1.055 -> 1.032 (cool graph), n 2 262144 1.248 -> 1.229: the squeeze's L2 intake is NOT the limit (-1..-2 %);
  (ii) e4m3 h with the bf16 residual path = what the chain already does (`resid` reads the bf16 x; the h round trip is e4m3); (iii) expand epilogue pipelining: the expand microbenchmark (`work/n13-fp8-a/mb_table2.txt`) reaches 770 TF/s e4m3 at K = 384 (35 % of the e4m3 peak,
  the plain GEMM with a bf16 output store: e4m3 / bf16 0.79 expand, 0.56 squeeze), i.e. the epilogue-bound tile (K = 384 = 3 K-blocks per tile) is the limit of the GEMM itself; with the measured one-lever gains of 1-2 % the best-effort chain does not beat the default
  (x0.96-1.07 at >= 65536 rows). Not served; `FOLDJET_ENGINE_TRANSITION_FP8_WIDTHS=384 _MINROWS=...` for experiments.
* **D 256 n 2 at >= 8192 rows**: fused kernel sustained 1.09 (16384), 1.03 (65536), 0.95 (147456), 0.96 (262144) rows, cool graph 1.00 / 1.06 / 1.09 / 1.11; chain 1.39 / 1.46 / 1.76 (graph); energy 0.87 / 0.85 / 0.94 / 0.95. Served only below 8192 rows (chain x0.78).
* **D 128 below 4096 rows (n 4) / 16384 (n 2)**: the default is at its 12.3 / 10.3 us launch floor (fused kernel and default both 12.3 / 10.3 us at 1024-8192 rows, x1.00; chain 1.05-1.40).
* **D 64**: not attempted (`wide_d64_fwd_small_f3`-class kernel, the H100 doc itself only bounds it on paper at 0.72-0.76; not in this order's width list).

### 8.6 Gates (served rows; `ROW_GATE`, `ROW_MAX`, `FUSED_MIN_ROWS` in `fp8_sm100a.py`) and re-check of the shipped D 512 / D 768 gates

| (D, n) | served from | fused kernel from | measured (sustained x time / x energy) |
|---|---|---|---|
| (128, 4) | 4096 | 4096 | 1.00 / 0.92 (4096), 1.00 / 0.83 (16384), 0.95 / 0.79 (65536), 0.86 / 0.85 (147456), 0.85 / 0.86 (262144) |
| (128, 2) | 16384 | 16384 | 1.00 / 0.88, 0.93 / 0.79 (65536), 0.94 / 0.80, 0.87 / 0.87 |
| (256, 4) | 1024 | 12288 (chain below) | 0.54 / 0.61 (1024), 0.54 / 0.66 (2048), 0.62 / 0.78 (4096), 0.85 / 0.93 (8192) chain; 0.92 / 0.76 (12288), 0.92 / 0.73 (16384), 0.84 / 0.79 (65536), 0.83 / 0.82 (147456), 0.83 / 0.82 (262144) |
| (256, 2) | 1024 .. 8191 | never fused (chain) | 0.78 / 0.83-0.89 at 1024-4096 rows (chain); >= 8192 not served |
| (512, 4) / (512, 2) / (768, 4) / (768, 2) | 1024 / 4096 / 4096 / 4096 (unchanged) | chain | re-measured on the quiet GPU: table below |

Quiet-GPU re-check of the shipped D 512 / D 768 gates (`dev.sh -x`, GPU 2, no foreign process; `ab_wide_v1.jsonl`): the gates hold (D 512 n 4 0.47 sustained at 1024 rows; D 512 n 2 0.65 at 4096, 0.88 at 16384; D 768 n 4 1.02 at 1024 -> gate 4096 (0.57), D 768 n 2 1.25 at 1024 -> gate 4096 (0.50)):

| D | n | rows | default us (graph, flushed) | e4m3 us | x graph | default us (sustained) | e4m3 us | x sustained | W default / e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 4 | 1024 | 45.1 | 24.6 | 0.545 | 38.9 | 18.4 | 0.473 | 347 / 403 | 0.551 |
| 512 | 4 | 4096 | 55.3 | 32.8 | 0.593 | 47.1 | 27.0 | 0.573 | 585 / 675 | 0.661 |
| 512 | 4 | 16384 | 92.1 | 76.8 | 0.834 | 97.9 | 75.0 | 0.766 | 828 / 835 | 0.772 |
| 512 | 4 | 65536 | 311.3 | 241.7 | 0.776 | 369.8 | 286.0 | 0.773 | 867 / 853 | 0.761 |
| 512 | 4 | 262144 | 1255.4 | 956.4 | 0.762 | 1454.4 | 1113.9 | 0.766 | 900 / 891 | 0.758 |
| 512 | 2 | 4096 | 38.9 | 24.5 | 0.630 | 31.3 | 20.3 | 0.650 | 522 / 556 | 0.692 |
| 512 | 2 | 16384 | 57.3 | 55.3 | 0.964 | 53.5 | 47.0 | 0.878 | 821 / 797 | 0.853 |
| 512 | 2 | 65536 | 176.9 | 163.2 | 0.922 | 206.4 | 177.4 | 0.860 | 856 / 854 | 0.857 |
| 512 | 2 | 262144 | 657.4 | 604.2 | 0.919 | 767.2 | 692.2 | 0.902 | 885 / 877 | 0.893 |
| 768 | 4 | 1024 | 32.8 | 32.8 | 1.000 | 26.6 | 27.1 | 1.016 | 643 / 473 | 0.748 |
| 768 | 4 | 4096 | 83.8 | 49.2 | 0.587 | 75.4 | 42.8 | 0.568 | 824 / 813 | 0.560 |
| 768 | 4 | 16384 | 202.8 | 133.1 | 0.657 | 229.1 | 145.1 | 0.633 | 856 / 846 | 0.626 |
| 768 | 4 | 65536 | 740.3 | 464.9 | 0.628 | 875.9 | 552.6 | 0.631 | 877 / 877 | 0.631 |
| 768 | 2 | 1024 | 22.4 | 26.6 | 1.188 | 16.4 | 20.5 | 1.250 | 570 / 404 | 0.886 |
| 768 | 2 | 4096 | 67.4 | 34.8 | 0.517 | 57.4 | 28.7 | 0.500 | 558 / 693 | 0.620 |
| 768 | 2 | 16384 | 112.6 | 92.1 | 0.818 | 117.3 | 88.9 | 0.757 | 833 / 829 | 0.753 |
| 768 | 2 | 65536 | 380.8 | 288.8 | 0.758 | 440.8 | 334.4 | 0.759 | 862 / 857 | 0.754 |
| 768 | 2 | 262144 | 1488.9 | 1128.5 | 0.758 | 1708.2 | 1301.0 | 0.762 | 898 / 896 | 0.759 |

D 384 and D 256 on the three-launch chain, forced at every row count (`ab_chain_v1.jsonl`; the D 256 rows show why the fused kernel is used from 12288 rows):

| D | n | rows | default us (graph, flushed) | e4m3 us | x graph | default us (sustained) | e4m3 us | x sustained | W default / e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 4 | 4096 | 30.7 | 22.5 | 0.734 | 26.6 | 16.4 | 0.616 | 381 / 469 | 0.757 |
| 256 | 4 | 16384 | 34.8 | 38.9 | 1.118 | 26.9 | 32.8 | 1.220 | 810 / 711 | 1.070 |
| 256 | 4 | 65536 | 92.5 | 112.6 | 1.218 | 92.6 | 110.5 | 1.193 | 833 / 828 | 1.187 |
| 256 | 4 | 147456 | 168.0 | 229.4 | 1.365 | 194.8 | 244.8 | 1.257 | 856 / 856 | 1.257 |
| 256 | 4 | 262144 | 284.7 | 387.1 | 1.360 | 337.7 | 430.3 | 1.274 | 859 / 864 | 1.281 |
| 256 | 2 | 16384 | 24.5 | 32.8 | 1.335 | 18.8 | 26.6 | 1.413 | 669 / 573 | 1.212 |
| 256 | 2 | 65536 | 57.4 | 84.0 | 1.463 | 53.5 | 77.8 | 1.455 | 825 / 771 | 1.359 |
| 256 | 2 | 262144 | 161.8 | 284.7 | 1.759 | 191.9 | 292.7 | 1.526 | 854 / 850 | 1.518 |
| 384 | 4 | 1024 | 16.4 | 24.6 | 1.500 | 12.3 | 18.9 | 1.537 | 505 / 343 | 1.045 |
| 384 | 4 | 4096 | 30.7 | 28.7 | 0.933 | 24.6 | 24.6 | 0.999 | 693 / 535 | 0.771 |
| 384 | 4 | 16384 | 59.4 | 59.4 | 1.000 | 51.1 | 52.0 | 1.019 | 821 / 820 | 1.018 |
| 384 | 4 | 65536 | 186.4 | 178.2 | 0.956 | 196.0 | 195.8 | 0.999 | 849 / 858 | 1.010 |
| 384 | 4 | 262144 | 622.6 | 667.6 | 1.072 | 726.7 | 765.4 | 1.053 | 875 / 886 | 1.066 |
| 384 | 2 | 4096 | 24.6 | 26.6 | 1.083 | 20.5 | 20.5 | 1.000 | 534 / 444 | 0.833 |
| 384 | 2 | 16384 | 39.0 | 47.1 | 1.207 | 30.7 | 38.9 | 1.268 | 807 / 670 | 1.053 |
| 384 | 2 | 65536 | 116.7 | 127.1 | 1.088 | 112.9 | 128.0 | 1.134 | 829 / 833 | 1.139 |
| 384 | 2 | 262144 | 362.5 | 458.8 | 1.266 | 418.0 | 502.3 | 1.202 | 865 / 868 | 1.206 |

### 8.7 Verification

* FP8 off (every switch unset): the B200 default route is bit-identical to `n14-base`: `work/n14-f/shaf.py` (23 inference cases, eager AND captured graph, separate processes: bf16 D 64 / 128 n 4 / 2 (incl. 16421 ragged rows, 65536) / 256 n 4 / 2 / 384 / 512 / 768, and the fp32-activation TF32 / H16 generic builds D 64 / 128 n 4 / 2 / 256 / 384 / 512) 23 / 23 identical;
  training forward / backward at D 128 n 4 / n 2, D 256, D 384, D 512: y, x-grad and every parameter gradient identical except ONE parameter gradient (the last) of each case, which also differs between two runs of the BASE tree (`sha_base.json` vs `sha_base2.json`: atomic accumulation order, the known behaviour).
  `n1cfg/poison.py` (eager and captured-graph runs on 0xFF-poisoned allocator memory, 14 shapes D 384 / 512 / 768): ALL BITWISE EQUAL. Non-sm100 CUDA sources are byte-identical to `n14-base`; `backend_versions` unchanged (the default route does not change).
* Tests: `tests/gpu/test_transition_fp8_b200.py` 48 passed (24 existing + 24 new); the combined run (this file, `test_fp8_switch`, `test_capture_api`, `test_prebuild_plan`, `test_trimul_fp8_b200_cpu`, `test_trimul_fp8_b200`, `test_engine_fp8_backend`, `test_transition_b200_uninit`, `test_trimul_b200`,
  `test_triattn_b200`, `test_triattn_b200_capture_bits`, `test_misc_b200`, `test_local_dit_b200`): 517 passed, 7 skipped (H100-only), 2 min.
* `_prebuild --arch sm100 --check`: full output `work/n14-f/prebuild_check_f.txt` (base `prebuild_check_base.txt`): 813 -> 821 artifacts at the time of that run (the 4 fused cubins + the 4 `PAIRB` cubins; the `PAIRB` cubins are planned only with `FOLDJET_ENGINE_TRANSITION_FP8_PAIRB=1` in the final tree: 817), native sources compiled by a planned artifact
  216 -> 217; rc = 1 on both trees (pre-existing H100-family notes), no other line differs.

### 8.8 Open items (exact)

* D 256 n 2 above 8192 rows and D 128 n 2 / n 4 gains at the smallest row counts are bounded by the fixed per-tile critical path (LayerNorm of tile i + 1 and the epilogue of tile i share warpgroups; the output accumulator is single-buffered): the next lever is a second output accumulator
  (needs the 512 TMEM columns D 128 does not have free: AB 256 + h 64 + OUT 2 x 128 = 576) or the epilogue on its own warpgroup with the LayerNorm in the producer.
* D 384 fused (tensor memory: 384 output columns + [a|b]): not built; the chain is within 1.00-1.07 of the default at >= 65536 rows.
* D 64, D 1024 fused: not built (D 1024 stays on the shipped chain; D 64 is not a multiple of 128).
* End-level (structure) accuracy of the e4m3 Transition on the B200 was not measured (module level only).
* `FOLDJET_ENGINE_FP8=safe` now serves the Transition at D 128 / 256 as well (WIDTHS default `128,256,512,768`): revert to `512,768` in `WIDTHS` to keep the 3.0.3 `safe` scope on B200 unchanged.

# Agent B: TriMul FP8 (B200)

# B200 FP8 (e4m3) -- opt-in routes (docs/b200/fp8.md)

FP8 on B200 is DEVELOPED for tcgen05, not ported (docs/fp8.md of the H100 engine is the recipe reference: quantisation points, scale granularity, saturation
protocol, gates). Everything here is OPT-IN and OFF by default: with the switches unset the B200 default routes are bit-identical to the tree without it.

## 2026-11-19 -- TriMul e4m3 planes (agent B, branch tree n13-fp8-b)

### Design (what is e4m3, what is not)
| tensor | dtype | scale | where made |
|---|---|---|---|
| planes a / b (front output) | e4m3 | one power-of-two per TENSOR: a and b (per direction in the bidirectional block: 4 scales), static, HEADROOM 16 | `k1w_f8.cu`: plane value * `qinv[chunk]` -> `cvt.rn.satfinite.e4m3x2`, max |scaled value| -> guard (per group atomicMax) |
| contraction accumulator | fp32 (TMEM) | - | `ctr8.cu` `tcgen05.mma kind::f8f6f4` (e4m3 x e4m3, K = 32 per MMA, M128 N128/256) |
| t (contraction output) | bf16 | scale `sa sb` applied in ctr8's epilogue (`csc[2]` device scalars, exact: powers of two) | `ctr8.cu` |
| LN_in / projections / gates / K3 (LN_out, W_o, gate, residual) | bf16 / fp32 | - | unchanged B200 kernels (k3g / k3h / k3hf / k3w) -- K3 sees t at its TRUE scale, no folded operands |
Weights stay bf16 (e4m3 projection operands cost 0.8-3 % update error each on H100: accuracy map in docs/fp8.md; not built). t is not e4m3 (H100: tried, no gain).

Chain: `k1w8` (front: LN_in + the four projections + gate + mask, e4m3 plane store) -> `ctr8` (contraction, PDL-chained exactly like `ctr`) -> default K3.
The B200 cell logic (k1w FOLD / plain, k3hf / k3h / k3g / k3w, XF / OST flags, PDL triggers) is the bf16 chain's: `b200_fp8._body` mirrors `b200_infer._inference_body`
(k1h / M64 fronts and the one-launch L128 kernels are not used on the FP8 route).

Design decisions forced by tcgen05 (the H100 recipe otherwise transfers unchanged):
* **qinv / csc are device arrays read at run time** (the H100 pre-scales the packed weights per calibration). `rescale_all` / `refresh` update them IN PLACE, so captured graphs follow.
* **sa sb is applied in the contraction epilogue** (the H100 folds it into K3's projection operand because cuBLASLt returns the raw sum). K3 is untouched. RoseTTAFold3's length factor f rides on `csc[0]` (`lfac` from the mask at run time: one calibration for any length).
* **Token-shuffled plane layout (the trick that makes the e4m3 front epilogue cheap):** the front's TMEM fragment gives a thread tokens tr and tr + 8; those two bytes are ONE 16-bit element,
  so a single `stmatrix.trans` (+ 4 PRMT) writes the channel-major byte rows instead of a lane exchange + byte stores (first version: k1w8 +35-49 % over k1w; this: +4-12 %).
  The price: inside every aligned 16-token group the plane's contiguous axis is stored in the order p -> token (p >> 1) + 8 (p & 1). The contraction takes both operands in the same order for free
  where the contiguous axis is the contracted one (outgoing: k); for the incoming direction (the contiguous axis is the output index) ctr8's epilogue undoes it (staging row = natural row; the accumulator columns 2n / 2n + 1 become columns n / n + 8 of each 16-group). The bf16 chain is untouched.
* e4m3 MN-major operands (incoming direction) are legal for `kind::f8f6f4`: the same 128 B swizzle box [128 k rows][128 B], LBO = one box, 4096 B per K32 MMA -- no transposed front (the H100 needs one: wgmma e4m3 takes K-major only).
* **Any L:** ctr8 uses 3-D tensor maps ([plane][row][col]) so tiles clip at L (out-of-range rows / columns read zero and are not stored); L % 16 != 0 is padded on the host with masked tokens like the default (`_padded_inference`). The route declines lengths far from a multiple of 128 (tile quantisation: L400 D256 measured +5 % slower than bf16): `L / ceil128(L) >= 0.9`.

### Scales, calibration, guard (the H100 contract)
* First eager call of a module: a probe launch of the front with qinv = 1 gives the exact amax per group (a, b; per direction), `pow2_scale` (smallest 2^e with amax / 2^e <= 448, floor 2^-30, all-zero -> 1) x HEADROOM 16 (`FOLDJET_ENGINE_TRIMUL_FP8_HEADROOM`); static afterwards, cached per module (key: data_ptr of `to_left.weight`, dropped by `weakref.finalize`).
* `GUARD`: `sync` (default; eager: the call reads the peak, > 448 -> scales x 2^ceil(log2(peak / 448 x 2)), refresh, redo <= 4 times, then the bf16 route), `lazy` (capturable; kernel atomically maxes the scaled peak into a per-device buffer), `off`. A capture before any calibration takes the bf16 route (`capture_fallbacks`).
* `overflow_peak()` (one host sync, once per forward outside graphs), `rescale_all(peak)` (in place; captured graphs follow; tested: stale scales 64x too small -> peak > 448 on replay -> `rescale_all` -> the SAME graph is accurate again).
* LN_in makes the planes invariant to the input scale and offset: input stress (x scale 1e-3 .. 40, offsets 0 / 25) never saturates (peak 17-23 of 448 at 16x headroom); saturation can only come from stale scales (weights changed after calibration) or heavy outlier channels.
* Switches (same names as H100, read via `fp8.env`): `FOLDJET_ENGINE_TRIMUL_FP8` {calib, dynamic}, `_GUARD`, `_HEADROOM`, `_MINL` (replaces every gate, 0 = none). Counters: `b200_fp8.GUARD_STATS` (calls, retries, fallbacks, capture_fallbacks, calibrations). Registration for `fp8.py`: `work/n13-fp8-b/FP8_REGISTER.md`.

### Served
bf16 master modules, inference (no grad, no live dropout), TriangleMultiplication outgoing / incoming, Bidirectional, RF3 (D <= 128, one sample), D 64 / 128 / 256 / 384 / 512 with the bf16 route's hidden widths, any L with `L / ceil128(L) >= 0.9` and L >= MIN_L (below the gate the bf16 route runs unchanged), B up to one default launch.
MIN_L: see the follow-up section (D64 out/in 512, D64 bidir 384, D128 512, D256 out/in 384 and bidir 256, D384 512, D512 128).

### Results (module level, CUDA-graph replay, B = 1, bf16 master, `bench_mod.py`; interleaved A/B in ONE process, 3 rounds)
`gap` = the first replay after 0.15 s idle (the gap measurer's regime); `sus` = back-to-back replays for 1.5 s per arm (clock / power capped), power from nvidia-smi every 0.2 s; energy ratio = (sus time x mean power) FP8 / bf16.
| D | dir | L | gap bf16 / FP8 | gap ratio | sus bf16 / FP8 | sus time ratio | power bf16 / FP8 | sus energy ratio |
|---|---|---|---|---|---|---|---|---|
| D128 | out | 128 | 18.0 / 17.7 us | 0.981 | 17.2 / 16.9 us | 0.984 | 500 / 468 W | 0.919 |
| D128 | out | 256 | 41.4 / 41.0 us | 0.990 | 39.2 / 38.2 us | 0.975 | 717 / 652 W | 0.886 |
| D128 | out | 512 | 133.3 / 129.8 us | 0.974 | 144.4 / 133.6 us | 0.925 | 819 / 801 W | 0.905 |
| D128 | out | 1024 | 546.4 / 512.9 us | 0.939 | 630.5 / 576.6 us | 0.915 | 846 / 814 W | 0.880 |
| D128 | out | 2048 | 2787.3 / 2375.5 us | 0.852 | 3209.0 / 2715.0 us | 0.846 | 880 / 850 W | 0.818 |
| D128 | bidir | 128 | 25.0 / 23.8 us | 0.950 | 21.8 / 21.9 us | 1.006 | 578 / 551 W | 0.959 |
| D128 | bidir | 256 | 61.6 / 60.5 us | 0.982 | 61.3 / 57.9 us | 0.945 | 824 / 808 W | 0.927 |
| D128 | bidir | 512 | 213.6 / 200.3 us | 0.938 | 243.7 / 223.7 us | 0.918 | 824 / 842 W | 0.939 |
| D128 | bidir | 1024 | 1010.3 / 927.4 us | 0.918 | 1168.9 / 1042.0 us | 0.891 | 854 / 852 W | 0.889 |
| D128 | bidir | 2048 | 5222.2 / 4396.7 us | 0.842 | 6085.9 / 5043.5 us | 0.829 | 933 / 897 W | 0.796 |
| D64 | out | 128 | 12.5 / 14.4 us | 1.153 | 11.7 / 13.6 us | 1.163 | 417 / 371 W | 1.035 |
| D64 | out | 256 | 25.0 / 24.7 us | 0.989 | 22.9 / 23.4 us | 1.022 | 549 / 534 W | 0.994 |
| D64 | out | 512 | 67.2 / 65.1 us | 0.968 | 66.5 / 62.3 us | 0.938 | 778 / 826 W | 0.995 |
| D64 | out | 1024 | 257.8 / 244.3 us | 0.948 | 298.6 / 268.5 us | 0.899 | 835 / 808 W | 0.871 |
| D64 | out | 2048 | 1342.1 / 1129.4 us | 0.842 | 1533.3 / 1289.3 us | 0.841 | 857 / 841 W | 0.825 |
| D64 | bidir | 128 | 15.0 / 15.7 us | 1.044 | 14.4 / 14.9 us | 1.036 | 457 / 426 W | 0.966 |
| D64 | bidir | 256 | 32.8 / 32.7 us | 0.999 | 30.8 / 30.2 us | 0.981 | 686 / 604 W | 0.865 |
| D64 | bidir | 512 | 108.7 / 98.0 us | 0.902 | 112.1 / 98.5 us | 0.879 | 814 / 819 W | 0.885 |
| D64 | bidir | 1024 | 431.9 / 395.6 us | 0.916 | 500.3 / 446.0 us | 0.892 | 821 / 834 W | 0.905 |
| D64 | bidir | 2048 | 2383.6 / 1947.7 us | 0.817 | 2746.2 / 2217.0 us | 0.807 | 858 / 864 W | 0.812 |
| D256 | out | 128 | 29.7 / 29.5 us | 0.995 | 27.2 / 27.0 us | 0.993 | 687 / 679 W | 0.982 |
| D256 | out | 256 | 82.8 / 82.4 us | 0.996 | 88.2 / 85.6 us | 0.970 | 823 / 812 W | 0.958 |
| D256 | out | 512 | 299.4 / 282.2 us | 0.943 | 344.5 / 323.2 us | 0.938 | 831 / 841 W | 0.949 |
| D256 | out | 1024 | 1285.2 / 1177.2 us | 0.916 | 1497.6 / 1357.9 us | 0.907 | 856 / 839 W | 0.888 |
| D256 | out | 2048 | 6373.0 / 5466.4 us | 0.858 | 7455.9 / 6346.0 us | 0.851 | 921 / 898 W | 0.830 |
| D256 | bidir | 128 | 42.1 / 41.7 us | 0.991 | 39.3 / 38.3 us | 0.974 | 802 / 766 W | 0.930 |
| D256 | bidir | 256 | 140.7 / 129.3 us | 0.919 | 152.3 / 139.1 us | 0.913 | 845 / 800 W | 0.865 |
| D256 | bidir | 512 | 491.7 / 453.0 us | 0.921 | 569.1 / 522.6 us | 0.918 | 814 / 829 W | 0.936 |
| D256 | bidir | 1024 | 2229.8 / 2003.9 us | 0.899 | 2613.3 / 2321.9 us | 0.889 | 860 / 873 W | 0.902 |
| D256 | bidir | 2048 | 11835.4 / 9770.8 us | 0.826 | 13602.0 / 11322.5 us | 0.832 | 978 / 954 W | 0.813 |
| D512 | out | 128 | 69.6 / 67.0 us | 0.962 | 68.9 / 64.8 us | 0.940 | 797 / 813 W | 0.958 |
| D512 | out | 256 | 241.1 / 227.5 us | 0.944 | 268.1 / 251.3 us | 0.937 | 808 / 832 W | 0.965 |
| D512 | out | 512 | 895.3 / 841.9 us | 0.940 | 1028.7 / 970.0 us | 0.943 | 857 / 839 W | 0.923 |
| D512 | out | 1024 | 3947.3 / 3563.6 us | 0.903 | 4547.3 / 4109.4 us | 0.904 | 902 / 889 W | 0.891 |
| D512 | out | 2048 | 19642.7 / 16727.1 us | 0.852 | 21334.4 / 18421.6 us | 0.863 | 996 / 998 W | 0.865 |

More cells (same protocol, SUS 2 s): 
| case | gap bf16 / FP8 | gap ratio | sus time | sus energy |
|---|---|---|---|---|
| D384 out L512 | 550.9 / 519.1 us | 0.942 | 0.940 | 0.921 |
| D384 bidir L512 | 951.1 / 890.7 us | 0.937 | 0.928 | 0.940 |
| D384 out L1024 | 2403 / 2212 us | 0.921 | 0.911 | 0.901 |
| D128 out L3072 | 7717 / 6807 us | 0.882 | 0.818 | 0.801 |
| D256 in L3072 | 18441 / 14718 us | 0.798 | 0.824 | 0.824 |
| D128 out L1000 (padded 1008) | 1182 / 1179 us | 0.998 | 0.998 | 0.923 |
| D128 out L300 (padded 304), D256 out L400 | 116 / 121 us, 206 / 221 us | 1.036, 1.073 | 1.046, 1.011 | 0.965, 1.003 | (declined by the 0.9 tile-efficiency gate)

Reading: the gain is 5-10 % from L512, 10-18 % at L1024-3072 (time and sustained energy); at L128-256 level (D128 / D256 / D512 gain 1-6 %, D64 one direction loses 2-16 % -> gate 512).
It is smaller than the H100's (-15 .. -25 %) because (a) the B200 bf16 contraction is already at 1.3-1.45 PFLOP/s sustained, (b) the power cap: FP8 runs ~2.0 PFLOP/s in ctr8 at the same ~900-950 W, (c) the e4m3 front epilogue costs 4-12 % over k1w.
Per-kernel device time (profiler, graph-free eager, us; D128 out L1024): k1w 164 -> k1w8 183, contraction (cuBLAS nvjet) 175 -> ctr8 144, k3g 209 -> 196 (data-dependent power); D256 out L1024: k1w 444 -> 459, 379 -> 308, k3w 454 -> 430. D128 L512: 48.5 -> 56.3, 34.4 -> 28.4, 53.4 -> 53.6.
Contraction alone (microbenchmark, 20-launch graphs, burst, L2-warm): ctr8 reaches 1.7-2.0 PFLOP/s at L >= 1024 against cuBLAS bf16 1.35-1.45 (D128 L1024 138 vs 190 us, L2048 1089 vs 1631 us), 1.2-1.3 PF at L512 (26.4 vs 33.6 us), L384 15.9 vs 19.4 us, L256 level, L128 4.1 vs 3.3 us (cuBLAS) / 4.6 us (ctr bf16). cfg KB = 2 (256 k a stage) is slower at every L: default KB 1, BN 256 (BN 128 when 256 would waste a tile column).

### Accuracy against the float64 module (random N(0, 1/fan_in) weights, 5 masked tokens; update error = rms(y - y64) / rms(y64 - x))
Benign inputs (x ~ N(0, 1)); FP8 = static calibration on the same call, HEADROOM 16:
| case | bf16 default | FP8 | ratio | peak scaled plane value (of 448) |
|---|---|---|---|---|
| D128 out L256 / in L256 / bidir L256 | 0.0058 / 0.0058 / 0.0054 | 0.0379 / 0.0379 / 0.0378 | 6.5 / 6.5 / 7.0 | 19.1 / 16.7 / 20.1 |
| D64 out L256 / bidir L256 | 0.0059 / 0.0060 | 0.0381 / 0.0379 | 6.5 / 6.4 | 17.3 / 17.7 |
| D256 out L256 / bidir L256 | 0.0059 / 0.0060 | 0.0379 / 0.0378 | 6.5 / 6.4 | 18.5 / 18.0 |
| D384 out L256, D512 out L256 | 0.0058, 0.0064 | 0.0379, 0.0379 | 6.5, 5.9 | 19.1, 19.1 |
| D128 out L384 / L512 | 0.0058 / 0.0066 | 0.0379 / 0.0380 | 6.5 / 5.8 | 19.1 |
(the H100 e4m3 planes: 3.7-3.8 % update error, bf16 0.6 %: the error is the e4m3 storage of the planes, the same on both cards: tcgen05's fp32 accumulation adds nothing visible.) It fails the engine's 8e-3 module form by design (FP8 is outside the judge, as on H100).
Stress (x scale {1e-3, 1, 40} x mean offset {0, 25}, FP8 calibrated on the stressed input; `work/n13-fp8-b/stress_table.txt`, all widths, both directions, bidirectional):
x scale 1e-3, offset 0: bf16 0.0045-0.0052, FP8 0.0377-0.0381 (ratio 7.3-8.4); x 1, offset 0: 0.0054-0.0064 / 0.0378-0.0382 (5.9-7.0); x 1, offset 25: 0.0475-0.0673 / 0.0604-0.0771 (1.15-1.27); x 40, offset 0: 0.087-0.122 / 0.094-0.128 (1.05-1.09); x 40, offset 25: 0.102-0.143 / 0.108-0.148 (1.03-1.07).
At large activations / offsets both routes are limited by the bf16 rounding of y itself (the update is small against y) so the ratio -> 1; x 1e-3 with offset 25 is degenerate (constant rows in bf16: the update is exactly zero, 0 / 0 in the metric; all outputs finite). No NaN, no inf, no saturation (peak 17-23).

### Verification
* FP8 off (POLICY None) vs the pre-change tree `n13-base`, separate processes, 24 cases (widths 64-512 both modules, L 128-1024 incl. dropout p 0.25 eval): eager **24 / 24 outputs bitwise equal**, CUDA-graph replay **24 / 24 equal** (`work/n13-fp8-b/bit_*.pt`, `bitg_*.pt`, xchk / xchk_g / xcmp of `n2-trimul/_tools`). The default extension `trimul_b200_v6` is untouched (its sources are byte-identical); the hook in `integrations/trimul_b200.forward` is skipped when `b200_fp8.POLICY` is None.
* `tests/gpu/test_trimul_fp8_b200.py`: 38 passed (accuracy vs float64 and vs the bf16 route for D64-D512 both directions and bidirectional at L256 / 384, hidden = 2 D forms, padded lengths 1000 / 1008, lengths far from a tile multiple keep the default; stress inputs; saturation + `rescale_all` through a captured graph; graph == eager and capture-before-calibration fallback; default untouched and gates; RF3 length factor with one calibration, also through a captured graph after the mask changed; B = 3 equal to per-sample calls). `tests/gpu/test_trimul_b200.py`: 179 passed. `tests/cpu -k "fp8 or prebuild"`: 60 passed.
* `_prebuild --arch sm100 --check` (full output `work/n13-fp8-b/prebuild_check.txt`, base `prebuild_check_base.txt`): rc 1 in both (the pre-existing "no compiled kernel matches" / "compiled only by other recipes" notes, identical); the plan goes 804 -> 805 artifacts (+ `trimul_b200_fp8_v1` ext), call sites 221 -> 222, native sources 530 -> 533.

### Rejected / measured no (B200, this session)
* Separate quantise kernel (bf16 planes -> e4m3): extra 3 B / plane element of traffic = the whole contraction saving at L1024 (planes pass 0.19 ms vs 0.2 ms saved) -- the quantisation must be in the front's epilogue.
* First front epilogue (lane exchange + 16-bit byte stores, 32-token boxes): k1w8 +35 % (L1024) / +49 % (L2048); 64-byte-row boxes + the stmatrix shuffle: +4.5-12 %.
* ctr8 with 256 k a stage (KB 2): 10-30 % slower than KB 1 at L >= 512.
* Atomics / amax of the guard: ablation (atomics off, amax off) moved k1w8 by 0-3 % (noise); kept, flushed once per guard group instead of per chunk.
* Not built: e4m3 weights in k1w / k3w (projection operands: 0.8-3 % update error each, H100 accuracy map), e4m3 t, per-channel scales (pow2 rounding: no accuracy gain), training.

### Open items
* k1w8 is 4-18 % slower than k1w (D256 the most): the epilogue is the cost centre; the next levers are fewer cvt / amax instructions (a packed fp16 max) and the weight-stream-bound D256 front.
* Small L (<= 256): FP8 is level (the chain's fixed latencies dominate); the one-launch L128 kernels and the M64 fronts have no e4m3 twin. D64 one direction is gated at L512.
* The gain at L512-1024 (7-10 %) is bounded by the front (k1w8) and K3, neither of which gets faster with e4m3 operands; a gate-projection-only e4m3 front (H100 emulation: +0.3-0.5 % error) is the next candidate but needs two MMA chains per weight block.
* `fp8.py` registration (agent A): `FP8_REGISTER.md`. B > one default launch (chunked batches), fp32 and training stay on the bf16 route.

## 2026-11-19 (2) -- follow-up: front epilogue cost, preset path, gates (n13-fp8-b)
* **Count first**: `cuobjdump -res-usage` of `k1w8` D128 P256: 126 registers, no stack / local (no spills); SASS of the D128 P256 kernel 3.8k instructions against 2.2k of k1w (IMAD 593 vs 292, LOP3 364 vs 168, FMUL2 160 vs 96, F2FP 96 vs 64, FMNMX3 32: the guard amax is already one 3-input max per two values; the cvt is native `F2FP` e4m3x2; no SHFL / STSM beyond the baseline; files `work/n13-fp8-b/sass/`).
* **Ablations** (7 extension variants in ONE process, interleaved, min of 8 rounds; k1w8 / k1w, D128 P256 L1024 / D256 P512 L1024; a colleague's job shared GPU 1, so absolute times were 2x the quiet ones): full epilogue 1.025 / 1.005; guard flush off 1.009 / 0.943; amax off 1.010 / 0.944; qinv multiply off 0.977 / 0.999; cvt off 1.033 / 1.004; stmatrix off 0.993 / 0.930; scale as packed FMUL2 1.015 / 1.006. Reading: with the lanes' token shuffle the e4m3 front is within +0.5-2.5 % of k1w in isolation (the +4-18 % of the first measurement was clock / foreign-load noise); the only kept change is the packed FMUL2 scale (`F8_ABL` 128, default). The qinv multiply could be folded into the packed weights (H100 way) for another ~1-4 % of the FRONT = ~1 % of the module: not done (it forces a weight copy per rescale).
* **Gate-projection-only e4m3 front (H100 emulation: +0.3-0.5 % update error on top of the planes, 4.25 % vs 3.72 %)**: NOT implemented. It needs LN_in(z) quantised to e4m3 plus W_ag / W_bg e4m3 in a second MMA chain with a second xn image in TMEM per weight block (k1w's TMEM columns are full at D128: 3 x 128 accumulators + xn) -- a front rewrite, and the front is only 25-35 % of the module with the MMA a part of it; the upper bound (half of the gate chain at e4m3 rate) is at most ~3-5 % of the module at D256+ and ~1-2 % at D128 (H100's own estimate: -7 % of the module at D256 L2048). Below the cost of the rewrite for what is left of the session.
* **Gates re-measured** (graph replay, same protocol, GPU shared with a foreign job: +-3 %): D128 out L384 1.004 gap / 0.968 energy, D128 bidir L384 1.025 / 0.938, D256 out L384 0.969 / 0.975, D64 out L384 0.986 / 0.948, D64 out L512 0.980 / 0.861, D64 bidir L256 1.025 / 0.870, D64 bidir L384 0.976 / 0.939. New MIN_L = served only where the gap ratio is <= 0.99: D64 out / in 512, D64 bidir 384, D128 every direction 512, D256 out / in 384, D256 bidir 256, D384 512, D512 128 (`b200_fp8.MIN_L`).
* **`FOLDJET_ENGINE_FP8=safe` preset path verified end to end** with agent A's fp8.py (copy of the tree n13-fp8-bx = n13-fp8-b + A's `fp8.py` / `fp8_sm100a.py`, `work/n13-fp8-b/dev/py/t_preset.py`): `fp8.arch()` = sm100, `status()["trimul"]` = calib, `_modules()[1]` = `b200_fp8`; eager calls take the route (`route_stats()["trimul"]` calls 2, calibrations 1, fallbacks 0); a captured graph takes the route (`capture_fallbacks` 0, graph == eager bitwise); `check()` False when clean, True after the scales were made 64x too small (the replay saturated), False after the in-place rescale; `fp8.disable()` returns to the bf16 route.


# Agent C: token DiT / atom DiT FP8 (B200)

# FP8 (e4m3) on B200 (sm_100a): token DiT and atom DiT inference

Opt-in, OFF by default; the default route is bit-identical to the pre-change tree (section "Verification"). Product code: `kernels/conditioned_transition/cuda/token_dit_fp8_b200.py`,
`gemm_swiglu_f8_sm100.cu`, `token_dit_fp8_bind_sm100.cu` (+ the arch-neutral 3.0.3 `token_dit_fp8_rows.cu`, unchanged); call sites in `token_dit_fp8.py` (`ext()`, `active(..., b200=True)`) and
`conditioned_transition/cuda/__init__.py` (`f8` selection, the `and not b200` guard removed). Agent C, 2026-11-19, engine 3.0.3 + the B200 kernel set (branch b200/3.0.3).

## 2026-11-19 -- token DiT: e4m3 expand (+ SwiGLU) on tcgen05; atom DiT: measured, not built

### H100 design that was reused unchanged (digest: `$W/b2511/work/n13-fp8-c/h100_digest.md`)
Quantisation recipe of 3.0.3: activations get a DYNAMIC per-row scale (row max / 448) made by the row pass that produces the row (the quantising AdaLN passes `resgate_adaln_cf_q8`,
`adaln_in_rows_cf_par_q8` for Chai-1's identity block: the H100 `token_dit_fp8_rows.cu`, compiled for sm_100a as it is), weights a static per-output-channel scale (max / 448, once per weight pack);
no calibration, no overflow guard, no `check()` hook (a row's own maximum maps to 448: nothing saturates whatever a diffusion step's noise). Forms: default block and Chai-1's identity /
parallel block; `qkvg` never for the identity AdaLN. `safe` = the expand placement only. Counters: `route_stats()` keys unchanged.

### What was built (B200 specific)
`gemm_swiglu_f8_sm100.cu` = `gemm_swiglu2_sm100.cu` (2-CTA tcgen05, (tile pair, 128-unit chunk) items, TMEM double buffer, h staged for TMA stores) with e4m3 operands:
* X (e4m3 [M, 768], row scale sx[M]) and Wa / Wb (e4m3 [H, 768], channel scales) stream in 128-element K-blocks (one 128 B swizzle row = the same 32 KB per stage as bf16): NKB = 6, four
  `tcgen05.mma.cta_group::2.kind::f8f6f4` (M 256, N 256, K 32) steps per block, fp32 accumulators in TMEM (the unscaled e4m3 product).
* the epilogue applies the scales in fp32 before silu: `a = acc_a sx[row] swa[unit]`, `b = acc_b sx[row] swb[unit]`, `h = bf16(a sigmoid(a) b)` (TMEM lane = row, so the row scale is one
  register; the unit scales are float2 loads from L1). The sigmoid is `tanh.approx` (`SIG_TANH`, -7 % kernel time, error 2^-11 against 5 % of e4m3 operand error).
* h stays bf16 (the squeeze GEMM is the bf16 one); the e4m3 weights are made once per weight pack (`_capture.lookup`, like every pack).
Gates (`token_dit_fp8.active(..., b200=True)`): bf16 only, D 768, M = A B L a multiple of 128 and >= `MIN_ROWS_B200["expand"]` = 1024; served blocks: the default block (Protenix / MiniWorld /
OpenDDE ...) and Chai-1's identity block; NOT served: RoseTTAFold3 (k-q LN parallel block), Boltz-2 (up gate: no 3-operand kernel), fp32, training.
Chai-1 (parallel schedule): the transition runs on a side stream beside the attention core; a persistent expand grid over all 148 SMs waits for the core's CTAs (Chai-1 L512 A5:
1.044 with 148 CTAs, 0.964 with 96, 1.001 with 32): the grid is capped at `B200_PAR_CTAS` = 96 there.
Switches: `FOLDJET_ENGINE_TOKEN_DIT_FP8=expand` (module `PLACES`), `FOLDJET_ENGINE_TOKEN_DIT_FP8_MIN_ROWS`, `FOLDJET_ENGINE_TOKEN_DIT_FP8_B200` (placements the B200 serves, default `expand`),
`FOLDJET_ENGINE_TOKEN_DIT_FP8_B200_DEFS` (extra -D of the cubin, default `SIG_TANH`), `FOLDJET_ENGINE_TOKEN_DIT_FP8_B200_CTAS` (grid override for experiments).
Prebuild: `_prebuild --arch sm100` lists `token_dit_fp8_rows_sm100` and `augattn_sm100/gemm_swiglu_f8_sm100` for width-768 bf16 inference token rows.

### Kernel alone (graph, 50 back-to-back, median of 7; bf16 route = cuBLAS + `swiglu_rows` below 3840 rows, the sm_100a bf16 SwiGLU GEMM from there)
| M | bf16 route us | bf16 SwiGLU kernel us | e4m3 kernel us | e4m3 / route |
|---|---|---|---|---|
| 640 | 7.7 | 9.3 | 8.1 | 1.05 |
| 1280 | 10.5 | 9.5 | 8.3 | 0.79 |
| 1920 | 12.1 | 13.1 | 10.3 | 0.85 |
| 2560 | 16.1 | 13.3 | 10.5 | 0.65 |
| 3200 | 17.6 | 16.7 | 12.3 | 0.70 |
| 3840 | 16.7 | 16.7 | 12.6 | 0.76 |
| 5120 | 21.0 | 21.0 | 15.0 | 0.71 |
The e4m3 kernel is 21-29 % faster than the bf16 SwiGLU kernel but reaches only 1.6 PF/s of e4m3 at M = 5120 (epilogue / ring bound, not tensor bound); in a block it is 12 us of ~80.

### Placements measured and rejected on B200 (cuBLAS e4m3 `_scaled_mm` row-wise, bf16 out, graph timing; bf16 / e4m3 mm / the separate quantising pass, us)
| placement (K x N) | M 1280 | M 1920 | M 2560 | M 3840 | M 5120 |
|---|---|---|---|---|---|
| qkvg 768 x 3072 | 7.2 / 6.2 / +1.9 | 7.8 / 8.1 / +2.4 | 10.8 / 10.0 / +2.8 | 13.2 / 11.7 / +3.5 | 18.4 / 15.0 / +4.2 |
| out 768 x 768 | 4.1 / 4.0 / +1.9 | 4.5 / 4.7 / +2.4 | 4.9 / 4.9 / +2.8 | 5.5 / 5.9 / +3.5 | 7.1 / 6.8 / +4.2 |
| squeeze 1536 x 768 | 5.4 / 5.0 / +2.4 | 5.8 / 5.8 / +3.0 | 6.7 / 5.9 / +3.6 | 7.8 / 7.1 / +4.8 | 10.7 / 8.4 / +5.9 |
The e4m3 product saves at most 0-3 us, the quantising pass (qkvg: inside the AdaLN pass at +0.4 us per H100 data, but out / squeeze need their own) costs as much: no placement beyond expand
(as on H100: `out` / `squeeze` lost, `qkvg` lost below 2560 rows). Block-scaled MXFP8 (`_scaled_mm` with e8m0 block-32 scales, 128x4 swizzled layout; torch 2.13) is faster:
squeeze 5.9 -> 4.9 (M 1920), qkvg 7.7 -> 6.5, out 4.5 -> 3.8 us; an expand epilogue could write h as MXFP8 (32 units per thread = one block). Upper bound of the whole set ~3-4 us of an
80 us block (4 %) against another 3.7 % squeeze error and a swizzled-scale epilogue: not built (open item).

### Block results (B200 GPU 2, `dev.sh -x`, graph replay of one call under `static_weights()` + `static_inputs()` = the sampler's graphs; `_tools/blk2.py`)
Master weights (fp32 params, bf16 activations), A = 5 samples (A = 1 where marked), shared conditioning, 2 blocks per call (n = 2); e4m3 expand against the default in the SAME process, 9 rounds
interleaved with a 0.15 s idle gap before every timed 20-replay burst, order reversed every round, median; "ctrl" = a third capture of the DEFAULT timed the same way (address / order
effect of the A/B: +-4 % on single cells, so single ratios below ~3 % are noise); every row gate removed in the harness (`MIN_ROWS = 0`). `e4m3 / def` and `e4m3 / ctrl`. Raw rows: `$W/b2511/work/n13-fp8-c/final2/*.jsonl`.

| block (A:L) | M = A L | default us (2 blocks) | e4m3 us | e4m3 / def | ctrl / def | e4m3 / ctrl |
|---|---|---|---|---|---|---|
| Protenix 5:128 | 640 | 47.4 | 49.5 | 1.045 | 1.001 | 1.043 |
| Protenix 1:768 | 768 | 60.8 | 61.4 | 1.009 | 1.017 | 0.993 |
| Protenix 1:1024 | 1024 | 74.8 | 71.6 | 0.957 | 1.034 | 0.926 |
| Protenix 5:256 | 1280 | 67.4 | 64.0 | 0.950 | 1.040 | 0.914 |
| Protenix 1:1536 | 1536 | 98.9 | 96.4 | 0.975 | 1.003 | 0.973 |
| Protenix 5:384 | 1920 | 83.8 | 81.8 | 0.976 | 1.008 | 0.969 |
| Protenix 1:2048 | 2048 | 123.2 | 118.9 | 0.965 | 0.987 | 0.979 |
| Protenix 5:512 | 2560 | 116.7 | 109.6 | 0.939 | 0.984 | 0.955 |
| Protenix 5:768 | 3840 | 145.3 | 140.7 | 0.969 | 0.992 | 0.976 |
| Protenix 5:1024 | 5120 | 200.4 | 196.9 | 0.982 | 1.003 | 0.979 |
| Protenix 5:1536 | 7680 | 295.9 | 291.2 | 0.984 | 1.004 | 0.980 |
| MiniWorld 5:128 | 640 | 49.7 | 51.6 | 1.037 | 1.020 | 1.017 |
| MiniWorld 5:384 | 1920 | 88.7 | 87.6 | 0.987 | 1.005 | 0.983 |
| MiniWorld 5:640 | 3200 | 136.6 | 132.4 | 0.969 | 1.014 | 0.956 |
| MiniWorld 5:1024 | 5120 | 210.0 | 204.3 | 0.973 | 1.007 | 0.966 |
| Chai-1 5:128 | 640 | 38.1 | 37.2 | 0.974 | 1.014 | 0.961 |
| Chai-1 5:256 | 1280 | 56.9 | 53.4 | 0.940 | 0.994 | 0.945 |
| Chai-1 5:384 | 1920 | 82.2 | 71.9 | 0.874 | 0.964 | 0.907 |
| Chai-1 5:512 | 2560 | 95.2 | 90.7 | 0.952 | 1.009 | 0.944 |
| Chai-1 5:1024 | 5120 | 181.0 | 172.1 | 0.951 | 0.998 | 0.953 |
| Chai-1 5:1536 | 7680 | 277.9 | 265.9 | 0.957 | 1.001 | 0.956 |
| RoseTTAFold3 5:128 / 384 / 640 | 640 / 1920 / 3200 | 41.5 / 87.7 / 126.0 | declined (route is the default) | 1.043 / 0.969 / 0.982 | 1.042 / 0.988 / 0.990 | 1.001 / 0.981 / 0.992 |
| Boltz-2 | any | declined (up gate not built) | | | | |
The RoseTTAFold3 rows ARE the default route (the route declines it): their spread (0.97-1.05) is the noise floor of this A/B. The parallel block and the plain block at M = 640 (A5 L128: 1.043 over
the control, reproduced in three runs) lose; the gate `MIN_ROWS_B200["expand"]` = 1024 keeps them on bf16. Block-level result: plain blocks -2 .. -5 % from 1280 rows, Chai-1 -5 .. -9 %.
Why so little: the e4m3 expand kernel is 12 of ~80 us of a block; a quantising AdaLN pass is +0.4 us (the extra row-max reduction), the saving is the expand kernel (-2 us) plus a unified one-launch
expand against cuBLAS + row pass below 3840 rows (-2 us).

### Sustained time and energy per block (power cap ~1000 W; 8 s back-to-back replays per mode, nvidia-smi 0.2 s, two passes def / e4m3 / def / e4m3, mean of the pairs; `ENERGY=1`)
| block (A:L) | M | def us / W | e4m3 us / W | time | energy per call |
|---|---|---|---|---|---|
| Protenix 5:256 | 1280 | 64.7 / 709 | 62.3 / 674 | 0.963 | 0.917 |
| Protenix 5:512 | 2560 | 113.2 / 831 | 107.5 / 798 | 0.950 | 0.911 |
| Protenix 5:1536 | 7680 | 308.7 / 975 | 299.7 / 972 | 0.971 | 0.967 |
| Chai-1 5:256 | 1280 | 55.5 / 775 | 52.0 / 713 | 0.937 | 0.860 |
| Chai-1 5:384 | 1920 | 78.9 / 845 | 72.4 / 810 | 0.918 | 0.880 |
| Chai-1 5:512 (148 CTAs, before the grid cap) | 2560 | 92.6 / 933 | 96.3 / 835 | 1.040 | 0.929 |
| Chai-1 5:1024 | 5120 | 181.9 / 972 | 176.5 / 975 | 0.970 | 0.974 |
e4m3 saves energy before it saves time (the tensor pipe runs at half the cycles for the same work): 3-14 % less energy per block at 1280-2560 rows where the block is not yet at the cap;
at the cap (7680+ rows) time and energy agree at -3 %. (The Chai-1 5:512 row is the 148-CTA grid before the `B200_PAR_CTAS` cap: +4 % time, -7 % energy; the cap turned it into -5 % time.)

### Accuracy (random weights as the H100 protocol; fp64 PyTorch gold; update error = rms of (y - y64) over rms of (y64 - x); judge = max|y - y64| / max|y64|; `tests/gpu/test_token_dit_fp8_b200.py`)
| case | bf16 default route: rms / judge | e4m3 expand: rms / judge | ratio (rms) |
|---|---|---|---|
| Protenix 5:384, input x 1 | 0.0070 / 0.0050 | 0.0535 / 0.0306 | 7.6 |
| Protenix, input x 0.05 | 0.0072 / 0.0074 | 0.0536 / 0.0490 | 7.5 |
| Protenix, input x 20 | 0.109 / 0.0047 | 0.121 / 0.0048 | 1.1 |
| Chai-1 5:384, input x 1 | 0.0067 / 0.0061 | 0.0495 / 0.0453 | 7.4 |
| Chai-1, input x 0.05 | 0.0069 / 0.0084 | 0.0496 / 0.0558 | 7.2 |
| Chai-1, input x 20 | 0.042 / 0.0047 | 0.064 / 0.0061 | 1.5 |
Every block of the sweep above has e4m3 update rms 4.7-5.4 % and judge 2.5-5.1e-2 (table right-hand columns of `final2`), i.e. the H100 route's figures (5.4 % / 3-3.4e-2; Chai-1 4.7-6.4e-2): the 3.0.3 acceptance
is not the judge (the expand fails the 8e-3 judge 3-4x on H100 too, "FP8 outside the judge"), it is the end-level protocol of the 3.0.3 docs (not re-run here: same quantisation recipe, same
operand error). Saturation: impossible by construction (per-row scale = row max / 448; the stress inputs x 0.05 and x 20 stay finite and at the same relative error; graph replays are deterministic and equal
the eager route within 2 % of max |y|: test). The only B200-specific error source is the tanh.approx sigmoid (2^-11 relative; below the e4m3 rounding of a and b by 2 orders of magnitude).

### Verification
* FP8 OFF, bit-identity of the default route: the merge agent's scripts `$W/b2511/jobs/n12-bid/bid.py` (eager) and `bidg.py` (CUDA graph, replay twice), 398 measurer cases each (AttentionPairBias + five TokenDiT classes, bf16 / master /
  fp32, eval + train), sha1 of outputs and parameter gradients, tree n13-fp8-c against n13-base: **398 / 398 identical in both, 0 mismatches, 0 errors, 0 nondeterministic**
  (`$W/b2511/work/n13-fp8-c/bid/cmp.txt`, final tree). Non-sm100 CUDA sources byte-identical (diff list: only new files under `conditioned_transition/cuda/`), `backend_versions["cuda:b200_sm100"]` untouched.
* tests (B200): `tests/gpu/test_token_dit_fp8_b200_kernel.py` (the cubin against a torch reference on the same e4m3 operands at M 128 .. 5120), `tests/gpu/test_token_dit_fp8_b200.py` (accuracy next to the bf16 route at input x 1 / 0.05 / 20
  for Protenix and Chai-1, route on / off bit-equal below the gate, RoseTTAFold3 / Boltz-2 not served, graph == eager, route stats); the B200 suites trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit /
  atom_dit and `tests/cpu/test_fp8_switch.py`: 570 passed, 351 skipped (H100-only) (`$W/b2511/work/n13-fp8-c/tests/pytest.log`).
* `_prebuild --arch sm100 --check`: 804 -> 806 artifacts (739 cubin, 67 ext), 0 not built yet; the two new ones (`token_dit_fp8_rows_sm100`, `augattn_sm100/gemm_swiglu_f8_sm100`) are in the plan; rc = 1 only from the unchanged `*_sm90` / mpnn lines, and the
  line "not compiled: conditioned_transition/cuda/token_dit_fp8_rows.cu" is gone (full outputs: `$W/b2511/work/n13-fp8-c/prebuild/`).

### Rejected / not built (with data)
* `qkvg`, `out`, `squeeze` on B200 (cuBLAS e4m3 `_scaled_mm`): kernel table above; in the block (Protenix 5:384 / 768 / 1024 / 1536, e4m3 / default): all four placements 1.079 / 1.046 / 1.021 / 1.036; `qkvg + expand`
  1.007 / 0.986 / 0.983 / 0.983 against `expand` alone 0.976 / 0.969 / 0.982 / 0.984: no gain beyond the expand (and 1-3 % more error than expand alone: 5.4 % -> 6.6 % update rms with out / squeeze).
* RoseTTAFold3 `expand`: the parallel block with k-q LN needs a standalone quantising pass (`rows_q8(xt)`), which costs as much as the e4m3 expand saves (the transition runs beside the attention core on a side stream anyway); declined on B200.
* Boltz-2 (up gate): the bf16 B200 route is cuBLAS + `swiglu_rows` (unfused, [a | b | u] through HBM); a three-operand e4m3 tcgen05 kernel (64-unit chunks, N = 192 per pair step, B rows split a64 + b32 | b32 + u64) was NOT built (see open items); not served.
* A 148-CTA persistent expand grid in the parallel (Chai-1) schedule: serialises with the attention core (1.044); capped at 96 CTAs (0.964 / 0.965 at 96 / 64, 1.001 at 32).
* H100 transfers not taken: `ht = 64` tiling, wgmma n192 up kernel, the sm90 weight packs.

### Atom DiT: measured, no B200 route built
The 3.0.3 atom DiT route (`local_dit_h100.FP8_ROWS`: e4m3 `pre90_f8` / `post90_f8` row kernels) is an H100 wgmma kernel family; `FOLDJET_ENGINE_ATOMDIT_FP8` stays unserved on B200 (`serves()` is H100 only).
What the B200 chain looks like (docs/b200/atom-dit.md, "Second pass" / "Third pass"): the bf16 inference stages (`qkvg_fwd1t / fwd3`, `attn_fwd`, `ffn_fwd1t / fwd3`, K = 128 / 256, d_atom 128, hidden 256) are row-thread and TMA-latency bound, not tensor bound
("9500 -> 5650 ticks per tile, all of it row-thread work now"; "a cold TMA load lands ~2.3 us after its issue"); the modulation GEMMs run per row inside the kernels from smem operands.
Measured bf16 block time (graph, B200 GPU 2, A = 5, master, shared conditioning, `_tools/blk.py` default column): AtomDiTProtenix / OpenFold3 / Boltz2 / Chai1 5 x 1024 atoms 28.4-31.4 us, 5 x 4096 atoms 49.5-53.8 us, 5 x 8192 atoms
83.7-90.6 us (dense AF3-class classes with pair bias; Protenix D5 variant 65 / 191 / 343 us). Tensor time of the e4m3-capable GEMMs at 1.3 PF/s bf16: ~0.45 MF per row (qkvg 131 k, Wo 33 k, a | b 131 k, squeeze 66 k, modulation
~100 k) -> 20480 rows 9 GF = 7 us of 52 us (13 %); 40960 rows 18 GF = 14 us of 89 us (16 %). Halving the MMA time bounds the gain at ~6-8 % even if the MMAs were serial with the row work, and they are overlapped on B200 (the kernels
hide them under the epilogue and the TMA); the H100 ablation, where the row kernels ARE serial `cp.async -> wgmma -> epilogue` chains, found 9-13 % and delivered 5-10 % at 5 % update error (post: judge 5e-2). The B200 e4m3 route would
additionally add a per-row amax + cast to the row-thread critical path of qkvg / FFN (the part that is binding) and re-derive the modulation / smem layouts of kernels that were tuned for tens of rounds. Not built: the measured bound does not justify the
loss of the "bit-identical bf16 cubins" and the accuracy cost; no atom DiT file of the tree changes.
(This is an analytic bound from measured block times and the documented stage structure, not an ablation of the B200 kernels.)

### Open items (exact)
1. Boltz-2 up-gate e4m3 expand (3-operand tcgen05 kernel); bound ~3 % of a block (unfused bf16 path today).
2. MXFP8 (block-32 e8m0 scales, `kind::mxf8f6f4` / cuBLAS block-scaled `_scaled_mm`): h written as MXFP8 by the expand epilogue (one thread = 32 units = one scale block) feeding a block-scaled squeeze; measured GEMM
   gain 0.3-1.3 us per GEMM (table above), not built; needs the 128x4 swizzled scale layout in the epilogue.
3. The e4m3 expand at 64-unit chunks (more items than SM pairs below 2560 rows: ~1 us of 10 us at M 1920); PDL for the expand kernel (the row passes of the 3.0.3 file never trigger their dependents early).
4. End-level (200-step diffusion, FoldJet RMSD) acceptance is NOT re-run on B200: the quantisation recipe / operand error are the 3.0.3 ones; only the module-level fp64 protocol was run.
5. Activation scales of rows whose max is denormal (|row max| < 1e-35) are not floored (the H100 row kernels are shared byte for byte); AdaLN outputs cannot be such rows.
6. `fp8.py` registration: see `$W/b2511/work/n13-fp8-c/FP8_REGISTER.md` (agent A).

### H100 token / atom DiT design digest (3.0.3 read, basis of the recipe)
#### H100 token / atom DiT FP8 design digest (read from 3.0.3 fp8_ref; basis of the B200 recipe)

#### Placements (token DiT, bf16 master inference, width 768, CondFold route)
| placement | e4m3 tensors | scale | where made | in `safe` | why / why not |
|---|---|---|---|---|---|
| expand (+SwiGLU) | xt (AdaLN rows) x [Wa;Wb] | act: dynamic per-row max/448; weight: static per-output-channel (max/448), per weight pack (`_capture.lookup`) | quantising AdaLN row pass (`adaln_in_rows_cf_q8` qkvg-less: `resgate_adaln_cf_q8`; Chai-1 `adaln_in_rows_cf_par_q8`; RF3 parallel: quant in the one LN pass) ; epilogue multiplies a,b by sx[row]*sw[unit] before silu(a)*b | YES | wins at every M (0.94-0.98 module); error 5.4 % rms (both a and b e4m3) but accepted: end-level RMSD inside 47/48 |
| qkvg | xa (adaln_in_cf_q8) x Wqkvg | per-row / per-channel, `torch._scaled_mm` rowwise bf16 out + bias | row pass | NO | loses below 2560 rows (1.006-1.019), wins 0.97-0.99 above; error 1-1.3 % at x1 but 3.5-3.9 % at small input (x0.05, attention update dominates); Chai-1 identity AdaLN scale unbounded: never qkvg |
| out | og (attention output) | needs own row pass (16 head CTAs write a row) | extra pass | NO | 1.02-1.05 loss (row pass 5-9 us outweighs) |
| squeeze | h (e4m3 hidden) | own row pass (h row spans 12 tiles) | extra pass | NO | loss; error 3.7 % |
Scale floor: weights `clamp_min(1e-30)` (train NaN fix); activation scale 1.0 if row amax == 0. No saturation by construction -> no calibration / guard / take_overflow.
Forms: plain (AF3/Protenix/MiniWorld), identity+parallel (Chai-1: sig=false, eps 0.1; expand/out/squeeze, never qkvg), parallel (RF3, kq-LN: expand only, quant in the LN pass), up (Boltz-2: [Wa64|Wb64|Wu64] n192 tiles, silu(a) b u, three channel scales).
Gates: MIN_ROWS_PER = {qkvg 2560, expand 640, out 2560, squeeze 2560}; MIN_ROWS_UP = 2560 (kernel -27 % but graph block +2.7 % at 1920: the quantising row pass costs a third reduction, row passes got SLOWER 12.6->14.0 us); M % 128 == 0; D == 768; bf16 only (fp32 declined).
Accuracy acceptance (H100): judge = max|y-y64|/max|y64| vs max(err of PyTorch bf16, 8e-3); expand FAILS the judge by 3-4x (3e-2) -> FP8 stays outside the judge, acceptance = end level (RMSD to bf16 <= max(1.1x bf16 pair max, 0.5 A)) + finite + route stats. Stress cases: x0.05 and x20 input, 3 masked keys, random weights LN 1+-0.1.
Speedups H100 (module graph, sustained, qkvg+expand / expand): 5:256 .941/.977 ; 5:512 .948/.967 ; 5:1024 .927/.949 ; 5:2048 .939/.973; A1 768 .992/.976. Boltz-2 expand: 1.056 at M640, .998 at 2560, .954 at 7680. Chai-1: whole gain 0.6 % of kernel time.
Traps: row pass slower than bf16 (3rd block reduction); out/squeeze lose; qkvg loses small M; Boltz up gate bistable end-level case 7y37; per-row scale is independent of step noise level but not of row shape.
#### Atom DiT (H100)
pre90_f8 (q/g/k/v proj K=128) + post90_f8 (Wo, [a,b], SwiGLU, squeeze Wd K=256) in the fused row kernels, A operand per-row power-of-two scale cast once to 16 regs, weights per 64-row-unit pow2 scales, epilogue acc*inv*inv exact; fp32 sum across the 4 hidden chunks (rescaling the wgmma accumulator gave wrong results). Attention (window, 32-wide heads) NOT e4m3: ablation <=1.5 %. Upper bound from ablation: row-kernel e4m3 products = 9-13 % of step (weight bytes only 3-6 %). Measured pre+post .895-.968 module; error post 5 % rms (judge 5e-2), pre 1.6 %. Gates: shared conditioning only (per-sample tables lost 1.09-1.12 -> 9 of 27 calls/run fallback), UP from 2048 atoms (Boltz), pre alone never wins (post carries it). Cond tables / pair bias are hoisted per fold: no FP8 there.
#### Transfers unchanged to B200
quantization recipe (per-row dynamic act max/448, per-channel weight max/448, floor), form matrix (plain/identity/parallel/up), the quantising AdaLN row kernels token_dit_fp8_rows.cu (CUDA, arch-neutral: only gencode changes), accuracy protocol + stress cases, gates as STARTING values (expand 640 -> re-measure; up 2560), safe = expand only, fp32 never, qkvg never for identity, route-stats keys (calls/declined/expand/...), shared-conditioning-only for atom rows.
#### Must be redone for tcgen05
MMA instruction (kind::f8f6f4, K=32/step, 128 elems per 128B swizzle block -> NKB=6 for D=768, same smem bytes per stage as bf16), accumulator in TMEM (epilogue scale in TMEM->reg path, per-lane row scale), 2-CTA pair tiling of gemm_swiglu2_sm100 (not wgmma n192), TMA box in elements, PDL, fold/side streams/pair32/vmn B200 chain (the e4m3 path must live beside _b200_swiglu and the parallel side-stream fork), thresholds (B200 bf16 expand kernel only from 3840 rows; below cuBLAS+rows), power-cap energy judgement.

## 2026-11-19 (second session, agent C) -- OuterProductMean and MSAPairWeightedAveraging: e4m3 on B200 (opt-in, default OFF)

Product code: `kernels/integrations/opm_fp8_b200.py` + `csrc/sm100/opm_fp8_b200.cu` (cast helper), `pwa_fp8_b200.py`; hooks in `opm_sm100.py` (`_forward`, `_product_and_counts`) and `pwa_sm100.py` (`_contract`, `_forward`);
two new template flags / entry points in the sm_100a sources `csrc/sm100/opm_sm100.cu` (`opm_prologue_sm100<WT, F8>`, `opm_prologue_f8`) and `csrc/sm100/pwa_sm100.cu` (`ln_vg_body<WT, F8>`, `ln_vg_f8`): the F8 = false instantiations are the
default kernels (the default route stays bit-identical, see Verification). Switches are the H100 route's: `FOLDJET_ENGINE_OPM_FP8=bound` (`opm_fp8.POLICY`), `FOLDJET_ENGINE_PWA_FP8=1` (`pwa_fp8.ENABLED`), so `fp8.enable` already reaches them.

### Decision from the B200 blockers tables (which stage is tensor / HBM bound, where e4m3 can win)
Stage times, B200 GPU 2/3, graph replay, `torch.profiler`, default route (`_tools/prof3.py`):
| module / case | stage | us | share |
|---|---|---|---|
| OPM 64/128, L384 depth 2048 (485 us) | cuBLAS product `nvjet_tst_128x256_64x6` (K = S) | 371.7 | 77 % |
| | epilogue (O round trip, W_o 1024 -> 128) | 64.6 | 13 % |
| | prologue (LN + projections, writes A2 / BT) | 40.7 | 8 % |
| OPM 64/128, L256 depth 1024 (140 us) | product / epilogue / prologue | 84.0 / 34.7 / 17.0 | 60 / 25 / 12 % |
| PWA 8x8, L768 depth 1024 (208 us) | gate_out_w / bmm / ln_vg_w / pair_fwd | 73.2 / 52.6 / 42.6 / 39.7 | bmm 25 % |
| PWA 8x8, L1536 depth 1024 (518 us) | bmm 175.4 / gate_out 141.1 / pair_fwd 119.9 / ln_vg 81.8 | | bmm 34 % |
The OPM product is tensor bound (cuBLAS bf16 1.3-1.66 PF/s at the power cap) and 60-77 % of the call from depth 1024: the e4m3 product bound is -30 .. -40 % of the module. The PWA contraction is 16-34 % of the module (the rest is the
latency / memory bound gate pass, `ln_vg`, `pair_fwd`): bound ~-8 .. -17 % for the e4m3 product alone. Both above the 5 % bar: built. (Agent A's `mb_results.txt` was not needed; micro-benchmarks of this section: `_tools/bench_opm.py`, `bench_pwa.py`.)
cuBLAS e4m3 `_scaled_mm` against bf16 for the OPM product (K = S, M = N = 32 L; GEMM only, us): L128 / S1024 27.1 -> 16.2, L256 / S1024 87.4 -> 51.6, L384 / S2048 371.7 -> 195.9, L768 / S4096 2997 -> 1578: 0.52-0.60 at S >= 1024 (0.72-0.84 at S 512).

### OuterProductMean: design (H100 recipe unchanged where not instruction specific)
* e4m3 stage: the grouped product O[(i,c), (j,e)] = sum_s A2[(i,c), s] BT[(j,e), s] only (as H100). Policy `bound`: static power-of-two scales from the weights alone,
  |LN(x)_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k| (`opm_fp8.bound_scales`, reused): no input can saturate the cast, no calibration / guard / `check()` hook. Scale floor: `pow2_scale` of the H100 TriMul helper (2^-30).
* B200 specific: d_msa 64 / d_pair 128 (the narrow prologue): the prologue drain writes e4m3 A2 / BT itself (`opm_prologue_f8`: value x q, saturating `cvt.rn.satfinite.e4m3x2`, 64 B swizzled staging, u8 TMA maps: half the A2 / BT bytes,
  no cast pass; the fp32 accumulator is cast once: no double rounding). Other widths and the projection-bias variants: one cast launch (`opm_cast2`, bf16 A2 / BT -> e4m3, 6 TB/s) after the unchanged bf16 prologue.
  The product is `torch._scaled_mm` (cuBLASLt e4m3, fp32 accumulation, scalar dequantisation scales, bf16 O); count table (side stream) and epilogue are the default's.
* Gate (`opm_fp8_b200._gate`): padded S >= 1024 at every L, or S >= 512 from N 256 where the fused bf16 kernel does not serve; below it (and where the fused kernel serves: S <= 512, N >= 512) the bf16 route runs. Chai-1's grouped OPM stays bf16 (below).
* Not served: training, Chai-1 grouped OPM (declined, see "Rejected"), calibration policy (`bound` only).

### OuterProductMean results (B200 GPU 3, `dev.sh -x`, graph replay of one call under `static_weights()` + `static_inputs()`, master weights, one sample; `_tools/blk3.py`: 9 rounds interleaved, 0.15 s idle gap before every timed 10-replay burst, order reversed every round, median; "ctrl" = a third capture of the default timed the same way)
Module time e4m3 / default (e4m3 us); fp64-gold error as rms / judge (max |y - y64| / max |y64|) of the bf16 route and of the e4m3 route; sustained energy per call = (time x power) of the e4m3 route over the default's (8 s back-to-back, nvidia-smi 0.2 s, mean of two def / e4m3 passes):
| case (d_msa / d_pair, L, depth S) | default us | e4m3 / default | ctrl | gold bf16 rms / judge | gold e4m3 rms / judge | energy ratio |
|---|---|---|---|---|---|---|
| 64/128 L128 S1024 | 48.9 | 0.782 (38.2) | 1.03 | 0.0041 / 0.0047 | 0.0208 / 0.0127 | 0.74 |
| 64/128 L128 S2048 | 78.2 | 0.721 | 1.00 | 0.0040 / 0.0053 | 0.0161 / 0.0112 | 0.70 |
| 64/128 L256 S1024 | 138.1 | 0.755 | 1.01 | 0.0041 / 0.0044 | 0.0210 / 0.0129 | 0.73 |
| 64/128 L256 S2048 | 233.8 | 0.657 | 1.00 | 0.0040 / 0.0045 | 0.0163 / 0.0101 | 0.65 |
| 64/128 L384 S1024 | 284.6 | 0.708 | 1.00 | 0.0041 / 0.0046 | 0.0210 / 0.0137 | 0.71 |
| 64/128 L384 S2048 | 493.9 | 0.634 | 1.00 | 0.0040 / 0.0050 | 0.0163 / 0.0106 | 0.64 |
| 64/128 L768 S1024 / S2048 (sweep) | 1072 / 1863 | 0.693 / 0.620 | 1.00 | | | |
| 128/256 L256 S1024 / S2048 | 153.7 / 257.9 | 0.815 / 0.736 | 1.00 | 0.0044 / 0.0052 | 0.0270 / 0.0169 (S2048: 0.0222 / 0.0131) | |
| 128/256 L384 S1024 / S2048 | 312.0 / 532.0 | 0.762 / 0.698 | 1.00 | | | |
| 128/384 L256 S1024 / S2048 | 167.4 / 271.5 | 0.823 / 0.748 | 1.00 | 0.0044 / 0.0052 | 0.0271 / 0.0169 | |
| 128/384 L384 S1024 / S2048 | 350.2 / 571.4 | 0.783 / 0.720 | 1.00 | | | |
| 256/512 L256 S1024 / S2048 | 204.2 / 331.4 | 0.854 / 0.789 | 0.99 | 0.0045 / 0.0041 | 0.0258 / 0.0194 | |
| 256/512 L384 S1024 / S2048 | 405.9 / 660.7 | 0.815 / 0.758 | 1.00 | | | |
Short depths (64/128, e4m3 / default): S 512: L128 0.97 (ctrl 1.06: noise), L256 0.877, L384 0.872, L768 1.00 (the fused bf16 kernel serves: unchanged); S 1024 at L128: 0.757 (the gate takes every L at S >= 1024).
The first version (separate cast pass) gave 0.80 / 0.71 / 0.74 / 0.67 at L256 S1024 / L256 S2048 / L384 S1024 / L384 S2048; the fused prologue (e4m3 written by the drain, no cast launch) takes another 4-6 points
(L384 S2048: product 372 -> 188 us, cast 25 us gone, the prologue writes half the A2 / BT bytes). Energy tracks time at the cap (both routes ~965-975 W): the e4m3 route's energy per call is its time ratio.
Accuracy: e4m3 update error = one e4m3 rounding of A2 / BT (2.7 % rms each) averaged by the depth: judge 1.0-2.0 % against 0.4-0.5 % for the bf16 route (tolerance 8e-3: 1.3-2.4x it; the H100 route: 1.03-1.72x at depth 1024-4096, inside at depth >= 8192);
update rms 1.6-2.7 % (ratio to the bf16 route 4-6x). Large-weight stress (LN weights x 8, biases x 4; `test_large_weight_stress`): judge 1.3 % (bf16 0.46 %), finite: the bound policy's scales follow the weights. No saturation by construction.
The e4m3 judge error exceeds the 8e-3 judge, like every FP8 route ("FP8 outside the judge"); the end-level check (RMSD) was NOT run on B200.

### MSAPairWeightedAveraging: design
* e4m3 stage: the key contraction o[h] = w[h] . v[h] (K = j = N) only. w (a probability <= 1): fixed scale 2^8 (cannot saturate); v: static power-of-two scale from the weights alone (`bound` policy: |y_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|,
  max_c sum_k |Wv_ck| (...) x (1 + 2^-6)): no calibration, no guard (the H100 route needed 16 x headroom + `take_overflow()` because its v scale came from a first-call amax). Fp32 accumulation, bf16 o exactly where cuBLAS bmm writes it; `pwa_gate_out` is the default's.
* B200 specific: the default `ln_vg` output v is head-major [H, N, S C] = an MN-major B operand, which cuBLASLt's e4m3 GEMM takes as it is on Blackwell (measured: NN == TN speed) -- Hopper needed v transposed to K-major (`ln_vt8`, a new kernel).
  Eight per-head `torch._scaled_mm` GEMMs with `out=` o[h]. (a) MiniWorld's MSA side (d_msa 64, 8 x 32, inference at d_pair 128): `ln_vg_f8` -- the drain of `ln_vg` writes v as e4m3 (v x q, saturating cast, 32 B rows, no swizzle, u8 TMA map) instead of bf16: no cast pass over v
  (v is 0.4-1 GB at N 768-2048), half its bytes; `pwa_dual` (pair_fwd + ln_vg in one launch) is replaced by the split launches while the route serves. (b) every other build (Chai-1 d_pair 256, 8 x 16, IntelliFold 256/512): one cast launch for v (`opm_cast1`).
* Gates: fused (a): N^2 S >= 5e8 (padded); cast route (b): head dim 32 from N 1280, 16 from N 1792, 8 never (the below-5 % bar). Below them bf16 (cuBLAS bmm) runs unchanged.
* Not served: training, p_drop / gate / out passes stay bf16 (as H100: gate pass is latency bound, o e4m3 lost on H100).

### MSAPairWeightedAveraging results (B200 GPU 3, same protocol; e4m3 / default; d_msa 64 / d_pair 128 / 8 x 32 (MiniWorld, fused v), depth 1024)
| N (= L) | default us | e4m3 us | e4m3 / default | ctrl | gold bf16 rms / judge | gold e4m3 rms / judge | energy ratio |
|---|---|---|---|---|---|---|---|
| 768 | 467.7 | 413.5 | 0.884 | 1.00 | 0.00175 / 0.00295 | 0.00217 / 0.00350 | 0.86 |
| 1024 | 707.6 | 559.8 | 0.791 | 1.00 | 0.00175 / 0.00285 | 0.00207 / 0.00316 | 0.79 |
| 1536 | 1342.5 | 960.7 | 0.716 | 1.00 | 0.00175 / 0.00294 | 0.00197 / 0.00319 | 0.72 |
| 2048 | 2096.1 | 1449.6 | 0.692 | 1.00 | 0.00174 / 0.00285 | 0.00192 / 0.00303 | 0.71 |
Depth sweep (e4m3 / default, fused, gate removed, `pwa/s2_fused.jsonl`): L512: S384 1.25, S1024 0.99, S2048 0.87; L768: 1.02 / 0.86 / 0.79; L1024: 0.96 / 0.79 / 0.75; L1280: 0.97 / 0.76 / 0.72; L1536: 0.83 / 0.72 / 0.69; L2048: 0.79 / 0.69 / 0.66; L256-L384 lose (1.03-1.45).
The first (cast) version of the 8 x 32 route: 0.90 / 0.89 / 0.84 at N 768 / 1536 / 2048 (1.10 at N 768): the fused v halves its cost.
Other builds (cast route, S1024): Chai-1 d_pair 256 8x32 N 1536 / 2048: 0.885 / 0.847; 128/256 8x16 N 1536 / 2048: 0.962 / 0.907; 256/512 8x32 N 1536 / 2048: 0.940 / 0.911; 8x8 builds 1.01-1.13 at N 1024-1536, 0.95-0.97 at N 2048 (not served).
Accuracy: PWA's update is small next to the residual, so the e4m3 contraction adds little: judge 0.30-0.35 % (bf16 0.28-0.29 %, tolerance 8e-3), update rms 0.19-0.22 % (bf16 0.175 %); large-weight stress (LN weights / biases x 8): 0.92 % judge (bf16 0.57 %), finite.
H100 route: judge 0.30-0.37 %. Energy at the cap = time ratio.

### Verification (OuterProductMean / MSAPairWeightedAveraging)
* FP8 OFF, bit-identity of the default route: the merge agent's `bid.py` (eager) and `bidg.py` (CUDA graph, replay twice) over the measurer cases of `OuterProductMean` and `MSAPairWeightedAveraging` (both modes: inference + training; L 128 / 384 / 640, every depth /
  width / variant, bf16 / master / fp32: 174 cases, `_tools/msabid.yaml`), sha1 of outputs and parameter gradients, tree n13-fp8-c (final) against n13-base: **174 / 174 identical in both, 0 mismatches, 0 errors**
  (`$W/b2511/work/n13-fp8-c/bid_msa2/cmp.txt`; the 30 cases flagged non-deterministic are the training cases with dropout draws, flagged identically in both trees). The default kernels are the F8 = false instantiations of
  `opm_prologue_sm100` / `ln_vg_body` (default template argument, defaulted trailing kernel parameters): same code, same bits. Their artifacts are re-keyed (source text changed): `miniworld_opm_sm100`, `miniworld_pwa_sm100`.
* Tests: `tests/gpu/test_opm_fp8_b200.py` (accuracy at 64/128 plain / projection bias / count_offset and 128/256, large-weight stress, below-gate and off bit-equal, graph == eager), `tests/gpu/test_pwa_fp8_b200.py` (accuracy + stress at N 1280 depth 512, below-gate
  and head-dim-8 bit-equal, graph == eager); with the token DiT tests, the B200 suites (trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit / atom_dit / opm / pwa / engine_fp8 / route_stats / i64_opm_pwa) and
  `tests/cpu/test_fp8_switch.py`: **583 passed, 491 skipped (H100-only), 0 failed** (`$W/b2511/work/n13-fp8-c/tests2/pytest.log`).
* `_prebuild --arch sm100 --check`: 807 artifacts (739 cubin, 68 ext), 0 not built yet (base 804 / 66); new: `opm_fp8_b200` (the cast helper), plus the re-keyed opm / pwa extensions and the token DiT pair of the first session; rc 1 only from the unchanged `*_sm90` / mpnn lines (`prebuild2/`).

### Rejected / not built (with data)
* Chai-1's grouped OPM (8 groups of 8, product LayerNorm): kept bf16. Product only, B200, us (bf16 `bmm` / eight e4m3 GEMMs / the separate cast): N256 S1024 46.9 / 56.5 / 53.5 (a loss), N384 S1024 100 / 81 / 80, N512 176 / 125 / 106, N384 S4096 383 / 217 / 312, N768 S1024 394 / 244 / 159.
  The cast costs what the GEMM saves below S 4096 and the H100 census puts the grouped OPM at 0.7 % of Chai-1's kernel time: no route.
* OPM O as e4m3 (the epilogue reads 302 MB at L384 S2048: 13 % of the module, 25 % at L256 S1024): not built (adds a third e4m3 rounding and an epilogue rewrite; the H100 doc estimated <= 2 % there).
* PWA 8 x 8 builds: the e4m3 contraction is 5 % of the module at best at N 2048 (0.95-0.97), a loss below: not served. PWA o as e4m3 / gate-out in e4m3: lost on H100 (the gate pass is latency bound), not retried.
* OPM / PWA training: no e4m3 placement (as 3.0.3).

### Open items (OPM / PWA)
1. OPM wide prologue (`opm_wprologue_sm100`: d_msa 128 / 256, projection biases): write e4m3 A2 / BT from its drain as the narrow prologue now does (today one cast launch: ~6-8 % of the module at S 2048).
2. PWA wide builds (`ln_vg_w`, Chai-1 d_pair 256 / 8 x 16 / IntelliFold 256-512): fused e4m3 v (the cast pass is ~15 % of the 8 x 32 module at N 2048).
3. OPM O as e4m3 end to end (cuBLASLt e4m3 D + an e4m3-reading epilogue); an e4m3 version of the fused bf16 kernel for S <= 512 (N >= 512), a new tcgen05 kernel.
4. PWA: the eight per-head `_scaled_mm` launches -> one batched / grouped e4m3 GEMM; `pwa_dual` (pair_fwd + ln_vg in one launch) has no e4m3 form: the route runs the split launches.
5. End-level (diffusion RMSD) acceptance on B200 not run; judge errors above the 8e-3 tolerance for OPM (1.0-2.0 %), inside it for PWA (0.30-0.35 %).
6. `fp8.py` registration: `$W/b2511/work/n13-fp8-c/FP8_REGISTER.md` (OPM / PWA section).




## TriAttn FP8 on the B200 (agent D, 2026-11-19; tree n14-d)

Opt-in, OFF by default. The H100 digest (what each route quantises, scales, saturation, whitelist, numbers, traps, transfers / redo table) is `docs/b200/fp8_h100_triattn_digest.md`.
What the framework owner registers: `$W/b2511/work/n14-d/FP8_REGISTER.md` (`fp8.B200_ROUTES["triattn"]`; `fp8.py` itself was not edited). Tools and raw logs: `$W/b2511/work/n14-d/` (`t_*.py`) and `$W/b2511/jobs/n14-d-*.log`.

### D.1 Design (the H100 FP8 PV route redone for tcgen05)

* **What is e4m3**: P (the softmax numerator) and V; q, k, the bias, the logits, the row offset, O and l stay bf16 / fp16 / fp32. New core `b200_sources/triattn_f8_sm100.cu` (the shipped bf16 core `triattn_sm100.cu` and every other source are untouched):
  head dim 32 (`TF_D = 32`: d_pair 128 x 4, 64 x 4 x 32, 64 x 2 x 32, 256 x 8 x 32, 384 x 12 x 32 share the core) and 64 (`-DTF_D=64`: Chai-1 / AF3 d_pair 512 / RF3 64 x 4 x 64).
* **P without `ex2`** (the H100 trick, which is what makes the H100 core faster; the e4m3 MMA rate is not): the e4m3 byte `b = 8 (x - m) + 120` IS `2^(x - m + 8)` to ~3 bits (a byte is `2^((b >> 3) - 7) (1 + (b & 7) / 8)`).
  One packed-fp16 chain a key pair after `x = f16(S) * scl + bias` (the bias converted bf16 -> fp16 by one small exact pass, -inf kept): `HFMA2.RELU` (`t = relu(x - m + 15)`), `HFMA2` (`8 t + 1024`: the byte is the low 8 bits of the fp16 mantissa),
  one `PRMT` a word of 4 keys; the words go to TMEM as the A operand (4 e4m3 a 32-bit column, 8 columns a K = 32 MMA: the 128 keys are 32 columns of the S buffer; the PV is 4 MMAs of K = 32 instead of 8 of K = 16). No MUFU, no row-sum FADD2 in the softmax warps.
* **Row sum on the tensor core**, `[O | l] += P [V | 1]`: one `tcgen05.mma.kind::f8f6f4` M128 N(D + 16) K32 (B = V^T rows 0 .. D-1 + 16 ones rows = e4m3 1.0, written once a ring stage): l is the sum of the QUANTISED P (self-normalising).
  TMEM: S x 3 (128 columns each) | O, l x 2 rows x (D + 16) = 480 of 512 columns at D = 32; at D = 64 S x 2 + 2 x 80 = 416. Two pair rows a task (the bf16 core uses three above L384; the e4m3 core needs R x (D + 16) accumulator columns).
* **V^T planes** (`f8_vtq`, one pass: [B N H D][S] bytes, K-major, 128-byte swizzle through a TMA box 128 B x D rows; the H100's `vt_quant`): per-head static power-of-two scale `vinv = 2^floor(log2(448 / (4 amax)))` (headroom 4) calibrated on the first eager call of a site
  (`FP8_VSCALE_AUTO`); saturation guard = an int32 [2] per device (running max of |v vinv|, flag > 448) written by the pass; `take_overflow` / `rescale_all` widen the scales in place (captured graphs follow). A site = the weight tensor object (fused: the V weight, wide: the packed weights):
  weakref-dropped with it, recalibrated when the address is taken by another tensor or the weights change in place (`_version`: a first version keyed by `data_ptr` alone read a stale scale of a freed module in the tests).
* **Offset / window**: tile 0 and any row whose tile maximum leaves the window top (`> m + 0.5`) put the offset at `max + RAISE` (RAISE = 3 log2 units, P_max = 2^5); the byte window is [max - 11.9, max + 3.5]. The raise changes the conversion's constant (free) and rescales O / l in TMEM (one warp-uniform
  `tcgen05.ld` / `st` round trip after the previous PV of the row retired; counted in the kernel's `stat[1]`). A row whose final l is below 2^(8 - RAISE) / 2 (every key under the window) or not finite is recomputed exactly at the drain (`exact_row`, fp32 from q / k / v / bias in global memory; `stat[0]`).
  The e4m3 NaN byte (127) cannot be formed: after the vote t <= 15.5, so no upper clamp is needed; a non-finite input gives a NaN byte, then the exact recompute.
* **Where the H100 routes went**: `pv_fused` (front writes V^T) -> the d_pair-128 call (the B200 front writes bf16 v, the pass converts; the front is not touched); `pv_front` / `pv_pass` -> every other head-dim-32 site (the same pass; counted as `pv_front`);
  `pv_d64` -> the `TF_D = 64` build (built, gate closed, D.5); `qk` (e4m3 q / k): ceiling measured, route not built (D.5). The H100's NaN-byte clamp is the `relu` + the vote. Model whitelist unchanged and enforced by `fp8.enable_triattn` (af3 protenix1 opendde protenix2 openfold3 rosettafold3 chai1; Boltz-2 and IntelliFold2 stay OFF).
* **Shipped gates** (below them the bf16 core runs, bit-identical to FP8 off): padded S >= 1024 for every head-dim-32 kind (`FOLDJET_ENGINE_TRIATTN_B200_FP8_MINS`); the 64-channel-head core closed (`..._MINS64`, default 2^30).

### D.2 Kernel level: e4m3 core against the bf16 core (same process, interleaved, 3 rounds, GPU 0 exclusive; events after a 0.15 s idle gap; bf16 core = the shipped best build of the length: HB + QS, 3 rows; 4 heads x 32, N = L rows, B = 1; `t_speed.py`, `t_abl*.py`)

| L | bf16 core ms | e4m3 core ms | core x | pass ms | core + pass x |
|---|---|---|---|---|---|
| 512 | 0.2072 | 0.1968 | 0.948 | 0.0311 | 1.097 |
| 1024 | 1.3446 | 1.1635 | 0.865 | 0.1229 | 0.957 |
| 2048 | 9.4665 | 7.9544 | 0.840 | 0.4567 | 0.889 |

(`raise2speed`, a later same-build run: L1024 1.3440 -> 1.1561 = 0.860, L2048 9.4677 -> 7.9359 = 0.838.) How the core got there (L1024, ms, always against the bf16 core 1.34): first build (bf16 logits by FFMA2 + 2 FHADD, exact recompute from q / k of a tile that left the window, 2 rows, no TMEM rescale) 1.3757 (x1.02);
the whole tile in packed fp16 + the one-vote raise with a TMEM rescale instead of a global recompute 1.2020; batched O / l rescale round trip (one `tcgen05.ld` wait, l is one column) 1.1635. RAISE trades accuracy for the rescale events:
RAISE 2 / 3 / 4 = 1.2866 / 1.2035 / 1.1452 ms (L1024), 8.949 / 8.338 / 7.864 ms (L2048); RAISE 4 judge error 0.0088 / 0.0094 at L512 / 768 (> 8e-3) against 0.0056 / 0.0067 at RAISE 3 and 0.0058 at RAISE 2: **RAISE 3 shipped**; a larger raise only for rows that
leave the window after tile 0 (`TF_RAISE_E` 4 / 5 / 6) gives no speed (1.1599 / 1.1599 / 1.1571 against 1.1561) and a larger max error (0.0645 / 0.0995 / 0.1484 at L1024): rejected. Ablations of the packed-fp16 build (L1024, `TF_ABL`): no logits / bias loads 1.094 (-9 %: what the logit step costs), no tile-maximum vote and the byte conversion / P-store ablations produce
wrong bytes (NaN bytes send every row to the exact recompute, 312 ms): no timing information. e4m3 QK ceiling, built as a timing-only variant (one QK^T MMA of the two, `TF_ABL=64`): L1024 1.1574 -> 1.0999 (-5.0 %), L2048 7.9420 -> 7.5562 (-4.9 %), before the quantisation pass of q and k
(2 x the measured V pass = 0.25 ms at L1024, 0.9 ms at L2048): not served (D.5).

### D.3 Module level: the whole TriangleAttention graph, front + core (+ pass) + tail, default route vs FP8 route (same process, 3 interleaved rounds, master fp32 weights, bf16 x, B = 1, starting node, no mask; `t_en.py`, logs `n14-d-en_*.log`)
"gap" = CUDA graph, 256 MB L2 flush before each replay, median of 5 replays after a 0.15 s idle gap; "sustained" = the graph replayed back to back for 3 s a round, `nvidia-smi` power and SM clock every 0.2 s, J = mean W x ms. Ratios are FP8 / default (< 1: FP8 faster / cheaper). dev.sh flagged no foreign process on GPU 0 in any run cited here.

| d_pair x heads x ch (kind) | L | default ms (gap) | FP8 ms (gap) | x gap | x sustained | x J / step | W default / FP8 | MHz default / FP8 |
|---|---|---|---|---|---|---|---|---|
| 128 x 4 x 32 (fused) | 256 | 0.0717 | 0.0779 | 1.086 | 1.118 | 1.050 | 779 / 727 | 1965 / 1965 |
| | 384 | 0.1577 | 0.1679 | 1.065 | 1.042 | 1.017 | 898 / 877 | 1900 / 1965 |
| | 512 | 0.2970 | 0.3072 | 1.034 | 0.999 | 0.995 | 900 / 915 | 1771 / 1892 |
| | 768 | 0.7537 | 0.7844 | 1.041 | 0.989 | 0.993 | 905 / 903 | 1717 / 1837 |
| | **1024** | 1.5584 | 1.5790 | 1.013 | 0.973 | 0.971 | 915 / 914 | 1720 / 1808 |
| | **1536** | 4.5425 | 4.4677 | 0.984 | 0.935 | 0.934 | 902 / 907 | 1736 / 1841 |
| | **2048** | 10.0086 | 9.5743 | 0.957 | 0.908 | 0.905 | 918 / 914 | 1745 / 1843 |
| 128 x 4 x 32, ending node | 384 | 0.1597 | 0.1596 | 0.999 | 0.998 | 0.993 | 918 / 904 | 1906 / 1904 |
| | 768 | 0.7659 | 0.7680 | 1.003 | 1.002 | 0.997 | 940 / 910 | 1729 / 1736 |
| | 1024 | 1.5749 | 1.6005 | 1.016 | 0.966 | 0.966 | 895 / 888 | 1733 / 1826 |
| | 1536 | 4.5772 | 4.5097 | 0.985 | 0.937 | 0.935 | 914 / 910 | 1743 / 1850 |
| | 2048 | 10.0987 | 9.6481 | 0.955 | 0.905 | 0.914 | 934 / 917 | 1730 / 1850 |
| 64 x 4 x 32, ending node | 1536 | 4.4168 | 4.3203 | 0.978 | 0.927 | 0.943 | 903 / 919 | 1764 / 1875 |
| 256 x 8 x 32, ending node | 1536 | 9.5877 | 9.4740 | 0.988 | 0.936 | 0.932 | 925 / 918 | 1696 / 1792 |
| | 2048 | 21.0123 | 20.1255 | 0.958 | 0.904 | 0.896 | 946 / 926 | 1698 / 1784 |
| 64 x 4 x 32 (unfused) | 384 | 0.1434 | 0.1556 | 1.086 | 1.092 | 1.016 | 938 / 831 | 1943 / 1965 |
| | 768 | 0.7148 | 0.7453 | 1.043 | 0.988 | 0.991 | 898 / 946 | 1762 / 1890 |
| | **1024** | 1.4889 | 1.5073 | 1.012 | 0.960 | 0.948 | 910 / 906 | 1750 / 1869 |
| | **1536** | 4.4022 | 4.3059 | 0.978 | 0.927 | 0.907 | 927 / 917 | 1757 / 1874 |
| 64 x 2 x 32 (unfused) | 384 | 0.0901 | 0.0922 | 1.023 | 1.056 | 1.009 | 750 / 733 | 1965 / 1965 |
| | 768 | 0.3994 | 0.4116 | 1.031 | 0.989 | 0.989 | 884 / 894 | 1814 / 1949 |
| | **1024** | 0.8130 | 0.8131 | 1.000 | 0.959 | 0.948 | 909 / 906 | 1789 / 1901 |
| | **1536** | 2.3634 | 2.2682 | 0.960 | 0.921 | 0.925 | 923 / 927 | 1786 / 1893 |
| 256 x 8 x 32 (unfused) | 384 | 0.3236 | 0.3502 | 1.082 | 1.048 | 1.079 | 904 / 931 | 1728 / 1839 |
| | 768 | 1.5892 | 1.6486 | 1.037 | 1.001 | 0.991 | 913 / 912 | 1653 / 1754 |
| | **1024** | 3.2746 | 3.3022 | 1.008 | 0.971 | 0.975 | 912 / 924 | 1673 / 1762 |
| | **1536** | 9.4986 | 9.3697 | 0.986 | 0.937 | 0.932 | 935 / 923 | 1686 / 1778 |
| 384 x 12 x 32 (unfused) | 384 | 0.5530 | 0.5878 | 1.063 | 1.038 | 1.034 | 897 / 907 | 1706 / 1798 |
| | 768 | 2.6870 | 2.7822 | 1.035 | 0.998 | 0.994 | 905 / 920 | 1669 / 1750 |
| | **1024** | 5.4989 | 5.5429 | 1.008 | 0.972 | 0.960 | 924 / 913 | 1669 / 1758 |
| | **2048** | 33.8749 | 32.5150 | 0.960 | 0.895 | 0.902 | 960 / 965 | 1660 / 1782 |
| | **3968** | 236.8564 | 199.7947 | 0.844 | 0.875 | 0.855 | 993 / 953 | 1682 / 1833 |

(d_pair 384 L3968: 2 s sustained rounds; the V calibration of this call reads V with amax / amin, without an fp32 copy: the first version OOMed at 22 GB there.)

Reading: the e4m3 core is 14-16 % faster than the bf16 core from L1024 (D.2) but the standalone V^T pass (5-8 % of the core) and the unchanged front / tail eat part of it, so the module gains 3-4 % (cool) / 3-9 % (sustained) / 3-10 % (energy) from L1024 to L2048 and loses below L768 (the pass and the 2-row core). Under the
power cap the e4m3 route runs ~90-120 MHz higher at the same power (the core draws less per work): sustained and energy gains exceed the cool ones. Gate S >= 1024 (bold rows: gap <= +1.3 %, sustained <= 0.973, J <= 0.975). The ending node behaves the same from L1024 (the ending-node rows below L1024 are level: 0.999 / 1.003 at L384 / 768).
The front-fused form of the planes (the front writing V^T e4m3, as the H100's `tri_front_vt`) is not built (it changes the shared front kernels): its ceiling is the pass, 0.123 / 0.457 ms at L1024 / 2048 (8 / 5 % of the module).

### D.4 Accuracy (fp64 gold, non-zero random weights `randomize`, `t_mod64.py`; judge form `max|y - y64| / max|y64|`; update rms of `y - x`)

| case (gate opened for L < 1024 in the first run, default gate in the second) | default route | FP8 route | ratio | update rms default / FP8 |
|---|---|---|---|---|
| 128 x 4 x 32 L256 / 384 / 512 / 768 (fp32 PyTorch statements) | 0.0033 / 0.0030 / 0.0031 / 0.0030 | 0.0066 / 0.0061 / 0.0056 / 0.0067 | 2.0 / 2.0 / 1.8 / 2.2 | L256 0.0123 / 0.0273, L512 0.0131 / 0.0240, L768 0.0134 / 0.0219 |
| 128 x 4 x 32 L1024, fp64 gold | 0.0030 | 0.0060 | 2.00 | 0.0136 / 0.0207 |
| AF3 128 x 4 x 32 L1024, fp64 gold | 0.0030 | 0.0060 | 2.00 | 0.0136 / 0.0207 |
| 64 x 4 x 32 L1024, fp64 gold | 0.0030 | 0.0064 | 2.16 | 0.0152 / 0.0223 |
| 64 x 2 x 32 L1024, fp64 gold | 0.0030 | **0.0088** (> 8e-3) | 2.95 | 0.0133 / 0.0195 |
| 256 x 8 x 32 L1024, fp64 gold | 0.0029 | 0.0055 | 1.92 | 0.0140 / 0.0210 |
| 384 x 12 x 32 L1024 | the fp64 gold does not fit (as for every backend); L384 in `test_triattn_fp8_b200.py` against fp32: inside 8e-3 | | | |
The H100 FP8 PV band is 0.43-0.66 % judge form and 3.3-3.7 % update rms (docs/fp8.md): the B200 route is at the same band (0.55-0.67 %, 2.0-2.3 %), except 64 x 2 x 32 at L1024 (0.88 %, 10 % over the tolerance; a max-norm statistic of one element, rms 1.95 %): of the models that serve that site only
protenix2 / opendde whitelist the `unfused` kind (end-level evidence on the H100, same numerics class). Kernel level (`t_core.py`, fp64 gold, random q / k / v / bias, 8 pair rows x 4 heads): max 0.040-0.065, rms 0.0405-0.0415 for e4m3 against 0.004 / 0.0036-0.0039 for the bf16 core (rms ratio 10-11: P and V on e4m3 alone are ~2.2 %
rms each); the 64-channel core 0.0476-0.0507 max, 0.0409-0.0412 rms.

Large logits (n7 `qs_big` / `qs_big2` form: weights N(0, 0.5), Wq / Wk x s, bias x s, L384 / L640, `t_stress.py`, error = max |y - fp32 PyTorch| next to the default route's; |y|max 22-34):

| stress | default err | FP8 err | FP8 / default |
|---|---|---|---|
| Wq, Wk x1 (start / end), L384 | 0.250 / 0.297 | 0.500 / 0.500 | 2.0 / 1.7 |
| Wq, Wk x3 (start / end), L384 | 2.586 / 2.125 | 2.211 / 2.344 | 0.85 / 1.10 |
| Wq, Wk x3 (start / end), L640 | 2.859 / 3.016 | 2.891 / 3.289 | 1.01 / 1.09 |
| Wq, Wk x8 (logits ~9x the bf16 range), L384 / L640 | **NaN** | finite, 9.6-11.7 (the logits are beyond bf16: both routes are wrong, only the default is NaN) | |
| bias x4 / x10 / x25 (start), L384 | 0.812 / 2.723 / 5.633 | 1.031 / 2.645 / 5.547 | 1.27 / 0.97 / 0.98 |
| bias x4 / x10 / x25 (end), L384 | 0.875 / 2.125 / 7.395 | 0.988 / 2.250 / 7.301 | 1.13 / 1.06 / 0.99 |
Extreme single-core inputs (`t_extreme.py`: bias spikes +100 / +1000 in the last / first / every tile, 40 / 130 masked first keys, rows with 3 live keys, q x6, flat rows; L256 / 1024): the e4m3 core is finite in all 20 cases with max 0.031-0.067 / rms 0.026-0.042 against the fp64 gold; **no row needed the exact
recompute** (`stat[0]` = 0); the bf16 default core (HB, non-QS build) returns NaN on the late-spike rows (+100 / +1000) of this harness. Saturation protocol (`test_saturation_guard`): a 64x stale scale sets the flag (`peak 9152 > 448`), `fp8.check()` widens x4 three times (through the H100 module's registry, no change in `fp8.py`), the redone call is finite and clean;
graph replay == eager bytes (`test_graph_equals_eager_and_uncalibrated_capture`); an uncalibrated capture takes the default route and is counted (`capture_fallbacks`).

### D.5 Built, measured, NOT served (and what is not built)

* **64-channel-head core** (`TF_D = 64`, 2 S buffers, N = 80, 3-stage ring; Chai-1 trunk / confidence 256 x 4 x 64, AF3 512 x 8 x 64, RF3 64 x 4 x 64): kernel `t_core64.py` (4 heads, L = N): e4m3 core 0.0545 / 0.2585 / 1.6151 / 11.528 ms against the bf16 core 0.0604 / 0.2755 / 1.7583 / 12.170 ms at L256 / 512 / 1024 / 2048
  (x0.90 / 0.94 / 0.92 / 0.95), the V^T pass 0.0207 / 0.0634 / 0.2169 / 0.7853 ms: core + pass x1.25 / 1.17 / 1.04 / 1.01. Module (chai1_256, start): L512 gap x1.122 / sustained 1.086 / J 1.101, L1024 x1.054 / 1.037 / 1.032, L2048 x1.046 / 1.016 / 0.988;
  AF3 512 x 8 x 64: L768 x1.072 / 1.054 / 1.050, L1536 x1.034 / 1.025 / 1.033. Accuracy is in band (judge 0.0040-0.0047 at L768, 1.5-1.6x the default). The two-buffer S ring (TMEM: 80 accumulator columns a row) costs the lookahead the 3-buffer core has: the gain is 5-10 %, below the pass: gate closed (`MIN_S64` 2^30; `test_d64_core_when_opened`).
  Open: a 3-buffer d64 form needs N = 64 PV halves or a second CTA a task; the front-fused planes would remove the pass (0.2 / 0.79 ms at L1024 / 2048 = 8 % / 6.5 % of the module).
* **e4m3 q / k** (the H100 `FP8_QK`): NOT built as a route. Built: the timing ceiling (D.2): one QK^T MMA of the two = -5.0 % of the e4m3 core at L1024 and L2048 before quantising q and k; a q / k pass costs two V passes (0.25 ms at L1024, 0.9 ms at L2048 = 21 % / 11 % of the core): net loss unless the
  front writes the e4m3 rows (a change of the shared front kernels, the H100's `QK8` front). `fp8.status()["triattn"]["qk"]` is False on a B200 (`b200_fp8.FP8_QK`).
  **Superseded by D.7**: the front-written e4m3 rows were built and measured; the route is served from S >= 1024 (status qk follows the whitelist and the gate).
* **RAISE 4 / 5 / 6, `TF_RAISE_E`**: faster (L1024 1.1452 ms at RAISE 4 against 1.2035 at 3) but judge error 0.0088-0.0094 at RAISE 4 and max errors 0.06-0.15: rejected (D.2).
* **Gates**: every head-dim-32 kind below S = 1024: module x1.00-1.09 cool, 1.00-1.12 sustained (table D.3).
* Not done: the 3-row (`G3`) form of the bf16 core needs the row sum ring (`ORING`: 3 x 32 + 32 + S x 3 = 512 columns): the e4m3 core stays at 2 rows; at L1024 the bf16 core with 2 rows is 1.4022 ms against 1.3284 with 3 (-5 %), so a 3-row e4m3 core would be worth up to ~5 % more of the core.

### D.6 Verification (tree n14-d)
* FP8 off: the default route is bit-identical to n14-base: `bitscan_d.py` (TriangleAttention d_pair 128 / 64 x 4 x 32 / 64 x 2 x 32 / 64 x 4 x 16 / 256 / 384, AF3, RF3, both orientations, masks, ragged L1030, B = 2, L256 .. 1030, eager and CUDA-graph replay): 90 cases = 180 tensors, 0 differ (the scan was repeated on the final tree, after the last edit of the route: the same, 0 differ).
  The route is reached through one `if kind is not None and hb and not need and not d16` branch of `_core_forward` / `_core64_forward` that imports `h100_sm90` and tests `FP8_PV`; `backend_versions["cuda:b200_sm100"]` is unchanged.
* `tests/gpu/test_triattn_b200.py` + `test_triattn_b200_capture_bits.py`: 95 passed. `tests/cpu/test_fp8_switch.py`, `test_capture_api.py` (two lines added to `NOT_WEIGHT_DERIVED` for `b200_fp8._SITES` / `_GUARDS`), `test_prebuild_plan.py`: 42 passed.
* `tests/gpu/test_triattn_fp8_b200.py` (new, 30 passed): on / off, accuracy against fp32 at every head-dim-32 geometry and both nodes, ragged + batched + mask, large logits, graph == eager, uncalibrated capture, saturation guard through `fp8.check()`, whitelist (af3 protenix1 openfold3 rosettafold3 fire; opendde (no fused site) / boltz2 / intellifold2 / no model do not), shipped gates, d64 core.
* `_prebuild --arch sm100`: +2 artifacts (`triattn_b200_f8_v2`, the `TF_D = 64` build).

### D.7 TriAttn e4m3 q / k (n21-qk, 2026-11-21): built, measured, served from S >= 1024 (fused d_pair 128; unfused d_pair 64 fronts only on request)

Opt-in like the P / V route (it needs `fp8.enable_triattn` with qk, i.e. the models af3 / protenix1 / openfold3 / rosettafold3 of `fp8.TRIATTN_MODELS`, fused kind; `FOLDJET_ENGINE_TRIATTN_B200_FP8_QK_KINDS=unfused` adds the d_pair-64 fronts); FP8 off is bit identical to n18-cache (below).
Tools and raw logs: `$W/b2511/work/n21-qk/` (`t_q8.py` core, `t_mod3.py` module, `rg.sh` / `q*.sh` runners, `core_speed.log`, `mod_acc_*.log`, `fin_<gpu>_<case>.log` final timings, `m2_ / m3_time_*.log` the iterations, `pt*_*.log` tests, `bits_*`).

**What is built.**
* Core: `triattn_f8_sm100.cu -DTF_Q8=1` (D = 32 and, built, D = 64): QK^T = `tcgen05.mma kind::f8f6f4` e4m3 x e4m3 -> fp32 in TMEM, K = 32 in ONE MMA (the bf16 core issues two of K = 16); q / k tiles are TMA boxes of 32 B (64 B at D = 64) with the 32B (64B) swizzle, smem descriptors SBO 256 (512). Scales: per head powers of two `qa0` (q8 = e4m3(q qa0 qvi)), `ka0` (k8), product 1 at calibration (H100 `qk_scale_from_amax`, headroom 2: q, k scaled so that |q8|, |k8| <= 224, absmax product must fit), a logit multiplier `lm0 qv^2` (`qv` = the in-place widening factor `fp8.check()` -> `rescale_all` multiplies, `qvi = 1 / qv`): `x = f16(S) * scale * lm + bias`; the softmax statistics, the row offset, the P / V path, `exact_row8` (an e4m3 recompute for the rows the window cannot hold) are the P / V core's. Saturation guard = the P / V route's (`guard[1]`, `fp8.check()`, captured graphs follow `qv`).
* Where the rows come from: (1) the FRONT writes them: `tri_front_sm100<..., Q8>` (fused d_pair 128 = `tri_front_q8`, d_pair 64 with 4 x 32 / 2 x 32 = `tri_front64_q8`): the drain scales the fp32 accumulators, converts (`cvt.rn.satfinite.e4m3x2`), stages 32 x 32 B in the 32B-swizzled ring and TMA-stores; the bf16 q / k stores are gone; the guard max is kept in a register and written once after the last tile. (2) a standalone pass (`f8_qkq`, the experiment `FOLDJET_ENGINE_TRIATTN_B200_FP8_QK_PASS=1`): used for the d_pair 256 / 384 and 64-channel sites, which have no e4m3 front (wide fronts not changed).
* Calibration: the first eager call of a site (`_calibrate_qk`: absmax per head of q and k) runs the P / V core on the bf16 rows; from the second call the front writes e4m3 rows. A capture of an uncalibrated site declines (counted, `qk_declined`).

**Core, QK^T operands only (fp64 gold, random q / k / v / bias, 4 heads x 32, N = 16 rows; judge `max|o - o64| / max|o64|`, rms of the error / rms of o64)**

| S | bf16 core max / rms | e4m3 P / V max / rms | e4m3 P / V + q / k max / rms | rms ratio qk / pv | NaN | exact rows |
|---|---|---|---|---|---|---|
| 256 | 0.0041 / 0.0037 | 0.0338 / 0.0409 | 0.0545 / 0.0560 | 1.37 | 0 | 0 |
| 512 | 0.0064 / 0.0039 | 0.0524 / 0.0411 | 0.0818 / 0.0567 | 1.38 | 0 | 0 |
| 1024 | 0.0076 / 0.0041 | 0.0551 / 0.0416 | 0.0813 / 0.0583 | 1.40 | 0 | 0 |

**Core speed (GPU 0 exclusive, 0.15 s gap, ms; q8 / k8 prepared; N = S, 4 heads)**

| S | bf16 core | e4m3 P / V | e4m3 P / V + q / k | x of P / V | q / k pass | V pass | qk route with passes x P / V route |
|---|---|---|---|---|---|---|---|
| 512 | 0.2041 | 0.1828 | 0.1647 | 0.901 | 0.0455 | 0.0293 | 1.129 |
| 1024 | 1.3392 | 1.1399 | 1.0543 | 0.925 | 0.1481 | 0.1218 | 1.050 |
| 2048 | 9.4658 | 7.8865 | 7.3422 | 0.931 | 0.5736 | 0.4572 | 1.004 |

The ablation of D.2 (one of two QK^T MMAs, -5.0 %) under-estimated it: the e4m3 operands also halve the q / k smem and TMA bytes (-7.5 % / -6.9 % at L1024 / L2048). A standalone pass cannot pay for it (x1.05 / x1.00); the front-written rows can.

**How the front route got faster than the P / V route (module, gap protocol, first iteration -> final, C / B = e4m3 q / k over the P / V route)**: the first drain version took the guard max with a warp shuffle and an `atomicMax` on one global word per 32-channel chunk: C / B 1.03-1.055 at L1024 / 2048 (slower than the P / V route, only L4096 0.98 sustained); reading the running maximum first: 1.00 / 1.03; keeping the maximum in a register and one atomic after the last tile (final): 0.95-0.965. (`m2_time_*`, `m3_time_*`, `mod_time_*` logs.)

**Module level, final (CUDA graph of one call, starting node, no mask, master fp32 weights, bf16 x, B = 1; same process, routes A = default bf16, B = e4m3 P / V, C = e4m3 P / V + q / k interleaved, 3 rounds; "gap" = 0.15 s idle + 256 MB L2 flush before each of 5 replays, median, min over rounds; "sust" = graph replayed back to back for 3 s a round with `nvidia-smi` power / clock sampled every 0.2 s; J = mean W x ms; each row measured twice, on GPU 0 and on GPU 1, ratios below are the two runs (neighbouring GPUs idle: no foreign process on GPU 0 / 1 in any run, whole-node `nvidia-smi` state logged at start and end in each log). Ratios are C / B (< 1: the q / k route is faster than the P / V route), C / A against the default.**

| case | L | gap C/B (gpu0 / gpu1) | gap C/A | sust C/B | sust C/A | J C/B | J C/A |
|---|---|---|---|---|---|---|---|
| AF3 128x4x32 (fused) | 1024 | 0.965 / 0.965 | 0.976 | 0.935 / 0.936 | 0.904 / 0.906 | 0.938 | 0.900 / 0.905 |
| | 2048 | 0.954 / 0.956 | 0.912 / 0.917 | 0.921 / 0.923 | 0.835 / 0.841 | 0.921 / 0.919 | 0.846 / 0.832 |
| | 3072 | 0.949 / 0.949 | 0.873 / 0.876 | 0.918 / 0.916 | 0.789 / 0.793 | 0.923 / 0.919 | 0.784 / 0.783 |
| | 4096 | 0.957 / 0.955 | 0.841 / 0.845 | 0.898 / 0.886 | 0.760 / 0.762 | 0.900 / 0.890 | 0.756 / 0.758 |
| Protenix 128x4x32 (`TriangleAttention`, fused) | 1024 | 0.964 / 0.964 | 0.974 / 0.975 | 0.934 / 0.936 | 0.905 / 0.907 | 0.940 / 0.940 | 0.900 / 0.896 |
| | 2048 | 0.954 / 0.955 | 0.913 / 0.916 | 0.924 / 0.924 | 0.837 / 0.842 | 0.921 / 0.919 | 0.836 / 0.828 |
| | 3072 | 0.949 / 0.949 | 0.872 / 0.876 | 0.917 / 0.918 | 0.789 / 0.794 | 0.910 / 0.910 | 0.781 / 0.785 |
| | 4096 | 0.956 / 0.956 | 0.841 / 0.851 | 0.907 / 0.885 | 0.762 / 0.764 | 0.901 / 0.887 | 0.758 / 0.763 |
| RF3 128x4x32 (fused) | 1024 | 0.964 / 0.966 | 0.974 / 0.977 | 0.932 / 0.935 | 0.904 / 0.905 | 0.934 / 0.916 | 0.900 / 0.889 |
| | 2048 | 0.955 / 0.955 | 0.909 / 0.916 | 0.924 / 0.923 | 0.838 / 0.842 | 0.914 / 0.916 | 0.834 / 0.831 |
| | 3072 | 0.949 / 0.949 | 0.872 / 0.875 | 0.918 / 0.918 | 0.791 / 0.796 | 0.907 / 0.902 | 0.775 / 0.776 |
| | 4096 | 0.957 / 0.955 | 0.842 / 0.851 | 0.907 / 0.879 | 0.758 / 0.763 | 0.905 / 0.881 | 0.757 / 0.760 |
| 64x4x32 (unfused, d_pair-64 front) | 1024 | 0.960 / 0.961 | 0.971 | 0.932 / 0.933 | 0.896 | 0.919 / 0.935 | 0.888 / 0.897 |
| | 2048 | 0.952 / 0.953 | 0.904 / 0.909 | 0.922 / 0.922 | 0.828 / 0.830 | 0.922 / 0.924 | 0.832 / 0.834 |
| | 3072 | 0.947 / 0.947 | 0.866 | 0.915 / 0.915 | 0.780 / 0.788 | 0.913 / 0.909 | 0.766 / 0.779 |
| 64x2x32 (unfused, d_pair-64 front) | 1024 | 0.967 / 0.965 | 0.962 / 0.968 | 0.942 / 0.946 | 0.903 / 0.910 | 0.927 / 0.915 | 0.887 / 0.873 |
| | 2048 | 0.961 / 0.963 | 0.903 / 0.909 | 0.930 / 0.933 | 0.838 / 0.843 | 0.930 / 0.926 | 0.833 / 0.831 |
| | 3072 | 0.954 / 0.957 | 0.869 / 0.875 | 0.923 / 0.924 | 0.802 / 0.808 | 0.920 / 0.922 | 0.796 / 0.800 |

Not served (the route would be the standalone pass; measured with `QK_PASS`, first run, same protocol; C / B gap, sustained): 256x8x32 L1024 1.042 / 1.023, L2048 1.011 / 0.986; 384x12x32 L1024 1.037 / 1.020, L2048 1.006 / 0.982; Chai-1 256x4x64 (d64 core, gate opened) L1024 1.071 / 1.039, L2048 1.032 / 0.992; RF3 64x4x64 (d64) L1024 1.063 / 1.061, L2048 1.032 / 0.997. The 64-channel-head core does not benefit from e4m3 q / k either (no e4m3 front for it; with the pass it loses 3-7 %).

**Gates set from these numbers** (`b200_fp8.QK_MINS_SHIPPED`; rule: open only where the q / k route is >= 2 % faster than the P / V-only route in the gap protocol AND sustained, reproduced on two GPUs): `fused` 1024, `unfused` 1024 (effective for the d_pair-64 fronts only, and only with `FOLDJET_ENGINE_TRIATTN_B200_FP8_QK_KINDS=unfused`: the whitelist's qk flag covers the fused kind), `d64` closed. Below S = 1024 the P / V route itself is not served. `status()["triattn"]["qk"]` is True only when the H100-module switch (`enable_triattn` qk, whitelisted model) is on AND a gate is open. L1024 gain 3.5 % (gap) / 6.5 % (sustained), L4096 4.4 % / 9-11 %; the gain under the power cap is larger because the e4m3 q / k route draws less power at the same work (1906-1964 MHz against 1836-1896).

**Accuracy (module, judge form max|y - y64| / max|y64| against the fp32 PyTorch statements, random non-zero weights, key mask; A default, B P / V, C q / k from the front, P calibration call (pass)): 'x' = the P / V route's value**

| case | L | A | B | C | C / B |
|---|---|---|---|---|---|
| 128x4x32 (AF3, Protenix) | 384 / 1024 | 0.0031 / 0.0030 | 0.0060 / 0.0058 | 0.0119 / 0.0103 | 2.0 / 1.8 |
| RF3 128x4x32 | 384 / 1024 | 0.0031 / 0.0029 | 0.0068 / 0.0051 | 0.0093 / 0.0089 | 1.4 / 1.7 |
| 64x4x32 | 384 / 1024 | 0.0033 / 0.0030 | 0.0058 / 0.0058 | 0.0107 / 0.0088 | 1.8 / 1.5 |
| 64x2x32 | 384 / 1024 | 0.0032 / 0.0030 | 0.0060 / 0.0087 | 0.0108 / 0.0131 | 1.8 / 1.5 |
| 256x8x32 (pass) | 384 / 1024 | 0.0030 / 0.0029 | 0.0056 / 0.0047 | 0.0074 / 0.0074 | 1.3 / 1.6 |
| Chai-1 256x4x64 (pass, d64 core) | 384 / 1024 | 0.0031 / 0.0028 | 0.0054 / 0.0057 | 0.0084 / 0.0087 | 1.6 / 1.5 |

Update rms (L1024): default 0.0089-0.0152, P / V 0.0135-0.0223, q / k 0.0181-0.0315. The error of the q / k route is 1.3-2.0 x the P / V route's and 2.5-4.4 x the default's; all values finite; the judge tolerance of the H100 FP8 PV (8e-3) is crossed (as on the H100: 0.85-0.95 %); `TOL_QK` of the tests = 1.5e-2 (the largest measured value 0.0131 plus 15 %). The front route and the calibration-call pass route give the same error (difference <= 0.0015). Stress (`test_triattn_fp8_qk_b200.py`): extreme logits with q, k x 6 flip the winning key of some rows (the 6 % rounding of q / k moves a logit of ~36 sigma by ~2): max error 1.04 x max|o| (a wrong V row, finite, no NaN); bias x 25 with 130 masked keys and 250 masked keys: inside 0.35 max|o| + 0.2; no row needed the exact recompute in any core test. **End-level (structure) accuracy on the B200 has NOT been run**; the H100 evidence for q / k (af3 / protenix1 / openfold3 / rosettafold3) does not cover tcgen05 accumulation / the B200 scales.

**Saturation, graph, off**: a Q weight x 200 saturates q8 (guard peak > 448), `fp8.check()` widens `qv` in place (x4 per step), the redone call is clean and the scales were widened, not recalibrated (`test_saturation_guard_qk`); graph replay == the eager call (calibrated site); an uncalibrated capture declines the q / k part (counted); a closed gate calibrates nothing and returns the P / V route's bytes.

**Verification.** FP8 off (default route) against n18-cache, `bitscan_d.py` (TriangleAttention d_pair 128 / 64 / 256 / 384, AF3, RF3, both orientations, masks, ragged L1030, B = 2, L256 .. 1030; eager and CUDA-graph replay; random non-zero weights): 90 cases = 180 tensors, 0 differ, 0 missing. Tests: `test_triattn_fp8_qk_b200.py` (new) + `test_triattn_fp8_b200.py` 62 passed; `test_triattn_b200.py` + `test_triattn_b200_capture_bits.py` (+ the new file) 127 passed; `tests/cpu` 3504 passed, 2 skipped (n18-cache: 3500; `test_fp8_switch.py` no longer overrides qk to False on sm100). `_prebuild --arch sm100`: plan 837 artifacts (+2: the q / k build and its 64-channel form), 0 not built; `--check` rc 1 exactly as the base tree (same 1545 lines + 2 `have ext` lines).
**Not done**: end-level accuracy; the wide fronts (d_pair 256 / 384) and the 64-channel-head fronts do not write e4m3 rows (the pass loses); fp32 / training / bidirectional unchanged.




# Agent E (n14-e, 2026-11-19): atom DiT FP8 rows route on tcgen05, Boltz-2 token DiT up-gate expand, RoseTTAFold3 expand, token DiT placements re-measured

Opt-in, OFF by default, every default route bit-identical (section "Verification"). H100 digest (what is e4m3, scales, per-sample fallback, gates and why, accuracy acceptance, speedups, traps; transfers vs redo): `docs/b200/fp8_h100_atomdit_digest.md`
(copy `$W/b2511/work/n14-e/h100_digest.md`). Registration for `fp8.py`: `$W/b2511/work/n14-e/FP8_REGISTER.md`. Order of the user: build the best-effort B200 kernel first, measure at the requirement shapes, then decide; every decision below
cites numbers of a BUILT kernel.

## E.0 Summary

| route | built | served (gated, opt-in) | measured result (B200 GPU 1, graph, 0.15 s idle gap, sustained, same-process interleaved) |
|---|---|---|---|
| atom DiT `pre` (q / k / v / gate projections, `pre_fwd_f8.cu`) | yes: tcgen05 e4m3, weights in TMEM, x1 / xkv quantised by the AdaLN threads, plain / cross / PARALLEL / RMS / KQLN / rotary builds | rows <= 10240 (2048 atoms x 5) | stage alone 0.88 (5120 rows) .. 1.06 (40960); free-quantisation ablation 0.85 .. 0.99 |
| atom DiT `post` (Wo, SwiGLU expand incl. Boltz-2 up gate, `post_fwd_f8.cu`) | yes: Wo (smem A) and [Wa; Wb; Wu] (TMEM A) on e4m3, gated / x2 quantised by the row threads | same | stage alone 0.94 .. 1.04; ablation 0.90 .. 0.97 |
| atom DiT squeeze (Ws, K 256) | NOT built (needs a cross-lane row amax of h and an e4m3 transposed store); measured bound: `SQABL` build (e4m3 squeeze MMA on garbage, free quantisation) | no | see E.2 (SQABL row) |
| Boltz-2 token DiT up-gate expand (`gemm_swiglu_f8_up_sm100.cu`) | yes: 2-CTA M256 N192 tiles [a64 | b64 | u64], 64-unit items | `expand` placement, rows >= 1280 (default on with `expand`) | kernel 0.45-0.77 of the bf16 route (cuBLAS + `swiglu_rows`) from M 1280, block 0.89-0.96 (M 1280 .. 5120) |
| RoseTTAFold3 token DiT expand (`rows_q8` + the e4m3 expand, parallel non-identity block) | the B200 guard of `token_dit_fp8.active` opened (`FOLDJET_ENGINE_TOKEN_DIT_FP8_B200_RF3`, default on) | `expand` placement, rows >= 2560 | 0.95-0.99 of the block from 2560 rows (E.5), coin flip below |
| token DiT qkvg / out / squeeze, Chai-1 identity out / squeeze | the H100 forms (cuBLAS e4m3 `_scaled_mm` + quantising row passes) re-measured alone and in the H100 combinations | no | qkvg 0.99-1.01, out 1.02-1.05, squeeze 1.02-1.05, qkvg + expand = expand (0.93-0.985), all four 1.00-1.04 (E.5) |

## E.1 Atom DiT rows route: design (the H100 recipe on tcgen05)

Files (all new; `pre_fwd.cu` / `post_fwd.cu` and every sm90 / sm80 source are untouched, so no default cubin re-keys): `kernels/augmented_attention/cuda/b200_atom/{pre_fwd_f8.cu, post_fwd_f8.cu, f8.py}` (the two `.cu` are the sm_100a halves of
`pre_fwd.cu` / `post_fwd.cu` with the `F8` sections switched on), call sites in `integrations/local_dit_b200.py` and `local_dit_esm_b200.py` (`f8.serves(stage, rows, save)` guards, `ENABLED` empty = never taken).

* Quantisation (H100 recipe): activation scale DYNAMIC per row, max / 448 (exact, not a power of two: the epilogue is fp32 anyway), made by the eight AdaLN threads of the row (16 channels each, `quant_row16`: |max| + 3 xor shuffles,
  `cvt.rn.satfinite.e4m3x2.f32`, one 16-byte store = chunk `ak` of the 128-byte swizzled row); weights static per OUTPUT CHANNEL (max / 448), once per weight pack (`f8.pre_pack8` / `post_pack8`, `_capture.param_forms(capture_reads_eager)`),
  K axis permuted so that thread `ak`'s 16 channels are k-slots `16 ak ..` (the `_kperm` table). No calibration, no guard, no saturation possible.
* `pre`: the transposed 32-row tile structure of `pre_fwd.cu` is kept (p^T = W_p x1^T, W in TMEM as the A operand, four accumulators double buffered): W_p e4m3 `[128][64 B]` SW64 atoms (16 KB per projection, 4 `tcgen05.cp` per projection), x1 as one
  `[32][128 B]` SW128 e4m3 tile, `umma8_ts` kind::f8f6f4 M128 N32 K32 x 4 steps per projection (was 8 bf16 steps). Cross builds quantise xkv separately (its own row scale). Epilogue: `acc * sx[row] * sw[channel]` in fp32 BEFORE bias / sigmoid
  (the TMEM fragment has channel = lane, row = column: the row scale is a `float2` smem read per tile, the channel scale an L1 load). KQLN (RoseTTAFold3) / QKROPE (ESMFold2) epilogues run unchanged on the staged rows.
* `post`: P1 quantises gated = sg o (Wo e4m3 in smem, `umma8_ss` M128 N16 K32 x 4), P2 quantises x2 after the AdaLN (Wa | Wb | Wu e4m3 in TMEM, `umma8_ts` x 4 steps, Boltz-2's Wu in smem `umma8_ss`); h stays bf16 and the squeeze Ws stays the bf16
  TMEM product (an e4m3 h needs a row amax across the 128 accumulator lanes and an e4m3 transposed store: not built, see E.3). Row scales live in a 768-byte smem table per (warpgroup, stage).
* Not changed: pair bias, conditioning tables, attention, residual / gate arithmetic, PDL, EB prefetch, side streams, folds.

## E.2 Atom DiT: the built kernels at the requirement shapes (B200 GPU 1, `dev.sh -x`, graph, 7 interleaved rounds, 0.15 s idle gap, median; us)

Stage alone (random rows, one call per graph node x 20; bf16 = the shipped `pre_fwd` / `post_fwd`; rows M = A N; 5120 = 1024 atoms x 5 samples, 20480 = 4096 x 5, 40960 = 8192 x 5):

| stage | M 5120 | M 12800 | M 20480 | M 40960 |
|---|---|---|---|---|
| `pre` plain (AdaLN 1 + q k v g), bf16 -> e4m3 us | 7.15 -> 6.28 (0.878) | 8.79 -> 8.41 (0.957) | 10.77 -> 10.85 (1.008) | 16.62 -> 17.62 (1.060) |
| `pre` cross (Protenix / OpenFold3: + K / V AdaLN) | 7.87 -> 7.03 (0.893) | 10.50 -> 10.28 (0.979) | 13.48 -> 13.77 (1.021) | 21.86 -> 23.65 (1.082) |
| `pre` plain, FREE quantisation (ablation `F8ABL`: no amax, no cast, garbage values) | 0.845 | 0.904 | 0.952 | 0.988 |
| `pre` cross, free quantisation | 0.868 | 0.904 | 0.941 | 0.977 |
| `post` (Wo + SwiGLU expand on e4m3; squeeze bf16) | 9.48 -> 8.92 (0.940) | 14.36 -> 14.37 (1.001) | 18.87 -> 19.08 (1.011) | 31.25 -> 32.41 (1.037) |
| `post`, free quantisation (`F8ABL`) | 0.904 | 0.944 | 0.951 | 0.967 |
| `post`, free quantisation + e4m3 squeeze MMA (`F8ABL` + `SQABL`: K 256 e4m3 on the h bytes) | 0.895 | 0.934 | 0.937 | 0.950 |

What this says: the e4m3 MMAs are worth 5-15 % of a row stage at 5120 rows and 1-5 % at >= 20480 even with the quantisation made free; the per-row amax + cast the AdaLN / row threads add (the binding resource of
these kernels: `pre_fwd` is TMA / row-thread bound, tensor time 13-16 % of the block) costs 4-8 % at every size. The B200 stages are not the serial `cp.async -> wgmma -> epilogue` chain of the H100, where the e4m3 product
was 9-13 % of the step. The e4m3 squeeze (Ws, K 256) is worth at most 0.9-1.7 % of `post` more (the `SQABL` row against the `F8ABL` row: 0.1-0.5 us of a 30-90 us block call), and needs a row amax of h across the 128
accumulator lanes plus an e4m3 transposed store: NOT built, bound measured.

### Block level (graph replay of a captured module call, 2 blocks per call, master weights, shared conditioning, A = 5 samples; every row gate removed; `blk_atom.py`)

Ratio e4m3 / default, sustained (2 s idle + 1 s back-to-back burst, then 3 interleaved blocks; mean over the registry variants of the class, [min .. max]); default us -> e4m3 us in the last column (class mean):

| class (variants) | 1024 atoms (5120 rows) | 2048 (10240) | 4096 (20480) | 8192 (40960) |
|---|---|---|---|---|
| Protenix (4) | 0.960 [0.935..0.997] 29.9 -> 29.0 | 0.994 [0.976..1.001] 36.2 -> 35.9 | 1.016 [1.002..1.032] 52.0 -> 52.6 | 1.025 [1.018..1.035] 89.2 -> 91.5 |
| OpenFold3 (6) | 0.954 [0.936..1.000] 32.0 -> 30.6 | 1.005 [0.962..1.044] 37.9 -> 37.7 | 1.030 [1.015..1.040] 52.9 -> 54.1 | 1.033 [1.021..1.037] 90.9 -> 93.7 |
| Chai1 (8) | 0.992 [0.965..0.999] 30.0 -> 29.4 | 0.999 [0.991..1.002] 36.7 -> 36.7 | 1.006 [0.989..1.032] 52.9 -> 53.1 | 1.022 [1.013..1.027] 89.3 -> 91.0 |
| Boltz2 (4) | 0.964 [0.934..0.997] 29.9 -> 28.9 | 0.980 [0.963..0.999] 37.0 -> 36.5 | 0.989 [0.964..1.000] 52.4 -> 51.9 | 1.006 [1.000..1.011] 90.2 -> 90.7 |
| RoseTTAFold3 (4) | 0.981 [0.958..0.999] 30.0 -> 29.8 | 1.006 [1.000..1.022] 37.0 -> 37.1 | 1.027 [1.014..1.033] 52.6 -> 54.2 | 1.037 [1.034..1.044] 91.8 -> 95.2 |
| ESMFold2Pair (2) | 0.946 [0.946..0.946] 30.9 -> 29.6 | 1.000 [0.999..1.000] 37.3 -> 37.2 | 1.034 [1.032..1.035] 57.0 -> 58.6 | 1.031 [1.023..1.040] 102.5 -> 105.7 |


(Variants: pad and slide key windows, OpenFold3 `slide_qblock` / `mask_form=af3`, Chai-1 identity / `adaln_eps`; `D5` = 5 folds (`depth` = F: rows = A F N, so a D5 case at L1024 is 25600 rows): 1.018 / 1.033 / 1.047 / 1.057 at 1024 / 2048 / 4096 / 8192 atoms,
Protenix.) Small shapes are launch bound (~20 us a block): 256 / 512 atoms x 1 / 5 samples (256 .. 2560 rows) Protenix / OpenFold3 / Chai-1: 0.92-1.01 cool, no trend (noise +-0.03).
Stage selection (Protenix / OpenFold3 / Chai-1 at 1024 / 2048 atoms x 5, sustained): `pre` only 0.99-1.00 (error 1.8-4.3 %), `post` only 0.94-1.01 (error 5.3 %): the gain is in `post` and `pre` alone buys nothing (the H100 finding).

### Sustained power and energy per block call (`ENERGY=1`: 8 s back-to-back replays per mode, nvidia-smi 0.2 s, def / e4m3 / def / e4m3, mean of the pairs)

| class, atoms x 5 (rows) | default us / W | e4m3 us / W | time | energy per call |
|---|---|---|---|---|
| Protenix 1024 (5120) | 28.8 / 440 | 28.1 / 422 | 0.974 | 0.934 |
| Protenix 2048 (10240) | 34.9 / 504 | 35.0 / 482 | 1.000 | 0.956 |
| Protenix 4096 (20480) | 52.0 / 622 | 52.9 / 594 | 1.017 | 0.971 |
| OpenFold3 1024 | 30.9 / 436 | 28.9 / 428 | 0.936 | 0.919 |
| OpenFold3 2048 | 35.0 / 514 | 35.0 / 490 | 1.000 | 0.951 |
| OpenFold3 4096 | 52.1 / 620 | 53.2 / 594 | 1.022 | 0.979 |
| Chai-1 1024 | 28.7 / 423 | 26.8 / 414 | 0.932 | 0.912 |
| Chai-1 2048 | 33.0 / 497 | 32.9 / 476 | 0.998 | 0.957 |
| Chai-1 4096 | 47.6 / 592 | 47.7 / 560 | 1.002 | 0.946 |

The blocks are far below the 1000 W cap at these sizes (420-620 W), so time and energy decouple: e4m3 saves 4-9 % energy through 2048 atoms at equal time, and loses time above that.

### Decision (served / not served)

* Served (opt-in, `FOLDJET_ENGINE_ATOMDIT_FP8_B200=1`): rows <= 10240 (`f8.MAX_ROWS`; 1024 atoms x 5 samples: -1 .. -5 % time (class means, up to -7 % on single variants), -7 .. -9 % energy; 2048 x 5: level time, -4 .. -5 % energy). Above it the measured time is +1.5 .. +6 % (4096 atoms: 1.002-1.030,
  8192 atoms: 1.019-1.038, five-fold stacks 1.018-1.057), the energy gain (2-5 %) does not pay for it.
* The gain is 1-2 us of a ~30 us block call at best; end to end the atom DiT is 3-12 % of a warm forward, so the route is worth <= 0.3 % of a forward. It stays out of `safe`.
* Not served: fp32 / training / `save`, `AtomDiTMiniWorld` and `SWAAtomTransformer` (the swa_dit chain, no H100 route either), the non-pair `AtomDiTESMFold2` (never reaches the tcgen05 block).

## E.3 Accuracy (fp64 gold; `acc_atom.py`, L 1024, A 5, master bf16 weights `N(0, 1/fan_in)`-style registry modules; update error = rms(y - y64) / rms(y64 - x); judge = max|y - y64| / max|y64|)

Per class, x = input scale, off = input offset, w8 = weights x 8; cell = bf16 default route -> e4m3 route (rms ratio):

| class | x 1e-3 | x 1 | x 40 | x 1, off 25 | w8, x 1 |
|---|---|---|---|---|---|
| Protenix | 0.0095 -> 0.0724 (7.6x) | 0.0093 -> 0.0551 (5.9x) | 0.2306 -> 0.2368 (1.0x) | 0.1370 -> 0.1473 (1.1x) | 0.0165 -> 0.0693 (4.2x) |
| OpenFold3 | 0.0093 -> 0.0717 (7.7x) | 0.0093 -> 0.0550 (5.9x) | 0.2302 -> 0.2365 (1.0x) | 0.1372 -> 0.1474 (1.1x) | 0.0153 -> 0.0701 (4.6x) |
| Chai1 | 0.0065 -> 0.0558 (8.6x) | 0.0075 -> 0.0630 (8.4x) | 0.0932 -> 0.1125 (1.2x) | 0.0727 -> 0.0958 (1.3x) | 0.0063 -> 0.0514 (8.1x) |
| Boltz2 | 0.0104 -> 0.0855 (8.2x) | 0.0094 -> 0.0661 (7.0x) | 0.1996 -> 0.2102 (1.1x) | 0.1217 -> 0.1381 (1.1x) | 0.0195 -> 0.0845 (4.3x) |
| RoseTTAFold3 | 0.0071 -> 0.0540 (7.6x) | 0.0087 -> 0.0540 (6.2x) | 0.2347 -> 0.2409 (1.0x) | 0.1396 -> 0.1495 (1.1x) | 0.0067 -> 0.0522 (7.8x) |
| ESMFold2Pair | 0.0116 -> 0.0939 (8.1x) | 0.0080 -> 0.0538 (6.7x) | 0.1539 -> 0.1625 (1.1x) | 0.0772 -> 0.0910 (1.2x) | 0.0145 -> 0.1183 (8.2x) |


Every output finite; no saturation exists (row max maps to 448). At x 40 and offset 25 both routes are bound by the input's bf16 rounding (the update is a small fraction of the stream): ratio 1.0-1.3. The route is outside the 8e-3 judge (as every FP8 route and as the H100: update
error 5.4-8.5 %, 4-8x the bf16 route), acceptance is the H100's (end level, not re-run here: same quantisation recipe). Graph == eager bitwise, also after replaying with x 40 + 25 and x 1e-3 contents in the static input (`test_graph_equals_eager`).

## E.4 Token DiT: Boltz-2 up-gate expand on tcgen05 (built, served with `expand`)

`gemm_swiglu_f8_up_sm100.cu` (new; `gemm_swiglu_f8_sm100.cu` untouched): h = bf16(silu(a) b u), [a | b | u] = (sx x8)(W8 sw)^T. W8 [3 H, 768] is the H100 `_bf16_wab64` pack ([Wa 64 | Wb 64 | Wu 64] per chunk); one item = (tile pair, 64-unit chunk) = a
2-CTA `cta_group::2` M 256 N 192 K 768 product (CTA c of the pair holds B rows 96 c .. 96 c + 95 of the chunk = one 96-row TMA box: Wa64 + Wb32 | Wb32 + Wu64, so the split is contiguous), accumulators [a 64 | b 64 | u 64] in TMEM double
buffered, epilogue warpgroup s (= TMEM buffer s) takes items q = s, s + 2, ...: acc x sx[row] x sw[unit] for the three operands in fp32, silu(a) b u (tanh.approx sigmoid as the plain kernel), one 128 x 64 bf16 tile staged for TMA stores. 5 stages of 28 KB.
Hooks: `SwigluF8Up`, `_Ext.gemm_swiglu_f8(..., up=True)` (the assert is gone), `token_dit_fp8.active` (`B200_UP`, `MIN_ROWS_B200_UP` = 1280), prebuild item.

Kernel alone (graph, 50 back-to-back, median of 7; bf16 route = cuBLAS [a | b | u] GEMM + `swiglu_rows`, the shipped Boltz-2 B200 path; M = A L rows, K 768, H 1536):

| M | 128 | 384 | 640 | 1280 | 1920 | 2560 | 3200 | 3840 | 5120 | 7680 |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 route us | 5.94 | 7.49 | 8.86 | 12.21 | 17.18 | 20.43 | 24.20 | 27.97 | 35.93 | 55.99 |
| e4m3 up kernel us | 7.90 | 8.20 | 8.23 | 9.40 | 12.00 | 13.56 | 15.41 | 16.06 | 19.78 | 25.38 |
| ratio | 1.332 | 1.094 | 0.929 | 0.770 | 0.698 | 0.664 | 0.637 | 0.574 | 0.550 | 0.453 |

Correctness: against a torch reference on the SAME dequantised operands (fp64): relative rms 1.7e-3 (bf16 output + tanh.approx), M 128 / 384 / 512 / 1920 / 5120.

Block (`TokenDiTBoltz2`, A = 5, master, shared cond, graph, 7 rounds + 0.15 s gap; e4m3 up expand / default; two registry variants r108 / r185):

| M | 640 | 1280 | 1920 | 2560 | 3840 | 5120 |
|---|---|---|---|---|---|---|
| r108 cool / sustained | 1.033 / 1.038 | 0.939 / 0.938 | 0.977 / 0.974 | 0.959 / 0.964 | 0.931 / 0.931 | 0.926 / 0.927 |
| r185 cool / sustained | 0.961 / 0.956 | 0.922 / 0.932 | 0.890 / 0.892 | 0.945 / 0.941 | 0.921 / 0.920 | 0.924 / 0.924 |

Sustained power and energy per block call (`ENERGY=1`, def / e4m3 / def / e4m3, 8 s each; r108 / r185): M 1280: time 0.951 / 0.933, energy 0.876 / 0.855 (780 -> 718 W, 693 -> 635 W); M 2560: 0.956 / 0.930, energy 0.878 / 0.842; M 5120: 0.916 / 0.911,
energy 0.923 / 0.885 (the block sits at the 1000 W cap: 973 -> 981 W, 976 -> 948 W). Time and energy both fall; the energy gain is the larger one below the cap.


Gate 1280 rows (640 rows is a coin flip: 1.033 / 0.961 on two variants). Error: update rms 6.5 % (bf16 route 0.75 %; 8.7x), `acc_tok.py` table below. The plain expand placement of Protenix / MiniWorld / Chai-1 stays as shipped by agent C
(0.94-0.98 of the block from 1280 rows).

## E.5 Token DiT: placements re-measured the H100 way, alone and in the H100 combinations

Same protocol as above (`blk.py` of agent C, B200 placements selected by `FOLDJET_ENGINE_TOKEN_DIT_FP8_B200`, every row gate removed, cool = 0.15 s gap, 7 rounds, median over the registry variants [min .. max]; Protenix 13 variants, Chai-1 6; A = 5, L 256 / 512 / 768 / 1024).
The forms are the H100 ones (cuBLAS e4m3 `_scaled_mm` row-wise for qkvg / out / squeeze, the quantising AdaLN passes `adaln_in_rows_cf_q8` / `resgate_adaln_cf_q8` / `adaln_in_rows_cf_par_q8` of the arch-neutral `token_dit_fp8_rows.cu` built for sm_100a,
`rows_q8` for the standalone passes):

| placements | class | M 1280 | M 2560 | M 3840 | M 5120 | update rms (bf16 route 0.0065-0.0070) |
|---|---|---|---|---|---|---|
| expand | Protenix | 0.965 [0.930..0.995] | 0.951 [0.923..1.010] | 0.983 [0.957..0.999] | 0.976 [0.967..0.994] | 0.0499 |
| expand | Chai1 | 0.951 [0.943..1.032] | 0.937 [0.876..0.957] | 0.968 [0.958..0.988] | 0.961 [0.956..0.971] | 0.0506 |
| qkvg | Protenix | 1.003 [0.968..1.024] | 1.004 [0.987..1.022] | 1.008 [0.993..1.021] | 0.996 [0.988..1.006] | 0.0097 |
| qkvg | Chai1 | 1.004 [0.967..1.047] | 0.993 [0.987..1.007] | 0.995 [0.993..1.021] | 1.007 [0.996..1.009] | 0.0063 |
| out | Protenix | 1.051 [1.005..1.075] | 1.028 [0.999..1.042] | 1.033 [1.007..1.051] | 1.025 [1.002..1.034] | 0.0075 |
| out | Chai1 | 1.033 [1.002..1.075] | 1.028 [1.023..1.041] | 1.042 [1.039..1.057] | 1.025 [1.024..1.033] | 0.0140 |
| squeeze | Protenix | 1.040 [0.994..1.066] | 1.025 [1.001..1.036] | 1.032 [1.008..1.048] | 1.015 [1.002..1.030] | 0.0349 |
| squeeze | Chai1 | 1.048 [1.018..1.124] | 1.042 [1.036..1.051] | 1.050 [1.047..1.063] | 1.030 [1.013..1.035] | 0.0353 |
| qkvg+expand | Protenix | 0.956 [0.903..0.993] | 0.964 [0.926..1.009] | 0.985 [0.964..0.999] | 0.976 [0.964..0.995] | 0.0503 |
| qkvg+expand | Chai1 | 0.957 [0.922..1.025] | 0.931 [0.903..0.953] | 0.964 [0.952..0.983] | 0.960 [0.955..0.972] | 0.0506 |
| all4 | Protenix | 1.010 [0.988..1.064] | 1.026 [0.948..1.037] | 1.043 [1.000..1.060] | 1.023 [1.002..1.037] | 0.0608 |
| all4 | Chai1 | 0.997 [0.974..1.049] | 0.991 [0.968..1.007] | 1.020 [1.008..1.032] | 0.970 [0.967..0.976] | 0.0627 |


(Chai-1 `qkvg` is never served for the identity AdaLN: that row is the default route, i.e. the noise floor of this A/B, +-0.03; Chai-1 serves expand / out / squeeze.)

* `expand` alone 0.94-0.98; `qkvg + expand` 0.93-0.985 = `expand` alone within the spread (qkvg alone 0.99-1.01, error 1.0-1.5 %); `out` alone 1.02-1.05 (error 0.8-1.4 %); `squeeze` alone 1.02-1.05 (error 3.5 %); all four 1.00-1.04 on Protenix,
  0.97-1.02 on Chai-1 (error 6.1-6.3 % against 5.0 % for `expand`). No placement beyond `expand` pays on B200, at any of the four sizes: the verdict of the first agent holds with the H100 forms measured alone and in every H100 combination.
* Why `out` / `squeeze` cannot follow the H100 recipe "quantise in the previous kernel's epilogue": og is written by 16 head CTAs (48 channels each) so no CTA owns the row maximum, and h (the expand's output) spans 12 expand tiles per row; a per-row scale
  therefore needs its own pass (`rows_q8`: 3.6-5.9 us at M 2560-5120), which is what these placements lose. The only epilogue-quantisation form that fits is block scaling along K (MXFP8, `kind::mxf8f6f4`, 32-element e8m0 blocks; 48-channel heads straddle blocks).
  Upper bound with a FREE quantisation (cuBLAS e4m3 GEMM alone, agent C's table: out 4.9 -> 4.9 us at M 2560 and 7.1 -> 6.8 at 5120; squeeze 6.7 -> 5.9 and 10.7 -> 8.4): <= 0.8 us at M 2560 (0.8 % of the block), <= 2.6 us at 5120 (1.4 %). Not built.
* RoseTTAFold3 `expand` (parallel non-identity block, `rows_q8(xt)` + the e4m3 expand, guard opened by `FOLDJET_ENGINE_TOKEN_DIT_FP8_B200_RF3=1`): the block (RoseTTAFold3, A = 5, r109 / r186) e4m3 / default at M 640 / 1280 / 2560 / 3840 / 5120: 1.043 / 0.966, 0.978 / 1.038, 0.995 / 0.951, 0.963 / 0.973, 0.971 / 0.970 (update rms 5.3-5.4 % against 0.65-0.70 %, 7.7-8.9x): a 3 % gain from 2560 rows, a coin flip below. Served with the `expand` placement from 2560 rows (`MIN_ROWS_B200_RF3`, default on, `FOLDJET_ENGINE_TOKEN_DIT_FP8_B200_RF3=0` turns it off). The first agent declined this block on an estimate (the quantising pass costs what the expand saves): the measured pass (`rows_q8`) is hidden behind the side-stream schedule.


### Accuracy, token DiT (fp64 gold, `acc_tok.py`, L 384, A 5; update rms bf16 default -> e4m3; same grid as E.3)

| placements | class | x 1e-3 | x 1 | x 40 | x 1, off 25 | w8, x 1 |
|---|---|---|---|---|---|---|
| expand (+ RF3 flag) | Protenix | 0.0076 -> 0.0538 (7.1x) | 0.0070 -> 0.0535 (7.6x) | 0.2166 -> 0.2231 (1.0x) | 0.1193 -> 0.1306 (1.1x) | 0.0149 -> 0.0547 (3.7x) |
| expand (+ RF3 flag) | Boltz2 | 0.0090 -> 0.0653 (7.3x) | 0.0075 -> 0.0650 (8.7x) | 0.1897 -> 0.2004 (1.1x) | 0.1064 -> 0.1245 (1.2x) | 0.0193 -> 0.0673 (3.5x) |
| expand (+ RF3 flag) | RoseTTAFold3 | 0.0060 -> 0.0534 (8.9x) | 0.0070 -> 0.0536 (7.7x) | 0.2160 -> 0.2225 (1.0x) | 0.1189 -> 0.1303 (1.1x) | 0.0060 -> 0.0529 (8.9x) |
| expand (+ RF3 flag) | Chai1 | 0.0061 -> 0.0521 (8.6x) | 0.0067 -> 0.0495 (7.4x) | 0.0829 -> 0.0963 (1.2x) | 0.0669 -> 0.0829 (1.2x) | 0.0063 -> 0.0527 (8.4x) |
| qkvg+expand+out+squeeze (RF3 off) | Protenix | 0.0076 -> 0.0818 (10.8x) | 0.0070 -> 0.0659 (9.4x) | 0.2166 -> 0.2265 (1.0x) | 0.1193 -> 0.1361 (1.1x) | 0.0149 -> 0.0816 (5.5x) |
| qkvg+expand+out+squeeze (RF3 off) | Boltz2 | 0.0090 -> 0.0964 (10.7x) | 0.0075 -> 0.0751 (10.0x) | 0.1897 -> 0.2037 (1.1x) | 0.1064 -> 0.1301 (1.2x) | 0.0193 -> 0.0966 (5.0x) |
| qkvg+expand+out+squeeze (RF3 off) | RoseTTAFold3 | 0.0060 -> 0.0060 (1.0x) | 0.0070 -> 0.0070 (1.0x) | 0.2160 -> 0.2160 (1.0x) | 0.1189 -> 0.1189 (1.0x) | 0.0060 -> 0.0060 (1.0x) |
| qkvg+expand+out+squeeze (RF3 off) | Chai1 | 0.0061 -> 0.0639 (10.5x) | 0.0067 -> 0.0618 (9.3x) | 0.0829 -> 0.1031 (1.2x) | 0.0669 -> 0.0908 (1.4x) | 0.0063 -> 0.0644 (10.3x) |


## E.6 Verification

* FP8 OFF is bit-identical: `$W/b2511/jobs/n12-bid/bid.py` (eager) / `bidg.py` (CUDA graph, replayed twice), sha1 of every output and parameter gradient, separate processes, n14-base against n14-e:
  atom classes (`work/n14-e/bid/atom_bid.yaml`: AtomDiTProtenix / OpenFold3 / Chai1 / Boltz2 / RoseTTAFold3 / ESMFold2Pair / ESMFold2 / SWAAtomTransformer, L 128 / 1024 / 4096, bf16 / master / fp32, eval + train, all sample counts): eager 258 / 258 identical, 0 errors, 0 non-deterministic;
  graph 252 / 258 identical -- the 6 that differ are 6 of the 18 fp32 eval cases that are non-deterministic IN THE BASE tree itself (`det` false across two replays of the same graph; 240 / 240 deterministic cases identical), 4 errors equal on both sides (the stale AccumulateGrad capture error of the measurer).
  Token DiT classes (`n12-bid/bid.yaml`: AttentionPairBias + five TokenDiT classes, bf16 / master / fp32, eval + train, 398 cases): eager 398 / 398 and graph 398 / 398 identical, 0 errors, 0 non-deterministic (`work/n14-e/bid/tok.*`). `AtomDiTMiniWorld` has no case in the measurer's requirement table (0 cases at lengths 128 / 1024 / 4096 and at all lengths): not testable by this harness; its route (swa_dit) contains no line of this change.
  The default atom DiT / token DiT sources are the base ones: `pre_fwd.cu`, `post_fwd.cu` and every sm90 / sm80 source are byte-identical to n14-base; `backend_versions` untouched.
* Tests (B200): new `tests/gpu/test_atom_dit_fp8_b200_kernel.py` (rows kernels against the fp64 reference at M 32 / 1024 / 5120, x 1e-3 .. 40, offset 25, weights x 8; the up-gate post build) and `tests/gpu/test_atom_dit_fp8_b200.py` (six classes: accuracy next to the bf16 route
  on the stress grid, route stats, off == bit-identical, graph == eager also after replaying x 40 + 25 / x 1e-3 contents, row gate, declines); `tests/gpu/test_token_dit_fp8_b200.py` extended (Boltz-2 up gate served and accurate, RoseTTAFold3 opt-in, the switch off, default RF3 unserved).
  The B200 suites of agent C's list (trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit / atom_dit / engine_fp8 backend / route stats) + the new files: **587 passed, 351 skipped, 0 failed** (final run on an otherwise idle GPU; n14-base, same list without the new files: 547 passed)
  (`$W/b2511/work/n14-e/reg/pytest3.n14-e.log`. An earlier run while other jobs shared the GPU had 3 failures in `test_atom_dit_slide_tail.py::test_padded_inference_matches_reference[float32-...-slide*]` (fp32 path, no FP8 code in it): the same test also failed once on n14-base under that load with other parameters, and 3 repeated runs of each tree on an idle GPU pass 122 / 122: a load-sensitive flake of the base.)
  CPU: `tests/cpu/test_fp8_switch.py` + `test_trimul_fp8_b200_cpu.py` 30 passed (shim `PYTHONPATH=$TREE/src:$W/b2511/agents/_trimul_pylib`).
* `_prebuild --arch sm100 --check`: plan 813 (base) -> 828 artifacts (759 cubin, 69 ext), 0 not built yet; new: 10 `pre_fwd_f8`, 4 `post_fwd_f8`, 1 `gemm_swiglu_f8_up_sm100`; the base line "no compiled kernel matches: cond_transition_gemm_swiglu_up_sm90" is gone;
  rc = 1 in both trees from the unchanged `*_sm90` / mpnn lines (full outputs `$W/b2511/work/n14-e/reg/n14-{base,e}.prebuild-check.txt`).

## E.7 Not done / open (exact; every non-served case cites a measured number of a built kernel)

1. Atom DiT squeeze e4m3: not built; `SQABL` ablation of the built `post` kernel: <= 0.9-1.7 % of the post stage more (0.1-0.5 us of a 30-90 us block call).
2. Atom DiT above 2048 atoms x 5 (rows > 10240): measured loss 1.5-6 % (E.2 tables); free-quantisation ablation of the same kernels 0.95-0.99 of the stages: no form of these kernels wins there; a different kernel structure (quantisation inside the producer of x1, i.e. in `cond_fwd`'s consumer) is the only
   remaining lever and was not built.
3. Token DiT `out` / `squeeze` / `qkvg`: not served (E.5 numbers); fused-epilogue MXFP8 not built (bound <= 1.4 % of a block).
4. fp32 / training atom DiT and token DiT: no e4m3 form (as the H100).
5. End-level (200-step diffusion RMSD) acceptance not re-run on B200: the recipe and operand error are the 3.0.3 ones; only the fp64 module protocol was run.
6. `fp8.py` registration: `$W/b2511/work/n14-e/FP8_REGISTER.md`.




# Agent G (2026-11-19): the REMAINDER of the MSA-family and TriMul routes -- built first, measured, then gated

Rule of this section (user order): every H100 FP8 route that the 3.0.3 engine has gets a best-effort B200 kernel BUILT first; what is served (gated) and what stays bf16 is decided from the measured
numbers of the BUILT kernel. Everything stays opt-in and OFF by default (`FOLDJET_ENGINE_OPM_FP8=bound`, `FOLDJET_ENGINE_PWA_FP8=1`, `FOLDJET_ENGINE_TRIMUL_FP8=calib`). H100 digests (what is e4m3, scales, gates and why, accuracy,
speedups, traps, "transfers unchanged vs redo for tcgen05"): `docs/b200/fp8_h100_msa_digest.md` (OPM / PWA / PWA BIG / grouped OPM), `docs/b200/fp8_h100_trimul_digest.md` (TriMul).
Protocol of every table: B200 GPU 3 (`dev.sh -x`, no foreign pid), same-process interleaved A/B (default vs e4m3 vs a same-route control; `ctrl` 0.99-1.01 everywhere), CUDA graph of one call,
0.15 s idle gap before each timed burst, medians of 5-9 rounds; "sustained" = back-to-back replays (power cap), energy = mean `nvidia-smi` power x time. Scripts: `work/n14-g/{py,tools_msa}`, raw rows `work/n14-g/{opm,pwa}/*.jsonl`.

## 0. Summary (ratio = e4m3 route / bf16 default route, module level)

| route | what was built | result | served |
|---|---|---|---|
| OPM wide prologue (d_msa 64 + projection bias / 128 / 256) | `opm_wprologue_f8`: the drain writes e4m3 A2 / BT (bias added before the cast, no `opm_cast2` pass) | 128/256 L256 S1024 0.815 -> 0.774, S2048 0.736 -> 0.691; 128/384 0.823 -> 0.789; 256/512 0.854 -> 0.823; bias 64/128 L384 S2048 0.634 | yes; wide builds (no bf16 fused kernel) from padded S 256 (0.84-0.95), the narrow gate otherwise |
| OPM fused e4m3 kernel, S <= 512 | `opm_fused_infer_f8` (`opm_fused_sm100<CZF, DB, F8>`: kind::f8f6f4 mainloop, K = 128 per stage, drain scales the fp32 product) | S 256 0.94-0.95, S 384 0.87-0.89, S 512 0.78-0.87 (L 384-1024); S 128 flat 0.985-0.996 | padded S >= 256 |
| Chai-1 grouped OPM (8 x 8, product LN) | `opm_chai_prologue_f8` (drain writes group-major e4m3 a / b) + eight `_scaled_mm` (MN-major operands) | N 384-768: S 1024 0.77-0.90, S 2048 0.68-0.76, S 4096 0.62-0.66; N 256 loses below S 2048; N <= 128 loses (1.07-1.50) | (N >= 384, S >= 1024) or (N >= 256, S >= 2048) or (N >= 768, S >= 512) |
| OPM O as e4m3 | NOT built (measured bound of the e4m3-output product, section 3.4) | product -> e4m3 O saves 4-12 % of the product | no |
| PWA wide builds (8 x 8 / 8 x 16 / 8 x 32 at d_msa 64 / 128, d_pair 128-384) and BIG (256, 512, 8 x 32) | `ln_vg_w_f8` (`ln_vg_w_sm100<LT, F8>`): the drain writes the e4m3 v | 8 x 8 N2048 S1024 0.870 (was 0.95-0.97), 8 x 16 N1536 0.862 (0.962), 8 x 32 d_pair 256 N1536 0.732 (0.885), BIG N1536 0.862 (0.940) | padded N^2 S >= 2.4e9 (8 x 8), 1.1e9 (8 x 16), 6e8 (8 x 32 / BIG) |
| PWA `pwa_dual` e4m3 form | `pwa_dual_f8` (one launch pair_fwd + `ln_vg` with e4m3 v) | 0 .. -1 % against the split-launch e4m3 route (N 640-1024) | yes (`_DUAL_F8`) |
| PWA 8 per-head GEMMs -> ONE launch | `pwa8_forward` (`pwa8_fp8_b200.cu`): all heads in one tcgen05 e4m3 kernel (A K-major, B MN-major, bf16 out; bitwise == the per-head cuBLASLt GEMMs) + a 2-stream per-head fallback | kernel alone 0.41 / 0.80 / 0.80 of the eight GEMMs at N256 SC8192 / N768 SC16384 / N1280 SC8192, 0.98-1.21 above N^2 SC 1.5e10 (cuBLAS 2.8 PF/s vs 2.4); the module cells it opens: MiniWorld 8 x 32 L256 S1024 0.916, L384 0.931; 8 x 8 L1280 S1024 0.994 -> 0.932 | yes: grouped up to per-head N^2 SC <= 1.5e10, 2 streams above |
| TriMul quiet-GPU gates | re-measured 68 cells (+ D384 / D512 bidirectional, B > 1, RF3) | the shipped MIN_L stand except (128, bidir) 512 -> 384 and D384 512 -> 128 (bidir) / 256; RoseTTAFold3 loses below L 1024 (1.11-1.39 at L 256-512): new RF3 floor | yes |
| TriMul gate-projection-only front, L <= 256 `tf` / M64 twins | NOT built | stage profile + built-GEMM bounds in section 5 | no |

## 1. Quiet-GPU TriMul gate re-measurement (decides whether the shipped MIN_L stand)

`work/n14-g/py/bench_mod.py` (agent B's script, unchanged), GPU 3 quiet (no foreign pid in any run), rounds 3, gap = first replay after 0.15 s idle, cool = 10 replays, sustained 1.5 s per arm, energy = sustained energy ratio.
L192 rows are omitted (L / ceil128(L) < 0.9: the route declines, ratio 1.0 by construction). Rule: served from the smallest L (multiple of 128) where gap, cool and sustained are all <= 0.99 and the cell above it also holds.
| D | dir | L | B | gap bf16 / fp8 us | gap | cool | sustained | energy |
|---|---|---|---|---|---|---|---|---|
| D64 | out | 128 | 1 | 16.6 / 20.1 | 1.211 | 1.179 | 1.185 | 1.041 |
| D64 | out | 256 | 1 | 25.8 / 25.1 | 0.972 | 1.028 | 1.029 | 0.983 |
| D64 | out | 384 | 1 | 44.8 / 41.7 | 0.932 | 0.988 | 0.983 | 0.905 |
| D64 | out | 512 | 1 | 68.8 / 64.9 | 0.942 | 0.962 | 0.927 | 0.916 |
| D64 | in | 128 | 1 | 16.4 / 18.8 | 1.145 | 1.176 | 1.184 | 1.048 |
| D64 | in | 256 | 1 | 27.2 / 29.9 | 1.098 | 1.025 | 1.027 | 0.977 |
| D64 | in | 384 | 1 | 47.0 / 42.5 | 0.905 | 0.983 | 0.980 | 0.887 |
| D64 | in | 512 | 1 | 71.2 / 68.6 | 0.964 | 0.956 | 0.922 | 0.909 |
| D64 | bidir | 128 | 1 | 20.3 / 19.7 | 0.974 | 1.047 | 1.040 | 0.969 |
| D64 | bidir | 256 | 1 | 37.7 / 35.2 | 0.932 | 1.006 | 1.006 | 0.873 |
| D64 | bidir | 384 | 1 | 69.2 / 65.9 | 0.951 | 0.966 | 0.931 | 0.921 |
| D64 | bidir | 512 | 1 | 111.8 / 105.1 | 0.940 | 0.930 | 0.898 | 0.913 |
| D128 | out | 128 | 1 | 20.3 / 20.6 | 1.018 | 0.986 | 0.990 | 0.996 |
| D128 | out | 256 | 1 | 41.7 / 41.2 | 0.988 | 0.981 | 0.982 | 0.940 |
| D128 | out | 384 | 1 | 78.5 / 79.6 | 1.014 | 1.007 | 0.961 | 0.965 |
| D128 | out | 512 | 1 | 132.7 / 128.7 | 0.970 | 0.957 | 0.924 | 0.881 |
| D128 | in | 128 | 1 | 17.9 / 17.6 | 0.984 | 0.987 | 0.997 | 0.999 |
| D128 | in | 256 | 1 | 41.7 / 41.3 | 0.991 | 0.982 | 0.980 | 0.943 |
| D128 | in | 384 | 1 | 78.9 / 79.5 | 1.007 | 0.999 | 0.955 | 0.944 |
| D128 | in | 512 | 1 | 132.7 / 128.8 | 0.970 | 0.960 | 0.926 | 0.997 |
| D128 | bidir | 128 | 1 | 22.4 / 22.9 | 1.022 | 1.013 | 1.013 | 0.978 |
| D128 | bidir | 256 | 1 | 60.8 / 60.3 | 0.992 | 0.977 | 0.942 | 0.926 |
| D128 | bidir | 384 | 1 | 126.0 / 122.2 | 0.970 | 0.951 | 0.940 | 0.961 |
| D128 | bidir | 512 | 1 | 213.1 / 199.4 | 0.936 | 0.913 | 0.917 | 0.901 |
| D256 | out | 128 | 1 | 29.4 / 29.3 | 0.997 | 0.991 | 0.995 | 0.950 |
| D256 | out | 256 | 1 | 82.1 / 82.9 | 1.009 | 1.004 | 0.977 | 1.002 |
| D256 | out | 384 | 1 | 174.1 / 169.0 | 0.971 | 0.947 | 0.958 | 0.992 |
| D256 | out | 512 | 1 | 304.6 / 287.3 | 0.943 | 0.937 | 0.936 | 0.957 |
| D256 | in | 128 | 1 | 29.3 / 29.4 | 1.002 | 0.997 | 0.998 | 0.962 |
| D256 | in | 256 | 1 | 82.0 / 83.3 | 1.016 | 1.007 | 0.982 | 0.955 |
| D256 | in | 384 | 1 | 173.3 / 169.0 | 0.975 | 0.949 | 0.958 | 0.987 |
| D256 | in | 512 | 1 | 303.7 / 287.6 | 0.947 | 0.941 | 0.936 | 0.956 |
| D256 | bidir | 128 | 1 | 41.2 / 41.5 | 1.009 | 1.005 | 0.970 | 0.897 |
| D256 | bidir | 256 | 1 | 140.3 / 129.8 | 0.925 | 0.917 | 0.911 | 0.884 |
| D256 | bidir | 384 | 1 | 284.2 / 268.5 | 0.945 | 0.941 | 0.950 | 0.915 |
| D256 | bidir | 512 | 1 | 504.2 / 465.5 | 0.923 | 0.924 | 0.911 | 0.939 |
| D384 | out | 256 | 1 | 148.3 / 142.9 | 0.964 | 0.939 | 0.945 | 0.999 |
| D384 | out | 384 | 1 | 317.2 / 303.0 | 0.955 | 0.951 | 0.960 | 0.977 |
| D384 | out | 512 | 1 | 565.0 / 531.9 | 0.942 | 0.944 | 0.934 | 0.927 |
| D384 | bidir | 256 | 1 | 262.5 / 249.4 | 0.950 | 0.934 | 0.938 | 0.935 |
| D384 | bidir | 384 | 1 | 548.7 / 523.6 | 0.954 | 0.941 | 0.956 | 0.988 |
| D384 | bidir | 512 | 1 | 976.6 / 913.2 | 0.935 | 0.933 | 0.926 | 0.936 |
| D512 | out | 128 | 1 | 69.6 / 67.0 | 0.962 | 0.944 | 0.938 | 0.982 |
| D512 | out | 256 | 1 | 241.8 / 228.4 | 0.945 | 0.930 | 0.933 | 0.958 |
| D384 | bidir | 128 | 1 | 70.1 / 68.3 | 0.974 | 0.959 | 0.943 | 0.950 |
| D384 | bidir | 768 | 1 | 2350.5 / 2153.6 | 0.916 | 0.917 | 0.915 | 0.904 |
| D384 | in | 256 | 1 | 148.4 / 143.2 | 0.965 | 0.939 | 0.946 | 0.951 |
| D384 | in | 512 | 1 | 564.2 / 531.4 | 0.942 | 0.939 | 0.935 | 0.918 |
| D384 | out | 128 | 1 | 45.4 / 45.7 | 1.007 | 0.998 | 0.979 | 0.941 |
| D512 | out | 384 | 1 | 519.3 / 492.1 | 0.948 | 0.949 | 0.955 | 0.956 |
| D512 | out | 512 | 1 | 921.3 / 863.4 | 0.937 | 0.945 | 0.938 | 0.947 |
| D512 | in | 128 | 1 | 69.6 / 67.3 | 0.967 | 0.945 | 0.937 | 0.956 |
| D512 | in | 256 | 1 | 241.0 / 228.1 | 0.946 | 0.931 | 0.935 | 0.933 |
| D512 | in | 512 | 1 | 919.3 / 863.6 | 0.939 | 0.924 | 0.938 | 0.946 |
| D512 | bidir | 128 | 1 | 116.4 / 104.1 | 0.894 | 0.881 | 0.908 | 0.912 |
| D512 | bidir | 256 | 1 | 424.2 / 387.6 | 0.914 | 0.911 | 0.923 | 0.954 |
| D512 | bidir | 384 | 1 | 901.2 / 834.8 | 0.926 | 0.907 | 0.941 | 0.923 |
| D512 | bidir | 512 | 1 | 1589.4 / 1455.5 | 0.916 | 0.897 | 0.917 | 0.900 |
| D128 | out | 256 | 2 | 69.5 / 67.1 | 0.965 | 0.963 | 0.943 | 0.934 |
| D128 | out | 256 | 4 | 133.4 / 124.2 | 0.931 | 0.918 | 0.917 | 0.909 |
| D128 | out | 128 | 4 | 40.5 / 39.0 | 0.964 | 0.965 | 0.961 | 0.954 |
| D128 | out | 128 | 8 | 67.5 / 65.1 | 0.964 | 0.961 | 0.949 | 0.911 |
| D128 | bidir | 256 | 2 | 106.6 / 100.7 | 0.944 | 0.929 | 0.915 | 0.904 |
| D256 | out | 256 | 2 | 154.7 / 146.5 | 0.947 | 0.930 | 0.933 | 0.902 |
| D256 | out | 256 | 4 | 306.9 / 295.0 | 0.961 | 0.950 | 0.955 | 0.948 |
| D256 | out | 384 | 2 | 339.8 / 325.1 | 0.957 | 0.951 | 0.960 | 0.926 |
| D64 | out | 512 | 2 | 122.6 / 116.2 | 0.948 | 0.933 | 0.907 | 0.897 |
| D64 | out | 512 | 4 | 241.6 / 242.6 | 1.004 | 1.002 | 1.003 | 1.034 |

Result: the shipped values stand for (64, out / in) 512, (64, bidir) 384, (128, out / in) 512, (256, out / in) 384, (256, bidir) 256, (512, *) 128 (every cell below them has a gap or sustained ratio above 0.99:
D64 out L384 gap 0.932 but cool 0.988 / sustained 0.983, D128 out / in L384 gap 1.014 / 1.007, D256 L256 gap 1.009 / 1.016).
Changed (agent B measured under a colleague's foreign load, +- 10 %): `MIN_L[(128, 0)]` 512 -> 384 (gap 0.970, cool 0.951, sustained 0.940, energy 0.961; L256 gap 0.992), `MIN_L[(384, 0)]` 512 -> 128 (gap 0.974 / 0.959 / 0.943 / 0.950),
`MIN_L[(384, 1)]` and `[(384, 2)]` 512 -> 256 (gap 0.964 / 0.965, cool 0.939, sustained 0.945 / 0.946; L128 gap 1.007). B > 1 (the rows with B 2 / 4 / 8) gain at least as much as B = 1 (D128 out L256 B4 0.931 / 0.918 / 0.917).
D384 / D512 bidirectional and incoming are served and gain 0.89-0.95 (D512 bidir L128 0.894). Tests: `test_remeasured_gates`, `test_accuracy_bidirectional_d384_d512_and_batches` (update error 3.77-3.79 % against 0.54-0.64 % bf16 at D384 bidir L128 / L256, D512 bidir / in L256, B 2 / 4).

### 1b. RoseTTAFold3 and batches (same protocol, quiet GPU)
RoseTTAFold3 (`model="rf3"`, one sample, all-ones mask; e4m3 chain / bf16 chain, gap | cool | sustained | energy): the FP8 chain LOSES below L 1024 -- D64 out L384 1.363 | 1.391 | 1.396 | 1.035, L512 1.254 | 1.270 | 1.216 | 1.074, L768 1.071 | 1.068 | 0.967 | 0.948, L1024 0.996 | 0.973 | 0.922 | 0.918, L1280 0.931 | 0.918 | 0.890 | 0.867;
D128 out L256 1.367 | 1.376 | 1.377, L384 1.225 | 1.241 | 1.160, L512 1.108 | 1.109 | 1.009, L1024 0.947 | 0.944 | 0.919 | 0.925, L1280 0.905 | 0.893 | 0.893 | 0.889; D128 in L384 1.217, L512 1.113, L768 1.022 | 1.006 | 0.959 | 0.993, L1024 0.945 | 0.927 | 0.919 | 0.957
(the plain module's chain is level or better at those lengths, section 1: the RF3 call adds the length-factor kernels and its bf16 chain is the single-launch one). The shipped plain gates served RF3 from L 512 (an 11 % loss at D128 L512): **new `MIN_L_RF3 = {64: 1280, 128: 1024}`** (floor on top of `MIN_L`).
Batches (e4m3 / bf16, gap | cool | sustained | energy): D128 out L128 B4 0.964 | 0.965 | 0.961 | 0.954, B8 0.964 | 0.961 | 0.949 | 0.911, B16 0.926 | 0.912 | 0.920 | 0.939; L256 B2 0.965 | 0.963 | 0.943 | 0.934, B4 0.931 | 0.918 | 0.917 | 0.909, B8 0.941 | 0.927 | 0.930 | 0.849, B16 1.001 (above one default launch: the bf16 route runs);
D128 bidir L256 B2 0.944 | 0.929 | 0.915 | 0.904; D256 out L256 B2 0.947 | 0.930 | 0.933 | 0.902, B4 0.961 | 0.950 | 0.955 | 0.948, B8 0.966 | 0.959 | 0.958 | 0.899, L384 B2 0.957 | 0.951 | 0.960 | 0.926; D64 out L512 B2 0.948 | 0.933 | 0.907 | 0.897, B4 / B8 1.00 (declined), L256 B16 1.00 (declined).
So B > 1 is served up to the default route's chunk (the largest B that is one default launch, `TB._plan`) and gains 0.91-0.96 there; above it the bf16 route runs unchanged (measured 1.00).

## 2. OPM e4m3 (details)

### 2.1 Wide prologue (the 6-8 % `opm_cast2` pass is gone)
`opm_wide_sm100.cu`: `opm_wprologue_sm100<CM, WT, F8>` (default F8 = false = the default kernel) -- the drain of the d_msa 64 (+ projection bias) / 128 / 256 prologue casts `(acc + masked bias) * q` with `cvt.rn.satfinite.e4m3x2`
into the 64-B-swizzled u8 staging tile and stores it with the u8 TMA maps (the narrow route's layout); `opm_wprologue_f8(m, mask, lnw, lnb, eps, wa, wb, pbias, pb_roff, qa, qb)` -> (A2q, BTq, bits); `opm_sm100._forward` calls it where the e4m3 route serves.
| case (module us, e4m3 / default, 5-7 rounds, ctrl 1.00) | L256 S1024 | L256 S2048 | L384 S1024 | L384 S2048 | cast-pass version (agent C) L256 S1024 / S2048 |
|---|---|---|---|---|---|
| d_msa 128 / d_pair 256 | 0.774 | 0.691 | 0.736 | 0.665 | 0.815 / 0.736 |
| 128 / 384 | 0.789 | 0.706 | 0.762 | 0.686 | 0.823 / 0.748 |
| 256 / 512 | 0.823 | 0.760 | 0.794 | 0.730 | 0.854 / 0.789 |
| 64 / 128, projection bias | 0.755 | 0.661 | 0.709 | 0.634 | (cast pass; not measured) |
Sustained (8 s back-to-back, def -> e4m3, us / W / mJ): 128/256 L384 S2048 629.0 / 962 / 605 -> 431.4 / 948 / 409 (time 0.686, energy 0.676); 256/512 774.0 / 969 / 750 -> 566.7 / 962 / 545 (0.732 / 0.727); bias 64/128 581.9 / 964 / 561 -> 375.8 / 957 / 360 (0.646 / 0.641).

### 2.1b Low depths on the wide builds (no bf16 fused kernel there)
d_msa 128 / 256 and c_z 256 / 384 / 512 have no fused bf16 kernel, so the e4m3 composition (wide e4m3 prologue + cuBLASLt product + `opm_wepilogue`) replaces cuBLAS + epilogue at every depth. e4m3 / default (padded N x padded depth S, 7 rounds, ctrl 1.00):
| build | N | S 128 | S 256 | S 384 | S 512 |
|---|---|---|---|---|---|
| 128 / 256 | 256 / 384 / 512 / 768 | 0.938 / 0.940 / 0.923 / 0.916 | 0.938 / 0.942 / 0.897 / 0.891 | 0.936 / 0.918 / 0.874 / 0.863 | 0.900 / 0.878 / 0.844 / 0.835 |
| 128 / 384 | same | 0.951 / 0.950 / 0.936 / 0.933 | 0.947 / 0.939 / 0.915 / 0.910 | 0.939 / 0.921 / 0.891 / 0.884 | 0.903 / 0.890 / 0.863 / 0.858 |
| 256 / 512 | same | 0.960 / 0.954 / 0.946 / 0.942 | 0.950 / 0.945 / 0.926 / 0.919 | 0.940 / 0.931 / 0.905 / 0.897 | 0.917 / 0.902 / 0.881 / 0.873 |
Gate `opm_fp8_b200._gate(s, n, wide)`: wide builds from padded S >= 256 (every cell <= 0.950); S 128 (0.92-0.96) stays bf16; the d_msa 64 / c_z 128 build keeps `S >= 1024 or (S >= 512 and N >= 256)` (plus the fused-kernel gate of 2.2).

### 2.2 Fused e4m3 kernel for S <= 512 (where the bf16 fused kernel serves)
`opm_sm100.cu`: `opm_fused_sm100<CZF, DB, F8>` (default F8 = false). F8: A2 / BT are u8 maps with a 128-byte box (128 e4m3 per swizzle row: K = 128 per stage, the same bytes per stage), the mainloop MMA is
`tcgen05.mma.cta_group::2.kind::f8f6f4` (idesc without the bf16 format bits, 4 MMAs of K = 32 per stage: the descriptor advance is the same 32 bytes), the drain multiplies the fp32 product by `dq = sa sb` before its bf16 rounding;
Opair, the projection (bf16 `kind::f16`), count, bias and residual are the bf16 kernel's. `opm_fused_infer_f8`; operands from `opm_prologue_f8` / `opm_wprologue_f8`.
| e4m3 / default (64 / 128, depth S padded) | S 128 | S 256 | S 384 | S 512 |
|---|---|---|---|---|
| L384 | 0.985 | 0.946 | 0.887 | 0.874 |
| L512 | 0.989 | 0.940 | 0.877 | 0.822 |
| L768 | 0.992 | 0.951 | 0.875 | 0.798 |
| L1024 | 0.996 | 0.937 | 0.871 | 0.784 |
(projection-bias and normalize-before-projection variants within +- 0.005 of these rows). Sustained L768 S512: 732.6 us / 970 W / 710.5 mJ -> 553.7 / 976 / 540.3 (time 0.756, energy 0.760).
Gate `opm_fp8_b200._gate_fused`: padded S >= 256 (S 128 is flat: bf16 runs). The route counts in `opm_fp8_b200.STATS["calls"]`.

### 2.3 Chai-1 grouped OPM (the previous decision came from the product alone: now the whole module)
`opm_chai_prologue_f8` (drain writes group-major e4m3 a / b [8, S N, 8], u8 maps) + eight `torch._scaled_mm(a8_g.t(), b8_g)` (cuBLASLt takes both MN-major operands as they are on Blackwell: verified, `work/n14-g/py/t_scaled.py`) + the unchanged
product-LayerNorm epilogue. Product alone (bf16 `bmm` / 8 e4m3 GEMMs, us): N256 S1024 49.9 / 58.9; N384 S1024 107.7 / 81.2; N384 S2048 202.6 / 127.6; N512 S4096 690.3 / 399.8; N768 S1024 410.9 / 256.8.
Whole module (e4m3 / default, 7 rounds, ctrl 0.99-1.02; `normalize_before_proj` True / False identical within 0.002):
| N \ S | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 128 | 1.50 | 1.28 | 1.16 | 1.07 |
| 256 | 1.23 | 1.11 | 0.92 | 0.77 |
| 384 | 1.06 | 0.90 | 0.755 | 0.66 |
| 512 | 1.05 | 0.83 | 0.72 | 0.65 |
| 768 | 0.89 | 0.77 | 0.68 | 0.62 |
Sustained N512 S2048: 613.1 us / 964 W / 591 mJ -> 441.2 / 968 / 427 (0.72 / 0.72). Error against fp64 (N384 S1024): rms 2.27e-2, judge 2.55e-2 against the bf16 route's 4.7e-3 / 5.8e-3 (4.8x / 4.4x; the H100 doc: 8x). Gate `_gate_chai`:
(padded N >= 384 and S >= 1024) or (N >= 256 and S >= 2048) or (N >= 768 and S >= 512): every served cell <= 0.92. This reverses the previous B200 decision (made from the product alone, which pays the cast pass: the cast is now free in the prologue drain).

### 2.4 O as e4m3 (NOT built)
Built-kernel measurement of the producing side only (cuBLASLt `_scaled_mm`, e4m3 operands, `out_dtype` bf16 vs e4m3 with `scale_result`, graph, us): L256 S1024 53.4 -> 47.1 (0.88), L256 S2048 94.6 -> 89.8 (0.95), L384 S1024 120.6 -> 106.3 (0.88),
L384 S2048 212.6 -> 204.0 (0.96), L512 S2048 377.2 -> 361.4 (0.96): the e4m3 write saves 4-12 % of the product. The remaining half (the epilogue kernel reading O: 13 % of the module at L384 S2048, 25 % at L256 S1024) needs a new `kind::f8f6f4`
epilogue (A = O e4m3, B = W_o e4m3) AND a data-dependent scale for O (O = sum over s: no static bound like A2 / BT's LayerNorm bound; a worst-case bound puts typical O in the e4m3 subnormal range): not built in this session
(open item 2). It would add a third e4m3 rounding.

## 3. PWA e4m3 (details)

### 3.1 Wide builds + BIG: fused e4m3 v
`pwa_wide_sm100.cu`: `ln_vg_w_sm100<LT, F8>` (default F8 = false): the TMEM drain writes `v * qv` as e4m3 bytes (`cvt.rn.satfinite`) into an unswizzled C-byte row (C = 8 / 16 / 32; the C = 8 head block goes through the flat 256-byte-row u8 map as the bf16 kernel's),
`head_map8`; the same code serves the ring-streamed d_msa 256 build (BIG). `ln_vg_w_f8` -> (v8, y or empty); `pwa_sm100._forward` takes it where `pwa_fp8_b200.serves_fused_wide` says so, followed by `contract8` (the 8 head GEMMs). The old cast route
(`opm_cast1` over v) is no longer used on the wide builds.
Module, e4m3 / default (every gate removed, 5 rounds, ctrl 0.99-1.01), padded N x depth S:
| build (d_msa / d_pair, heads x C) | N768 S512 | N768 S1024 | N1280 S1024 | N1536 S1024 | N2048 S1024 | N1536 S2048 | N2048 S2048 | previous (cast) N1536 / N2048 S1024 |
|---|---|---|---|---|---|---|---|---|
| 64 / 128, 8 x 8 | 1.180 | 1.062 | 0.994 | 0.929 | 0.870 | 0.851 | 0.795 | 1.01-1.13 / 0.95-0.97 |
| 128 / 256, 8 x 8 | 1.103 | 1.030 | 0.986 | 0.943 | 0.905 | 0.888 | 0.848 | 1.01-1.13 / 0.95-0.97 |
| 128 / 384, 8 x 8 | 1.099 | 1.030 | 0.992 | 0.952 | 0.917 | 0.897 | 0.861 | |
| 128 / 256, 8 x 16 | 1.046 | 0.997 | 0.912 | 0.862 | 0.815 | 0.798 | 0.778 | 0.962 / 0.907 |
| 64 / 256, 8 x 32 (Chai-1) | 1.048 | 0.887 | 0.775 | 0.732 | 0.711 | 0.707 | 0.673 | 0.885 / 0.847 |
| 256 / 512, 8 x 32 (BIG, IntelliFold2) | 1.002 | 0.949 | 0.885 | 0.862 | 0.839 | 0.848 | 0.807 | 0.940 / 0.911 |
The 8 x 8 builds that the previous agent declined (0.95-0.97 at N2048) reach 0.87-0.92 once the fused v removes the cast; N768 and S512 rows lose or are flat. Gates `pwa_fp8_b200.MIN_WORK_WIDE` (padded N^2 S), re-derived AFTER the grouped contraction kernel of section 3.3 (table 3.3b): 8 x 8 1.6e9 (N1280 S1024 0.932 / 0.945, N1536 S512 0.952 / 0.959 stays bf16), 8 x 16 6e8 (N768 S1024 0.947, N1280 S512 0.945; N768 S512 0.964 stays bf16),
8 x 32 and BIG 6e8 (N768 S1024 0.867 / 0.941, N1280 S512 0.860 / 0.928; BIG N768 S512 0.965 stays bf16). The table above is the version BEFORE the grouped kernel (cast removed, serial 8 GEMMs).
Sustained (8 s, def -> e4m3, us / W / mJ), N1536 / N2048 S1024: 64/128 8x8 N2048 899.7 / 963 / 866 -> 740.1 / 985 / 729 (time 0.823, energy 0.842); 64/256 8x32 N1536 1583.6 / 980 / 1552 -> 1153.7 / 971 / 1120 (0.728 / 0.721); 128/256 8x16 N1536 1149.5 / 973 / 1118 -> 969.0 / 980 / 950 (0.843 / 0.849);
BIG N1536 2860.8 / 970 / 2776 -> 2413.9 / 981 / 2368 (0.844 / 0.853).

### 3.2 `pwa_dual` e4m3 form
`pwa_sm100.cu`: `pwa_dual_fwd_sm100<T, DZ, F8>` runs `ln_vg_body<T, F8>` on the MSA-side CTAs (`pwa_dual_f8`: one launch -> (w, v8)); `_forward` uses it where `serves_fused` applies and `_DUAL` applies (d_pair 128, N <= 1024, shared affine dtype).
MiniWorld 64 / 128 8 x 32, depth 1024 / 2048, split launches -> one launch (module e4m3 / default): N640 S2048 0.8166 -> 0.8087, N768 S2048 0.7905 -> 0.7838, N1024 S2048 0.7485 -> 0.7453, S1024 rows 0.883 / 0.857 / 0.781 -> 0.884 / 0.857 / 0.781:
0 .. 1 %, identical numerics (same gold errors), so it stays enabled (`_DUAL_F8`).

### 3.3 The eight per-head `_scaled_mm` -> ONE launch (built: `pwa8_forward`)
`pwa8_fp8_b200.cu` (new, a derivative of `ctr8.cu`; extension `pwa8_fp8_b200`, in the prebuild plan): `o[h][i][c] = dq sum_k w8[h][i][k] v8[h][k][c]` for ALL heads in one tcgen05 `kind::f8f6f4` launch: TMA producer, one MMA thread, TMEM double buffer, 8 epilogue warps, BN 256 x BM 128 x K 128 per stage, the plane index = the head,
A (w8) a K-major SW128 box, B (v8 = `ln_vg_w_f8`'s head-major output) an MN-major SW128 box (`desc_mn_sw128`, the same descriptors as ctr8's incoming form), idesc `a_mn = 0, b_mn = 1`, `dq = sw sv` a device scalar, bf16 TMA store; row tiles are the fastest item index so the 148 CTAs
reuse each v8 column tile from L2 (this ordering took N1536 SC32768 from 589 to 525 us). Result: **bitwise equal to the eight cuBLASLt `_scaled_mm` calls** (`test_grouped_kernel_equals_per_head_gemms`, max diff 0), module output equal bit for bit with the grouped / 2-stream / serial routes (`test_grouped_route_equals_streams_route`).
Kernel alone (CUDA graph, us; 8 per-head cuBLASLt GEMMs serial / 2 streams / ONE grouped tcgen05 launch / ONE equal-flop cuBLASLt launch = the earlier bound):
| N, S C (per head) | 8 heads | 2 streams | grouped kernel | equal-flop cuBLAS launch |
|---|---|---|---|---|
| 256, 8192 | 27.9 | 21.9 (0.79) | 11.4 (0.41) | |
| 768, 16384 | 102.3 | 89.8 (0.88) | 82.3 (0.80) | 71.0 (0.69) |
| 1280, 8192 | 123.2 | 113.6 (0.92) | 98.9 (0.80) | 80.5 (0.64) |
| 1536, 8192 | 135.7 | 121.9 (0.90) | 133.1 (0.98) | 109.9 (0.80) |
| 768, 32768 | 154.6 | 150.1 (0.97) | 160.9 (1.04) | 141.8 (0.88) |
| 2048, 8192 | 206.8 | 196.1 (0.95) | 222.9 (1.08) | 188.5 (0.86) |
| 1536, 16384 | 233.6 | 227.9 (0.98) | 264.3 (1.13) | 221.9 (0.89) |
| 2048, 16384 | 367.9 | 362.3 (0.985) | 441.4 (1.20) | 377.3 (0.95) |
| 1536, 32768 | 435.6 | 421.4 (0.97) | 525.2 (1.21) | 473.1 (1.01) |
Variants tried (KB 2, BN 128, grid 74): all slower than BN 256 / KB 1 / 148 CTAs (N1536 SC32768: 560 / 666 / 809 us). The kernel reaches 2.4 PF/s of e4m3 where cuBLASLt reaches 2.8 PF/s (short graph bursts), so above per-head N^2 S C = 1.5e10 the route runs the 8 GEMMs on 2 streams (fork / join inside
the capture: 0.3-1.7 % of the module) -- `GROUPED_MAX_WORK`; open item 3 is a 2-CTA (`cta_group::2`) version of the grouped kernel for the large cells.
Module cells (e4m3 / default, 5-7 rounds; "before" = the cast-free fused v with serial per-head GEMMs, "after" = grouped / 2 streams; ctrl 0.99-1.01):
| build | N768 S512 | N768 S1024 | N1280 S512 | N1280 S1024 | N1536 S512 | N1536 S1024 | N2048 S1024 |
|---|---|---|---|---|---|---|---|
| 64/128 8x8 | 1.180 -> 1.003 | 1.062 -> 0.977 | 1.042 -> 0.963 | 0.994 -> 0.932 | 1.028 -> 0.952 | 0.929 -> 0.902 | 0.870 -> |
| 128/256 8x8 | 1.103 -> 0.989 | 1.030 -> 0.972 | 1.025 -> 0.979 | 0.986 -> 0.945 | 1.021 -> 0.959 | 0.943 -> 0.933 | |
| 128/256 8x16 | 1.046 -> 0.964 | 0.997 -> 0.947 | 0.997 -> 0.945 | 0.912 -> 0.893 | 0.937 -> 0.918 | 0.862 -> 0.858 | |
| 64/256 8x32 | 1.048 -> 0.947 | 0.887 -> 0.867 | 0.892 -> 0.860 | 0.775 -> 0.769 | 0.823 -> 0.815 | 0.732 -> 0.723 | |
| 256/512 BIG | 1.002 -> 0.965 | 0.949 -> 0.941 | 0.943 -> 0.928 | 0.885 -> 0.881 | 0.906 -> 0.899 | 0.862 -> 0.860 | |
MiniWorld 64/128 8 x 32 (the narrow fused route; the earlier agent measured L256-L384 as losses, 1.03-1.45): L256 S512 / 1024 / 2048 / 4096 1.017 / 0.916 / 0.928 / 0.918; L384 0.980 / 0.931 / 0.914 / 0.910; L512 0.986 / 0.909 / 0.857 / 0.833; L640 0.939 / 0.871 / 0.817 / 0.791;
the gate `pwa_fp8_b200._gate_fused` is now (S >= 1024 and N >= 256) or (S >= 512 and N >= 640) (was N^2 S >= 5e8). gold errors unchanged (judge 0.0034-0.0036 against 0.0029-0.0032 bf16).

## 4. Accuracy against the fp64 gold (random non-zero weights; rms = rms(y - y64) / rms(y64), judge = max|y - y64| / max|y64|; bf16 default route next to the e4m3 route)
| case | bf16 route rms / judge | e4m3 rms / judge | stress x8 LN weights (x4 bias): bf16 -> e4m3 rms / judge |
|---|---|---|---|
| OPM 128/256 L256 S1024 | 0.0044 / 0.0052 | 0.0268 / 0.0185 | (128/384) 0.0049 / 0.0047 -> 0.0361 / 0.0177 |
| OPM 256/512 L256 S1024 | 0.0045 / 0.0041 | 0.0256 / 0.0176 | 0.0049 / 0.0045 -> 0.0361 / 0.0207 |
| OPM 64/128 projection bias L256 S1024 | 0.0038 / 0.0060 | 0.0124 / 0.0120 | (nbp, bias) 0.0048 / 0.0059 -> 0.0338 / 0.0158 |
| OPM fused e4m3 64/128 L768 S384 | 0.0044 / 0.0051 | 0.0278 / 0.0206 | 0.0049 / 0.0046 -> 0.0366 / 0.0203 |
| OPM fused e4m3 + projection bias L768 S512 | 0.0039 / 0.0051 | 0.0164 / 0.0171 | 0.0049 / 0.0056 -> 0.0360 / 0.0229 |
| OPM fused e4m3 nbp L512 S256 | 0.0028 / 0.0046 | 0.0116 / 0.0166 | 0.0049 / 0.0043 -> 0.0369 / 0.0206 |
| Chai-1 grouped OPM N384 S1024 | 0.0047 / 0.0058 | 0.0227 / 0.0255 | (weights: bound scales follow them) |
| PWA 64/128 8x8 N1536 S1024 | 0.00176 / 0.00292 | 0.00198 / 0.00333 | 0.0041 / 0.0061 -> 0.0073 / 0.0091 |
| PWA 128/256 8x16 N1280 S1024 | 0.00174 / 0.00291 | 0.00201 / 0.00322 | 0.0040 / 0.0054 -> 0.0079 / 0.0101 |
| PWA 64/256 8x32 N768 S1024 | 0.00175 / 0.00297 | 0.00216 / 0.00332 | 0.0042 / 0.0056 -> 0.0095 / 0.0101 |
| PWA BIG 256/512 N768 S1024 | 0.00176 / 0.00285 | 0.00217 / 0.00339 | 0.0041 / 0.0059 -> 0.0094 / 0.0119 |
| PWA dual_f8 MiniWorld N768 S1024 / S2048 | 0.00175 / 0.00295 | 0.00217 / 0.00350 | = the split-launch route's (bitwise same errors) |
| PWA grouped kernel (MiniWorld 64/128 8x32 N768 S1024) | 0.00175 / 0.00295 | 0.00217 / 0.00350 | bitwise equal to the per-head GEMM route |
TriMul (update error rms(y - y64) / rms(y64 - x)): D384 bidir L128 / L256 0.0054 -> 0.0377, D512 bidir L256 0.0060 -> 0.0379, D512 in L256 0.0064 -> 0.0379, D128 out L256 B2 0.0058 -> 0.0379, D256 out L256 B2 0.0059 -> 0.0379, D128 out L128 B4 0.0058 -> 0.0379.
All OPM / PWA scales are the `bound` policy's static power-of-two scales (from the weights alone: |LN(x)_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|, w = 2^8): a cast cannot saturate for any input, so no calibration / guard / `fp8.check()` hook exists
for them (the x8 / x4 weight stress above is exactly the case that moves the scales). Input stress (tests `test_input_stress_scale_offset`, `test_wide_input_stress_scale_offset`: x scale {1e-3, 40} x offset {0, 25} on OPM 128/256, OPM fused 64/128, the Chai-1 grouped route, PWA 8x16 and 8x8; finite, no saturation, 20 cells): LayerNorm removes scale and offset, so the e4m3 error does not move: OPM 128/256 judge 0.017 at x40 / offset 25 (bf16 0.0046), 0.0096 at x1e-3 (bf16 0.0058, the LayerNorm eps dominates), Chai-1 0.026 (bf16 0.006), PWA e4m3 = bf16 to three digits at x40 / offset 25 (judge 0.0018-0.0022) and 0.0104 / 0.0118 at x1e-3 (bf16 0.0089 / 0.0119). TriMul keeps the guard
protocol of agent B (sync / lazy, `rescale_all` through captured graphs: `test_saturation_rescale_through_graph`).
The e4m3 errors exceed the engine's 8e-3 judge on the OPM routes (as on the H100: FP8 is outside the judge); the PWA routes stay inside it at x1 (judge 0.33-0.34 %), 0.9-1.2 % with the x8 weight stress (bf16: 0.55-0.61 %).

## 5. Not built (every item with the measured numbers of what exists)

* **TriMul gate-projection-only front (LN_in -> e4m3, W_ag / W_bg e4m3, a second MMA chain), D256 / D384 / D512.** NOT built: `k1w_f8.cu` is 891 lines of tcgen05 pipeline in which the LayerNorm output (or, for D256 / D384, the raw x of the FOLD form with the correction MMA)
  is written into TMEM as the A operand of one N128 chunk (64 gate + 64 projection rows) per K-block ring slot; a gate-only e4m3 chain needs the e4m3 LN output in TMEM in the kind::f8f6f4 A layout, a second weight ring of e4m3 K-blocks and a second accumulator per chunk, in all of the FOLD / non-FOLD
  variants. What was measured (stage profile of the BUILT e4m3 chain, `work/n14-g/py/prof_mod.py`, L512, us, bf16 chain -> e4m3 chain): D256 out k1w 121.4 -> k1w8 127.5 | ctr 67.3 -> ctr8 58.7 | k3w 117.0 -> 113.8 (total 305.7 -> 300.0);
  D384 out 251.6 -> 251.8 | 95.3 -> 88.5 | 217.2 -> 209.3 (564.1 -> 549.6); D384 bidir 476.1 -> 479.0 | 188.4 -> 173.8 | 310.4 -> 299.1 (974.9 -> 951.9); D512 out stats 51.6 + k1w 401.9 -> 389.0 | 124.3 -> 130.1 | 344.3 -> 336.8 (922.2 -> 907.1);
  D128 out 49.9 -> 55.2 | 34.8 -> 28.6 | 52.9 -> 53.7 (137.6 -> 137.5). The front is 40-50 % of the e4m3 chain and the e4m3 front is itself 0-10 % slower than the bf16 one (the byte store). The built e4m3 tcgen05 GEMM (agent A's `tf8_gemm`, the K <= 512 rows of section 2 of this doc)
  reaches 0.92 / 0.79 / 0.79 / 0.74 of the bf16 time at K = 128 / 256 / 384 / 512 for the WHOLE N; half of the front's N (the two gate projections) at that rate on 40-50 % of the module is at most 4.7 % (D256), 5.9 % (D512), before the second chain's overhead. Accuracy cost on the H100 emulation: 4.2 % update error against 3.7 %.
  Open item 1. (The gate-front is not declined by this bound: it was not built in the time available.)
* **L <= 256 e4m3 twins of the single-launch `tf` kernel and of the M64 front (`k1h` / `k3h`).** NOT built. Measured on the built e4m3 chain at these lengths (section 1): D64 L128 1.18-1.21 (sustained), D128 L128 0.99-1.01, D128 L256 0.98 / 1.00, D256 L256 0.98 / 1.00, D64 L256 1.03:
  the cells are launch- / latency-bound (chain of 3 kernels, 17-41 us), not tensor-bound, so e4m3 in the same chain structure is flat; the bf16 single-launch kernel (`tf_sm100.cu`, D64 L128 12.5 us against the chain's 14.4-20.1) is the one to beat and has no e4m3 twin.
* **OPM O as e4m3**: section 2.4. **`pwa_dual` for N > 1024**: the dual kernel is bounded by N <= 1024 (the pair side's row of logits).

## 6. Verification
* FP8 OFF, bit-identity of the default route against `n14-base` (separate processes, sha1): OPM + PWA 174 cases of `msabid.yaml` (`work/n14-g/tools_msa/`; `jobs/n12-bid/{bid,bidg}.py`, eager and CUDA graph): **174 / 174 identical in both, 0 mismatches, 0 errors**,
  30 non-deterministic (training dropout draws) in both trees (`work/n14-g/bid/cmp_msa.txt`); TriMul 24 cases (`work/n13-fp8-b/bitcases.txt`, `xchk.py` eager, `xchk_g.py` graph): 0 differing (`cmp_tr.*.txt`). The default instantiations are the `F8 = false` template arguments (`opm_wprologue_sm100`, `opm_chai_prologue_sm100`,
  `opm_fused_sm100`, `ln_vg_w_sm100`, `pwa_dual_fwd_sm100`; defaulted trailing kernel parameters): same code, same bits; their artifacts are re-keyed (source text changed). `backend_versions["cuda:b200_sm100"]` is NOT bumped.
* Tests (B200): `tests/gpu/test_opm_fp8_b200.py`, `test_pwa_fp8_b200.py`, `test_trimul_fp8_b200.py` pass (100 passed; new: accuracy + stress for every new route, input scale / offset stress, grouped kernel bitwise == per-head GEMMs, gates): accuracy next to the bf16 route + x8 / x4 weight stress for every new route, below-gate and off bit-equal, graph == eager, the re-measured gates; with the B200 suites
  (trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit / opm / pwa / engine_fp8 / route_stats / i64_opm_pwa) and `tests/cpu` fp8 switch + capture_api (two new module-level caches listed): 566 passed, 174 skipped (H100-only), 0 failed (`work/n14-g/tests/pytest_final.log`, tree final). The 3 TriMul front / L<=256 / O items are not covered by tests because they are not built.
* `_prebuild --arch sm100 --check`: 814 artifacts (744 cubin, 70 ext), 0 not built (the base tree: 813 / 69 / 0): the one new artifact is `pwa8_fp8_b200` (the grouped PWA kernel; call sites 224 -> 225, native sources compiled 216 -> 217); the other lines that differ from the base.s check are the re-keyed `miniworld_opm_sm100`, `miniworld_opm_wide_sm100`, `miniworld_pwa_sm100`, `miniworld_pwa_wide_sm100_*` hashes (full output `work/n14-g/tests/prebuild_check_g_full.txt`).

## 7. Open items (exact)
1. TriMul gate-projection-only e4m3 front (section 5): build the e4m3 A (LN_in) + W_ag / W_bg path in `k1w_f8.cu`; built-GEMM bound <= 4.7-5.9 % of the module at D256-D512.
2. OPM O as e4m3: e4m3-output product (saves 4-12 % of the product, section 2.4) + an e4m3 epilogue (`kind::f8f6f4`, W_o e4m3) + a calibrated O scale.
3. e4m3 twins of the bf16 `tf` single-launch kernel and the M64 front for L <= 256 (the cells are latency-bound: only worth it with a new fused kernel).
3b. The grouped PWA kernel loses to cuBLASLt above per-head N^2 S C = 1.5e10 (2.4 vs 2.8 PF/s): a `cta_group::2` (256 x 256) version; end-level acceptance of the new routes (item 5).
5. End-level (diffusion RMSD) acceptance on B200 is not run for any of these routes.
6. `fp8.py` registration: `work/n14-g/FP8_REGISTER.md`.
