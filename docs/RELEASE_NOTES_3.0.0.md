# miniworld-engine 3.0.0 (DRAFT, not released)

Base: 2.5.14 (9dd18e507). Release branch `rel/3.0.0` = 2.5.14 + `fp8/3.0.0-all` (the FP8 routes, built on 2.5.12) + the release work below.
**3.0.0 adds opt-in FP8 (e4m3) inference routes on H100 (sm_90a).** Nothing changes unless a switch is set: with every switch off the outputs
are the 2.5.14 bytes (registry modules eval + train, AF3 end to end), no module `version` / `backend_versions` changed, judge owed 0 on all
three cards. A100 / RTX 5090 / B200 builds and plans are unchanged.

## What FP8 is here

Inference only, bf16 (`master`) modules, H100. Operands of the large GEMMs / attention products are rounded to e4m3 with power-of-two scales
and multiplied on the e4m3 tensor cores (fp32 / fp16 accumulation); inputs, outputs, LayerNorms, softmax statistics and residuals stay bf16 /
fp32. Training, fp32 modules and other GPUs take the default route. Full spec, per-route design and every measurement: `docs/fp8.md`.

## One switch

```
MINIWORLD_FP8=safe                                 # environment, before the engine's kernels are imported (TriAttn stays bf16 / fp16)
MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=af3         # + the TriAttn FP8 routes validated for that model
import miniworld_engine.fp8 as fp8
fp8.enable("safe")                                 # run time (TriAttn stays bf16 / fp16); fp8.disable(); fp8.status()
fp8.enable("safe", model="af3")                    # + the TriAttn FP8 routes validated for that model
...
out = model(x)
if fp8.check():                                    # once per forward, outside any CUDA graph (one host sync per route)
    out = model(x)                                 # some route saturated and was rescaled: redo (at most 3 times)
```

`safe` = the recommended set ("fp8safe2"): TriMul e4m3 planes (`calib`); OuterProductMean e4m3 product (`bound`); PWA e4m3 key contraction
(8 x 32 and the wide 64:8 / 128:8 / 128:16 widths); Transition e4m3 GEMMs (`calib`, D128 / D256 / D384 / D512); token DiT e4m3 `expand` +
SwiGLU; atom DiT e4m3 row kernels (`pre` + `post`): on for every model. **TriAttn FP8 only for a validated model**
(`fp8.TRIATTN_MODELS`, named by `MINIWORLD_FP8_MODEL` or `enable(model=)`): `af3` = FP8 PV on the fused d_pair-128 front + the unfused /
front-made sites with e4m3 q / k on the fused route; `protenix1` = PV on the fused front; `opendde` = PV on the unfused sites; any other model
(Boltz-2, OpenFold3, ...) or no model: the default fp16 TriAttn core. The per-route switches of the 3.0.0 line (`MINIWORLD_TRIMUL_FP8`, `MINIWORLD_OPM_FP8`, `MINIWORLD_PWA_FP8`,
`MINIWORLD_TRANSITION_FP8`, `MINIWORLD_TOKEN_DIT_FP8`, `MINIWORLD_ATOMDIT_FP8`, their `_GUARD` / gate variables, `h100_sm90.enable_fp8_pv_for`)
keep working; with `MINIWORLD_FP8` set, a per-route variable set explicitly (also to "") wins.

**Guard contract.** TriAttn (V, q / k), TriMul, Transition and PWA use static scales: they are calibrated on each site's first eager call
in a process and then kept. A later value past the e4m3 range saturates and is recorded; eager TriMul / PWA / Transition calls (guard `sync`,
the default) redo themselves, under CUDA-graph capture they only record. Call `fp8.check()` once per forward outside graphs and redo the
forward when it returns True (it has already raised the saturated scales in place; captured graphs follow). OPM `bound`, token DiT and atom
DiT cannot saturate (weight-bound / exact per-row scales). **Calibration note:** the scales depend on the first inputs a process sees, so two
runs whose first target differs are not byte-identical (same-sample RMSD 0.23 A between two otherwise identical AF3 runs).

## Routes: module time and error against the 2.5.14 default (H100, CUDA graph, same process, interleaved; `docs/fp8.md` "Release line 3.0.0")

FP8 / 2.5.14 default (cool / sustained; < 1 = faster), FP8 with its gates removed; error against fp64 as update rms (or rms rel) and the judge
form max |y - y64| / max |y64| (engine tolerance max(err_pytorch, 8e-3)).

| route (switch) | module time FP8 / 2.5.14 default | error default -> FP8 (judge form) | gates as shipped |
|---|---|---|---|
| TriAttn FP8 PV, 32-ch heads (128:4:32 fused; 64:4:32, 64:2:32 front-made) | L256 0.91-0.97, L1024 0.85-0.91, L2048 0.84-0.87; L128 1.03-1.07 | 0.27-0.32 % -> 0.43-0.75 % (inside tol) | fused S >= 256, d_pair 64 sites from L256 |
| TriAttn FP8 PV, 16-ch heads (64:4:16, AF3 / OpenFold3 templates) | L256 0.96, L1024 0.88, L2048 0.83-0.84 | as above | from L256 |
| TriAttn FP8 PV, 256:8:32 / 384:12:32 | L1024 0.93-0.95, L2048 0.87-0.97 | as above | 768 / 1024 |
| TriAttn + e4m3 q / k (fused 128:4:32) | L512-2048 0.79-0.85 (0.93-0.99 of PV alone) | -> 0.95 / 0.85 % at L512 / 1024 (**outside** tol) | rides on the fused route |
| TriMul e4m3 planes, D128 one direction / bidirectional | L256 0.88-0.91 / 0.96, L2048 0.73-0.75 / 0.76 | 0.42-0.63 % -> 2.2-3.5 % (**outside** tol; update rms 3.5-3.8 %) | D128 out 144 / in 192, bidir 256 |
| TriMul, D256 one direction / bidirectional | L256 0.95-0.97 / 1.01-1.03, L2048 0.81-0.82 / 0.83-0.85 | as above | D256 out 224 / in 144, bidir 1024 |
| TriMul, D64 / D384 (B > 1 per sample) | D64 L2048 0.78, D384 L128 0.96 .. L2048 0.84 | as above | D64 512, D384 128, bidir 2048 / 1024 |
| OPM `bound` (all widths, both orders) | S1024 0.74-0.93, S4096 0.57-0.75, S16384 0.52-0.65; S512 1.05-1.36 (gated) | 0.31-0.59 % -> 0.62-1.37 % (tol x 0.38-1.72; inside from depth 8192) | S >= 2048, or S >= 1024 at L >= 256 |
| PWA 8 x 32 | N512-2048 0.71-0.82 at S1024; N384 0.96 | 0.27-0.29 % -> 0.30-0.37 % (inside) | N >= 384, S >= 1024 below N512 |
| PWA wide 64:8 / 128:8 / 128:16 | 64:8 0.74-0.86 (N768-2048), 128:x 0.85-0.96 | -> 0.30-0.38 % (inside) | N >= 256, S x N >= 3 x 2^16 |
| Transition D128 / D256 / D384 / D512 (n 4; n 2) | 0.66-0.75 / 0.69-0.72 / 0.63-0.65 / 0.57-0.59 (n 2: 0.62-0.75) | 0.35-0.42 % -> 3.5-4.5 % (**outside** tol) | rows >= 16384 / 8192 / 65536 |
| token DiT `expand` (A 5, M 1280-10240) | 0.94-0.98 | 0.66 % -> 5.4 % update rms (judge 2.1-4.4e-2, outside) | M >= 640 |
| atom DiT `pre` + `post` (shared cond, A 5) | 0.90-0.97 | rms rel 0.38 % -> 2.9 % (Chai-1 5.6 %); judge 4.8-5.7e-2 | shared conditioning only |
| *not in `safe`*: token DiT `qkvg` | 0.97-0.98 from M 2560 | judge 5.3-7.0e-3 (inside) | M >= 2560 |

Each row's same-cubin control is 0.98-1.02 except a few long-L cells (up to 0.94-1.08, listed in `docs/fp8.md`). Against the 2.5.12-base
tables the errors are identical to the printed digit; 2.5.13's faster default (TriAttn TA_RC, TriMul LT_UNI) raises the FP8 ratios at mid L by
0.01-0.03 and moves no gate (each gate region re-measured; two candidates worth a repeat, not changed: 256:8:32 front gate 768 -> 512 and the
32-channel standalone-pass gate 1024 -> 768).

Several routes are outside the engine's per-module accuracy tolerance at module level (TriMul, Transition, token DiT `expand`, atom DiT, TriAttn
q / k): the recommendation rests on the end-level evidence below, not on the module tolerance. **FP8 is outside the judge:** no registry
row, tolerance or `backend_versions` entry exists for it (registry dtypes `float32|master`); the judge tokens (owed 0 on all three cards) cover
the default bf16 route only. Do not run the judge or `miniworld_engine.measure` with an FP8 switch set.

## End-level evidence (AF3, FoldBench, H100)

AF3 on the 27 FoldBench targets of the paper run (3 additional trunk passes, 200 steps, 5 samples, trunk_seed = diffusion_seed = 0, bf16, eager,
FoldBench's evaluator, all 27 in one process, FoldForge 399d0b4); flag = mean of 5 worse than bf16 by more than max(2 x sample sd, 0.02 lDDT /
0.05 DockQ). Measured on the 2.5.12 base (`fp8/3.0.0-all` c6a1e10ff), whose FP8 kernels are the ones in 3.0.0 (TA_RC is bit-identical, every
FP8 error identical):

| arm | flagged | lDDT mean5 vs bf16 | same-sample RMSD median | whole forward / bf16 (128 .. 2008 tokens; compile + graph, warm) |
|---|---|---|---|---|
| **safe** (fp8safe2) | **0 / 27** | **+0.0000** | 0.54 A | **0.973 .. 0.795** (trunk 0.971 .. 0.734) |
| every route incl. token DiT `qkvg` (all-FP8) | 3 (8j62 -0.045, 7uyn -0.023, 8xgu -0.021) | -0.0148 | 0.83 A | 0.974 .. 0.791 |

Reference: bf16 2.5.12 vs PyTorch 0.38 A, different samples of bf16 1.71 A. Earlier end-level checks of the TriAttn PV route alone (4 trunk
seeds per target): Protenix v1 9 + 3 targets, OpenDDE 16 + 3, Protenix v2 11, OpenFold3 4, AF3 4, IntelliFold2 6 inside the bf16 trunk-seed
spread; RoseTTAFold3 3 / 4 (one failed prediction); Boltz-2 6 / 7 (9c90, below).

## Not recommended

- token DiT `qkvg`: the only placement that moved FoldBench (the all-FP8 flags reproduce with `qkvg` alone; `expand` alone is clean), although
  its module error is the smallest. Random-weight module error did not predict the end-level effect.
- TriAttn hi + lo planes, TriMul FP8-compute K1 (`MINIWORLD_TRIMUL_FP8_K1`, 6.9 % update error), fast accumulation of OPM (4-6 x tol at depth
  16384), token DiT `out` / `squeeze` (slower): experimental switches, off in `safe`.

## Limits

- Validated end to end on **AF3 only** (FoldBench 27 targets with the full `safe` set). Every other model has at most the TriAttn PV route checked
  end to end. TriAttn FP8 runs only for a model named in `fp8.TRIATTN_MODELS` (af3: PV fused + unfused and e4m3 q / k; protenix1: PV fused;
  opendde: PV unfused; `h100_sm90.FP8_PV_WHITELIST` is derived from it); the non-TriAttn routes of `safe` are on for every model.
- **Boltz-2 is not whitelisted**: 9c90 is bistable under any 0.1-2 % numerical noise (its own bf16 trunk seeds fall into 5 modes, 15.3 A apart);
  FP8 V moves 9 of 24 same trunk seeds to another mode while the mode population stays equal.
- Calibration / first-target dependence (above); a caller that never calls `fp8.check()` runs graph replays unguarded (no saturation seen in any
  run: peaks <= 0.93 of the e4m3 range, 0 redos in 288 + 27 + 2 FP8 runs).
- The scale caches are keyed by weight addresses and dropped when the weight is freed; `weights_changed()` does not invalidate them (after an
  in-place weight update call `fp8.disable()` / re-create the modules).
- **FoldForge drift caveat:** the end-level numbers use FoldForge 399d0b4. Later FoldForge commits (da381a5: engine LayerNorm everywhere) change
  bf16 outputs by ~0.2 A by themselves; compare FP8 against bf16 of the same FoldForge commit.
- cuBLASLt e4m3 heuristics and `cvt.satfinite` depend on the CUDA 12 toolchain.

## Prebuilt kernels

Every FP8 kernel the `safe` set can reach is in the sm90a prebuild plan (no first-call JIT): sm90a 513 -> **560** artifacts (+47; no 2.5.14 name
dropped), sm80 72 and sm120 64 unchanged (same names and keys). New names (29; the other 18 are TriMul FP8 K1 / K3 cubins under existing family
names: `mw_b200style_k1_direct` D64 / D128 / D256 / D384 FP8 tiles, `mma_n` K3 with the channel scale, one direction and bidirectional):
`triattn_h100_core_h16_ta_f8pv1_v3`, `..._ta_f8pv1_ta_f8hl1_v3`, `..._ta_f8pv1_ta_f8qk1_v3`, `triattn_h100_front3_mw_f3_vtw_{1,2}_sm90_v2`,
`triattn_h100_front3y16_mw_f3_vtw_{1,2}_sm90_v1`, `miniworld_opm_epilogue_native_fp8{,_128_256,_128_384,_256_512}`, `miniworld_opm_lt_f8`,
`miniworld_pwa_vt8{,_64_8,_128_8,_128_16}`, `transition_fp8_sm90a_c132_n{2,4}_f1_g1_t1_s4_a0_v1`, `transition_fp8_d256_sm90a_n{2,4}_c132_f1_s4_v1`,
`transition_fp8_wide_sm90a_d{384,512}_n{2,4}_c132_f1_v1`, `token_dit_fp8_sm90`, `atom_sm90/{pre,post}90_f8_bf16_w4m{1,3}`. 101 default-route
sm90a artifacts re-key because their source text changed under FP8 `#if` blocks (TriMul `tmn_kernels.cuh` / `k3.cu` families, OPM epilogue,
TriAttn h16 core / fronts): their outputs are bitwise unchanged, but the 3.0.0 sm90 kernels wheel must be rebuilt (all 560 built in the release
cache: 0 failed).

## Verification (rel/3.0.0)

- Default route bit-identical to 2.5.14 (origin/main tree, separate processes, sha1 of every output; training: output, input and parameter
  gradients): **64 / 64 cases** (209 tensors): the registry rows of every module FP8 touches, eval + train (TriAttn incl. AF3 64:4:16 / 128:4:32 /
  512:8:64, 64:2:32, 256:8:32, 384:12:32; TriMul D64-D384 both directions, bidirectional, train; OPM, PWA 8 x 32 + wide incl. training; Transition
  D64-D768 incl. the FP8 row-gate sizes; token DiT, atom DiT) plus direct TriMul calls at L300 / 1104 with and without a mask (the 2.5.13 LT_UNI
  table and exact-key launch cells).
- AF3 end to end (FoldForge 399d0b4, 8gzl + 7sxp, 5 samples each): every switch off, **10 / 10 CIFs byte-identical** to 2.5.14.
- `MINIWORLD_FP8=safe` through FoldForge (AF3 8gzl + 8e4h, `fp8.check()` after every forward), before the decisions: every route fires, 0
  redos, and the 10 CIFs are byte-identical to the fp8safe2 FoldBench run on the 2.5.12 base: the FoldBench evidence carries over to 3.0.0.
- After decisions 1-4: `MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=af3`: **10 / 10 CIFs byte-identical** to that run (same route calls: TriAttn PV
  208 / q k 592, TriMul 704, OPM 24, PWA 24, Transition 356, token DiT 9600, atom DiT 4800; 0 redos). `MINIWORLD_FP8=safe` without a model:
  TriAttn PV / q k 0 calls, every other route the same calls; module level (4 TriAttn configs: fused 128:4:32, front-made 64:4:16, AF3 / 384:12:32
  pass) the TriAttn output bytes equal FP8 off with 0 FP8 core calls, by environment and by `fp8.enable("safe")`; with af3 (environment or
  `enable(model="af3")`) the FP8 core runs. Default route again 64 / 64 bit-identical to 2.5.14; the 13 FP8 GPU files + `test_fp8_switch`
  (12 cases, whitelist rule) 222 passed; `_prebuild --check` sm90a 560 / sm80 72 / sm120 64 unchanged; judge owed 0 on the three cards.
- GPU tests (one queue job, 3 H100): the 13 FP8 files 210 passed (TriAttn PV 31, PV + QK 6, TriMul planes 25 / bidir 16 / widths 18 / K1 6, OPM
  16, PWA 6 / wide 10, Transition 37 / D256 17, token DiT 11, atom DiT 11) and the default-route files the merge touches 471 passed, 24 skipped
  (TriAttn h16 core / tail gate / key tiles / fronts / variants / bidir, TriMul LT_UNI tables / K3 pipe / workspace).
- CPU: 3218 passed, 2 skipped (`test_fp8_switch` (new), `test_capture_api`, `test_init`, `test_imports`, `test_supports_signature`,
  `test_prebuild_plan`, `test_judge`, `test_requirements`, `test_modules_cpu`; 3226 passed after decisions 1-4); `pyright` clean on every
  changed `src` / `tests` file; whole-tree `ruff check` / `ruff format --check`: the same findings as the 2.5.14 tree (all in pre-existing
  `benchmarks/` / older files), nothing from 3.0.0 after the experiment harnesses left the tree; `_prebuild --check` sm90a / sm80 / sm120: no new finding (the FP8
  loaders / sources that were "not reached" / "not compiled" are now planned).
- Judge: owed 0 (accuracy and speed) on h100_sm90, a100_sm80, 5090_sm120: no `version` / `backend_versions` changed by the merge.

## Decisions (owner, 2026-11-17)

Decided:

1. **TriAttn FP8 only for validated models.** `MINIWORLD_FP8=safe` / `fp8.enable("safe")` without a model leave TriAttn on the default fp16
   core; the TriAttn FP8 routes come on only for a model in `fp8.TRIATTN_MODELS`, named once by `MINIWORLD_FP8_MODEL=<model>` (environment, with
   `MINIWORLD_FP8`) or `fp8.enable(preset, model=<model>)` (run time; `enable()` does not read the environment variable). Table: af3 (PV fused +
   unfused, e4m3 q / k), protenix1 (PV fused), opendde (PV unfused): the routes each model's end-level runs used. Boltz-2 stays out (9c90);
   OpenFold3 is not added. The other `safe` routes are unchanged (on for every model).
2. **e4m3 q / k stays in `safe`** for af3: module judge-form error above tolerance (0.95 / 0.85 % at L512 / 1024), end-level AF3 FoldBench 0 / 27
   flagged with it on.
3. **No FP8 registry rows / tolerances.** FP8 is an opt-in path outside the judge; judge tokens cover the default bf16 route only. No judge /
   measure / requirements code names an FP8 route (`test_fp8_switch::test_fp8_is_outside_the_judge`).
4. **Experiment harnesses out of the release tree.** `benchmarks/fp8_e2e/` and `benchmarks/opm_fp8/` (scripts + raw rows; no shipped test
   imported them) were removed; they stay on branch `fp8/3.0.0-all` (e5ce31d0c) and in the `rel/3.0.0` history at 80ecfe71f (`docs/fp8.md` says
   where). Whole-tree `ruff check` / `ruff format --check` now report exactly the 2.5.14 tree's findings (173 `ruff check` findings, 162 files `ruff format` would change; all pre-existing).

Open:

1. Version bump to 3.0.0, push, tag, upload (not done).
2. Gate re-measure: two candidates (not changed): 256:8:32 front 768 -> 512, 32-channel standalone pass 1024 -> 768.
3. The experimental FP8-compute TriMul K1 (`TMN_K1_F8C`, `MINIWORLD_TRIMUL_FP8_K1`) is still in the code and the shared header: remove or keep.
4. FoldForge: wire `fp8.check()` and the model name (`MINIWORLD_FP8_MODEL` / `enable(model=)`) into its forward loop (the earlier patch file
   covers TriMul only) before offering `MINIWORLD_FP8=safe` there.
