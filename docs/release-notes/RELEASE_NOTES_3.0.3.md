# miniworld-engine 3.0.3

Base: 3.0.2 (`db223038b`, tag `v3.0.2` -> `c13fc0b9a`). Integration branch `rel/3.0.3` = 3.0.2 + `fp8t4/tokendit-bytes-default` + `fp8t4/opm` + `fp8t4/pwa`.
One default change (the token DiT training byte savings, bit-identical outputs and gradients); everything else is **opt-in and off by default**: the FP8 inference
routes of 3.0.2 are unchanged, and the new FP8 training routes need an explicit switch.

**Honest summary of the FP8 training work.** FP8 training mostly does not pay off. Where it wins on step time it wins by 4-11 % at the shapes it is gated to, with a
gradient error of 3-4 % rms (7-10x the bf16 default's), and the one end-level check that was run (a real Protenix v1 fine-tune with FP8 token DiT `expand`) raised the
diffusion loss by +2 %. The real gains are the **byte / memory savings**, which need no FP8 (token DiT: default now) or are opt-in memory modes (OPM `recompute`, PWA `O8`).
Evidence: `docs/fp8_training.md` (token DiT rounds 1-4, OPM rounds 1-4, PWA rounds 1-4: step-time tables, gradient errors against fp64, memory tables, training curves).

## Default change: token DiT training byte savings (`nox` + `hsave`)

`TokenDiTProtenix` (and every subclass that runs the same default-option bf16 block: `TokenDiTMiniWorld`; Boltz-2 / RF3 / Chai-1 keep their own blocks and are unaffected)
on H100, bf16 master, width 768, training: the training block no longer keeps an fp32 copy of the input (`nox`) and the forward keeps `h` (`hsave`).
Outputs and all gradients are **bit-identical** to 3.0.2 (and 3.0.1) (6 shapes, sha1 of the output and 27 gradients; re-checked on the installed wheels, see below).
Step time -0.9 .. -1.1 % sustained (-1.5 .. -2.1 % cool) at M = A x L >= 18432, -0.5 .. -1.1 % below; peak memory -3 %; the largest A at L 384 grows from 1536 to 1792 (+17 %).
`MINIWORLD_TOKEN_DIT_TRAIN_BYTES=0` returns to the 3.0.2 path. Also in this release: the scale floor 1e-30 in every FP8 quantizer of the training module (a real-weight NaN: channels with
amax about 1e-37 gave scale 1e-39 and `1 / scale = inf`).

| class | 3.0.2 | 3.0.3 | cause |
|---|---|---|---|
| TokenDiTProtenix (`backend_versions["cuda:h100_sm90"]`) | 5 | 6 | the default training block keeps no fp32 input copy and keeps `h` (bit-identical; the re-measure is record keeping) |

`TokenDiTMiniWorld` inherits the value. No module `version` was bumped (backend-only rule). The judge owed 292 accuracy + 292 speed cells on `h100_sm90` because of the bump;
they were re-measured (section "Judge re-measure").

## New opt-in FP8 training routes (all OFF by default)

| route | switch | what it does | measured (`docs/fp8_training.md`, H100, CUDA graph, vs the 3.0.1 default) |
|---|---|---|---|
| token DiT, recommended | `MINIWORLD_FP8_TRAIN=1` | = `nox` + `hsave` (now the default anyway) | see above |
| token DiT FP8 `expand` | `MINIWORLD_TOKEN_DIT_FP8_TRAIN=expand` (gate: 16384 rows) | e4m3 expand GEMM | step -0.7 .. -0.9 % (inside the 1 % control spread), transition gradient error 4-5 %, **end-level fine-tune: +2 % diffusion loss. No-go for recommended use**; kept opt-in for the record |
| OPM | `MINIWORLD_OPM_FP8_TRAIN=e4m3` (d_pair 128 / 256; 384 borderline; 512 not served; gate `opm_fp8_train.MIN_L_BY_CZ`) | e4m3 prologue (both planes), `dg_wbuf`, single-pass dual-plane dgrad | deep MSA (S >= 2048): 0.58-0.72x of the module step; L 256-768 at depth 1024, d_pair 128 / 256: 0.81-0.95x; not at d_pair 384 / 512 or L 128; gradient error 3-4 % rms (7-12x bf16); 300-step distillation curve within 0.2 % |
| OPM memory mode | `MINIWORLD_OPM_FP8_TRAIN_OSTORE=recompute` (use at d_pair 128) | recompute O in the backward instead of saving it | saved bytes -70 .. -85 %, max L +40 % for the 64 / 128 widths |
| PWA backward-only | `MINIWORLD_PWA_FP8_TRAIN=bwd` (S >= 1024, N 512 .. 1024, d_pair 128 / 256) | e4m3 backward, forward numerics unchanged | step 0.95-0.96x, peak -12 .. -17 %, max S +10 .. 15 %; gradient error 2.3-4 % rms |
| PWA full | `MINIWORLD_PWA_FP8_TRAIN=1` | e4m3 forward + backward (also N 384) | step 0.89-0.93x; forward error and +0.3-1 % loss |
| PWA memory mode | `MINIWORLD_PWA_FP8_TRAIN_O8=1` (needs `bwd` or `1`) | `o` saved as e4m3 (`csrc/pwa_o8.cu`), early release of do8 / doT8 | peak -20 .. -23 %, max S +20 .. 23 %, step +7 .. 11 %, +0.7 % loss |

With no switch set every one of these routes is bit-identical to 3.0.2 (verified on the installed wheels: default token DiT / OPM / PWA / TriMul / Transition / TriAttn training paths).

## Not included in 3.0.3

- TriMul FP8 training `k1` (branch `fp8t3/trimul`; wins only at L >= 1024).
- Transition `g8` / `h8` (branch `fp8t4/transition`: narrow D384 shapes only).
- TriAttn and atom DiT FP8 training (no-go).
- The D384 "wide path instead of generic" training default (an FP8-independent finding, saved bytes -57 %; a possible later default change).
- TriAttn whitelist lines for Boltz-2 and IntelliFold2 (owner decision pending); Chai-1 grouped OPM stays OFF.

## Prebuild regeneration (release build)

Kernels wheels (sm90 / sm80 / sm120, cu129, cp312) rebuilt incrementally from the 3.0.2 release cache with GCC 12.4 (CUTLASS 4.2, MathDx): sm90a 39 built, 579 artifacts, 0 absent (3.0.2: 574);
sm80 / sm120: 0 built. The prebuild plan gained the training builds (no first-call JIT of the ~100 s OPM variants): the OPM e4m3 prologue at the widths of `opm_train.SHAPES` for training rows
(`miniworld_opm_epilogue_native_fp8`, `_128_256`, `_128_384`, `_256_512`, whose names were already in 3.0.2 but whose source changed: re-keyed), the PWA FP8 training builds, and
`token_dit_fp8_train_sm90` (the row kernels of the default nox / hsave blocks at width 768). The changed `token_dit_train_rows.cu` re-keys the default token DiT training row artifacts (outputs unchanged).
New names (sm90): `miniworld_pwa_vt8_nat`, `miniworld_pwa_vt8_nat_bf`, `miniworld_pwa_glue3_f8`, `miniworld_pwa_o8`, `token_dit_fp8_train_sm90`. The names carried since 3.0.1 are carried again.
**No name of 3.0.2 is dropped** (manifest names 0 dropped / 5 added for sm90, 0 / 0 for sm80 and sm120); the sm80 and sm120 payloads are unchanged (every path and CRC identical to 3.0.2).

## Judge re-measure (H100, `h100_sm90`, tree `rel/3.0.3`)

`TokenDiTProtenix.backend_versions["cuda:h100_sm90"]` 5 -> 6 (TokenDiTMiniWorld inherits). Owed before: accuracy 292 + speed 292 (all "adapter:cuda 5->6"; 0 for every other class);
cycle via `tools/slurm/cycle.sh` (2 groups, 1 H100 / 10 min per job, the release prebuild cache): 1168 records issued; owed after: **0 / 0**, pending 0. **RED: 0**; lights unchanged
(TokenDiTProtenix 811 GREEN / 513 YELLOW, TokenDiTMiniWorld 46 / 26, before and after). Speed, `cuda` median ms against the 3.0.2 token of the same case (316 cells): geometric mean 0.999
(training cells, 132: 0.998, range 0.977 - 1.007); no cell more than 5 % slower or faster. `a100_sm80` and `5090_sm120`: 96 / 96 owed before and after (the `PairformerBlock` chai1 rows without a token
since 3.0.1; unchanged by this release; those cards were not measured here).

## Verification (rel/3.0.3, H100, Slurm)

Installed-wheel runs use the five release artifacts extracted without the repo on the path, an empty JIT cache, no CUTLASS / MathDx and no `nvcc` on `PATH`.

| item | result |
|---|---|
| manifest names against the 3.0.2 upload | sm90 0 dropped / 5 added, sm80 0 / 0, sm120 0 / 0; wheel families 0 dropped / 5 added; sm80 and sm120 payloads identical to 3.0.2 (path + CRC) |
| installed-wheel load of every indexed sm90 artifact (579 = 163 + 190 + 226) | 579 present, 579 resolved, **0 compiled**; the plan check reaches 94 call sites (3.0.2: 89) and leaves none of the new `.cu` files unplanned |
| default routes, FP8 off, installed 3.0.2 vs installed 3.0.3 (separate processes, each twice) | inference outputs bitwise identical (token DiT x4, PWA, OPM, TriMul, TriAttn, atom DiT x5, default fp16 h16 core) |
| default TRAINING paths, installed 3.0.2 vs 3.0.3, output + input gradients + every parameter gradient | **token DiT (TokenDiTProtenix A8 L128 / A6 L200 / A48 L128 / A24 L384, TokenDiTMiniWorld): all 27 / 29 tensors bitwise identical, run to run and across releases (the nox / hsave claim holds on the shipped wheels)**; OPM 64/128 and 128/256, PWA 64/128, Transition D128: bitwise identical. Boltz-2 / RF3 token DiT blocks, TriMul D128 / D256 and TriAttn differ between two runs of 3.0.2 itself (nondeterministic backward / harness inputs; their sources are unchanged in this release); the tensors that differ across releases are exactly the ones that differ run to run |
| FP8 + training GPU / CPU test files (23 + 7 files) installed | all pass except the cases that also fail on the installed 3.0.2 (identical counts): JIT-ablation variants of Transition (24 + 6), TriMul K1 / k3-pipe / planes (6 + 1 + 2), `test_pwa_bf16_train` native-any-shape (2), `test_capture_api` run from a copy (repo-layout test, 13 / 13 in the repo tree). New: `test_opm_fp8_train` 16, `test_pwa_fp8_train` 10, `test_token_dit_fp8_train` 16, `_train_bytes` 23, token DiT train accumulation / shared cond / pair apb / call cache 3 + 12 + 8 + 2, OPM wide prologue / deep product 7 + 25: all passed |
| re-keyed routes + restored Triton backends, installed | 11 + 2 + 113 passed; 63 passed; 0 builds |
| judge owed (versions only) | `h100_sm90` 292 / 292 -> 0 / 0, RED 0; `a100_sm80` / `5090_sm120` 96 / 96 before and after |
| CPU gate (`tools/check.sh cpu`) | ruff check clean (23 lint findings of the merged training tests / tools fixed, one real gap fixed: `opm_fp8_train._SITES` was missing from `test_capture_api`'s list); pyright 0 errors; tests/cpu 3400 passed, 2 skipped; `ruff format --check` reports files that are not mass-formatted (not enforced) |
