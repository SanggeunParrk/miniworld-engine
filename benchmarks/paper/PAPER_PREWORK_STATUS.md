# Paper pre-work status (engine + FoldForge, four cards), 2026-11-22 (B200 updated 2026-11-23)

Everything below is on `main`: engine `miniworld-engine` (this repo) and FoldForge `origin/main`. Numbers come from the CSVs; see `PAPER_DATA_SUMMARY.md` (engine, computed from `E*/E*_<card>.csv`) and the FoldForge `benchmarks/paper/F*/README.md` DATA_STATUS sections.

## Engine releases used by the paper

| release | content | PyPI / tag |
|---|---|---|
| 3.0.6 | 5090 AF3 NaN fix (Transition fp16 residual accumulator), long-L PWA / TokenDiT, harness | tag `v3.0.6` |
| 3.0.7 | H100 fp32 Transition with TF32 off on CUDA, bf16 opt-in for the bias-only pair-bias logits (fp32 default), B200 kernel set (source / JIT only) | tag `v3.0.7` |
| 3.0.8 | A100 / RTX 5090: OpenDDE `TokenDiTProtenix` `extra_pair_bias` on CUDA (was Triton) | tag `v3.0.8` |
| 3.0.9 | B200: TriAttn `ex2_poly2` upper clamp (non-finite on real AF3 / Chai-1), fp32 Transition `ieee_f32_sm100`, OpenDDE `extra_pair_bias` | tag `v3.0.9` |
| 3.0.10 | RTX 5090 FP8 routes (every H100 slot; e4m3 Transition opt-in only), Boltz-2 up-gate + token pair-bias hoist on sm_120 | tag `v3.0.10` |
| main after 3.0.10 (unreleased) | capability cache (host time), B200 PWA e4m3 off by default (a888a8143), B200 bf16 token DiT fork off by default (4915ee25a, TokenDiT* b200 26), Chai-1 unblocked on the B200 (37f357dbf) | -- |

There is no B200 kernels wheel: B200 rows are measured on source trees (labels in the CSVs, `engine_label`).

## Engine experiments (E1-E6)

| card | E1 speed | E2 sweeps | E3 accuracy | E4 FP8 | E5 external kernels | E6 memory |
|---|---|---|---|---|---|---|
| H100 | done | done | done | done | done (+ ds4sci eager correction) | done |
| A100 | done | done | done | n/a | done | done |
| RTX 5090 | done | done | done | FP8 routes since 3.0.10 (module-level E4 not re-run; end level: F3 / F4 / F5 / F9 FP8 arms in FoldForge) | done | done |
| B200 | done | done | done | done | done (+ fp64 accuracy) | done |

Timing: CUDA graph, cold L2, medians; the iteration count (100 or 20) does not change the quantity measured and is not a caveat. E5: AF3 setup (d_pair 128, 4 heads), L 512 / 1024 / 2048, inference and training, all op classes, 13 external candidates; cells without a candidate (OPM, LayerNorm) or without support are marked, not missing.

## FoldForge experiments (F-series)

| experiment | H100 | A100 | RTX 5090 | B200 |
|---|---|---|---|---|
| F1 / F2 FoldBench (AF3) | done (3.0.1; 32 targets > 2048 tokens re-run on 3.0.6); launch mode stated (multi, GPU-h = 2 x wall) | n/a | n/a | n/a |
| F3 5I28 time | 3.0.1 / 3.0.2 values (labelled) | 3.0.6, 33/33 | 3.0.6 (9 of 11 predictors; ESMFold2 and ESMFold2-Fast not feasible: ESM-C 6B loading exceeds the 16 GB host RAM cap) | done on engine main a553001e1 (66 cells; boltz2 / chai1 / rosettafold3 per-size processes; ESMFold2 family at the release 134 steps) + Anthropic kit arm (3 reps, random in-process weights; OpenDDE / OpenFold3 / ESMFold2(-Fast) kits not runnable on sm_100): engine faster in all 47 comparable cells, min 1.70x |
| F4 27 targets | 3.0.1 (labelled; largest target 2008 tokens, outside the PWA hole) | 3.0.6 + OpenDDE 3.0.8, 22/22 | 3.0.6 + OpenDDE 3.0.8 (9 of 11 predictors; ESMFold2 family not feasible) | done, engine 291 ok / PyTorch 268 ok; ESMFold2 re-measured at 134 steps on a553001e1 (27/27 both arms) |
| F5 sample comparison | 3.0.1 | done | done (9 of 11 predictors) | done (1,340 pairs); ESMFold2 at 134 steps: 135 pairs, engine 0.66 A vs baseline 2.66 A |
| F6 training speed | engine arm re-measured, Transition on CUDA (0.8121 / 0.7870 s per example) | 3.0.6 | 3.0.6 | done (8/8 configurations); engine arm re-measured on a553001e1: AF3 0.4375 / Protenix v1 0.4291 s per example (H100 0.812 / 0.787), Triton 0 |
| F7 finetune loss | PyTorch run 433 steps vs engine 1424: compared on the 433 common steps | n/a | n/a | n/a |
| F8 finetuned FoldBench | done | n/a | n/a | n/a |
| F9 inference limit | 3.0.6 | 22/22 | 9 of 11 predictors (ESMFold2 and ESMFold2-Fast not feasible: 16 GB host RAM cap) | done (22 limits; chai1 n=40 CUDA IMA at 98 % reserved classified OOM) |
| F10 training limit | dropped from the paper | | | |
| F11 FP8 accuracy | done | n/a | n/a | n/a |

### 2026-11-24 update: engine 3.0.10, FP8 arms everywhere, F4 under compile + CUDA graphs

F4 now runs under compile + CUDA graphs (`scope: trunk`, as F1 / F3; FoldForge 16d094c): eager timing measured host overhead (B200 FP8 looked
1.10-1.27x slower than bf16 eager, faster everywhere under graphs). New FoldForge data files (`*_3010*.csv`; bf16 and FP8 arms on the same engine):

| card | F3 | F4 (graph) + F5 (FP8 vs bf16) | F9 |
|---|---|---|---|
| H100 | 3.0.10, 11 predictors, FP8 / bf16 n = 16 0.777-0.922 | 11 predictors, 0.842-0.939; F5 337 keys, monomer -0.0087 | 3.0.10, both arms |
| RTX 5090 | FP8 arm on 5090-int (= 3.0.10) | 10 predictors, 0.852-0.965 (105 OOM cells on 32 GB under graphs); F5 230 keys | 3.0.10, both arms |
| B200 | main fixes (a888a8143 + 4915ee25a), 0.802-0.966 at n = 16 | 0.914-0.978, 21 / 21 finite | FP8 3.0.10; re-run on the fixes queued |
| A100 | -- | bf16 graph re-run (cssb3) | -- |

FoldBench AF3 (F1 / F2) on 3.0.10: FP8 and bf16 arms re-run (multi mode, 2 GPUs).

## Defects found by the end-to-end runs and fixed (all on `main`)

- 5090 AF3 NaN (3.0.6); H100 fp32 Transition on Triton (3.0.7); A100 / 5090 OpenDDE `extra_pair_bias` on Triton (3.0.8); B200 TriAttn non-finite, fp32 Transition and OpenDDE on Triton (3.0.9).
- 2026-11-23 (engine main a553001e1, pushed; no release): B200 atom-local attention ran the slid key window's TAIL (padding-atom query windows) on CUDA cores (F6 1.15 -> 0.44 s / example; 9760cfe4b); B200 CUDA-graph capture skipped `record_stream` (`_streams._skips_in_capture` meant for sm_80 only) -> bf16 confidence collapse at 512 / 1,024 tokens in captured runs since 3.0.4 (AF3 5I28 n = 4 pLDDT 43.8 -> 88.3, H100 88.6; a553001e1; F4 / F5 are eager and unaffected; A100 / 5090 keep the skip and are unverified); B200 FP8 Transition D128 gate limited to the rows where e4m3 is faster (6189cf1e0).
- Open, documented, not in the paper's experiments: B200 fp32 inference S=1, L=100 / 104 NaN (composed runner; bf16 unaffected); B200 multi-size CUDA-graph replay non-finite inside one process (workaround: one process per size); B200 is not a supported release target (no B200 kernels wheel). B200 judge tokens: issued 2026-11-24 after the `on_kit_stack` fix (e7054a74e; the host's CUDA-13 driver had made every B200 engine record look like a kit-stack record): owed 0 (accuracy + speed), 18,300 tokens (70479d8ca), engine accuracy RED 0; B200 cuda >= 1.5x H100 cuda in 3,398 of 3,679 cases (H100 >= 20 us), slower than H100 only in SmoothLddtLoss (declared exception, worst 1.13x).
- Memory: engine 3.0.8 keeps 18.6 MiB per OpenDDE TokenDiT block resident (0.44 GiB at 24 blocks); one 5090 cell (7wvs) therefore runs out of memory on 3.0.8 (documented in the F4 README).

## What the data supports (scope)

- Across the measured operator classes the engine CUDA backend is faster than PyTorch eager on every card (geometric means in `PAPER_DATA_SUMMARY.md`), and no surveyed public implementation in E5 (AF3 setup) is faster than the engine in any measured cell.
- E1 (all requirement rows): external baselines (cuEquivariance, Anthropic kit) are faster than the engine CUDA backend by more than 3 % (ratio < 0.97) only in LayerNorm (declared exception). The three RTX 5090 cells the 3.0.6 E1 campaign had listed (`TriangleAttentionAF3` training bf16 L=384 d_pair 64 vs cuEquivariance 0.922x; `OuterProductMean` inference bf16 L=256 0.929x / L=384 0.958x vs the Anthropic kit) were re-measured on 2026-11-24 on branch `5090-int` (tree eff985cfe, contains the TriAttn fix 5f9fae307 that postdates the campaign) with a paired protocol: every backend of a case (pytorch, cuda, triton, cueq, anthropic) built once in ONE process and timed in 6 interleaved rounds (order rotated per round; CUDA graph, cold L2, warmup 5, 20 iterations per round, as the campaign), ratio per round then median (`E1_operator_speed/paired.py`, `rtx5090_paired.sh`, `paired_summary.py`). All 330 cases of the two classes: no baseline below 0.97 any more; the three cells are now 1.659x (cueq), 1.031x and 1.086x (kit). The old kit ratios were cross-process (engine stack and kit stack are separate processes in the campaign; on the 5090, clocks unlocked, the kit process's own pytorch reference ran 4-9 % faster than the engine process's on the same cases), the old TriAttn ratio predates 5f9fae307. Write 'external implementations are ahead only in LayerNorm'. Remaining paired ratios < 1 of the two classes are within noise (worst: TriangleAttentionAF3 eval L=256 row 78 vs the Anthropic kit 0.979x, cross-process campaign 1.061x; Triton 0.984-1.000x). Cells where only the engine's own Triton backend is ahead are a path-selection matter, not a comparison with another implementation (lists: `PAPER_DATA_SUMMARY.md` section 1.3 / 1.3b, `paper_common/out/<card>/engine_not_fastest_cells.csv`; paired: `paper_common/out/rtx5090/engine_not_fastest_cells_paired_triattn_opm.csv`, rows `E1_operator_speed/E1_rtx5090_paired_triattn_opm.csv`). The other RTX 5090 classes keep the cross-process campaign numbers (only LayerNorm, an exception, has a kit/cueq cell < 0.97 there).
- E3 RED rows (96 in total over four cards): the engine-side ones are SmoothLddtLoss training fp32 `grad:x_pred` at L=5120 / 7168 on all four cards (cuda and Triton alike: error about 1e-3 against a 1e-5 tolerance while PyTorch is about 1e-7; a user-declared exception class, but an accuracy fact that must be stated) and AtomDiTRoseTTAFold3 training fp32 `grad:attention.norm_key.bias` on H100 at L=1024 (analytically zero gradient: the fp64 reference is noise, and PyTorch's own error happened to be ~1000x smaller on that one cell, which set the tolerance; the same tensor is YELLOW on the other cards). All other RED rows belong to the Anthropic kit baseline.
- The comparison set is the public implementations found by the survey (listed in `E5_other_kernels/E5_results.md` and `third_party/README.md`); claims should name that set. Non-public or unsurveyed implementations are outside the data.
- The wording of claims in the paper is the authors' decision; this file only records what was measured.
