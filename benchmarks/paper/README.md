# Paper experiments (E1-E6)

The engine-side experiments of the paper, one folder each, re-runnable on every card with the commands below. Everything here only *calls*
the engine's measurers (`miniworld_engine.measure.speed` / `accuracy`) or small drivers built on the same pieces; nothing under `src/` is
changed, no judge token is touched. All experiments are measured on **one release** (one tag); every raw record and CSV row carries the engine
version, the commit, torch / CUDA / driver, the GPU and the node.

| ID | Folder | Measures | Paper location | Cards |
|---|---|---|---|---|
| E1 | `E1_operator_speed/` | speed of all 39 operator classes (every requirements row), inference + training; backends pytorch, cuda, triton, cueq, anthropic | Fig 1c, Supp 2-4, abstract | H100, A100, 5090 |
| E2 | `E2_sweeps/` | TriMul (outgoing) / TriAttn (starting, 32 channels per head): L sweep (d=128) and D sweep (L=768), inference + training | Supp 5-7 | H100, A100, 5090 |
| E3 | `E3_accuracy/` | accuracy against float64 (judge tolerances), the E1 cases | Methods | H100, A100, 5090 |
| E4 | `E4_fp8/` | FP8 path of TriMul / TriAttn inference against the default path, same process, alternating | Fig 1c, Supp 5 | H100, B200 (5090: no sm120 FP8 path; A100 n/a) |
| E5 | `E5_other_kernels/` | 12 other public candidates (ds4sci, trifast, fast_trimul, gpumode, geonn, gnops, of3tri, megafold, megafold_tr, bionemo_ir, turbopf, ligo_pwa) + cuEquivariance as external backends of every Pairformer operator class (TriAttn, TriMul, AttentionPairBias, OPM, PWA, Transition); results in `E5_other_kernels/E5_results.md` | Fig 1c, Supp 2-7 | H100, A100, 5090 |
| E6 | `E6_memory/` | peak memory per operator / backend on the E2 grid, inference + training | new (Supp) | H100, A100, 5090 |

Each folder has `README.md` (what, conditions, commands, output), config `*.yaml` / extra requirements CSV, one script per card
(`h100.sbatch`, `a100.sbatch`, `rtx5090.sh`), `collect.py` (raw records -> CSV). `paper_common/` holds the shared code.

## Results layout

- raw records: `<ID folder>/records/<card>/<stage>.r<k>.jsonl` (+ `.meta.json` provenance, `.done` marker when a file is complete);
  stage `engine` = engine environment (pytorch, cuda, triton, cueq), `kit` = the Anthropic kernels on the card's kit stack;
  `r<k>` = repeat k (1-3), each in a different Slurm job / ssh run;
- CSV: `<ID folder>/<ID>_<card>.csv` with `card` in `h100 | a100 | rtx5090 | b200`. Columns:
  `id, card, gpu, engine_version, commit, module, mode, backend, L, d, n_head, dtype, ms_median, ms_min, ms_max, peak_mem_gib, status, reason`,
  then experiment-specific trailing columns (`stack, row, variant, samples, depth, repeats, timed_as` for speed; light and errors for E3; ...).
  `ms_median` = median over the repeats of the per-repeat medians, `ms_min` / `ms_max` = their min / max. `mode` is `eval` (inference) or `train`.
  Failures, OOM (`skipped`) and refusals (`unsupported`) stay as rows with `status` + `reason`; nothing is re-routed, figures draw only `ok`.
- `paper_common/tables.py` converts the CSVs to the shapes the paper scripts read (`sweeps.csv`, `operator_paired_ratios.csv`,
  `supp_operator_speed_3cards.csv`, `accuracy_lights.csv`).

## Measurement conditions (E1, E2, E5, E6 speed)

`timing: graph` (CUDA graph replay), cold L2 flush, warmup 5, 20 iterations (the configs), median; TF32 rule and inputs as in `measure/speed.py`.

**Iteration note (methodology).** The kernel timing was changed from 100 to 20 iterations (commit d9e8a3105) while the campaigns were running; nothing was re-measured because of it. H100 E1-E4 and E6 were measured at 100 iterations (before the change); E5 (all cards) was measured at 20. On A100 and RTX 5090 the records are a mix, because the change landed mid-campaign: E1 (the later passes, and on the 5090 repeat 3) and the A100 kit stage of E2 hold 20-iteration records, the rest (E2 engine, E1 repeat 1 on the 5090, most of E1 on the A100) 100. Every raw record carries its own `iterations`. All reported numbers are medians (of the per-repeat medians), so the iteration count changes the noise of a sample, not its expected value; the 100- and 20-iteration records are not biased against each other and are not corrected or re-measured.
Anthropic kernels run through each card's kit stack: H100 `third_party/anthropic/kit_stack.sh` (CUDA-13 container; needs `KITS_DIR`, `FF_SCRATCH`,
`MINIWORLD_ANTHROPIC_ROOT`), A100 `third_party/anthropic/kit_stack_compat.sh` with `MINIWORLD_ANTHROPIC_ROOT` = a release root holding the
cu129 / py312 prebuilt (cssb3: `~/w45-jobs/upstream`) and `KIT_COMPAT_DIR` (CUDA-13 compat libcuda, default `~/cuda-compat-13.0`); the RTX 5090
needs none (driver is CUDA 13). The `pytorch` reference of the `kit` stage is measured in that same process (pairing within a stack).

## Run order

1. Release tag checked out on each card (H100: this repo; A100: cssb3, `~/miniworld-engine-scratch/<tag>` or a copy; 5090: a `~/mwe/` copy), kernels prebuilt for the card.
2. For each experiment `E1 E2 E3 E4 E5 E6` and each repeat `REP=1 2 3`:
   - **H100** (login node, only sbatch): `cd benchmarks/paper && REP=1 paper_common/chain.sh 12 E1_operator_speed/h100.sbatch`
     (chain of 12 resumable 10-minute `--qos=cssb_h100` jobs; the jobs exit at once when the file is done). E4: `E4_fp8/h100.sbatch`.
   - **A100** (cssb3): the same with `a100.sbatch` (E4 not applicable).
   - **RTX 5090** (cssb-rtx): `REP=1 benchmarks/paper/E1_operator_speed/rtx5090.sh` (flock + 16 GB cap via `~/mwe/gpu.sh`; run again to resume).
   - E5 first needs the libraries: see `E5_other_kernels/README.md`.
   - **B200** (host `b200`, no Slurm, shared account: GPUs 0-3 only): from a clone of the source tree (the PyPI wheels carry no sm100 kernels), prebuild
     (`python -m miniworld_engine.kernels._prebuild --arch sm100 --jobs N`), then `nohup paper_common/b200_driver.sh 0 1 2 3 &` (one lane per GPU, tasks of
     `paper_common/queue_b200.txt` via `b200_lane.sh` / `b200_task.sh`; environment `paper_common/env_b200.sh`; resumable, `.done` markers, logs in `logs_b200/`).
     Kit stack: `third_party/anthropic/kit_stack_b200.sh` with a torch 2.13.0+cu130 venv (`KIT_VENV`). E5 graph check: `E5_other_kernels/gc_b200.sh`.
3. Collect: in a CPU job, `python <ID folder>/collect.py --card h100` (E4 likewise), then `python paper_common/tables.py --card h100 ...`.

Smoke run (1-2 cases, no records committed): `EXTRA="--module TriangleMultiplication --lengths 128 --max-cases 2" RECORDS=$SCRATCH/smoke sbatch --export=ALL,EXTRA,RECORDS,REP=1 ...`.
(Every `sbatch` log `slurm-*.log` written into a folder is git-ignored.)

## Status (3.0.6 candidate; full measurements wait for the released tag)

Smoke = 1-2 cases, 1 repeat, on the 3.0.6 candidate tree (rc-3.0.6 c40a761ef + this harness). **Full measurement: not yet, pending the 3.0.6 tag, on every card.**

| ID | H100 | A100 | RTX 5090 |
|---|---|---|---|
| E1 | smoke yes (engine + kit stack) | smoke yes (engine + kit) | smoke yes (engine + kit) |
| E2 | smoke yes (TriMul engine; TriAttn kit stage queued) | smoke yes (TriMul + TriAttn, engine + kit) | smoke yes (TriMul engine + kit) |
| E3 | smoke yes (TriAttn, engine + kit) | smoke yes (TriMul, engine + kit) | smoke yes |
| E4 | smoke yes (TriMul, both gate passes; TriAttn queued) | n/a (no FP8 path) | smoke yes: no sm120 FP8 route (`routes_fired` false, ratio ~ 1) |
| E5 | measured 2026-11-21, engine 3.0.6 (3 repeats, 20 iterations; CUDA-graph validity checked) | measured as H100 (A100: `cssb3`, 3 repeats, graph check) | measured (5090: 3 repeats; part of the TriMul records 100 iterations or re-measured at 20: see `E5_other_kernels/README.md`; graph check partial: out-of-memory cells unchecked) |
| E6 | smoke queued | smoke yes | smoke yes |

Smoke caveats: kernels were JIT-built on first use in the worktree (a cold cache ate a 10-minute job on H100): prebuild the tag's kernels
before a campaign (`python -m miniworld_engine.kernels._prebuild --arch sm90a`, sm80 on the A100; `tools/slurm/cycle.sh` shows the job shape).
Where there is no `.git` (cssb3, cssb-rtx copies) set `MWE_COMMIT=<tag commit>` (and `MWE_VERSION` if the tree has no pyproject) so records carry the commit.
Library accuracy of the E5 candidates against float64: `E5_other_kernels/E5_results.md` section 4 (E5 speed records are graph timings; a candidate whose kernels leave the CUDA graph is `invalid_graph`, section 0 of that file).
