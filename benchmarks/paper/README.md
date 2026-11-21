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
| E4 | `E4_fp8/` | FP8 path of TriMul / TriAttn inference against the default path, same process, alternating | Fig 1c, Supp 5 | H100 (5090: no sm120 FP8 path; A100 n/a) |
| E5 | `E5_other_kernels/` | other public kernels (DeepSpeed4Science EvoformerAttention, trifast) as external backends | Fig 1c, Supp 2-7 | H100, A100, 5090 |
| E6 | `E6_memory/` | peak memory per operator / backend on the E2 grid, inference + training | new (Supp) | H100, A100, 5090 |

Each folder has `README.md` (what, conditions, commands, output), config `*.yaml` / extra requirements CSV, one script per card
(`h100.sbatch`, `a100.sbatch`, `rtx5090.sh`), `collect.py` (raw records -> CSV). `paper_common/` holds the shared code.

## Results layout

- raw records: `<ID folder>/records/<card>/<stage>.r<k>.jsonl` (+ `.meta.json` provenance, `.done` marker when a file is complete);
  stage `engine` = engine environment (pytorch, cuda, triton, cueq), `kit` = the Anthropic kernels on the card's kit stack;
  `r<k>` = repeat k (1-3), each in a different Slurm job / ssh run;
- CSV: `<ID folder>/<ID>_<card>.csv` with `card` in `h100 | a100 | rtx5090`. Columns:
  `id, card, gpu, engine_version, commit, module, mode, backend, L, d, n_head, dtype, ms_median, ms_min, ms_max, peak_mem_gib, status, reason`,
  then experiment-specific trailing columns (`stack, row, variant, samples, depth, repeats, timed_as` for speed; light and errors for E3; ...).
  `ms_median` = median over the repeats of the per-repeat medians, `ms_min` / `ms_max` = their min / max. `mode` is `eval` (inference) or `train`.
  Failures, OOM (`skipped`) and refusals (`unsupported`) stay as rows with `status` + `reason`; nothing is re-routed, figures draw only `ok`.
- `paper_common/tables.py` converts the CSVs to the shapes the paper scripts read (`sweeps.csv`, `operator_paired_ratios.csv`,
  `supp_operator_speed_3cards.csv`, `accuracy_lights.csv`).

## Measurement conditions (E1, E2, E5, E6 speed)

`timing: graph` (CUDA graph replay), cold L2 flush, warmup 5, 20 iterations (the configs; 100 before commit d9e8a3105), median; TF32 rule and inputs as in `measure/speed.py`.

**RTX 5090 (cssb-rtx) iteration note.** The configs were switched from 100 to 20 iterations at 2026-11-21 ~11:30 KST (d9e8a3105), while the campaign was running. Records made before that point stay at 100 iterations (not re-measured): E1 repeat 1 (engine + kit), E2 repeats 1-3, and the first passes of E1 repeat 2; passes started afterwards (rest of E1 repeats 2/3, E5) use 20. A raw `.jsonl` of E1 repeat 2/3 can therefore hold both (warmup 5 in both); each record carries its own `iterations`. E5 on the 5090: the first (100-iteration, full E2-grid) E5 records are kept in `E5_other_kernels/records/rtx5090/old_it100/` (not in the CSV); the CSV is the AF3-only 20-iteration run.
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
| E5 | adapters installed; smoke queued | smoke yes: trifast runs, ds4sci fails to JIT-build (needs CUTLASS_PATH + nvcc; recorded as failed with the reason) | smoke queued (GPU lock) |
| E6 | smoke queued | smoke yes | smoke yes |

Smoke caveats: kernels were JIT-built on first use in the worktree (a cold cache ate a 10-minute job on H100): prebuild the tag's kernels
before a campaign (`python -m miniworld_engine.kernels._prebuild --arch sm90a`, sm80 on the A100; `tools/slurm/cycle.sh` shows the job shape).
Where there is no `.git` (cssb3, cssb-rtx copies) set `MWE_COMMIT=<tag commit>` (and `MWE_VERSION` if the tree has no pyproject) so records carry the commit.
Library accuracy of the E5 adapters is not verified against float64 (speed only).
