# tools/

Developer scripts. Nothing here is part of the wheel; every Python script runs on a compute node (a GPU node for the benchmarks).

| path | what |
|---|---|
| `check.sh` | the repository gate: ruff, pyright and the CPU tests (`tools/check.sh`), the GPU tests (`tools/check.sh gpu`); `pixi run check` calls it |
| `slurm/cycle.sh` | one judge cycle on a Slurm cluster: plan the owed cases, prebuild, measure, issue tokens ([docs/judge.md](../docs/judge.md)) |
| `bench/` | one-off timing helpers for kernel work: `ab_cells.py` (same-process A-B timing of variants of one cell), `prof_cell.py` (per-kernel time of one cell), `cublas_kernel_names.py`, `contract_bounded_sm80.py` / `trimul_masked.py` (A100 bounded TriMul contraction), `pwa_long_bench.py` (A100 PWA past L 2048) |
| `fp8_train/` | FP8 training step-time benchmarks ([docs/fp8_training.md](../docs/fp8_training.md)): `opm_step_bench.py` (OuterProductMean), `pwa_step_bench.py` + `run_pwa_step_bench.sbatch` (MSA pair-weighted averaging), `tokendit/` (token DiT; `pin.py` pins the package to a tree given by `PIN_TREE`) |

The measurers that produce judge results are not here: they are `python -m foldjet_engine.measure.<kind>` ([docs/measure.md](../docs/measure.md)).
Older commits and records name the previous locations `tools/bench_opm_fp8_train.py`, `tools/fp8_train_bench/` and `pwa_long_bench.py` (repository root).
