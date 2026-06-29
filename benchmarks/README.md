# Benchmarks

This directory is the boundary for benchmark definitions, runners, reports, and
generated artifacts.

Detailed methodology and plotting conventions are in `CONVENTIONS.md`.

## Policy

- `configs/` contains tracked benchmark inputs.
- `runners/` contains executable entry points.
- `suites/` contains op-specific benchmark definitions.
- `reports/` contains curated human-readable benchmark reports.
- `artifacts/` contains generated logs, CSVs, images, slides, and profiler
  outputs. It is gitignored by default.

Package code under `src/miniworld_kernels/` should not own ad hoc benchmark
entry points. Kernel-local scripts that still live under `src/` are migration
debt: port them into `suites/`, then remove the old entry point after the
output is reproduced.

GPU work must run through `srun` or an allocated compute node. Login-node work
is limited to lightweight repo-local inspection and editing.
