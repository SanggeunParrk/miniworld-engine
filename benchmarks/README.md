# Benchmarks

This directory contains the benchmark execution system only. It is intentionally
small: definitions, entry points, inputs, and generated outputs.

Detailed methodology and plotting conventions are in `../docs/benchmarks.md`.

## Directory Map

- `configs/`: tracked benchmark inputs, currently the Hydra config consumed by
  `runners/bench.py`.
- `runners/`: executable benchmark and rendering entry points. These are the
  only supported CLI targets.
- `suites/`: op-specific standalone benchmark definitions for kernels/modules
  not yet covered cleanly by the unified runner.
- `artifacts/`: generated logs, CSVs, images, HTML, slides, profiler output,
  and temporary reports. This directory is gitignored except `.gitkeep` files.

Do not add `reports/`, `archive/`, or kernel-local benchmark folders. If a
result is worth preserving, keep the source data in `artifacts/` and summarize
the conclusion in `../docs/`.

Runtime dispatch caches are not benchmark artifacts. They live outside the repo
by default; see `../docs/operations/dispatch-cache.md`.

GPU work must run through `srun` or an allocated compute node. Login-node work
is limited to lightweight repo-local inspection and editing.
