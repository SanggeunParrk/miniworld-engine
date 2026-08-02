# dev/

Development-only material. **Not part of the shipped package** — the wheel is built from
`src/` alone (`[tool.setuptools.packages.find] where = ["src"]`), so nothing here is
importable as `miniworld_engine` and none of it ships to users.

Contents:

- `research/` — exploratory kernel variants and one-off probes kept alongside the shipped
  kernels for reference (perf sweeps, readable rewrites, race-condition repros). These mirror
  the package layout under `research/miniworld_kernels/…` but are **not** imported by `src/`.
- `scratchpad_ncu/` — Nsight Compute / nsys harnesses and the CUDA-kernel-optimizer prompt
  logs (`prompt_b2b_v*.md`). The `docs/kernel-optimization/**` notes reference scripts here by
  path; some cited scripts were themselves scratch and have since been pruned — the notes are a
  historical record of how each kernel was tuned.
- `bench/` — legacy standalone bench scripts (e.g. `bench/bidir/`) superseded by the top-level
  `benchmarks/` runners; kept for reproducing older numbers.

If you are looking for the maintained benchmarks, they live in `benchmarks/` at the repo root.
