# Benchmarking

## Directory Boundary

`benchmarks/` has three meanings:

- `kernels/<kernel>/`: isolated kernel benchmarks, with target-local
  `configs/` and `artifacts/`.
- `modules/<module>/`: composed module benchmarks, with target-local
  `configs/` and `artifacts/`.
- `runners/`: supported executable entry points.

Do not add benchmark code under `src/`. Do not add archive folders. Do not add
curated markdown reports under `benchmarks/`; write durable explanations under
`docs/` and keep generated tables/plots under the target's `artifacts/`.

## Hard Rule: All Benchmarks Run Compiled

**Every benchmark MUST measure the `torch.compile`d path, never eager.** Eager PyTorch
is launch-bound and gives meaninglessly slow baselines — comparing a fused kernel to
eager is not a fair or valid result.

- The PyTorch-naive baseline is **always** `torch.compile(ref)` (reduce-overhead /
  default), warmed up before timing — NEVER the eager module.
- Time only steady-state (post-warmup) so compilation cost is excluded.
- TE / cute / triton are already compiled kernels; the rule is mainly about the
  PyTorch baseline — but the principle is absolute: **no eager numbers in any
  benchmark table or graph.** An eager measurement is a debug probe, not a result.

## Hard Rule: Follow The Team-GM Bench Harness

Do **not** invent ad hoc benchmark methodology for this repo unless the user
explicitly asks for a one-off experiment.

- If an op is already covered by the repo's unified bench harness, use
  `benchmarks/runners/bench.py` +
  `benchmarks/modules/<module>/configs/bench.yaml` + `tests/run_bench.sbatch`.
- That harness is the descendant of the `team-gm` benchmarking flow. Follow its
  shapes, dtype mode, compile/eager mode, and reporting format unless there is a
  concrete reason not to.
- Do **not** replace the harness with custom timing loops, notebook cells,
  random one-off `python - <<'PY'` snippets, or hand-written markdown tables for
  the "real" benchmark result.
- If a kernel is **not yet wired** into the unified runner, a temporary
  kernel-local `bench.py` is acceptable only as migration debt, and it must
  still mimic the team-gm style:
  - emit the same parseable log structure (`=== M=.. d_in=.. d_out=.. ===`,
    backend timing lines, correctness lines),
  - compare against the canonical baselines for that op,
  - produce the standard output schema consumed by the shared renderer.

The standard for "done" is: **team-gm-style harness, team-gm-style output,
benchmark registered under `benchmarks/kernels/<kernel>/` or
`benchmarks/modules/<module>/`, and generated artifacts under that target's
`artifacts/`.** Anything else is only a debug probe, not a benchmark result.

**Every benchmark in this repo ships a table AND a graph together.** A table
alone hides trends; a graph alone hides exact numbers. Always produce both, and
save them next to each other so a result is never just a wall of text.

**Generated benchmark results do not live in package code.** Raw logs, CSVs,
PNGs, SVGs, PDFs, slide exports, profiler outputs, and rendered markdown tables
go under `benchmarks/kernels/<kernel>/artifacts/` or
`benchmarks/modules/<module>/artifacts/` by default. Durable interpretation
belongs in `docs/`, not in a benchmark archive tree.

## Workflow (example: `layernorm_linear`)

Prefer this exact flow over ad hoc measurement.

Let `A=benchmarks/kernels/layernorm_linear/artifacts`.

1. **Run** the bench on a GPU compute node, capturing stdout into `$A`. Uses the
   repo's unified pixi env (`.pixi/`); `--frozen` keeps the cu12 TE core fix in
   place (a bare `pixi run`/`install` re-pins cu13 — see pyproject `fix-te-cu12`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 \
     --mem=64G --cpus-per-task=8 --time=00:30:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python benchmarks/runners/bench.py kernel=<kernel>"' \
     | tee "$A/<name>.out"
   ```
2. **Render** the table + graphs into the same artifact directory. **Never run this on the
   login node** — route it through `srun` (CPU only, no `--gres`). matplotlib is
   in the unified env; `LD_LIBRARY_PATH` picks up its libstdc++ (`CXXABI_1.3.15`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --mem=16G --cpus-per-task=4 --time=00:10:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python benchmarks/runners/plot_bench.py '"\"$A/<name>.out\" \"$A\""' --name <name>"'
   ```
   This writes `<name>.md` (markdown tables + embedded graphs) and one PNG per
   metric (`_fwd.png`, `_fwd_bwd.png`) into `$A`. Keep that generated output in
   artifacts; summarize durable conclusions in `docs/`.

`benchmarks/runners/plot_bench.py` parses the standard bench-output format
(`=== M=.. d_in=.. d_out=.. ===` blocks + `torch.compile`/`TE` timing lines), so
any bench that prints in that format gets table + graph for free.

## Visual style (single source of truth)

All figures — both plotting paths — share one palette/theme so the benchmark
figures read as **one coherent set** (this matters for the paper: a reviewer
sees the same backend in the same colour in every plot). Defined once in
`src/miniworld_kernels/viz/style.py` and imported by both:

- `benchmarks/runners/plot_bench.py` (grouped bars from `.out`) and
- `benchmarks/runners/bench.py` (Triton `perf_report` line plots).

Rules baked into the module:

- **Colour by meaning, not by position.** `color_for(name)` maps *any* spelling
  of a backend (`cuequiv`/`cuequivariance`, `cute`/`cute-fused`/`ours v4`, …) to
  one canonical identity → one fixed colour. Unknown names get a deterministic
  hash colour (stable across figures, never index-dependent). Never hand-assign
  colours in a kernel-local bench — call `color_for` / `style_for`.
  - **ours / cute family → hot (red/orange)** — the winner pops.
  - **NVIDIA family (cuequivariance / dtv1 / TE) → greens & teal.**
  - **baselines (pytorch / torch.compile / triton) → grey & blue** (recede).
- **`apply_theme()`** installs the publication rcParams (fonts, clean spines,
  y-grid). Call it once before plotting.
- **Vector output for the paper.** `save_figure(fig, path)` writes `.png` (for
  markdown/slides) **plus `.svg` and `.pdf`** (LaTeX `\includegraphics`,
  infinite zoom). All of those are generated artifacts.

## Conventions

- **Generated results go in the target's `artifacts/`**; durable explanations
  go in `docs/`.
- **Use the shared style** (`miniworld_kernels.viz`) for every figure — never
  ad-hoc colours.
- **Both forward and forward+backward** when the op is used in training.
- Bold the faster backend in each table cell; "lower is better" on graphs.
- Report numerical agreement (max abs error, relative Frobenius error, cosine)
  alongside latency — never just speed.

## Runtime Dispatch Caches

Benchmark output and runtime dispatch caches are separate. Generated benchmark
files stay under target-local `artifacts/`; runtime dispatch caches stay outside
the repo by default. See `operations/dispatch-cache.md`.
