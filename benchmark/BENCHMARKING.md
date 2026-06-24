# Benchmarking convention

## ⛔⛔ HARD RULE: ALL BENCHMARKS RUN COMPILED ⛔⛔

**Every benchmark MUST measure the `torch.compile`d path, never eager.** Eager PyTorch
is launch-bound and gives meaninglessly slow baselines — comparing a fused kernel to
eager is not a fair or valid result.

- The PyTorch-naive baseline is **always** `torch.compile(ref)` (reduce-overhead /
  default), warmed up before timing — NEVER the eager module.
- Time only steady-state (post-warmup) so compilation cost is excluded.
- TE / cute / triton are already compiled kernels; the rule is mainly about the
  PyTorch baseline — but the principle is absolute: **no eager numbers in any
  benchmark table or graph.** An eager measurement is a debug probe, not a result.

## HARD RULE: FOLLOW THE TEAM-GM BENCH HARNESS

Do **not** invent ad hoc benchmark methodology for this repo unless the user
explicitly asks for a one-off experiment.

- If an op is already covered by the repo's unified bench harness, use
  `scripts/bench.py` + `config/bench.yaml` + `tests/run_bench.sbatch`.
- That harness is the descendant of the `team-gm` benchmarking flow. Follow its
  shapes, dtype mode, compile/eager mode, and reporting format unless there is a
  concrete reason not to.
- Do **not** replace the harness with custom timing loops, notebook cells,
  random one-off `python - <<'PY'` snippets, or hand-written markdown tables for
  the "real" benchmark result.
- If a kernel is **not yet wired** into `scripts/bench.py`, a temporary
  kernel-local `bench.py` is acceptable, but it must still mimic the team-gm
  style:
  - emit the same parseable log structure (`=== M=.. d_in=.. d_out=.. ===`,
    backend timing lines, correctness lines),
  - compare against the canonical baselines for that op,
  - produce a checked-in `.out` log plus rendered `.md` + `.png` outputs.

The standard for "done" is: **team-gm-style harness, team-gm-style output,
results stored in the kernel's `benchmark/` folder.** Anything else is only a
debug probe, not a benchmark result.

**Every benchmark in this repo ships a table AND a graph together.** A table
alone hides trends; a graph alone hides exact numbers. Always produce both, and
save them next to each other so a result is never just a wall of text.

**All benchmark results live inside the kernel they belong to:**
`src/miniworld_kernels/kernels/<kernel>/benchmark/`. The raw log, the rendered
markdown report, and the PNGs all go in that one folder — NOT in this top-level
`benchmark/` directory (which is only for cross-kernel/unified runs and these
docs).

## Workflow (example: `layernorm_linear`)

Prefer this exact flow over ad hoc measurement.

Let `K=src/miniworld_kernels/kernels/layernorm_linear/benchmark`.

1. **Run** the bench on a GPU compute node, capturing stdout into `$K`. Uses the
   repo's unified pixi env (`.pixi/`); `--frozen` keeps the cu12 TE core fix in
   place (a bare `pixi run`/`install` re-pins cu13 — see pyproject `fix-te-cu12`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 \
     --mem=64G --cpus-per-task=8 --time=00:30:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m miniworld_kernels.kernels.<kernel>.bench"' \
     | tee "$K/<name>.out"
   ```
2. **Render** the table + graphs into the same `$K`. **Never run this on the
   login node** — route it through `srun` (CPU only, no `--gres`). matplotlib is
   in the unified env; `LD_LIBRARY_PATH` picks up its libstdc++ (`CXXABI_1.3.15`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --mem=16G --cpus-per-task=4 --time=00:10:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python scripts/plot_bench.py '"\"$K/<name>.out\" \"$K\""' --name <name>"'
   ```
   This writes `<name>.md` (markdown tables + embedded graphs) and one PNG per
   metric (`_fwd.png`, `_fwd_bwd.png`) into `$K`.

`scripts/plot_bench.py` parses the standard bench-output format
(`=== M=.. d_in=.. d_out=.. ===` blocks + `torch.compile`/`TE` timing lines), so
any bench that prints in that format gets table + graph for free.

## Conventions

- **Results go in `kernels/<kernel>/benchmark/`** — log + report + PNGs together.
- **Both forward and forward+backward** when the op is used in training.
- Bold the faster backend in each table cell; "lower is better" on graphs.
- Report numerical agreement (max abs error, relative Frobenius error, cosine)
  alongside latency — never just speed.

## Existing reports (`kernels/layernorm_linear/benchmark/`, H100 bf16, d_in=d_out ∈ {128..768} × M ∈ {16384,65536,262144})

- `layernorm_linear.md` — the canonical comparison: **pytorch (compiled) / TE / triton / cute**,
  both **inference** (`_fwd.png`) and **training fwd+bwd** (`_fwd_bwd.png`). cute = best variant
  (inference: tuned M2/M1 dispatch w/ cached fold; training: `layernorm_linear_fn`). triton =
  portable path (`layernorm_linear_triton` / `_triton_fn`). All compiled (see top rule).
