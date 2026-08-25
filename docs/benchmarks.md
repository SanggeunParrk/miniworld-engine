# Benchmarking

## Directory Boundary

`benchmarks/` has three meanings:

- `kernels/<kernel>/`: isolated kernel benchmarks.
- `modules/<module>/`: composed module benchmarks.
- `runners/`: shared, kernel-agnostic executable entry points (`bench.py`,
  `plot_csv.py`, and any reusable harness usable across targets).

### Per-target subdirectories (strict)

Every `kernels/<k>/` and `modules/<m>/` target uses exactly these subdirs, each
with one job. Nothing else belongs at the target root.

- `configs/` — Hydra bench config (`bench.yaml`), whose `target:`/`level:` name this
  very directory: `level` is `kernel` or `module` (the parent dir) and `target` is the
  folder name. One target owns exactly one directory; no folder is shared. This file is
  the run's **complete** config, selected by the `target=`/`level=` you pass on the
  command line — `bench.py` reads that pair off argv and points hydra at this directory,
  then applies the rest of the command line on top. There is no shared base file and no
  `defaults:` list, so a key missing here does not exist for this target's runs.
  (`config_path` used to be the constant `../modules/triangle_multiplication/configs`, so
  every target loaded that one file and the other 25 configs were read by nothing —
  `augmented_attention_atom` declared a 128–384 ladder and was swept at 384–1024.
  `tests/layout/test_bench_config_per_target.py` is what keeps it honest.)
- `artifacts/` — **generated benchmark outputs only**: `*.csv`, `*.svg`,
  `*_autotune_summary.txt`. No Python, no profiler captures, no repro trees.
  (`.gitignore` already drops everything under `artifacts/` except the data
  file types, so stray code dumped here is silently untracked — do not rely on
  that; just don't put it here.)
- `profiles/` — **profiler capture outputs**: `*.nsys-rep`, `*.ncu-rep`,
  `*.sqlite`. Sibling of `artifacts/`, never nested inside it. Captures are
  git-ignored (large binaries); the dir is kept via `.gitkeep`.

A target holds **no Python at all** — only `configs/`, `artifacts/`, `profiles/`.

Rules of thumb:
- **`benchmarks/` contains no target-specific Python.** The ONLY Python is the shared,
  kernel/module-agnostic harness in `benchmarks/runners/` (`bench.py`, `plot_csv.py`). No
  per-kernel/per-module bench scripts, no nsys capture-replay scripts, no `diagnostics/`
  drivers anywhere under `benchmarks/`.
- **Profiling (nsys/ncu) is instrumented in `src/`, not as a benchmarks script.** If a kernel
  needs profiling or logging, add the nsys/ncu hooks to that kernel's own
  `src/miniworld_engine/...` code and drive it through the normal harness — do not drop a
  one-off capture script under `benchmarks/`. Capture outputs still land in the target's
  `profiles/`.
- Profiler capture files go in the target's `profiles/`, never `artifacts/`.

Do not add archive folders or repro source trees under a target. Do not add curated markdown
reports under `benchmarks/`; write durable explanations under `docs/` and keep generated
tables/plots under the target's `artifacts/`.

## Hard Rule: All Benchmarks Run Compiled

**Every benchmark MUST measure the `torch.compile`d path, never non-compiled PyTorch.**
Non-compiled PyTorch is launch-bound and gives meaninglessly slow baselines; comparing a
fused kernel to that path is not a fair or valid result.

- The PyTorch-naive baseline is **always** `torch.compile(ref)` (reduce-overhead /
  default), warmed up before timing; never the non-compiled module.
- Time only steady-state (post-warmup) so compilation cost is excluded.
- TE / cute / triton are already compiled kernels; the rule is mainly about the
  PyTorch baseline, but the principle is absolute: **no non-compiled numbers in any
  benchmark table or graph.** A non-compiled measurement is a debug probe, not a result.

## Hard Rule: Follow The Team-GM Bench Harness

Do **not** invent ad hoc benchmark methodology for this repo unless the user
explicitly asks for a one-off experiment.

- If an op is already covered by the repo's unified bench harness, use
  `benchmarks/runners/bench.py` +
  `benchmarks/modules/<module>/configs/bench.yaml`, launched however your
  cluster launches things. (`submits/` was removed in 511d905 when its work
  moved into the package; anything still naming `submits/run_*.sbatch` is a
  stale reference.)
- That harness is the descendant of the `team-gm` benchmarking flow. Follow its
  shapes, dtype mode, compilation policy, and reporting format unless there is a
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

**Every benchmark in this repo writes a complete CSV first.** The CSV is the
source of truth: it must include the method, dimensions, dtype/precision, mode,
metric, device, compile flag, and measured value. Plots are a separate step that
read the CSV; benchmark code must not draw figures.
Shape sweeps are explicit: use `sweep_axis=seq_len` for L sweeps and
`sweep_axis=d_pair` for channel-width sweeps. The CSV must record both the
swept axis and the fixed dimensions. If a backend does not support a shape,
write a `status=failed` row with the error and leave `value` empty so plotting
can skip that point without hiding the unsupported case.
For d sweeps, prefer explicit `d_pair_values` when the paper/report only wants
canonical widths; trimul uses `128, 256, 512` rather than an arithmetic range
that accidentally includes unsupported or irrelevant intermediate widths.

**Generated benchmark results do not live in package code.** Raw logs, CSVs,
SVGs, slide exports, and profiler outputs go under
`benchmarks/kernels/<kernel>/artifacts/` or
`benchmarks/modules/<module>/artifacts/` by default. Durable interpretation
belongs in `docs/`, not in a benchmark archive tree.

## Workflow (example: `gemm_epilogue`)

Prefer this exact flow over ad hoc measurement.

Let `A=benchmarks/kernels/gemm_epilogue/artifacts`.

1. **Run** the bench on a GPU compute node. Uses the
   repo's unified pixi env (`.pixi/`); `--frozen` keeps the cu12 TE core fix in
   place (a bare `pixi run`/`install` re-pins cu13 — see pyproject `fix-te-cu12`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 \
     --mem=64G --cpus-per-task=8 --time=00:30:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python benchmarks/runners/bench.py target=gemm_epilogue level=kernel"'
   ```
   This writes a long-form CSV under the target-local artifact directory.
2. **Render** plots from the CSV into the same artifact directory. **Never run this on the
   login node** — route it through `srun` (CPU only, no `--gres`). matplotlib is
   in the unified env; `LD_LIBRARY_PATH` picks up its libstdc++ (`CXXABI_1.3.15`):
   ```bash
   srun --account=cssb --qos=cssb_h100 --partition=h100 --mem=16G --cpus-per-task=4 --time=00:10:00 \
     bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
       python benchmarks/runners/plot_csv.py '"\"$A/<name>.csv\" \"$A\""' --name <name>"'
   ```
   This writes grouped bar plots (`*_latency.svg` and `*_speedup.svg`). If
   `--name` is omitted, the renderer uses short mode-aware names such as
   `trimul_inference_manual_L_sweep_latency.svg` and
   `trimul_training_manual_d_sweep_speedup.svg` (the `manual` segment reflects
   the `cudagraph=manual` deployment config these kernels are benched under).
   Keep those generated SVGs in
   artifacts; derive PNG/PDF from SVG only when a downstream tool explicitly
   needs that format. Summarize durable conclusions in `docs/`.
   The plot caption must include the fixed sweep dimensions, e.g. `d_pair=128`
   for an L sweep or `L=384` for a d sweep.

## Visual style (single source of truth)

All figures share one palette/theme so the benchmark
figures read as **one coherent set** (this matters for the paper: a reviewer
sees the same backend in the same colour in every plot). Defined once in
`src/miniworld_engine/viz/style.py` and imported by:

- `benchmarks/runners/plot_csv.py` (grouped bar plots from benchmark CSVs).

Rules baked into the module:

- **Colour by meaning, not by position.** `color_for(name)` maps *any* spelling
  of a backend (`cuequiv`/`cuequivariance`, `cute`/`cute-fused`/`ours v4`, …) to
  one canonical identity → one fixed colour. Unknown names get a deterministic
  hash colour (stable across figures, never index-dependent). Never hand-assign
  colours in a kernel-local bench — call `color_for` / `style_for`.
  - **MiniWorld / cute family → gold** — the repo's kernels are
    visually fixed across every figure.
  - **NVIDIA family (cuequivariance / dtv1 / TE) → greens & teal.**
  - **baselines (pytorch / torch.compile / triton) → grey & blue** (recede).
- **`apply_theme()`** installs the publication rcParams (fonts, clean spines,
  y-grid). Call it once before plotting.
- **SVG-only output.** `save_figure(fig, path)` writes `.svg` only. SVG is the
  canonical vector artifact; convert it to PNG/PDF outside the benchmark runner
  if a paper, slide deck, or website requires that derivative format.

## Conventions

- **Generated results go in the target's `artifacts/`**; durable explanations
  go in `docs/`.
- **Use the shared style** (`miniworld_engine.viz`) for every figure — never
  ad-hoc colours.
- **Both inference and training** when the op is used in training.
- Latency plots use "lower is better"; speedup plots use "higher is better".
- Report numerical agreement (max abs error, relative Frobenius error, cosine)
  alongside latency — never just speed.

## Benchmark Acceptance Checklist

Before treating a benchmark artifact as final, check every item below.

- [ ] **Compile path:** `compiled=True` is present in the CSV, and the run used
  the repo benchmark entry point with `compile=true`; PyTorch baseline is never
  non-compiled. For Transition manual CUDA graph artifacts, `compiled=False`
  means the measured module was captured eagerly because CUDA graph capture is
  the timing regime; the `cudagraph` column must record `manual`.
- [ ] **Dtype and autotune:** CSV rows record `input_dtype` and
  `parameter_dtype`; each CSV has a matching `<run_name>_autotune_summary.txt`
  for Triton-autotuned kernels, and that file shows both the candidate config
  set and the selected cache entries for the measured shapes. `autotune_summary.txt`
  is only the latest-run compatibility copy, not the full audit record.
- [ ] **Kernel tiling:** repo-developed kernels have an explicit tiling strategy
  appropriate for the measured shape family. The benchmark notes or autotune
  summary must make the relevant tile dimensions visible, e.g. `BLOCK_M`,
  `BLOCK_N`, `BLOCK_K`, warp/stage counts, or the equivalent CuTe/quack tile
  shape.
- [ ] **Reference agreement:** CSV rows include `reference`, `output_max_abs`,
  `output_rel_frob`, and `output_cosine`; training rows also include
  `grad_max_abs`, `grad_rel_frob`, and `grad_cosine` when the runner has a
  reference path wired.
- [ ] **Inference/training separation:** implementation rows identify the
  `execution_path`, and MiniWorld-style kernels must use distinct inference and
  training paths when the kernel design has separate save/no-save behavior.
- [ ] **Both modes:** final artifacts include both inference and training CSVs
  and SVGs for training-relevant ops.
- [ ] **Both sweeps:** final artifacts include both L sweep (`sweep_axis=seq_len`)
  and d sweep (`sweep_axis=d_pair`). For trimul, the d sweep uses
  `d_pair_values=[128,256,512]` at fixed `L=384`.
- [ ] **All applicable methods:** rows include every applicable implementation,
  including PyTorch, and the repo-developed kernel is routed through its intended
  production path rather than a debug/prototype path.
- [ ] **CUDA graph option:** the benchmark design explicitly decides whether
  CUDA graph capture is part of the fair regime for each implementation. If it
  is used, the CSV/report must identify that regime; if it is not used, the
  docs or benchmark notes must explain why `torch.compile`/steady-state timing
  is the intended comparison.
- [ ] **Approved runner and plotter:** CSVs come from
  `benchmarks/runners/bench.py`; figures come from
  `benchmarks/runners/plot_csv.py`; benchmark code writes CSV only and plotting
  remains a separate step. Canonical CSV files should be replaced atomically
  after a run finishes, not truncated in place at job start.

## Module Benchmark Matrix

`triangle_multiplication_bidirectional` already has its own final artifact set;
the matrix below is the remaining repo-developed module kernels, one
`bench.py target=<target> level=module` run each:

| target | implementations | sweeps | modes | notes |
| --- | --- | --- | --- | --- |
| `attention_pair_bias` | `pytorch`, `cuequivariance`, `old_triton`, `miniworld` | `seq_len`, `d_pair` | inference, training | This benches the production `AttentionPairBias` module -- the bias-only TriangleAttention case (`use_self_attention=False`). (The `bias_only_attention` NAME belongs to the kernel-level target, which benches the `bias_only_attention` kernel family.) `old_triton` is the Team-GM vendored bias-only Triton attention kernel. MiniWorld covers the developed LN/projection/gate dispatch path. See `docs/kernels/bias-only-attention.md`. |
| `triangle_attention` | `pytorch`, `cuequivariance`, `miniworld` | `seq_len`, `d_pair` | inference, training | Full triangular self-attention (`use_self_attention=True`). MiniWorld maps to the canonical Triton pair-bias attention kernel plus the module LayerNorm/projection/gate path; `old_triton` is not a separate public full-attention implementation. See `docs/kernels/triangle-attention.md`. |
| `transition` | `pytorch`, `old_triton`, `miniworld` | `seq_len`, `d_pair` | inference, training | `old_triton` is the Team-GM Triton transition path (`LayerNorm(TRITON)` + `kernels.transition.triton.main`). MiniWorld means the production d-aware fused route: Triton for small `d_pair`, CuTe for large `d_pair`. Component-only `cute` runs are diagnostics, not final plots. |
| `conditioned_transition` | `pytorch`, `miniworld` | `seq_len`, `d_pair` | inference, training | benchmarks the post-AdaLN tail only; inference dispatch is fused at `d_pair<=128` and composed above that, training uses the custom autograd path. |
| `adaptive_layernorm` | `pytorch`, `miniworld` | `seq_len`, `d_pair` | inference, training | benchmark treats `d_pair` as the AdaLN hidden/condition width so d sweeps change the real tensor shape. |
| `augmented_attention_token` | `pytorch`, `miniworld` | `seq_len`, `d_pair` | inference, training | token path; `d_pair` sweeps pair-bias width. |
| `augmented_attention_atom` | `pytorch`, `miniworld` | `seq_len`, `d_pair` | inference, training | atom path; L sweep still includes `L=384`; unsupported/OOM points stay as failed CSV rows. |

Final plots must show a single `MiniWorld` series. If a diagnostic CSV includes
component aliases such as `cute` plus `miniworld`, the shared plotter collapses
them to the canonical `MiniWorld` backend before drawing.

## Runtime Dispatch Caches

Benchmark output and runtime dispatch caches are separate. Generated benchmark
files stay under target-local `artifacts/`; runtime dispatch caches stay outside
the repo by default. See `operations/dispatch-cache.md`.
