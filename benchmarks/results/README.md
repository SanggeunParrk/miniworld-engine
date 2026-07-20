# Benchmark results (tracked)

Curated benchmark results live here and **are committed to git**. This is the
authoritative, in-repo record of measured numbers — regenerate the plots/logs, but
these files are the source of truth.

## Layout

Split into `kernels/` and `modules/` (mirrors `benchmarks/{kernels,modules}/`), one
subdir per GPU, `<mode>_<axis>` files (mode ∈ inference/training, axis ∈ seq_len/d_pair):

```
benchmarks/results/kernels/<kernel>/<gpu>/<mode>_<axis>.csv          # kernels: CSV ONLY
benchmarks/results/modules/<target>/<gpu>/<mode>_<axis>.csv          # modules: CSV ...
benchmarks/results/modules/<target>/<gpu>/<mode>_<axis>_speedup.svg  #   ... + plots (SVG)
benchmarks/results/modules/<target>/<gpu>/<mode>_<axis>_latency.svg
```

e.g. `benchmarks/results/kernels/transition_b2b/b200/inference_seq_len.csv` (csv only),
`benchmarks/results/modules/transition/b200/inference_seq_len.csv` + `..._speedup.svg`.
GPU slug is `b200` (sm100), `h100` (sm90), etc.

**CSV is tracked for both; plots (SVG) are tracked for modules only.** An `.svg` committed
under `kernels/` is git-ignored on purpose — kernel benches stay CSV-only. plot_csv.py emits
SVG (vector/text: delta-compresses in git, scales cleanly).

## Rules (keep the repo lean)

1. **Overwrite in place.** One canonical path per `(module, gpu)`. Do NOT write
   timestamped files (`h100_2026-07-20.csv`) — those accumulate forever. Git stores only
   the diff when you overwrite, so re-running is nearly free for CSV and cheap for PNG.
   If you need history-over-time, append one row per run to the CSV (git diffs one line).
2. **Commit only curated results**, not every exploratory run. Regenerate + commit when
   the numbers actually change.
3. **Never commit raw dumps here.** `*.out` sweep logs, `*.ncu-rep`/`*.nsys-rep` profiler
   captures, `*.so` binaries are git-ignored globally and must stay out — a single one can
   be tens–hundreds of MB. Only `.csv/.png/.pdf/.md/.json` under this dir are tracked.

## Why

Repo weight = filesize × versions × (binary ⇒ no delta compression). CSV is small and
diffable; a ~150 KB PNG overwritten in place stays bounded. Raw logs/profiles are large
*and* regenerated often — the exact combination that bloats history — so they stay out.
