# Benchmark results (tracked)

Curated benchmark results live here and **are committed to git**. This is the
authoritative, in-repo record of measured numbers — regenerate the plots/logs, but
these files are the source of truth.

## Layout

Split into `kernels/` and `modules/` (mirrors `benchmarks/{kernels,modules}/`):

```
benchmarks/results/kernels/<kernel>/<gpu>.csv    # kernel benches: CSV ONLY (no plots)
benchmarks/results/modules/<module>/<gpu>.csv    # module benches: CSV ...
benchmarks/results/modules/<module>/<gpu>.png    #   ... + curated plot
benchmarks/results/**/results.md                 # human-readable table (from the CSVs)
```

e.g. `benchmarks/results/kernels/trimul_inproj/h100.csv` (csv only),
`benchmarks/results/modules/transition/h100.csv` + `.../h100.png`.

**CSV is tracked for both; PNG plots are tracked for modules only.** A `.png` committed
under `kernels/` is git-ignored on purpose — kernel benches stay CSV-only.

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
