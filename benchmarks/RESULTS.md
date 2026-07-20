# Benchmark results (tracked)

Curated benchmark results are **committed to git**, co-located with each target under its
own `results/` folder (sibling of the regeneratable `artifacts/` and `plots/` scratch).

## Layout

```
benchmarks/modules/<target>/
  artifacts/                 # raw bench output (CSVs, autotune summaries) — SCRATCH, git-ignored
  plots/                     # raw plot_csv.py SVG output                   — SCRATCH, git-ignored
  results/                   # curated, TRACKED
    <gpu>/<mode>_<axis>.csv          # numbers = source of truth
    plots/<gpu>/<mode>_<axis>_<latency|speedup>.svg

benchmarks/kernels/<target>/
  artifacts/ , plots/        # scratch, git-ignored
  results/
    <gpu>/<mode>_<axis>.csv          # kernels: CSV ONLY (no plots)
```

- `<gpu>`: `h100` (sm90), `b200` (sm100), …
- `<mode>`: `inference` | `training`; `<axis>`: `seq_len` | `d_pair`
- `augmented_attention` holds two targets in one folder, so its files are prefixed
  `atom_` / `token_` (e.g. `results/h100/atom_inference_seq_len.csv`).

## Rules

1. **CSV** (the numbers) is tracked for **both** module and kernel benches.
2. **Plots (SVG)** are tracked for **modules only** — an `.svg` under `kernels/*/results/`
   is git-ignored on purpose. plot_csv.py emits SVG (vector/text → delta-compresses, scales).
3. **Overwrite in place** per `(gpu, mode, axis)` so git stores diffs, not new blobs.
4. Only `.csv`/`.svg` land here. Raw dumps (`*.out`, `*.ncu-rep`, `*.so`) are ignored
   globally and stay out; `artifacts/` and `plots/` are scratch and never tracked
   (only their `.gitkeep` scaffolding is).

## Why

Repo weight = filesize × versions × (binary ⇒ no delta compression). CSV/SVG are small and
diffable; raw logs/profiles are large *and* regenerated often, so they stay in scratch dirs.
