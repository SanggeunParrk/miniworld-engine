# LayerNorm backward — per-GPU dispatch cache

`miniworld` LayerNorm has three correct backward implementations and picks the
fastest one per shape. The pick was only *measured* on H100, so on any other GPU we
**measure once and cache** instead of guessing. This doc describes that mechanism.

Code: [`dispatch_cache.py`](dispatch_cache.py) (cache I/O) +
[`compile_native.py`](compile_native.py) (`_resolve_bwd_path`). Background and the
H100 numbers that motivated it: [`benchmark/CUTE_INVESTIGATION.md`](benchmark/CUTE_INVESTIGATION.md).

## The three backward paths

| path | what it does | wins when |
|---|---|---|
| `atomic` | `atomic_add` into `dw`/`db` | small `d` (cheap atomics) |
| `partial` | `[cdiv(M, block_m), d]` partials → reduce | `d == 256`, large `M` |
| `persistent` | persistent `NUM_SM*waves` grid, vectorized 2D tiles, `[grid, d]` partials | `d >= 384` (grows with `d`) |

All three are **plain Triton** — they recompile per arch and Triton autotunes their
inner config (BLOCK_M / warps / stages). The persistent kernel reads the live SM
count and uses **no Hopper-only features** (no clusters / DSMEM / TMA). So every
path is correct and self-tuning on Ampere / Ada / Hopper / future archs; only the
*choice between them* is hardware-dependent.

## Resolution order

`_resolve_bwd_path(m, n, …)` decides per backward call:

1. **Env override** `MINIWORLD_LN_BWD=persistent|partial|atomic` → use it (debug /
   manual pin), bypassing everything below.
2. **`MINIWORLD_LN_AUTOTUNE=off`** → static H100 heuristic, no measuring.
3. **H100 (sm_90)** and mode ≠ `force` → static heuristic directly (already
   measured; calibration on H100 reproduces it exactly).
4. **Cache hit** for `(d, M-bucket)` on this GPU → use the cached path.
5. **CUDA-graph capturing** → static heuristic (never time inside a capture).
6. **Otherwise → calibrate:** time the three paths on the *real* tensors with
   `triton.testing.do_bench`, pick the winner, persist it, use it.

The static heuristic (also the universal fallback) is:
`d >= 384 → persistent`; `d == 256 → partial` (M ≥ 262144) else `atomic`;
`d < 256 → atomic`.

## The cache

- **Location:** `~/.cache/miniworld_kernels/ln_bwd_dispatch/<gpu>.json`
  (override with `MINIWORLD_KERNELS_CACHE_DIR`; honours `XDG_CACHE_HOME`). Lives
  **outside the repo**, so using this repo as a submodule needs no `.gitignore`
  entry and multiple GPUs coexist.
- **Key:** GPU name + compute capability + Triton version
  (e.g. `NVIDIA_H100_80GB_HBM3_sm90_triton3.6.0.json`) — a driver/Triton upgrade or
  a different GPU gets its own entry.
- **Entry:** keyed by `"<d>|<M-bucket>"` (M-bucket = largest power of two ≤ M, so we
  calibrate once per scale, not per exact M). Value stores the chosen `path` and the
  measured `ms` for all three, e.g.:

  ```json
  {
    "768|131072": { "path": "persistent",
                    "ms": { "atomic": 0.514, "partial": 0.352, "persistent": 0.267 } }
  }
  ```

- **Learns the shapes you actually use** — calibration is lazy and per-shape, so
  there is no upfront sweep; the first occurrence of each new `(d, M-bucket)` pays a
  one-time timing cost (a few `do_bench` reps), then it is instant forever.

## Safety

The cache only ever selects among **correct** kernels, so it can never change
numerics. A stale or corrupt cache at worst picks a slower (still correct) path; any
read/parse/timing error falls back to the static heuristic. Calibration is skipped
during CUDA-graph capture.

## Controls (env)

| var | values | effect |
|---|---|---|
| `MINIWORLD_LN_AUTOTUNE` | `auto` (default) / `off` / `force` | `off`: static only; `force`: calibrate even on H100 |
| `MINIWORLD_KERNELS_CACHE_DIR` | path | cache root (default `$XDG_CACHE_HOME` or `~/.cache`) |
| `MINIWORLD_LN_BWD` | `persistent` / `partial` / `atomic` | hard override, bypasses cache + heuristic |

## Using it on a new GPU

Nothing to do — just run. The first training step (or first time each
`(d, M-bucket)` is hit) calibrates and writes the cache; subsequent runs and
re-imports read it. To pre-warm explicitly, run one forward+backward per shape you
care about. To inspect/clear, look at / delete the JSON under the cache dir.
