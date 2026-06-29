# Runtime Dispatch Caches

Some kernels choose among multiple correct implementations at runtime. Those
choices are performance policy, not benchmark output, so their caches must not
live under any `benchmarks/**/artifacts/` directory and must not be committed.

The persistent on-disk cache root is:

- `$MINIWORLD_KERNELS_CACHE_DIR`, if set
- otherwise `$XDG_CACHE_HOME/miniworld_kernels`
- otherwise `~/.cache/miniworld_kernels`

This keeps the repo clean when it is used directly or as a submodule.

## LayerNorm Backward

Code:

- `src/miniworld_kernels/kernels/layernorm/dispatch_cache.py`
- `src/miniworld_kernels/kernels/layernorm/compile_native.py`

`miniworld` LayerNorm has three correct backward implementations and picks the
fastest one per shape. The H100 heuristic is the static default; unknown GPUs
measure once and cache the winner instead of guessing.

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

## LayerNorm Cache Format

- **Location:** `<cache-root>/ln_bwd_dispatch/<gpu>.json`.
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

## Bias-Only Attention Dispatch

Code:

- `src/miniworld_kernels/kernels/bias_only_attention/dispatch.py`

This cache uses the same root policy and GPU key helper as LayerNorm. It chooses
among correct attention backend variants and stores the per-GPU crossover. A bad
or missing cache only changes speed; it falls back to the static H100 heuristic.

## In-Process Dispatch

`src/miniworld_kernels/kernels/trimul_inproj/cute/dispatch.py` keeps a process
local dispatch table for cuBLAS vs CuTe/quack candidates. It is intentionally
not persisted: the choices are tied to live thunks, current process imports, and
shape-specific warmup state.

## Safety

Dispatch caches only select among **correct** kernels, so they must never change
numerics. A stale or corrupt cache at worst picks a slower path; read, parse, or
timing errors fall back to static heuristics. Calibration is skipped during
CUDA-graph capture.

## Controls (env)

| var | values | effect |
|---|---|---|
| `MINIWORLD_LN_AUTOTUNE` | `auto` (default) / `off` / `force` | `off`: static only; `force`: calibrate even on H100 |
| `MINIWORLD_KERNELS_CACHE_DIR` | path | persistent cache root |
| `MINIWORLD_LN_BWD` | `persistent` / `partial` / `atomic` | hard override, bypasses cache + heuristic |

## Using it on a new GPU

Nothing to do — just run. The first training step (or first time each
`(d, M-bucket)` is hit) calibrates and writes the cache; subsequent runs and
re-imports read it. To pre-warm explicitly, run one inference+training pass per shape you
care about. To inspect/clear, look at / delete the JSON under the cache dir.
