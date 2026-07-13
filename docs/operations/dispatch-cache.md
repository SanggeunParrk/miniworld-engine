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


# Autotune Config Cache

Separate from the LayerNorm *path* cache above: this ships the top-K tuned **autotune configs**
per `(gpu, dtype, op, shape-bucket)` so runs skip the full-grid autotune tax and performance is
reproducible across machines. It is **backend-agnostic** — one cache format + storage layer
serves Triton, CuTe, and CUDA kernels. Code: `src/miniworld_kernels/autotune/` (`cache.py`,
`build.py`).

**INVARIANT:** config choice is performance-only — every candidate config computes the same
math — so a missing / stale / wrong cache is only ever slower, never incorrect.

## How it works

Two runtime entry points share the cache; a kernel uses whichever fits its backend:

- **Triton** — the kernel's `@triton.autotune` uses `make_cache_prune(op, ...)` as its
  `early_config_prune` (composed OVER any device-smem safety prune). Per call it reads the
  running `(gpu, dtype, shape-bucket)`, and on a **hit** returns only the cached top-K (autotune
  picks among ~5, not the full grid); on a **miss/stale** returns the full grid.
- **CuTe / CUDA** — these fix their tile/cluster/stage config at build time (no autotune loop),
  so they call `select_config(op, dtype=…, bucket=…, candidates=…)` to *pick one* cached config
  and apply its `kwargs` (`tile_m`/`tile_n`/`cluster`/`pingpong`/…); on a miss they fall back to
  the kernel's own `default_config`.

Both paths, on a **miss** (unknown GPU, unseen shape) or **stale** (the kernel's config grid
changed, detected by a `config_space_hash` mismatch) → warn ONCE and fall back to the full grid
/ default (still correct). A config is any of: a `triton.Config`, a plain dict of tile params
(CuTe/CUDA, cluster shapes as tuples), or a pre-shaped `{kwargs, num_warps, num_stages}` dict —
`as_cfg_dict` normalizes all three and `config_to_dict` serializes them JSON-safely.

## Cache location & format

- **shipped** (committed defaults): `src/miniworld_kernels/autotune/data/<op>/<gpu_key>.json`.
- **runtime** (builder / RUN_AUTOTUNE output, preferred over shipped):
  `<cache-root>/autotune/<op>/<gpu_key>.json` (same `<cache-root>` as above).
- `gpu_key` = device name + capability (e.g. `NVIDIA A100 80GB PCIe (sm80)`); entries keyed
  `"<dtype>|<shape-bucket>"` → list of `{kwargs, num_warps, num_stages, ms}` (top-K). A
  `config_space_hash` field invalidates the whole file's entries when the grid changes.

## Building the shipped cache (on the target GPU)

There are two builders; both write the runtime cache, which you then copy into
`src/miniworld_kernels/autotune/data/` and commit. A new GPU (H100/B200/…) is enabled by
running one of these on that box and committing its JSONs.

**1. Capture builder (preferred — covers every wired kernel automatically).** Instead of
hand-replicating each kernel's launch, `autotune/capture.py` instruments the Triton autotuner
(`Autotuner._bench`) and records every `(config -> measured ms)` as it is benched during a real
module forward/backward, keyed by the SAME `(op, dtype, bucket)` the runtime prune uses. So one
module run populates the caches of every autotune kernel it fires. Drive it through the existing
bench harness:

    MINIWORLD_RUN_AUTOTUNE=1 MINIWORLD_AUTOTUNE_CAPTURE=1 \
      python benchmarks/runners/bench.py kernel=<module> implementations='[miniworld]' \
      compile=false cudagraph=manual mode=training sweep_axis=seq_len ...

`RUN_AUTOTUNE=1` unlocks the full grid (no cached narrowing) so every config is benched;
`AUTOTUNE_CAPTURE=1` installs the capture and flushes top-5 per `(op,dtype,bucket)` at the end.
`submits/run_autotune_capture_a100.sbatch` runs this across all module targets + modes + sweeps.
Validated against the hand builder: capture reproduces its top-1 selections (near-ties aside).

**2. Explicit builder (per-kernel, for the pilot kernels).**

    PYTHONPATH=src python -m miniworld_kernels.autotune.build --op all     # or --op <name>

Its core `tune_bucket(op, gk, dtype, bucket, candidates, run_ms, csh)` is backend-agnostic: it
benches each candidate via a `run_ms(cfg) -> ms` closure and stores the top-K. Triton builders
point `run_ms` at `do_bench(kernel.fn[grid])`.

Coverage: all live Triton kernels are wired (32 op-tags across triangle_attention,
augmented_attention, adaln, conditioned_transition, bias_only, layernorm_linear, tm1/tm2,
transition) and populated on A100 by the capture builder.

## CuTe / CUDA autotune (sm90+)

CuTe/CUDA kernels have no Triton autotune loop — they fix `tile_m/tile_n/cluster/pingpong` at
build time (e.g. `dualgemm_kernel.py`'s `_CFG` or a `default_config(dev)`). They join the SAME
cache via the two backend-agnostic hooks; this must be built + verified on an sm90+ box (it
cannot run on Ampere):

- **runtime pick-one** — replace the hardcoded config with a cache lookup that falls back to the
  kernel's own default on a miss (so it is a no-op until a cache is shipped):

  ```python
  from miniworld_kernels.autotune import select_config
  CANDS = [dict(tile_m=128, tile_n=256, cluster_m=1, cluster_n=1, pingpong=False), ...]
  def pick(dev, M, N, dtype):
      best = select_config("trimul_front_cute", dtype=str(dtype),
                           bucket=shape_bucket(N=N), candidates=CANDS)
      return best["kwargs"] if best else _default_cfg(dev)   # miss/err -> default
  ```

- **sweep builder** — feed the candidate dicts + a `run_ms` that builds+runs the kernel with
  each config into the same `tune_bucket` core:

  ```python
  run_ms = lambda cfg: do_bench(lambda: run_cute_kernel(**cfg, ...))
  tune_bucket("trimul_front_cute", gpu_key(), dtype, bucket, CANDS, run_ms, config_space_hash(CANDS))
  ```

The cache format already stores cute configs (cluster tuples serialize to JSON lists and back);
`as_cfg_dict` normalizes triton.Config and plain dicts uniformly. Because config choice is
performance-only, an un-wired cute kernel simply keeps its default and loses nothing.

## Controls (env)

| var | values | effect |
|---|---|---|
| `MINIWORLD_RUN_AUTOTUNE` | `0` (default) / `1` | `1`: ignore the shipped cache and run the full autotune grid (re-tune) |

An unknown GPU with no cache warns like: *"[miniworld.autotune] no tuned autotune cache for
op '<op>' on '<gpu>' (<dtype>). Falling back to the full autotune grid — this run may be
slower and the chosen config may be suboptimal. Build a tuned cache …"*


# Systematic backend dispatch

`modules/dispatch.py` holds one declarative table `_MINIWORLD_KNOWN_BEST : {op -> backend | 
callable(device)->backend}` behind `resolve(op, impl, device)`. `MINIWORLD` (auto) resolves
via the table; an op/GPU it doesn't specially recognize falls back to **TRITON** (the portable
default for brand-new GPUs), while op/arch pairs with a measured-faster backend (e.g. trimul
cute on Hopper+) follow that. The per-op `resolve_*` are thin wrappers over `resolve`. This is
the single source of truth for the module-layer backend choice; kernel-internal shape/arch
sub-dispatch still lives next to the kernels.
