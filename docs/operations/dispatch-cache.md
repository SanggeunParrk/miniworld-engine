# Runtime Dispatch Caches

Some kernels choose among multiple correct implementations at runtime. Those
choices are performance policy tied to a specific GPU, so — like the autotune-config
caches — they are committed to git and shared across machines.

The persistent on-disk cache root is a single, canonical, in-repo location:

- `src/miniworld_engine/autotune/data/<subdir>/<gpu_key>.json`

where `<subdir>` is `ln_bwd_dispatch`, `bias_only_dispatch`, etc. There is
deliberately **no** `$MINIWORLD_ENGINE_CACHE_DIR` / `$XDG_CACHE_HOME` / `~/.cache`
override: reads and writes both target this in-repo path, so a stale per-user cache
can never shadow the repo's committed choices, and a checkout of the repo (direct or
as a submodule) already carries the calibrated caches.

## LayerNorm Backward

Code:

- `src/miniworld_engine/kernels/layernorm/dispatch_cache.py`
- `src/miniworld_engine/kernels/layernorm/compile_native.py`

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

1. **Explicit pin** `settings.layernorm_bwd_path = "persistent"|"partial"|"atomic"|"cuda"`
   → use it (debug / manual pin), bypassing everything below.
2. **`settings.layernorm_dispatch = "off"`** → static H100 heuristic, no measuring.
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

- **Location:** `src/miniworld_engine/autotune/data/ln_bwd_dispatch/<gpu>.json` (in-repo, committed).
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

- `src/miniworld_engine/kernels/bias_only_attention/dispatch.py`

This cache uses the same root policy and GPU key helper as LayerNorm. It chooses
among correct attention backend variants and stores the per-GPU crossover. A bad
or missing cache only changes speed; it falls back to the static H100 heuristic.

## In-Process Dispatch

`src/miniworld_engine/kernels/trimul_inproj/cute/dispatch.py` keeps a process
local dispatch table for cuBLAS vs CuTe/quack candidates. It is intentionally
not persisted: the choices are tied to live thunks, current process imports, and
shape-specific warmup state.

## Safety

Dispatch caches only select among **correct** kernels, so they must never change
numerics. A stale or corrupt cache at worst picks a slower path; read, parse, or
timing errors fall back to static heuristics. Calibration is skipped during
CUDA-graph capture.

## Controls (`miniworld_engine.settings`)

Set with `settings.configure(field=value)`; `settings.reset()` puts them back. These used to be
`MINIWORLD_LN_AUTOTUNE` / `MINIWORLD_LN_BWD` environment variables — settings.py replaced every
`MINIWORLD_*` switch with a field, and nothing reads the variables any more.

| field | values | effect |
|---|---|---|
| `layernorm_dispatch` | `"auto"` (default) / `"off"` / `"force"` | `off`: static only; `force`: calibrate even on H100 |
| `layernorm_bwd_path` | `"persistent"` / `"partial"` / `"atomic"` / `"cuda"` / `None` | hard override, bypasses cache + heuristic |

## Using it on a new GPU

Nothing to do — just run. The first training step (or first time each
`(d, M-bucket)` is hit) calibrates and writes the cache under
`src/miniworld_engine/autotune/data/ln_bwd_dispatch/`; subsequent runs and
re-imports read it. Commit the JSON so other machines get the calibrated choice.
To pre-warm explicitly, run one inference+training pass per shape you care about.
To inspect/clear, look at / delete the JSON under that in-repo dir.


# Autotune Config Cache

Separate from the LayerNorm *path* cache above: this ships the top-K tuned **autotune configs**
per `(gpu, dtype, op, shape-bucket)` so runs skip the full-grid autotune tax and performance is
reproducible across machines. It is **backend-agnostic** — one cache format + storage layer
serves Triton, CuTe, and CUDA kernels. Code: `src/miniworld_engine/autotune/` (`cache.py`,
`build.py`).

**INVARIANT:** config choice is performance-only — every candidate config computes the same
math — so a missing / stale / wrong cache is only ever slower, never incorrect.

## How it works

Two runtime entry points share the cache; a kernel uses whichever fits its backend:

- **Triton** — `install_cache_reader()` wraps every autotuner's `early_config_prune` (composed
  OVER any device-smem safety prune). Per call it reads the running `(gpu, dtype, shape-bucket)`,
  and on a **hit** returns only the cached top-K (autotune picks among ~5, not the full grid).
- **CuTe / CUDA** — these fix their tile/cluster/stage config at build time (no autotune loop),
  so they call `select_config(op, dtype=…, bucket=…, candidates=…)` to *pick one* cached config
  and apply its `kwargs` (`tile_m`/`tile_n`/`cluster`/`pingpong`/…); on a miss they fall back to
  the kernel's own `default_config`.

Both paths warn ONCE on a **miss** (unknown GPU, unseen shape) or **stale** entry. What they
fall back TO differs, and neither is the full grid:

- **Triton** narrows to `settings.autotune_miss_cap` (24) configs centred on the standard part of
  the space — `num_warps` in {4, 8}, `num_stages` in {2, 3, 4} — plus the middle of each block
  axis. Returning the full grid here would put a 205,266-config sweep inside a production forward
  on any GPU nobody has built a cache for. A build (`run_autotune=True`) still gets the whole
  space on purpose, and a grid already smaller than the cap is left alone.
- **CuTe / CUDA** falls back to the kernel's own `default_config` — already a bounded answer.

An entry is **stale** when any of these no longer match what it was measured against:

| field | invalidated by |
| --- | --- |
| `config_space_hash` | the kernel's config grid changing |
| `env_identity` | triton / torch / CUDA / ptxas version changing |
| `op_identity` | the kernel's source, or its `key=[...]` list, changing |

A tuned config is a claim about a compiler and a device, not only about a grid; an entry missing
`op_identity` (written before that field existed) still reads, so committed caches degrade to the
old behaviour rather than to a permanent miss. A config is any of: a `triton.Config`, a plain dict of tile params
(CuTe/CUDA, cluster shapes as tuples), or a pre-shaped `{kwargs, num_warps, num_stages}` dict —
`as_cfg_dict` normalizes all three and `config_to_dict` serializes them JSON-safely.

## Cache location & format

- Single canonical, in-repo location (committed to git): `src/miniworld_engine/autotune/data/<op>/<gpu_key>.json`.
  Reads (dispatch/prune) and writes (builder / `RUN_AUTOTUNE` regen) both target this path — no
  `~/.cache` / env-var override — so a stale per-user cache can never shadow the committed configs.
- `gpu_key` = device name + capability (e.g. `NVIDIA A100 80GB PCIe (sm80)`); entries keyed
  `"<dtype>|<shape-bucket>"` → list of `{kwargs, num_warps, num_stages, ms}` (top-K), plus the
  three identity fields above, which reset the file's entries when any of them changes.
  `<dtype>` is what the kernel was OBSERVED to run in, so a kernel with mixed operands records
  `bfloat16+float32`, not the driver's dtype.

## The config set (the candidate space)

A cache entry names configs; something has to produce the list those names are matched against.
That is a **config set**: one `<op>.csv` per op under a directory, either materialised (one row =
one config) or a grid spec (`axis,values`, expanded as a cartesian product).

- The default is `grid`, packaged at `src/miniworld_engine/autotune/configs/grid/` so it ships in
  the wheel. `MINIWORLD_CONFIG_DIR` overrides it and must be set **before any kernel module is
  imported** — `triton.Autotuner` keeps the list it was handed only if it is non-empty, so a set
  chosen after the import updates a list nobody reads and the kernel dies at launch with
  `dynamic_func() missing 2 required positional arguments` naming its tile axes.
- It has to be the set the cache was tuned over. `select_config` INTERSECTS a stored entry
  against the live list, so pointing a shipped cache at a narrower set resolves every entry to
  nothing and re-tunes on every call, silently — right answers, no warning, just slow.
- The repo root's `configs/` holds the A-B sets used during development (`blk16`…`blk128`,
  `warp4`, `warp8`, `mixed1`, `mixed2`, `accuracy`). They are build inputs, not runtime data, and
  are not packaged.

## Building the shipped cache (on the target GPU)

**The normal way is `miniworld-engine build all`.** It decomposes the work into one unit per
`(op, declared dtype, shape bucket)` — 922 units today — runs them across every GPU it is given,
merges the shards into `data/`, and reports what it could not measure. Coverage is DECLARED
(registry.csv crossed with each kernel's `level` and `dtypes`), not "whatever a module happened to
dispatch to": driving modules reached 48 of 91 triton kernels on an A6000.

    miniworld-engine build all                  # config set defaults to `grid`
    miniworld-engine dev audit                  # did every declared (op, dtype, bucket) land?

`dev audit` is the check that closes the loop — it compares the shipped cache against the declared
work list and names the holes. A hole is not a wrong answer, only a bucket that pays the bounded
fallback above at runtime.

Both builders write directly into `src/miniworld_engine/autotune/data/`, so after a build you
`git add` + commit the JSONs — as their own commit, not folded into a code change. A new GPU
(H100/B200/…) is enabled by running the build on that box and committing its JSONs.

**1. Capture builder (preferred — covers every wired kernel automatically).** Instead of
hand-replicating each kernel's launch, `autotune/capture.py` instruments the Triton autotuner
(`Autotuner._bench`) and records every `(config -> measured ms)` as it is benched during a real
module forward/backward, keyed by the SAME `(op, dtype, bucket)` the runtime prune uses. So one
module run populates the caches of every autotune kernel it fires. Drive it through the existing
bench harness:

    from miniworld_engine import settings
    settings.configure(run_autotune=True, capture=True)

then run the bench harness in the same process:

    python benchmarks/runners/bench.py kernel=<module> implementations='[miniworld]' \
      compile=false cudagraph=manual mode=training sweep_axis=seq_len ...

`run_autotune=True` unlocks the full grid (no cached narrowing) so every config is benched;
`capture=True` installs the capture and flushes top-5 per `(op,dtype,bucket)` at the end.
Validated against the hand builder: capture reproduces its top-1 selections (near-ties aside).

In practice you do not drive this by hand: `miniworld-engine build all` at the top of this
section runs the whole matrix through exactly this capture, and that is what the shipped caches
were built with. Reach for the recipe above only to capture ONE module's kernels.

**2. Explicit builder (per-kernel, for the pilot kernels).**

    PYTHONPATH=src python -m miniworld_engine.autotune.build --op all     # or --op <name>

Its core `tune_bucket(op, gk, dtype, bucket, candidates, run_ms, csh)` is backend-agnostic: it
benches each candidate via a `run_ms(cfg) -> ms` closure and stores the top-K. Triton builders
point `run_ms` at `do_bench(kernel.fn[grid])`.

Coverage: every live Triton kernel is wired — 91 ops in registry.csv, 922 declared
`(op, dtype, bucket)` units. `miniworld-engine dev audit` is what reports how many of them the
shipped cache actually holds on the card you are on; do not infer it from this paragraph.

## CuTe / CUDA autotune (sm90+)

CuTe/CUDA kernels have no Triton autotune loop — they fix `tile_m/tile_n/cluster/pingpong` at
build time (e.g. `dualgemm_kernel.py`'s `_CFG` or a `default_config(dev)`). They join the SAME
cache via the two backend-agnostic hooks; this must be built + verified on an sm90+ box (it
cannot run on Ampere):

- **runtime pick-one** — replace the hardcoded config with a cache lookup that falls back to the
  kernel's own default on a miss (so it is a no-op until a cache is shipped):

  ```python
  from miniworld_engine.autotune import select_config
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

## Controls (`miniworld_engine.settings`)

| field | values | effect |
|---|---|---|
| `run_autotune` | `False` (default) / `True` | `True`: ignore the shipped cache and run the full autotune grid (re-tune) |
| `autotune_miss_cap` | `24` (default), `0` = off | on a cache MISS, how many configs a heuristic subset may keep |

A miss does not fall back to the whole grid. The grid is 205,266 configs across 91 ops, and a
forward that searches it is not slow, it is stopped: `autotune_miss_cap` bounds the search to a
heuristic subset instead. `run_autotune=True` lifts the cap, because a build wants the whole
space on purpose.

An unknown GPU with no cache warns like: *"[miniworld.autotune] no tuned autotune cache for op
'<op>' on '<gpu>' (<dtype>). Falling back to a heuristic 24 of 1944 configs (run
`miniworld-engine build all` to tune this GPU properly) — this run may be slower and the chosen
config may be suboptimal. Build a tuned cache for this GPU with the autotune cache-builder (see
docs/operations/dispatch-cache.md)."*

The same exit reports a STALE cache, and names which of the four identities moved: the kernel's
config grid, the toolchain (`env_identity` — triton / cuda / ptxas), the kernel source or its
autotune key list (`op_identity`), or simply no entry for this shape.


# Systematic backend dispatch

`modules/dispatch.py` holds one declarative table `_MINIWORLD_KNOWN_BEST : {op -> backend | 
callable(device)->backend}` behind `resolve(op, impl, device)`. `MINIWORLD` (auto) resolves
via the table; an op/GPU it doesn't specially recognize falls back to **TRITON** (the portable
default for brand-new GPUs), while op/arch pairs with a measured-faster backend (e.g. trimul
cute on Hopper+) follow that. The per-op `resolve_*` are thin wrappers over `resolve`. This is
the single source of truth for the module-layer backend choice; kernel-internal shape/arch
sub-dispatch still lives next to the kernels.
