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

A read-only wheel can consume its shipped tuned caches. Building or merging new
tuned caches requires a writable installation (for example a user-owned virtual
environment) or source checkout. `build` checks the cache destination before
configuration, plan derivation or GPU work and reports an actionable error when
it is not writable. No tuned-cache overlay or alternate root is provided.
The derived-plan cache location below does not remove this build requirement.

## Derived dispatch plans

Dispatch plans are build evidence, separate from the tuned kernel choices above.
The wheel includes the canonical `kernels/registry_kernel.csv` and matching
`registry_kernel.units.json` evidence. When their architecture and source identity match,
they can be reused from a read-only installation. Historical revision directories are
kept outside the package and do not ship. Missing or stale plans are regenerated under
`$XDG_CACHE_HOME/miniworld-engine/plans` (default `~/.cache/miniworld-engine/plans`).
`MINIWORLD_PLAN_CACHE_DIR` selects an explicit writable plan directory, for example
shared storage visible to Slurm workers. This setting does not redirect tuned caches.
Plans are validated against their architecture, runtime source files and registry;
unshipped optimization notes do not change the dispatch identity.

## LayerNorm Backward

Code:

- `src/miniworld_engine/kernels/layernorm/dispatch.py`
- `src/miniworld_engine/kernels/layernorm/compile_native.py`

`miniworld` LayerNorm has three correct backward implementations and picks the
fastest one per shape. The H100 heuristic is the static default; unknown GPUs
measure once and cache the winner instead of guessing.

## The three backward paths

| path | implementation |
|---|---|
| `atomic` | Triton backward with atomic accumulation into weight and bias gradients |
| `persistent` | Triton grid-stride backward with partial gradients and a reduction |
| `cuda` | Native CUDA backward for matching activation and weight dtypes |

The two Triton paths tune their tile configurations per GPU. Persistent backward
uses the running device's SM count. The path selector accepts `atomic`,
`persistent`, and `cuda`; `partial` is no longer a selectable path.

## Resolution order

`_resolve_bwd_path(m, n, …)` decides per backward call:

1. An explicit `settings.layernorm_bwd_path` pin bypasses calibration. A CUDA pin
   with mixed activation/weight dtypes falls back to a compatible Triton path.
2. `settings.layernorm_dispatch = "off"` uses the static heuristic.
3. H100 (sm90), unless mode is `force`, uses the static heuristic.
4. A compatible per-GPU cache hit selects the recorded path.
5. During CUDA graph capture, a miss uses the static heuristic.
6. Otherwise, calibration times compatible paths on the actual tensors and stores
   the fastest result.

The current static heuristic chooses CUDA for matching BF16 activations and
weights at widths 128 through 512. Otherwise it chooses persistent for widths
at least 384, and atomic for smaller widths.

Module cache-build children set both `layernorm_dispatch` and `biasonly_dispatch`
to `"off"`, matching the verified derivation. Explicit backend pins still build
the declared alternatives. This prevents a stored runtime choice from redirecting
a successful build unit away from its planned keys. Production execution retains
automatic dispatch.

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
                    "ms": { "atomic": 0.514, "persistent": 0.267 } }
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
| `layernorm_bwd_path` | `"persistent"` / `"atomic"` / `"cuda"` / `None` | hard override, bypasses cache + heuristic |

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

The question a fingerprint has to answer is **"are the recorded measurements void?"** — which is
narrower than "did anything change?", and conflating the two cost this repository real measurement
twice. So the fields split into three tiers:

| field | on mismatch | why |
| --- | --- | --- |
| `build_rev` | **RESET** | The only invalidator a *person* writes: a column in `registry.csv`, bumped when the way a kernel is MEASURED changes. See below. |
| `key_scheme` | **RESET** | The bucket string means something else, so entries are *mislabelled*, not merely old. |
| `env_identity` | **RESET** | Another triton / CUDA / ptxas really does make a recorded time a claim about a different compiler. Not a person's to notice, so it stays automatic. |
| `config_space_hash` | **incremental build** | A grid edit is a top-up, never a reset — see below. |
| `op_identity` | **RESET** when the supplied current identity differs | Guards kernel source and autotune-key compatibility; source edits may require remeasurement. |
| `driver_identity` | reported only | Auto-hash of the build driver. Changes which buckets get BUILT, never whether a measured winner is right. |
| `driver_id_scheme` | (gates `driver_identity`) | How that hash is computed; a stamp from another scheme is skipped, not failed. |

### `build_rev` — the invalidator a human writes

`registry.csv` carries a `build_rev` column, one integer per kernel. Bump it when an edit means the
old numbers are void. Compiler, kernel-source and affected bucket-scheme compatibility
remain automatic checks; leaving `build_rev` unchanged does not override those checks.

Earlier fingerprint policies exposed two distinct maintenance costs:

* `op_identity` hashes the `@triton.jit` body and its `key=[...]`. Reformatting a comment moved it,
  and moving it discarded every tuned entry for that kernel.
* `driver_identity` hashes the build driver. Correcting which *shapes* a driver builds moved it —
  and that reset deleted **32 of 38** tuned buckets from `cond_transition_expand_swiglu_triton`, an
  edit that said nothing at all against the numbers it destroyed.

Both are still computed and still reported by `dev cache-status`, because they are exactly what
tells a person where to look. What they no longer do is decide.

### A grid change is an incremental build, not a reset

`cache.configs_to_bench(op, gpu, configs)` returns the current grid **minus the space already
searched** — each cache records `config_space`, the set of configs the build that wrote it actually
benched:

* grid **widened** → only the added configs are timed; `store_ranked_configs` merges them against
  the stored winners, which remain valid for everything they beat.
* grid **narrowed** → nothing to time. The write drops any stored config the new grid no longer
  contains, and each surviving winner is still the best of a *smaller* space.

This matters because narrowing a ladder to what the cache proves it needs is maintenance the
repository's own tests *demand* (`test_a_fully_measured_kernel_carries_its_own_ladder` derives the
narrowing and fails until it is applied). Under the old rule, applying it cost a full rebuild —
collapsing `GROUP_M` to its single winning value across 16 kernels priced out at **25 GPU-hours** to
remove configs no winner needed. The tests asked for a change the cache then punished.

Reading back is unaffected and was already safe: `_cached_subset` intersects the stored entry with
the live grid (*"the cache names configs, the grid decides what is launchable"*), so a config the
current grid does not contain can never be served.

`driver_identity` covers what the other three cannot see: the driver decides which
`(shape, dtype, flag)` buckets get tuned and with what arguments, so adding (say) an
`ADD_RESIDUAL=1` call changes what the cache *covers* while the kernel body and the grid stay
byte-identical.

It is **not** consulted by the runtime reader — only by `dev cache-status` and CI. Checking it per
launch would import build-harness driver modules inside a production `prune_configs` (dragging
their import-time env-var surface into a consumer process, at ~1.2 ms a call), and it would fail
CLOSED: the helper returns `None` on any import error, so `stored != None` would mark every stamped
cache stale and silently drop every launch to the heuristic subset.

Its value carries a scheme number (`driver_id_scheme`). A stamp written under a different scheme is
skipped, not failed — without that, any change to the hashing scope turns every stamped cache into a
false STALE that no rebuild-free fix can clear (this happened once already, in development).

Its **scope** is the op's own driver function, the driver module's shared scope (imports, module
constants, `_`-private helpers), and — since `driver_id_scheme` **2** — the definitions the module
**imports from sibling driver modules**, transitively. Deliberately *not* the whole file. Measured
over this repo's own history: hashing the whole driver file flagged **50%** of caches, because one
op's edit invalidates every other op sharing that file; the scoped hash flags **6%**, the same as
hashing the function alone but without going blind to a shared-helper (`_x` / `_bdll` / `_w`) change.

Scheme 2 exists because scheme 1's blindness was not theoretical. `drivers/adaln.py` takes its
shapes from `drivers/conditioned_transition.py`:

```python
from .conditioned_transition import _D, _DC, _M, _SHAPE_KEY
```

`_M` is the row count every adaln bucket is built at. Editing it rewrote what adaln's cache covers
— but the *import statement's text*, all scheme 1 hashed of the sibling, is byte-identical across
that edit. So adaln's stamp did not move, `dev cache-status` reported it fresh, and the damage was
found only by diffing cache entries against `HEAD` after a rebuild had already overwritten them.
Scheme 2 hashes the imported names plus the module-level names they transitively reference — not
the whole sibling file, which would flag every importer on an unrelated edit. On this repo's
current tree it adds **zero** new drift while catching that edit;
`tests/registry/test_cache_status_detects_changes.py::test_driver_identity_follows_cross_module_imports`
drives the real edit through the real hash. Like
`op_identity`, an entry written before the field existed reads normally, so it only starts guarding
once that cache is rebuilt — or backfilled from git (below).

A tuned config is a claim about a compiler and a device, not only about a grid; an entry missing
`op_identity` (written before that field existed) still reads, so committed caches degrade to the
old behaviour rather than to a permanent miss. A config is any of: a `triton.Config`, a plain dict of tile params
(CuTe/CUDA, cluster shapes as tuples), or a pre-shaped `{kwargs, num_warps, num_stages}` dict —
`as_cfg_dict` normalizes all three and `config_to_dict` serializes them JSON-safely.

## What `build all` verifies

`miniworld-engine build all` verifies the module-derived **Triton autotune key plan**
before starting expensive tuning. The full build first checks the cache destination is writable, then imports the selected FlashAttention
backend and checks CuTe imports/cuBLASDx headers on SM90+, failing before tuning when
those dependencies are broken. A missing, stale, or foreign-architecture plan is
regenerated in a separate process and checked before publication. Generated plans are isolated by architecture and dispatch-source hash under the
[writable plan cache](#derived-dispatch-plans); incomplete derivations stop the build. Editing the module shape declarations does not itself invalidate measured
kernel configs: the builder selects module invocations covering the keys still missing.

Build inputs and derivation use the same `registry_module.csv` declarations. Diffusion
modules use augmentation **5 for inference and 48 for training**, including all declared
1024-step atom lengths through 8192. Token attention uses single width 768 and condition
width 384. Pairformer modules remain unaugmented. These are input lengths; the cache
continues to use the existing packed shape buckets.

The command builds module-reachable keys, runs drivers for registered alternative Triton
kernels not reached by modules, then merges the resulting shards. For `all`, it checks the
merged runtime cache against the verified plan and returns nonzero if required keys remain
unusable. A completed derivation alone proves Python dispatch/shape coverage, not GPU
compilation, numerical correctness, or benchmark performance.

This verification currently does **not** certify every backend's cache. In particular,
`autotune.cute_config.sweep_and_cache` has no builder caller: its six CuTe configuration
families still resolve an existing cache or a default. CUDA extensions and external
FlashAttention binaries have their own compilation/dependency requirements. FakeTensor
recording replaces those native launches with shape contracts so it can record the surrounding
MiniWorld Triton calls; it does not test those native binaries on another architecture.
Therefore a successful A6000 run is not an H100/B200 qualification. Those devices still
require actual build and numerical runs with compatible CuTe/FlashAttention/CUDA dependencies.

## Checking freshness BEFORE you trust a cache (`dev cache-status`)

The staleness fields above are checked per *launch*, and a miss is only a warning — which means a
benchmark can measure the bounded heuristic fallback from end to end and report it as the tuned
kernel. That is not hypothetical: narrowing 18 kernels' warp/stage ladders invalidated their
`config_space_hash` without a rebuild, and every module those kernels back then benched ~10-20%
slow, silently.

```bash
miniworld-engine dev cache-status              # every committed cache, every GPU key
miniworld-engine dev cache-status --gpu A100   # substring filter
```

It recomputes the code-driven fingerprints (`config_space_hash` from the grid CSVs, `op_identity`
from the kernel's live autotuner, `driver_identity` from the driver module, `key_scheme`) and
diffs them against what each cache recorded. **No GPU and no kernel launch** — importing a
`@triton.autotune` kernel only defines it — so it can run on an allocated CPU node or in CI, and exits
non-zero for code-driven stale entries. `env_identity` is reported separately and never fails the command:
a cache built under another triton/cuda is legitimately "not this machine's" without the committed
code being wrong.

`tests/registry/test_no_stale_caches.py` runs the same scan in CI. Build-revision,
key-scheme and source changes can invalidate entries. Grid changes request an incremental
build, and driver changes report potentially different coverage without erasing old measurements. `tests/registry/test_cache_status_detects_changes.py` drives a
synthetic cache to prove each detector actually fires.

### Backfilling `driver_identity` from git

`driver_identity` was added after most caches were committed, and a field a cache does not carry
cannot guard it — so every pre-existing cache reported OK even if its build driver had since
changed. The provenance is recoverable: git knows the commit that last wrote each cache, and hence
what the driver looked like then.

```bash
miniworld-engine dev cache-backfill            # dry run: what would be stamped, and what drifted
miniworld-engine dev cache-backfill --apply    # write the recovered hashes
```

It stamps the *historical* hash, never the current one — stamping today's would assert the cache
was built by code it has never seen and permanently hide the drift the field exists to surface.
Caches already stamped **under the current scheme** are skipped; an older-scheme stamp is re-derived.

Finding the historical driver is not a path substitution: the drivers have been reorganised twice
(one `kernels/drivers.py`, then abbreviated family modules `drivers_attn.py` / `drivers_ln.py`, then
today's `kernels/drivers/<family>.py`), so the tool SEARCHES the commit's tree for the module that
actually defines the driver function. Guessing paths silently skipped 70 of 234 caches — the oldest,
most likely to have drifted. Every skip is now reported with a reason; on this repo it stamps 228
caches (the 6 it cannot are the runtime-dispatch caches, which have no driver).

What this does **not** cover: a cache whose fingerprints all match but that has no entry for the
shape a run asks for ("no tuned autotune cache entry for this shape"). That is a coverage gap, not
a code-drift one. `dev coverage` checks required keys against the current verified plan;
`dev audit --replay` additionally observes actual lookups on a GPU.

### Source/compiler invalidation and partial rebuilds

`store_ranked_configs` resets incompatible measurements when the build revision,
compiler environment, kernel identity or affected key scheme changes. A grid edit is
incremental, and a driver-only change does not clear existing entries.

A partial rebuild after a genuine reset can leave other required buckets missing.
`build all` checks its merged result against the verified module plan and fails if required
keys remain unusable; a narrower build is not a declaration of full-plan completion.

Historical examples that motivated these protections:

* `rmsnorm_adamod_{fwd,bwd}_triton` lost all 34 / 33 of their `float32` entries to a rebuild that
  ran only the bf16 unit. The registry declares `bf16|fp32`; the fp32 half simply vanished.
* `cond_transition_expand_swiglu_triton` went from 38 entries to 6 when a driver edit made it stale
  and a targeted rebuild refilled only the buckets that edit produced.

Under the earlier behavior neither showed up as an error: the build reported success, and `dev cache-status` reported OK —
correctly, because the file it now guards really was written by the current code. What was lost is
invisible to every fingerprint, since a fingerprint describes the CODE and never the COVERAGE.

So, before a targeted rebuild: check `dev cache-status` for the op. If it is STALE, either rebuild
the whole op (`build --per-op`) or diff the entry keys against `git show HEAD:<cache>` afterwards.
Run `dev coverage` against the current plan after a targeted build; GPU replay can additionally
confirm the actual module lookups.

### Plan coverage and GPU replay answer different questions

Every fingerprint above answers "was this cache built by the current code?". None of them answers
"does it hold the buckets a run will ask for?" — a cache can be perfectly fresh and still serve
nothing. `dev coverage` compares usable keys with the current verified derivation without
running modules. Its default GPU key is `NVIDIA RTX A6000 (sm86)`; for another card,
pass its exact cache-file GPU key with `--gpu` and its architecture with `--arch`. `dev audit --replay` drives `builder.cases()` on a GPU and reports actual
lookup misses. Neither command establishes numerical correctness or benchmark performance.

```bash
miniworld-engine dev coverage --arch sm86    # default cache key: NVIDIA RTX A6000 (sm86)
```

```bash
miniworld-engine dev audit --replay      # exits 1 if any lookup missed
```

Read the misses by AXIS, not one at a time — the key is `dtype|FLAGS,shape_key=N`, and each axis
fails for a different reason and takes a different fix. One A100 run measured 310 misses over 39
ops, and they sorted cleanly:

| axis | share | what it means | where the fix goes |
| --- | --- | --- | --- |
| shape | 204 | the driver never builds that `(length, width)` | the width ladders in `op_units`, or the driver's own shape block |
| dtype | 54 | the op runs at a precision the row does not declare, or an operand's dtype differs | `registry.csv`'s `dtypes`, or the driver's operand dtypes |
| flag | 52 | a constexpr in the key is only ever driven one way | the driver — drive both values |

Decoding a `shape_key` is worth the trouble, because "shape" almost never meant length. `pack`
lays out `base`, then each axis width in sorted-name order, then a CRC digit of the axis names,
all base-4096 — so with the kernel's axis names from its `*_key(...)` call the key inverts exactly,
and the CRC digit confirms the arity. Done that way, only 2 of those 204 misses were a missing
LENGTH. The rest were channel WIDTHS the build never drove:

* `augmented_attention`'s three kernels wanted `(H=16, HEAD_DIM=24)` and `(16, 48)` and were built
  at `(12, 32)` / `(24, 32)`: the driver derived the head COUNT from the width at a fixed head dim,
  while the DiT fixes `n_head=16` and lets the head dim follow `d_single`.
* `trimul_gemm_gate_mmajor_triton` wanted `H2 == K` — the UNIDIRECTIONAL front — and the driver
  only ever built the bidirectional `H2 = 2*K`.
* `d_pair=384` and `d_msa=64` were on no ladder at all, though `cases()` presents both.

### The two declarations of "what shapes the model runs"

That last point is the structural one. `builder.cases()` (module dims — what `--replay` drives) and
`op_units`'s `LADDER` (channel widths — what `build all` sweeps) both claim to say what the model
presents, and historically nothing checked them against each other. The earlier per-op-only
build followed the ladder while replay asked for whatever the cases presented: a width
in one and not the other is a bucket that can never be filled and is asked for on every run.

`tests/layout/test_one_source_of_shape_truth.py` pins the direction that matters — every width a
case presents is a rung on some ladder — with an explicit `NOT_DRIVEN` list for widths that reach
no keyed kernel. It does not try to derive one list from the other: the dim-name-to-stream mapping
is real knowledge, and guessing it is how a "obviously equivalent" driver edit collapsed the adaln
and conditioned_transition caches into two useless buckets.

## Cache location & format

- Single canonical, in-repo location (committed to git): `src/miniworld_engine/autotune/data/<op>/<gpu_key>.json`.
  Reads (dispatch/prune) and writes (builder / `RUN_AUTOTUNE` regen) both target this path — no
  `~/.cache` / env-var override — so a stale per-user cache can never shadow the committed configs.
- `gpu_key` = device name + capability (e.g. `NVIDIA A100 80GB PCIe (sm80)`); entries keyed
  `"<dtype>|<shape-bucket>"` → list of `{kwargs, num_warps, num_stages, ms}` (top-K), plus the
  identity fields above, which reset the file's entries when any of them changes.
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

**The normal way is `miniworld-engine build all`.** It verifies the current source/GPU
module plan, selects module invocations covering missing keys, then runs drivers for
registered alternative Triton kernels. It merges measured shards and fails when required
keys remain unusable. Declared module invocations, selected work units and required cache
keys are separate quantities; read the command's current counts rather than a fixed total.

    miniworld-engine build all                  # config set defaults to grid; resume is enabled
    miniworld-engine dev coverage --arch sm86    # default cache key: NVIDIA RTX A6000 (sm86)
    miniworld-engine dev audit                  # build-system contract checks

`--resume` reuses completed nonempty measurement shards with compatible provenance.
Completion requires the shard's `_unit_complete` marker. A failed module may leave useful
partial timings; those can still be merged, but do not count as a completed unit. Shards
written before this marker also require a successful rerun to skip the unit. Failed children
release their claims even when they captured partial timings.
Work generations include the current source, configuration and GPU/compiler identity;
a claim file alone does not make work complete. Claim files coordinate concurrent workers,
and `--reclaim` handles orphaned claims. `--no-resume` disables completed-shard reuse;
shared valid tuning-round measurements can still be reused.

**Point `TRITON_CACHE_DIR` at a directory of the build's own, and let the build empty it.** The
triton cache is a build artifact: what ships is the JSON under `autotune/data/`, and nothing reads
the cache afterwards. One A6000 rebuild left 221,487 entries and 40 GB of it on a filesystem
shared with the rest of the lab.

    export TRITON_CACHE_DIR=/scratch/$USER/build-cache
    miniworld-engine build all --gpus 8 --prune-cache

`--prune-cache` empties it only after the merge, required-key coverage and planned-unit
checks succeed. An incomplete build retains its compiled artifacts for resume; measured timings
are stored in shards. `miniworld-engine dev prune-cache` does it by
hand, `--dry-run` says what would go. Both refuse a directory that is not unmistakably a triton
cache, and both refuse outright when `TRITON_CACHE_DIR` is unset, because then the build is
sharing `~/.triton/cache` with everything else on the machine.

The build also writes only what a launch needs -- the cubin and the metadata -- rather than every
IR level, which is 71 KB an entry instead of 187. `--keep-ir` turns that off when you want the
ttgir for a kernel. It is safe as a default because the knob is not one of triton's
cache-invalidating variables: the same config compiles to the same hash either way, verified on an
A6000, with the warm-hit path returning metadata and launcher intact.

A unit alternates between compiling (a pool of processes, no card) and measuring (one card, one
core), roughly 72% and 20% of its wall on an A6000. With one unit per card neither overlaps the
other, so `--units-per-gpu 2` puts two units on each card and each one's compile fills the other's
measurement. They never measure at the same time -- two kernels sharing the SMs both read slower
by an amount that drifts over a round, which would change which config wins -- so a unit takes its
card's lock for a whole tuning round. Each unit's compile pool is sized `cores / slots`, so twice
the units means half the workers each, not twice the load.

    miniworld-engine build all --gpus 8 --units-per-gpu 2

**It is worth about 5%, and which way it goes depends on the units.** Measured twice.

First, on 28 units of 750-864-config grids: 4436 s at 1 against 5940 s at 2, **34% slower**. Two
reasons, and neither was the idea: `compile_jobs` divided the cores by the SLOTS so each unit got
half a pool (since fixed -- it divides by the cards), and those units spend 73% of their wall
BENCHING, so both units on a card wanted the lock at once and it became a queue.

Re-measured with that fixed, on compile-dominated units -- four units, two cards, 48 cores either
way:

    --units-per-gpu 1    6097 s
    --units-per-gpu 2    5793 s      5% faster

Four of six buckets chose the same config; the two that differed were 0.0% and 2.6% off, against a
control (the same settings twice) that disagreed about three of seven with a worst case of 12.5%.

Five percent and not more, and the log says why: one unit spent 3,962 s waiting on the other's
bench lock. The ceiling is the bench itself, about 18% of a unit's wall, and half of that went to
queueing. Making the bench cheaper should raise the ceiling; the two have not been measured
together.

Which way it goes is decided by the unit's grid, and the two ends are far apart:

| | compile | bench |
|---|---:|---:|
| the A6000 rebuild's 283 units (big grids dominate the total) | 72% | 20% |
| these 28 units (750-864 configs each) | 27% | 73% |

A config costs roughly 125 ms to bench whatever the grid size -- `do_bench` fills its 25 ms warmup
and 100 ms measurement budget by construction -- while compile cost grows with the grid. So raise
it for a run over the large grids and leave it at 1 for a run over small ones. The per-unit log's
`[bench-lock] ... waited on the other unit Ns` is the number that says which one you got.

The lock itself did its job: over 26 buckets both arms chose the SAME config, every one, with
measured times 0.3-5% apart -- the run-to-run drift that was there before.

One more thing to watch: both units hold their driver's tensors on the same card at once, which is
a real out-of-memory risk on a 24 GB card at large shapes.

`dev coverage` checks the usable shipped cache against the current verified module plan.
`dev audit` separately checks build-system contracts. A coverage hole can trigger runtime
fallback and prevents `build all` from reporting complete required-key coverage.

Both builders write directly into `src/miniworld_engine/autotune/data/`, so after a build you
`git add` + commit the JSONs — as their own commit, not folded into a code change. A new GPU
(H100/B200/…) also needs its native dependencies and numerical/module validation on that
hardware; committing tuning JSONs alone does not qualify it.

**1. Capture builder (preferred — covers every wired kernel automatically).** Instead of
hand-replicating each kernel's launch, `autotune/capture.py` instruments the Triton autotuner
(`Autotuner._bench`) and records every `(config -> measured ms)` as it is benched during a real
module forward/backward, keyed by the SAME `(op, dtype, bucket)` the runtime prune uses. So one
module run populates the caches of every autotune kernel it fires. The CLI configures
capture inside the worker process that actually performs the measurements.

For a targeted cache build, use the existing build command so settings, augmentation,
training/inference modes, cache keys and provenance follow the same contract:

    miniworld-engine build <case>               # named production module

For a benchmark, use the benchmark command and its measurement contract. Changing a
CSV flag or configuring another process does not apply `torch.compile` to the timed callable.

**2. One kernel, or one module.** There is no second builder — there used to be
(`python -m miniworld_engine.autotune.build --op ...`, two hand-written pilot builders) and it was
removed: it stored under `transition_split_fwd` / `trimul_bidir_front`, names retired in the kernel
rename (see `docs/kernels/rename-map.tsv`), so nothing could read what it wrote. Both ops are
covered now, with a driver, a checker and a shipped cache under their current names. Use the one
builder, narrowed:

    miniworld-engine build <op> --per-op        # one registry kernel, every shape bucket
    miniworld-engine build <case>               # one production module's dispatch path

`--per-op` selects the registered kernel-driver workload; it does not prove coverage of
every module constexpr combination that can reach that kernel.

**`build all` runs both, and needs no flag to.** Neither list is complete alone. `--per-op`
coverage is DECLARED — registry.csv × level — so every kernel with a driver is tuned, but each
through its own driver, which never produces the constexpr combinations a module's real dispatch
does (`SAVE_PREACT=1`, `ADD_RESIDUAL=0`, `H2=512,K=256`). Measured on an A6000, a cache built that
way answers `missing_pairs 0` to the declared question and misses 363 lookups the module matrix
makes, across 42 of 91 ops (`docs/records/cache-coverage-replay-a6000.md`). The module matrix
reaches those keys and reaches only 48 of the 91 kernels.

The current default runs the verified module plan first and the registered alternatives
second. `fill_gaps` and workload-attributed shared tuning records reuse compatible
measurements. `--per-op` and `--per-module` request one pass alone.

Use `dev coverage` for required-key coverage of the current verified plan, `dev audit`
for build-system contracts, and `dev audit --replay` for actual GPU lookup behavior.
The older counts in the replay example above describe that historical run, not today's grid.

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
| `run_autotune` | `False` (default) / `True` | Enables full-grid tuning; `fill_gaps` and shared valid measurements can still avoid repeat work |
| `autotune_miss_cap` | `24` (default), `0` = off | on a cache MISS, how many configs a heuristic subset may keep |

A runtime miss does not normally fall back to the whole grid. `autotune_miss_cap`
bounds the search to a heuristic subset instead of putting an unbounded tuning sweep
inside a forward call. `run_autotune=True` lifts the cap, because a build wants the whole
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
