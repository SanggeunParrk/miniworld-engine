# JIT build cache

Every CUDA kernel here is compiled on first use, never at import, and cached on disk. A call
whose artifact is cached loads it in well under a second; a call whose artifact is missing runs
nvcc first (seconds for a cubin, up to a few minutes for a multi-arch torch extension). This page
says where the artifacts go, what makes them rebuild, how concurrent builders coordinate, and how
to keep the first call of a training job from compiling.

## What a build prints

Each compile prints two lines to stderr, and a cache hit prints nothing:

```
[miniworld-engine build] compiling trimul_h100/b200style/k3.cu [3f2a9c1d] (cubin) ...
[miniworld-engine build] built trimul_h100/b200style/k3.cu [3f2a9c1d] in 14.2 s; 3 built in this process
```

A process waiting on another process's build of the same artifact says so once a minute:
`[miniworld-engine build] waiting for <name>, built by another process (65 s)`.
`settings.configure(jit_build_log=False)` silences the lines. `kernels._nvcc.built_this_process()`
lists what this process compiled (name, kind, seconds), and `kernels._nvcc.cache_roots()` returns
the cache directories that exist.

For a torch extension ninja decides whether to rebuild, so the build is confirmed from its library
afterwards (`<build dir>/<name>.so` new or relinked). The "compiling" line comes up front when that
is predictable: no library yet, a `build.ninja` that names other source paths (another checkout
built this cache), or a source newer than the library. A rebuild caused only by an edited header
gets just the "built" line.

## Where each family caches

| Family | Built by | Location |
|---|---|---|
| Raw cubins (sm_100a / sm_90a driver-API kernels: `augattn_sm100`, `atom_sm90`, `local_dit_*`, `atom_local_*`, `swa_dit_sm100*`, `token_pair_init_sm100`, `smooth_lddt_*`, `transition_sm100a`; the bias-only DiT cubins go through `augattn_sm100`) | `subprocess.run(nvcc ... -cubin)` | `engine_jit_root/<family>/<stem>_<hash>.cubin` |
| TriMul H100 cubins | `trimul_inproj/cuda/_h100_runtime.compile` | `engine_jit_root/trimul_h100/<key>.cubin` (+ `.ptxas.log`, generated `.cu`) |
| Torch extensions with a `build_directory` (OPM / PWA integrations, SWA DiT loader, owned-BLAS / cuBLASLt inline helpers) | `_nvcc.load_extension` / `load_inline_extension` | `engine_jit_root/<name or family>/` |
| Torch extensions without one (most of `kernels/*/cuda`: Transition, LayerNorm, TriAttn sm90, token DiT, ...) | `_nvcc.load_extension` | `$TORCH_EXTENSIONS_DIR/<name>/`, default `$XDG_CACHE_HOME/torch_extensions/py<XY>_cu<ZW>/<name>/` |

`engine_jit_root` is `settings.engine_jit_root` if set, else `$XDG_CACHE_HOME/miniworld_engine_jit`
(`~/.cache/...` when `XDG_CACHE_HOME` is unset). Torch's default root also follows
`XDG_CACHE_HOME`, so setting that one variable moves both.

## What a cache key depends on

| Family | Rebuilds when | Does not depend on |
|---|---|---|
| Torch extensions | In one process torch's versioner hashes the sources and build arguments under the extension name. Across processes ninja decides: `build.ninja` (flags, source paths) changed, or a source or an included header is newer than its object. | token length, batch, dtype of the call |
| TriMul H100 | the source bytes, the flags, `nvcc --version`, **and every file under `trimul_inproj/cuda/h100_sources`** (`_source_digest`). Any edit or deletion there rebuilds every TriMul cubin. | shapes (but flags carry the width / bucket of the variant) |
| Raw cubins | nvcc path (and release), flags (including `-D` variant defines), the source and the headers it names (`sm100.cuh`, `row90.cuh`, `wgmma_bf16.inc`, ...) | shapes |

Keys never contain a shape, but the variant that is compiled can: extension names carry widths
and configs (`triattn_b200_core_r{rows}_v5`, PWA/OPM names per width), and some cubins take a
define chosen from the shape (`-DTPI_RI` from the token length bucket, TriMul flags per width).
A shape class a process has not used before can therefore compile a variant the cache does not
have yet. A prebuild has to exercise the shapes the run will use.

## Locks: sharing one cache between jobs

- Torch extensions: torch serialises a build with a `FileBaton` on `<build dir>/lock`.
  `_nvcc.load_extension` / `load_inline_extension` reclaim a lock older than
  `STALE_LOCK_SECONDS` (30 min, `max(mtime, ctime)`, NFS-safe), and otherwise wait while another
  process builds -- printing the waiting line once a minute and reclaiming the lock if it goes
  stale meanwhile. Only a wait past `LOCK_WAIT_SECONDS` (2 h) raises, naming the file to `rm`.
- TriMul cubins: `fcntl.flock` on `<key>.lock` (released by the kernel when its holder dies), so
  waiting needs no stale check; the waiter prints the same once-a-minute line. A finished cubin is
  reused without taking the lock.
- Raw cubins: no lock. Each builder writes a private `*.<pid>.tmp` and renames it into place;
  concurrent builders produce identical bytes.

So **any number of jobs can share one cache when they run the same engine sources**: they wait
for each other's builds and then reuse them.

**Two environments with different engine versions must not share `TORCH_EXTENSIONS_DIR`** (or an
`engine_jit_root` used by `build_directory` extensions). Extension names are the same across
versions while the sources differ, so each environment's ninja sees the other's build as out of
date (different source paths or newer files) and rebuilds over it -- every switch between them
recompiles, and a job can load a library that another environment just relinked. Cubin caches are
content-addressed and safe to share, but there is no reason to split the roots.

## Recommended setup

1. One cache root per environment (per checkout / installed engine version and GPU arch), e.g.
   `XDG_CACHE_HOME=<env>/cache`, or explicitly `settings.engine_jit_root=<env>/jit` plus
   `TORCH_EXTENSIONS_DIR=<env>/jit/torch_extensions`.
2. A prebuild step after every install and every engine pin bump, on one GPU of the target arch,
   running the shapes the jobs use (MiniWorld: `pixi run engine-prebuild`). It can snapshot the
   artifact mtimes under `cache_roots()` and check that a second pass prints no
   `[miniworld-engine build]` line.
3. Jobs then reuse the cache; a stray compile shows up as the two build lines in the job log.

## Why the first call after 299a2007 / 9897f70e compiled

The no-Triton changes on H100 / B200 (`e99ffb63`, `275d36ee`, `1b91a01c`) send a call no CUDA path
serves to the PyTorch reference, not to new CUDA extensions, so they added no nvcc targets. The
first-call cost after `299a2007` and `9897f70e` came from source edits invalidating existing
caches: both commits edit or delete files under `trimul_inproj/cuda/h100_sources`
(`front_fused2.cu`, `projection_ln.cu`, `tail_fused3.cu`, `PROVENANCE.json`; the `d64_train`
backward sources), which changes `_source_digest` and so the key of every TriMul H100 cubin, and
`299a2007` also edits the generic sm_90a Transition binding (`generic_sm90a/bind.cu`, `tr90.cuh`),
`swa_ffn_bwd.cu`, `pv_gate_inf.cu` and `opm_epilogue.cu`, which rebuild those extensions / cubins.
Separately, `d4feea9b` / `ed23a97b` did add one new extension (`layernorm_general_cuda_v3`, the
general CUDA LayerNorm / RMSNorm, three arches, under `TORCH_EXTENSIONS_DIR`).
