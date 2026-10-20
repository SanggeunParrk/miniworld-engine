# JIT build cache

Every CUDA kernel here is compiled on first use, never at import -- unless the installed engine is a release wheel carrying **prebuilt kernels**
that has it ([prebuilt-kernels.md](prebuilt-kernels.md)): then the first call loads the shipped binary and nvcc never runs.
Otherwise the artifact is compiled once and cached on disk; a call whose artifact is cached loads it in well under a
second, a call whose artifact is missing runs nvcc first (seconds for a cubin, up to a few minutes for a torch
extension). This page says where the artifacts go, what makes them rebuild, how concurrent builders coordinate, and
how to keep the first call of a training job from compiling.

## What a build prints

Each compile prints its lines to stderr, and a cache or bundle hit prints nothing:

```
[miniworld-engine build] not in the prebuilt bundle: trimul_h100/b200style/k3.cu (no prebuilt kernels in this install for this torch / CUDA / GPU arch)
[miniworld-engine build] compiling trimul_h100/b200style/k3.cu [3f2a9c1d] (cubin) ...
[miniworld-engine build] built trimul_h100/b200style/k3.cu [3f2a9c1d] in 14.2 s; 3 built in this process
```

The "not in the prebuilt bundle" line comes once per artifact (the reason in parentheses once per process). A process
waiting on another process's build of the same artifact says so once a minute:
`[miniworld-engine build] waiting for <name>, built by another process (65 s)`.
`settings.configure(jit_build_log=False)` silences the lines. `kernels._nvcc.built_this_process()`
lists what this process compiled (name, kind, seconds), and `kernels._nvcc.cache_roots()` returns
the cache directories that exist. With `settings.jit_fallback = False` (`MINIWORLD_ENGINE_JIT_FALLBACK=0`) a miss
raises `kernels.prebuilt.PrebuiltMissing` instead of compiling.

## Where each family caches

Every artifact is resolved by `kernels/_nvcc.py` (installed bundle, then this cache, then a build) and named by its
content key (below), so its location changes exactly when its code does:

| Family | Built by | Location |
|---|---|---|
| Raw cubins (sm_100a / sm_90a driver-API kernels: `augattn_sm100`, `atom_sm90`, `local_dit_*`, `atom_local_*`, `swa_dit_sm100*`, `token_pair_init_sm100`, `smooth_lddt_*`, `transition_sm100a`; the bias-only DiT cubins go through `augattn_sm100`) | `_nvcc.cubin` | `engine_jit_root/<family>/<name>-<key>.cubin` (+ `.json`: its key material) |
| TriMul H100 cubins | `trimul_inproj/cuda/_h100_runtime.compile` / `compile_text` -> `_nvcc.cubin`; the register-pool patch -> `_nvcc.derived_cubin` | `engine_jit_root/trimul_h100/<name>-<key>.cubin` (+ `.ptxas.log`, `.json`; a generated source as `.cu`, generated headers under `trimul_h100/*_headers_*` only when a build needs them) |
| Torch extensions (every `load_extension` / `load_inline_extension`; a `build_directory` argument is ignored) | `_nvcc.load_extension` / `load_inline_extension` | `engine_jit_root/torch_extensions/<name>-<key>/` (`artifact.json` marks a finished build) |

`engine_jit_root` is `settings.engine_jit_root` if set, else `$XDG_CACHE_HOME/miniworld_engine_jit`
(`~/.cache/...` when `XDG_CACHE_HOME` is unset). `TORCH_EXTENSIONS_DIR` is no longer used.

## What a cache key depends on

One key for every family, the same one a prebuilt bundle is indexed by (`kernels/_codegraph.py`, detailed in
[prebuilt-kernels.md](prebuilt-kernels.md#the-content-key-the-code-graph)): the normalized content of the entry
source(s) and of every header actually reached through `#include` (resolved like the compiler: next to the including
file for quoted includes, then the `-I` directories in order), the compile flags with machine paths and the host
compiler removed, the target arch, torch, `torch.version.cuda` and, for torch extensions, the Python ABI. External
headers (torch, CUDA, CUTLASS, cuBLASDx) count by version.

| An edit to ... | rebuilds |
|---|---|
| a comment, whitespace, line breaks, an `#if 0` block | nothing |
| a header in the same directory that the artifact does not include | nothing (TriMul used to rebuild every cubin on any edit under `h100_sources`) |
| a header the artifact reaches, a `-D` define, a flag, the arch, torch / CUDA | that artifact |
| the token length, batch or dtype of a call | nothing, unless the variant compiled depends on it (below) |

`miniworld-engine dev code-graph --affected <file>` lists the artifacts an edit reaches.

Keys never contain a shape, but the variant that is compiled can: extension names carry widths
and configs (`triattn_b200_core_r{rows}_v5`, PWA/OPM names per width), and some cubins take a
define chosen from the shape (`-DTPI_RI` from the token length bucket, TriMul flags per width).
A shape class a process has not used before can therefore compile a variant the cache does not
have yet. A prebuild (or a bundle) has to exercise the shapes the run will use.

## Locks: sharing one cache between jobs

- Torch extensions: torch serialises a build with a `FileBaton` on `<build dir>/lock`.
  `_nvcc.load_extension` / `load_inline_extension` reclaim a lock older than
  `STALE_LOCK_SECONDS` (30 min, `max(mtime, ctime)`, NFS-safe), and otherwise wait while another
  process builds -- printing the waiting line once a minute and reclaiming the lock if it goes
  stale meanwhile. Only a wait past `LOCK_WAIT_SECONDS` (2 h) raises, naming the file to `rm`. A finished build is
  marked by `artifact.json`, written last; a process that finds it imports the library without running ninja.
- Cubins: `fcntl.flock` on `<name>-<key>.lock` (released by the kernel when its holder dies), so
  waiting needs no stale check; the waiter prints the same once-a-minute line. A finished cubin only appears by rename
  and is reused without taking the lock.

So **any number of jobs can share one cache**, and since every directory is named by its content key, so can
checkouts and environments with different engine versions: each version's artifacts have their own keys, nothing is
rebuilt over another's, and identical code shares one artifact.

## Recommended setup

1. Install a release wheel of the engine built for this torch / CUDA (it carries the kernels for H100 in
   `_prebuilt/`, [prebuilt-kernels.md](prebuilt-kernels.md#the-wheel)): no compile at all.
2. Without one (development, an sdist or git install), one cache root per machine is enough (`XDG_CACHE_HOME` or
   `settings.engine_jit_root`), plus a prebuild step after every install and engine pin bump on one GPU of the target
   arch, running the shapes the jobs use (MiniWorld: `pixi run engine-prebuild`). A second pass must print no
   `[miniworld-engine build]` line.
3. Jobs then reuse the bundle / cache; a stray compile shows up as the build lines in the job log, and
   `MINIWORLD_ENGINE_JIT_FALLBACK=0` turns it into an error.

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
