# Prebuilt kernels

The engine's CUDA kernels (torch extensions and driver-API cubins) ship compiled inside the engine's own **binary
wheel**, the way PyTorch ships its kernels: `pip install miniworld-engine` next to the matching torch, and the first
call loads binaries instead of running nvcc. An sdist or a git install carries none and JIT-compiles every artifact on
first use into a per-machine cache ([jit-cache.md](jit-cache.md)), tens of minutes for a new environment.

The binaries live in the package at `miniworld_engine/_prebuilt/<arch>/` (`sm90` for H100, `sm100` for B200): a
`manifest.json` mapping each artifact's **content key** to its file under `artifacts/`. The directory is produced at
release time by `miniworld-engine dev bundle-kernels` and is not in git (`.gitignore`).

## How an artifact is found

Every compile entry point goes through `kernels/_nvcc.py`: `load_extension` / `load_inline_extension` for torch
extensions, `_nvcc.cubin` for cubins (the raw sm_90a / sm_100a families and TriMul's `_h100_runtime.compile` /
`compile_text`), `_nvcc.derived_cubin` for a post-link patch (TriMul's register-pool rewrite). Each one:

1. computes the artifact's key (below) -- reading sources and stat'ing headers, never running a compiler;
2. looks the key up in the prebuilt kernels (`kernels.prebuilt.find(kind, name, key)`): a hit is imported / loaded
   straight from the package (`torch.utils.cpp_extension`'s module import for a `.so`, the family's own cubin loader
   for a `.cubin`) -- no nvcc, no ninja, no build directory, nothing written;
3. else looks in the local cache: `engine_jit_root/<family>/<name>-<key>.cubin`, or
   `engine_jit_root/torch_extensions/<name>-<key>/` for an extension;
4. else builds into that location, after printing `[miniworld-engine build] not in the prebuilt bundle: <name>`. With
   `settings.jit_fallback = False` (or `MINIWORLD_ENGINE_JIT_FALLBACK=0` in the environment) it raises
   `kernels.prebuilt.PrebuiltMissing` instead: that is how a release check proves the shipped kernels cover a run.

A `_prebuilt/<arch>/` is used only when its manifest's torch version and CUDA (`torch.version.cuda`) are the running
ones and the GPU is that arch; anything else is ignored with one line and the JIT path takes over.
`settings.prebuilt_kernels = False` ignores the shipped kernels; `MINIWORLD_ENGINE_KERNEL_BUNDLES=<dir>[:<dir>]`
(directories holding a `manifest.json`) adds bundles from elsewhere.

CUTLASS / CuTe headers are needed only to *build* the TriangleAttention kernels that include them: the call sites take
the include directory from `_nvcc.cutlass_include()` (a stand-in when `CUTLASS_PATH` is unset), the key is the same
either way (`cutlass@4.2.0`), and only a build without headers raises `_nvcc.CutlassMissing`.

## The content key: the code graph

`kernels/_codegraph.py` keys an artifact by the code the compiler actually sees, so a trivial unrelated edit never
forces a rebuild, and an edit that reaches the compiler always does:

* **Graph.** The entry source(s) -- or the generated text, for `load_inline` and TriMul's `compile_text` -- and the
  transitive `#include` closure, resolved the way the compiler resolves it: a quoted include first next to the
  including file, then the `-I` directories in order; an angle include through the `-I` directories only. Only files
  that are reached are nodes; a header next to the source that nothing includes is not.
* **Normalization.** Each node is hashed after removing comments, line continuations, blank lines, runs of whitespace,
  spaces next to punctuation, line breaks inside code and `#if 0` blocks (`#if 0 ... #else` keeps the `#else` side).
  String / character literals are kept byte for byte; `#define F(x)` and `#define F (x)` stay different. It is not a
  preprocessor: macros are not expanded and other conditionals stay as written.
* **External headers** (torch, CUDA, CUTLASS / CuTe, cuBLASDx) are not read; they are represented by versions:
  torch and `torch.version.cuda` for every key, `cutlass@<version>` (from `cutlass/version.h`, or the pinned
  `CUTLASS_PINNED = 4.2.0` when the headers are absent, as on a machine that only loads prebuilt artifacts) and
  `mathdx` for such include directories.
* **Flags** as they reach the compiler, normalized: `-I` into the package becomes `-I<pkg:path>`, into a generated
  directory under the JIT root `-I<gen>` (its files are graph nodes, labelled relative to it, so the directory's name
  does not matter), into the toolkit / environment `-I<cuda:...>` / `-I<prefix:...>`; the host compiler (`-ccbin`) is
  dropped; `host_flags()` and `gencodes()` are placeholders the key sees as requested (`--miniworld-gencode=90a`),
  expanded to real flags only when a build runs.
* **Target**: the arch of the device (`sm90` / `sm100`), torch, CUDA, and the Python ABI for torch extensions
  (cubins are Python-independent).

Per-file parses are cached by (path, mtime, size): a key costs a few stat calls once its files have been read.

TriMul's transformed upstream headers (`publish_header`) are registered in an in-memory overlay and only written to
disk when a cubin that includes them must be compiled, so a process that loads prebuilt or cached cubins writes
nothing. TriMul's old `_source_digest` (every file under `h100_sources`) only names those overlay directories now; no
key uses it.

The local JIT cache uses the same keys, so the cache directory of an extension changes exactly when its graph does
(an edit that does not reach an extension never rebuilds it, which ninja's mtime / path checks did).

```bash
miniworld-engine dev code-graph src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources/common/warp_primitives.cuh
miniworld-engine dev code-graph trimul_h100/b1/b1_fused.cu
miniworld-engine dev code-graph --affected src/miniworld_engine/kernels/augmented_attention/cuda/sm100/sm100.cuh
```

The first prints a file's include graph (with `-I` options for its include directories) and the known artifacts that
reach it; the second an artifact's recorded graph, flags and versions, marking nodes whose current content differs;
the third which artifacts a change to that file rebuilds. Known artifacts are the local cache's (each build leaves its
key material next to the artifact), the installed bundles' and any `--manifest` (a `manifest.json` or a records
`.jsonl`).

## Building the prebuilt kernels

```bash
miniworld-engine dev bundle-kernels --arch sm90 --out /scratch/bundle-sm90
miniworld-engine dev bundle-kernels --arch sm90 --out /scratch/bundle-sm90 --coverage engine --coverage pytest:tests/gpu/sm90
```

On a GPU of the target arch, in the release environment. Coverage drivers run with a recorder
(`MINIWORLD_ENGINE_BUNDLE_RECORD`) that logs every artifact any of their processes resolves, with the recipe that
rebuilds it, a JIT root of the build's own (`<out>/cache`), and `MINIWORLD_ENGINE_BUNDLE_ARCH` so a multi-arch
`gencodes` request compiles only the bundle's arch (one arch of code per artifact; a fat build of every arch is what
made `layernorm_general` 32 MB):

* `engine` (default) -- `build/bundle_coverage.py`: every case of `autotune.builder.cases()` (the modules and
  checkpoint leaves of `registry_module.csv`, each row at its own widths, in each declared dtype / compute dtype and
  bf16-mixed) at MiniWorld's inference ladder (token rungs 128 ... 2560, atom rungs 1024 ... 49152) and the training
  lengths (128 ... 768 tokens, the registry's atom ladder), two batched samples on the pair stream, and the
  architecture blocks (Pairformer with one- and two-directional TriMul, the token DiffusionTransformer with its
  pair-bias-hoist / core-precision / qk-norm options and batched pairs, the SWA atom transformer in its three styles,
  the block-local atom DiT), inference and training, in bf16, fp32, bf16-mixed (fp32 parameters under bf16 autocast,
  MiniWorld's precision) and fp32 parameters fed bf16 activations. Cases are dealt round-robin over modules to
  `--workers` processes per GPU, so different artifacts compile in parallel; a worker that needs an artifact another
  one is compiling waits on its lock. Shapes that need a whole GPU run last, one per GPU.
* `pytest:<path,...>` -- a test run (one process per file), e.g. the release smoke set.
* `cmd:<command>` -- anything else, e.g. MiniWorld's production models:
  `--coverage "cmd:cd ~/MiniWorld && python scripts/prebuild_engine.py --mode all --jobs 8"` (run it in an
  environment whose engine is this checkout).

The recorded artifacts then update `<prebuilt root>/<arch>/` (default: the package's own `_prebuilt/`):

* **incremental**: while the drivers run, the existing `_prebuilt/<arch>` is installed (it is inside the package), so
  every artifact whose key did not change is a prebuilt hit and is not compiled again -- a refresh after an engine
  commit compiles only the kernels whose code graph changed;
* artifacts of this Python ABI (and every cubin) that the run no longer reaches are dropped (`--keep` keeps them);
  other ABIs' extensions are kept;
* artifacts that compile NVIDIA MathDx / cuBLASDx are never bundled (not redistributed; they JIT where used).

`<out>/summary.json` has the counts, sizes and timings; `<out>/logs/engine_cases.jsonl` the result of every coverage
case.

**Other Python ABIs and torch / CUDA builds, without a GPU: replay.** A torch extension is specific to the Python ABI
and the torch / CUDA it was compiled against; a cubin only to torch / CUDA. `--replay <records.jsonl>` rebuilds every
recorded recipe in the current environment (`--jobs` processes, CPU only) and adds the results:

```bash
# cp310 extensions next to the cp312 ones (same torch / CUDA: the cubins are reused as they are)
/venv-py310/bin/python -m miniworld_engine.cli dev bundle-kernels --arch sm90 --out /scratch/replay-py310 --replay /scratch/bundle-sm90/records/engine.jsonl
# another torch / CUDA: its own prebuilt root
/venv-cu130/bin/python -m miniworld_engine.cli dev bundle-kernels --arch sm90 --out /scratch/replay-cu130 --prebuilt-root /scratch/prebuilt-cu130 --replay /scratch/bundle-sm90/records/engine.jsonl
```

## The wheel

```bash
python -m miniworld_engine.cli dev build-wheel --out dist        # once per Python ABI, in an env with that Python
python -m build --sdist                                           # the sdist: no binaries, the JIT path
python -m twine check dist/*
```

`dev build-wheel` builds the pure wheel with setuptools (`pip wheel --no-build-isolation`; `_prebuilt/` is excluded
from package data, so an sdist never carries it), adds `_prebuilt/<arch>/` with every cubin and the extensions of the
running Python ABI (a filtered manifest), pins `Requires-Dist: torch==<major>.<minor>.*` to the torch the artifacts
were built against, and tags it `cpXY-cpXY-manylinux_2_28_x86_64`. Its binary check (auditwheel-style, `readelf`)
reports any shared library an artifact needs beyond the manylinux system set and what torch's wheel provides (libtorch,
libc10, libcudart, cuBLAS), any RPATH, and any glibc / libstdc++ symbol version newer than manylinux_2_28 allows;
`--strict` fails on a finding. libtorch, libc10 and the CUDA libraries are never bundled: they come with torch.

**Release toolchain.** The C++ side of every extension is compiled and linked by one pinned host compiler:
`MINIWORLD_ENGINE_HOST_CXX=<g++>` (also passed to nvcc as `-ccbin` for extensions that name none) and `CXX=<same>`
(torch compiles and links with it). It decides which libstdc++ the result needs: a conda GCC 14 link required
`CXXABI_1.3.15` and left an RPATH into the build environment (the .so then fails to import on a manylinux_2_28 system,
seen on the cluster's own `/lib64/libstdc++.so.6`), while GCC 12.4 (`/opt/ohpc/pub/compiler/gcc/12.4.0`) needs at
most `GLIBCXX_3.4.21` / `CXXABI_1.3.9`, no RPATH. torch 2.13 compiles host code as C++20, so GCC 8 and 9 cannot build
it. The release build is therefore a replay (no GPU) in clean venvs with that compiler, one per Python ABI and torch /
CUDA build, from the records of one H100 coverage run.

A wheel holds one torch / CUDA build of the kernels. On PyPI that has to be the CUDA build `pip install torch` gives
(torch 2.13.0: CUDA 13.0, `2.13.0+cu130`); with any other torch build the wheel still works, through the JIT path.

## Release checklist

Per release, at the engine commit being released (set `version` in `pyproject.toml` first; PyPI takes no local
versions, so one wheel per Python ABI carries one torch / CUDA build -- the one `pip install torch` gives).

1. **Coverage, sm90** (an H100, any environment whose torch has the target's torch minor):
   `miniworld-engine dev bundle-kernels --arch sm90 --out <work> --coverage engine --coverage pytest:<smoke set>`,
   plus the model's driver (`--coverage "cmd:..."`). Its `<work>/records/*.jsonl` list every artifact with its recipe.
2. **Release build** (CPU): per target torch / CUDA a clean venv per Python ABI (`pip install torch==<x.y.z>` from the
   index the wheel targets, `ninja`, the engine's runtime dependencies, `setuptools build twine`; for a CUDA build
   other than the cluster's, `cuda-toolkit[nvcc,cudart,crt,cccl,nvvm]==<cuda>` provides nvcc), with the pinned
   compiler, replaying into one prebuilt root:

   ```bash
   export MINIWORLD_ENGINE_HOST_CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++ CXX=/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++
   export CUDA_HOME=<toolkit matching torch.version.cuda> CUTLASS_PATH=<CUTLASS 4.2>
   /venv-py312/bin/python -m miniworld_engine.cli dev bundle-kernels --arch sm90 --out <w>/py312 --prebuilt-root <root> --replay <work>/records/engine.jsonl <work>/records/pytest1.jsonl
   /venv-py310/bin/python -m miniworld_engine.cli dev bundle-kernels --arch sm90 --out <w>/py310 --prebuilt-root <root> --replay <work>/records/engine.jsonl <work>/records/pytest1.jsonl
   /venv-py312/bin/python -m miniworld_engine.cli dev build-wheel --prebuilt-root <root> --out dist --strict
   /venv-py310/bin/python -m miniworld_engine.cli dev build-wheel --prebuilt-root <root> --out dist --strict
   /venv-py312/bin/python -m build --sdist --no-isolation -o dist .
   /venv-py312/bin/python -m twine check dist/*
   ```

   The root is kept between releases: a key that did not change is reused, only changed kernels rebuild.
3. **sm100** (a B200): steps 1-2 with `--arch sm100` into the same root (one `manifest.json` per arch); not built yet.
4. **Check** each wheel in a fresh venv with its torch and an empty cache: `pip install dist/<wheel>`,
   `XDG_CACHE_HOME=$(mktemp -d) MINIWORLD_ENGINE_JIT_FALLBACK=0 python -m pytest <smoke set>` -- no
   `[miniworld-engine build]` line, nothing under `$XDG_CACHE_HOME/miniworld_engine_jit`. This needs a driver that runs
   the torch build (CUDA 13 needs driver >= 580).
5. **Upload**: `python -m twine upload dist/miniworld_engine-<version>-*.whl dist/miniworld_engine-<version>.tar.gz`.
