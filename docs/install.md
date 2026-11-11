# Installing miniworld-engine

## Requirements

| | |
|---|---|
| OS / CPU | Linux x86_64 |
| Python | >= 3.10 (the prebuilt H100 kernels: CPython 3.12 only) |
| PyTorch | >= 2.8 with CUDA 12.x (the prebuilt H100 kernels: torch 2.13 + CUDA 12.9, i.e. the `cu129` wheel) |
| GPU | H100 (sm_90a): CUDA and Triton backends. A100 (sm_80): CUDA kernels JIT-built on first use (no prebuilt wheel), Triton backends. Other GPUs: Triton / PyTorch backends only |
| NVIDIA driver | one that runs your torch build's CUDA; for the prebuilt kernels (CUDA 12.9) R575 or newer |
| To JIT-build kernels | a CUDA toolkit whose release matches `torch.version.cuda` (`nvcc`), a host `g++` 9 - 13 (nvcc 12.x cannot parse GCC 14's libstdc++; `ninja` comes with the install); network access once (or the alternatives below) for the CUTLASS / MathDx headers |

The PyTorch backend of every module (the reference) needs only torch.

## Install

```sh
pip install miniworld-engine                     # from PyPI
pip install .                                    # from a checkout
```

The PyPI default torch wheel may target a newer CUDA than your driver supports; take torch from the matching PyTorch index, e.g.

```sh
pip install --extra-index-url https://download.pytorch.org/whl/cu129 miniworld-engine
```

Extras: `baselines` (cuEquivariance and Anthropic's kit dependencies, for the external comparison backends `cueq` / `anthropic`),
`flash` (FlashAttention-4, the global-attention path of the SWA atom transformer's `cuda` backend; prerelease:
`pip install --prerelease=allow "flash-attn-4[cu13]"`), `dev` (ruff, pyright, matplotlib).
Anthropic's esmfold2 kit also needs a transformers fork, which PyPI metadata cannot carry (a git dependency); pip users install it
by hand: `pip install "transformers @ git+https://github.com/huggingface/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb"`
(the pixi environment pins it already).

Developers of the repository use the pixi environment in `pyproject.toml` (`pixi install`, then `pixi run fix-te-cu12`).

## Prebuilt kernels vs JIT

Every hand-written CUDA artifact is identified by a content key (its sources and the headers they reach, the flags, torch / CUDA /
Python versions, the CUTLASS / MathDx release; `kernels/_codegraph.py`). An artifact is looked up, in order, in

1. `miniworld_engine/_prebuilt/` -- filled by the `miniworld-engine-kernels-sm90-cu129` wheel, which `pip` installs automatically on
   Linux x86_64 with CPython 3.12 (the version is pinned to the engine's). It holds the H100 (sm_90a) builds for torch 2.13 / CUDA 12.9.
2. the JIT cache, `$XDG_CACHE_HOME/miniworld_engine_jit` (default `~/.cache/miniworld_engine_jit`).

A key found in neither is compiled on first use into the JIT cache, under a file lock (one process builds, the others wait). Each build
prints one line when it starts and one when it ends (`[miniworld-engine build] ...`); a run that prints nothing reused every artifact.
A build failure raises `miniworld_engine.kernels._nvcc.KernelBuildError`; nothing falls back to another backend silently.

What that means in practice:

- **H100, CPython 3.12, torch 2.13 cu129**: everything the requirement lists need is prebuilt; nothing compiles, no toolkit or headers
  needed.
- **H100 with another Python / torch / CUDA version**: the keys differ, so the CUDA kernels JIT-build on first use (minutes per
  family; once per machine and cache). Needs the build requirements above.
- **A100**: there is no sm_80 kernel wheel; the CUDA backends JIT-build on first use. Needs the build requirements above.

Prebuild ahead of time (a CPU node is enough, no GPU): `python -m miniworld_engine.kernels._prebuild --arch sm90a` (or `--arch sm80`),
`--list` to see the plan and what is missing. Triton kernels compile in seconds on first use and are cached by Triton.

## CUTLASS and MathDx headers

Some CUDA kernels compile against CUTLASS v4.2.0 / CuTe (the H100 TriangleAttention front, the A100 OuterProductMean GEMM and
AugmentedAttention, ...); MathDx (cuBLASDx) headers are resolved the same way for builds that include them. Only a build needs them.
Where the headers are taken from, first hit wins (`kernels/_headers.py`):

| | CUTLASS | MathDx |
|---|---|---|
| 1. environment | `CUTLASS_PATH` = a CUTLASS checkout (`$CUTLASS_PATH/include/cute/tensor.hpp`) | `MINIWORLD_MATHDX_HOME`, `MATHDX_HOME` or `NVIDIA_MATHDX_HOME` = a mathdx root (`include/cublasdx.hpp`) |
| 2. installed distribution | `nvidia-cutlass==4.2.0.0` (its `cutlass_library/source/include`) | `nvidia-mathdx` (`nvidia/mathdx`) |
| 3. header cache | `<cache>/cutlass-4.2.0.0` | `<cache>/mathdx-25.6.0` |

`<cache>` is `$MINIWORLD_ENGINE_HEADER_CACHE`, else `$XDG_CACHE_HOME/miniworld_engine_headers` (default
`~/.cache/miniworld_engine_headers`). When a build needs headers that none of the three provides, the pinned wheel is downloaded from
PyPI (`nvidia-cutlass` 4.2.0.0: 6.4 MB; `nvidia-mathdx` 25.6.0: 23 MB), its SHA-256 checked against the value pinned in
`_headers.py`, and only the header tree unpacked into the cache. The `nvidia-cutlass` 4.2.0.0 wheel carries the complete `include/` tree
of the CUTLASS v4.2.0 release, file for file identical to the v4.2.0 git tag the prebuilt kernels are built with.

The headers are not a pip dependency on purpose: most installs (prebuilt H100 kernels) never read them, and `nvidia-cutlass` depends on
`cuda-python`, whose current release pulls `cuda-bindings` 13 and the CUDA 13 runtime wheels into the environment.

The artifact keys do not depend on where the headers come from (an include directory is keyed by its library version), so a checkout,
the installed wheel and the cache give the same keys -- and the same prebuilt artifacts are found. A `CUTLASS_PATH` to another CUTLASS
release is honoured for builds; a prebuilt artifact of the pinned release is still used when present.

Machines without network access (e.g. offline cluster compute nodes), any one of:

```sh
python -m miniworld_engine.kernels._headers fetch            # on a node with network access, into a cache the build nodes share
pip download --no-deps nvidia-cutlass==4.2.0.0 -d wheels/    # elsewhere; then, on the offline machine:
python -m miniworld_engine.kernels._headers fetch cutlass --wheel wheels/nvidia_cutlass-4.2.0.0-py3-none-any.whl
pip install --no-deps nvidia-cutlass==4.2.0.0 nvidia-mathdx==25.6.0   # into the environment (no cuda-python pulled in)
export CUTLASS_PATH=/path/to/cutlass-v4.2.0                  # or point at a checkout
```

`python -m miniworld_engine.kernels._headers show` prints what is found. `MINIWORLD_ENGINE_NO_DOWNLOAD=1` forbids the download (a
build that needs missing headers then fails with the options above).

## Environment variables

| variable | effect |
|---|---|
| `CUTLASS_PATH` | CUTLASS checkout for builds (overrides the pinned headers) |
| `MINIWORLD_MATHDX_HOME` / `MATHDX_HOME` / `NVIDIA_MATHDX_HOME` | mathdx root for builds |
| `MINIWORLD_ENGINE_HEADER_CACHE` | where fetched headers are kept |
| `MINIWORLD_ENGINE_NO_DOWNLOAD` | `1`: never download headers |
| `XDG_CACHE_HOME` | base of the JIT cache (`miniworld_engine_jit`) and the header cache |
| `CUDA_HOME` / `CUDA_PATH` | CUDA toolkit for builds (an `nvcc` in the Python environment's `bin/` is preferred, then this, then `PATH`, then `/usr/local/cuda*`; the one matching `torch.version.cuda` wins) |
| `MINIWORLD_ENGINE_HOST_CXX` | host compiler for nvcc (default: the first `g++` that compiles a CUDA probe) |

## Troubleshooting

- **`CutlassMissing: ... is not prebuilt, and the pinned headers could not be fetched`**: no network on this machine. Use one of the
  offline options above.
- **`cuBLASDx headers not found`**: as above, for `mathdx`.
- **Every first call compiles on an H100**: the prebuilt wheel is absent or does not match. Check `pip show
  miniworld-engine-kernels-sm90-cu129` (CPython 3.12 only), `torch.__version__` (2.13) and `torch.version.cuda` (12.9).
  `python -m miniworld_engine.kernels._prebuild --arch sm90a --list` shows `have` / `todo` per artifact.
- **`nvcc fatal: Unsupported gpu architecture`** or a toolkit mismatch: set `CUDA_HOME` to a toolkit of the same release as
  `torch.version.cuda`.
- **Host-compiler parse errors in libstdc++ headers** (`__is_array`, `0.0bf16`): GCC 14 with nvcc 12.x; install GCC <= 13 and/or set
  `MINIWORLD_ENGINE_HOST_CXX`.
- **A build seems to hang**: another process holds the build lock of the same artifact (the waiter prints a line a minute). A killed
  build leaves no lock behind.
- **`CUDA driver version is insufficient`**: the torch build or the prebuilt kernels need a newer driver; install a torch wheel for your
  driver's CUDA (`--extra-index-url https://download.pytorch.org/whl/cu1xx`) -- the CUDA kernels then JIT-build for it.
