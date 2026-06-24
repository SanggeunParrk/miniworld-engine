# miniworld-kernels

Dedicated GPU kernel-development repo for MiniWorld / AF3-style ops. The idea is
to **cut one op out of the full model and optimize it in isolation**:

## Critical Safety

This repo is often accessed from a cluster login node.

- Do not run recursive scans outside this repo, especially commands like `find /home/psk6950 ...`.
- Do not run installs, builds, benchmarks, profiling, or GPU-dependent commands on the login node.
- Keep login-node activity limited to lightweight repo-local inspection.
- Use `srun` or an allocated compute node for GPU work or heavy filesystem activity.

Repo structure:

- A **kernel** (`kernels/<unit>/`) is a chunk you deliberately chose to fuse and
  hand-optimize. It owns its backend implementations (`triton/`, `cute/`,
  `cuda/`) plus a PyTorch `reference.py`, a public `interface.py`, and a
  `benchmark/` dir holding that kernel's results.
- A **module** (`modules/<op>/`) is a part cut from the model (e.g.
  `triangle_multiplication`). It only *connects* kernels — it has **no**
  `triton/cute/cuda` folders — and keeps its own `benchmark/` results.

Kernels and modules were consolidated here out of `team-gm`
(`src/team_gm/modules/`, across `psk/benchmark`, `perf/trimul`, `miniworld`,
`exp/miniworld`) and the FlashAttentionBias repo (cute trimul work).

## Layout

```
src/miniworld_kernels/
├── _typecheck.py                 # standalone team_gm.typecheck shim
├── kernels/                      # fusion units: backends + per-kernel results
│   ├── tm1/  tm2/                #   left/right- and output-gated GEMM kernels
│   │   ├── reference.py interface.py
│   │   ├── triton/ cute/ cuda/   #   backend implementations
│   │   └── benchmark/            #   this kernel's results
│   ├── transition/ layernorm/ adaln/
│   ├── triangle_attention/ augmented_attention/
│   ├── bias_only_attention/ gated_projection/
│   ├── fused_ln_mask/            #   LN+mask fusion (used by the trimul cute path)
│   └── __init__.py               #   flat re-export bridge: kernels.triton_tm1, ...
└── modules/                      # model ops: connect kernels (NO backend folders)
    ├── triangle_multiplication/  #   module.py (connects tm1/tm2/LN; pytorch/triton/
    │   ├── module.py             #     cute/cuequivariance via ImplementationType)
    │   ├── reference.py interface.py baseline_dtv1.py
    │   └── benchmark/            #   this op's results
    ├── triangle_attention/ transition/ adaptive_layernorm/ augmented_attention/
    ├── exceptions.py             #   ImplementationType (pytorch/triton/cuda/cute/cuequivariance)
    ├── primitives.py ops.py      #   shared connecting utilities (LayerNorm, Linear, gates)
    └── __init__.py
scripts/bench.py                  # THE single bench entry (hydra, team-gm style)
config/bench.yaml                 # bench config (kernel, implementations, compile, ...)
benchmark/logs/                   # raw SLURM bench logs (cross-kernel/unified runs)
pyproject.toml                    # [tool.pixi] = the unified env (triton+TE+cute+cuequiv); .pixi/ gitignored
tests/run_bench.sbatch            # single SLURM launcher for scripts/bench.py
```

In each kernel's `triton/`: `main.py` is the `psk/benchmark` variant (canonical),
`perf.py`/`miniworld.py` are alternates. Each vendored file carries a
`# vendored from team-gm <branch>@<sha>` header. Vendored kernel bodies
(`**/triton/*.py`, `**/cuda/*.py`) are not linted.

## Benchmarking

**Benchmark policy: follow the team-gm harness unless there is a specific reason
not to.** In this repo that means `scripts/bench.py` + `config/bench.yaml` +
`tests/run_bench.sbatch` are the default and preferred path. Do not replace
them with ad hoc timing snippets or custom markdown summaries for final
results.

One entry point — `scripts/bench.py`, driven by `config/bench.yaml` (hydra):

```bash
# Unified repo env (.pixi/). --frozen keeps the cu12 TE core fix (see CLAUDE.md).
srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 --mem=64G --cpus-per-task=8 \
  bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
    PYTHONPATH=src python scripts/bench.py kernel=triangle_multiplication \
      implementations=[pytorch,triton,cuequivariance] compile=false"'
# or submit: sbatch tests/run_bench.sbatch
```

`kernel=` selects the op (`triangle_multiplication`, `triangle_attention`,
`transition`, `adaptive_layernorm`, `augmented_attention_token/atom`).
`compile=true` benches the `torch.compile`'d variant — no separate script.
Results land in `benchmark/<gpu>/`; raw logs in `benchmark/logs/`.

If an op is not yet integrated into the unified harness, a kernel-local
`bench.py` is acceptable only as a temporary bridge. It should still emit the
same kind of machine-parseable output so `scripts/plot_bench.py` can generate
the `.md` + `.png` report pair in that kernel's `benchmark/` directory.

The **cute** path is `implementations=[cute]` (an `ImplementationType.CUTE`
implementation of `triangle_multiplication` that connects the tm1/tm2/fused-LN
cute kernels). cutlass-dsl + quack are in the unified env, so the same
`pixi run --frozen ... scripts/bench.py implementations=[cute]` runs it.

## Status

Restructured into the kernels/modules split above. The triangle_multiplication
cute path wins end-to-end at L ≥ 768 on H100 (≈1.75 ms at L=1024); the
from-scratch single-megakernel tm2 (`kernels/tm2/cute/tm2_cute_kernel.py`) is WIP.

## Toolchain

```bash
ruff check src/
ruff format src/
ty check
```
