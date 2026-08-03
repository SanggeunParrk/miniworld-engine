# miniworld-engine

Dedicated GPU kernel-development repo for MiniWorld / AF3-style ops. The idea is
to **cut one op out of the full model and optimize it in isolation**:

> **Where this fits.** miniworld-engine is the bottom layer of a three-layer
> stack: it owns the fused kernels + building-block ops; **team-gm** composes them
> into the representative AF3 blocks; terminal product repos assemble those into
> full models. The boundary rules (which layer owns an op vs. a block vs. a model,
> and where residuals live) are documented canonically in team-gm's
> `docs/ARCHITECTURE.md`.

## Critical Safety

This repo is often accessed from a cluster login node.

- Do not run recursive scans outside this repo, especially commands like `find /home/psk6950 ...`.
- Do not run installs, builds, benchmarks, profiling, or GPU-dependent commands on the login node.
- Keep login-node activity limited to lightweight repo-local inspection.
- Use `srun` or an allocated compute node for GPU work or heavy filesystem activity.

Repo structure:

- A **kernel** (`src/miniworld_engine/kernels/<unit>/`) is a chunk you deliberately chose to fuse and
  hand-optimize. It owns its backend implementations (`triton/`, `cute/`,
  `cuda/`) plus a PyTorch `reference.py` and a public `interface.py`.
- A **module** (`src/miniworld_engine/modules/<op>/`) is a part cut from the
  model (e.g. `triangle_multiplication`). It only *connects* kernels — it has
  **no** `triton/cute/cuda` folders.
- A **benchmark** lives next to its target type:
  `benchmarks/kernels/<kernel>/...` for isolated kernels and
  `benchmarks/modules/<module>/...` for composed model modules. Each benchmark
  target owns its own `configs/` and `artifacts/` folders.

Kernels and modules were consolidated here out of `team-gm`
(`src/team_gm/modules/`, across `psk/benchmark`, `perf/trimul`, `miniworld`,
`exp/miniworld`) and the FlashAttentionBias repo (cute trimul work).

## Layout

```
src/miniworld_engine/
├── _typecheck.py                 # standalone team_gm.typecheck shim
├── kernels/                      # fusion units: importable backends
│   ├── tm1/  tm2/                #   left/right- and output-gated GEMM kernels
│   │   ├── reference.py interface.py
│   │   └── triton/ cute/ cuda/   #   backend implementations
│   ├── transition/ layernorm/ adaln/
│   ├── triangle_attention/ augmented_attention/
│   ├── bias_only_attention/ gated_projection/
│   ├── fused_ln_mask/            #   LN+mask fusion (used by the trimul cute path)
│   └── __init__.py               #   flat re-export bridge: kernels.triton_tm1, ...
└── modules/                      # model ops: connect kernels (NO backend folders)
    ├── triangle_multiplication/  #   module.py (connects tm1/tm2/LN; pytorch/triton/
    │   ├── module.py             #     cute/cuequivariance via ImplementationType)
    │   ├── reference.py interface.py baseline_dtv1.py
    ├── triangle_attention/ transition/ adaptive_layernorm/ augmented_attention/
    ├── exceptions.py             #   ImplementationType (pytorch/triton/cuda/cute/cuequivariance)
    ├── primitives.py ops.py      #   shared connecting utilities (LayerNorm, Linear, gates)
    └── __init__.py
benchmarks/
├── kernels/<kernel>/             # isolated kernel benchmarks + configs/artifacts
├── modules/<module>/             # composed module benchmarks + configs/artifacts
└── runners/                      # shared benchmark/render CLI entry points
docs/                             # repo docs, cache policy, kernel notes
third_party/                      # external checkouts/submodules
benchmarks/runners/bench.py       # active bench runner
benchmarks/modules/triangle_multiplication/configs/bench.yaml
pyproject.toml                    # [tool.pixi] = the unified env (triton+TE+cute+cuequiv); .pixi/ gitignored
submits/run_bench.sbatch          # single SLURM launcher for benchmarks/runners/bench.py
```

In each kernel's `triton/`: `main.py` is the `psk/benchmark` variant (canonical),
`perf.py`/`miniworld.py` are alternates. Each vendored file carries a
`# vendored from team-gm <branch>@<sha>` header. Vendored kernel bodies
(`**/triton/*.py`, `**/cuda/*.py`) are not linted.

## Benchmarking

**Benchmark policy: follow the team-gm harness unless there is a specific reason
not to.** The active path is `benchmarks/runners/bench.py` +
`benchmarks/modules/<module>/configs/bench.yaml` + `submits/run_bench.sbatch`.
Do not replace the harness with ad hoc timing snippets or custom markdown
summaries for final results. A benchmark run writes CSV; plotting is a separate
CSV-rendering step.

Detailed benchmark conventions live in `docs/benchmarks.md`. Runtime dispatch
cache policy lives in `docs/operations/dispatch-cache.md`.

One entry point — `benchmarks/runners/bench.py`, driven by target-local Hydra
configs:

```bash
# Unified repo env (.pixi/). --frozen keeps the cu12 TE core fix.
srun --account=cssb --qos=cssb_h100 --partition=h100 --gres=gpu:h100:1 --mem=64G --cpus-per-task=8 \
  bash -c 'pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; \
    PYTHONPATH=src python benchmarks/runners/bench.py kernel=triangle_multiplication \
      implementations=[pytorch,dtv1,cuequivariance,miniworld] mode=inference"'
# or submit: sbatch submits/run_bench.sbatch
```

`kernel=` selects the op (`triangle_multiplication`, `triangle_attention`,
`transition`, `adaptive_layernorm`, `augmented_attention_token/atom`).
All final benchmarks run the `torch.compile`d path; non-compiled debug probes
are not valid final benchmark results.
Generated results land in the selected target's `artifacts/` directory, for
example `benchmarks/modules/triangle_multiplication/artifacts/`.
The benchmark CSV is the source of truth. It includes method, dimensions,
precision, mode, metric, device, compile flag, and value. Render SVG figures
from it with `benchmarks/runners/plot_csv.py`.
For trimul, run separate `sweep_axis=seq_len` and `sweep_axis=d_pair` jobs; the
figure caption records the fixed dimension (`d_pair=...` or `L=...`).

If an op is not yet integrated into the unified harness, keep local probes
untracked and move the stable definition into `benchmarks/kernels/<kernel>/`
or `benchmarks/modules/<module>/`.

The **cute** path is `implementations=[cute]` (an `ImplementationType.CUTE`
implementation of `triangle_multiplication` that connects the tm1/tm2/fused-LN
cute kernels). cutlass-dsl + quack are in the unified env, so the same
`pixi run --frozen ... benchmarks/runners/bench.py implementations=[cute]`
runs it.

### Ampere workstation cards (A5000 / A6000)

The RTX A5000 / A6000 (GA102 = `sm_86`) are **triton-only** targets, exactly like the
A100 (`sm_80`): the cute/cutlass paths are gated behind `sm_90+`
(`torch.cuda.get_device_capability()[0] < 9`), so `MINIWORLD` resolves to the portable
Triton family — no dispatch change. They live on the `cssb-master` cluster
(`partition=gpu`, `qos=normal`, A5000=`gpu02`/24 GB, A6000=`gpu01,03-05`/48 GB), which is
separate from the A100/H100/B200 cluster. One parameterized launcher per bench type covers
both cards (pick the card with `--gres`; the script auto-detects it and asserts `sm_86`):

```bash
# module bench / kernel sweep / per-target / tuned-cache capture
BENCH_TARGET=transition sbatch --gres=gpu:A6000:1 submits/run_bench_ampere.sbatch
sbatch --gres=gpu:A5000:4 submits/run_kbench_ampere.sbatch
CAPTURE_TARGET=all      sbatch --gres=gpu:A6000:1 submits/run_autotune_capture_ampere.sbatch
```

The A5000's 24 GB may OOM at the top of the sweep (L=1024, d=512); `bench.py` records those
points as `status=failed` rows rather than aborting, so the CSV still shows the memory cliff.

## Status

Restructured into the kernels/modules split above. The triangle_multiplication
cute path wins end-to-end at L ≥ 768 on H100 (≈1.75 ms at L=1024); the
from-scratch single-megakernel tm2 (`kernels/tm2/cute/tm2_cute_kernel.py`) is WIP.

## Toolchain

```bash
pixi run --frozen ruff-check
pixi run --frozen ruff-format
ty check
```
