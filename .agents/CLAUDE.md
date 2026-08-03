# CLAUDE.md - agent instructions for miniworld-engine

## HARD RULE: NEVER RUN ANYTHING ON THE LOGIN NODE

Do not execute real work on the cluster login node. This includes, but is not
limited to, `python`, `py_compile`, `ruff`, `pytest`, `pip`/`pixi install`,
`import torch`, benchmarks, or any build/lint/test sweep. The login node is
shared infrastructure.

Route everything through `srun`/`sbatch` onto a compute node:

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=64G --time=00:20:00 \
     bash -c 'cd /home/psk6950/miniworld-engine && PYTHONPATH=src <cmd>'
# or write an sbatch script (see submits/run_bench.sbatch) and `sbatch` it.
```

- Always pass `--mem=...`; omitting it requests the node's full RAM and the job
  can sit pending.
- Even a one-line import check or a `ruff check` must go via `srun`.
- Only trivial, instant shell ops (`ls`, `cat`, `git`, editing files) are OK
  locally.

## Environment

One unified pixi env at the repo root (`[tool.pixi]` in `pyproject.toml`,
materialized to `.pixi/`, gitignored) covers torch, triton, Transformer Engine
(cu12), CuTeDSL + quack, cuequivariance, matplotlib, and the Hydra bench stack.
Use `pixi run`/`pixi run --frozen`.

- Always use `--frozen` for run/bench: a bare `pixi install`/`pixi run` can
  re-pin the cu13 TE core, which the 575 driver cannot load. After any real
  `pixi install`, run `pixi run fix-te-cu12`.
- Runtime/matplotlib need the env's libstdc++:
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
- Build needs no GPU (`srun` without `--gres`); compute nodes are offline, so
  resolve/download runs on the login node wrapped in `timeout` only when needed.

## Layout / Benchmarking

See `README.md` and `docs/benchmarks.md`. One bench entry point only:
`benchmarks/runners/bench.py` (Hydra) +
`benchmarks/modules/<module>/configs/bench.yaml`, launched via
`submits/run_bench.sbatch`. Do not add new standalone bench scripts.

If a kernel is not yet wired into `benchmarks/runners/bench.py`, any temporary
local probe must stay untracked until it is promoted into
`benchmarks/kernels/<kernel>/` or `benchmarks/modules/<module>/`, and it must
still follow the team-gm harness style:

- same kind of baseline comparisons,
- same parseable stdout format,
- same capture flow (`.out` log first, then render `.md` + `.png`),
- same compute-node execution discipline.

Do not treat one-off Python snippets or hand-timed loops as benchmark results.
Those are debug probes only.
