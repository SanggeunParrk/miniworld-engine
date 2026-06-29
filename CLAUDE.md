# CLAUDE.md — agent instructions for miniworld-kernels

## ⛔⛔ HARD RULE: NEVER RUN ANYTHING ON THE LOGIN NODE ⛔⛔

Do **NOT** execute real work on the cluster login node. This includes — but is
not limited to — `python`, `py_compile`, `ruff`, `pytest`, `pip`/`pixi install`,
`import torch`, benchmarks, or any build/lint/test sweep. The login node is
shared infrastructure; a prior agent ran compiles, repeated `python -c` import
checks, ruff, and a multi-GB `pixi install` on it and **blocked/overloaded the
login node.** The user was (rightly) furious. This is a standing rule.

**Route EVERYTHING through `srun`/`sbatch` onto a compute node:**

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=64G --time=00:20:00 \
     bash -c 'cd /home/psk6950/miniworld-kernels && PYTHONPATH=src <cmd>'
# or write an sbatch script (see tests/run_bench.sbatch) and `sbatch` it.
```

- ALWAYS pass `--mem=...` — omitting it requests the node's full ~2 TB RAM and
  the job sits PENDING (Reason=Resources) forever.
- Even a one-line import check or a `ruff check` must go via `srun`. If you are
  about to type a bare `python`/`ruff`/`pixi` at the prompt — stop, wrap it.
- Only trivial, instant shell ops (`ls`, `cat`, `git`, editing files) are OK
  locally.

## Environment (driver 575 / CUDA 12.9 on h100 nodes)

One **unified pixi env at the repo root** (`[tool.pixi]` in `pyproject.toml`,
materialized to `.pixi/`, gitignored) covers everything: torch 2.10+cu128,
triton, Transformer Engine (cu12), CuTeDSL + quack, cuequivariance, matplotlib,
and the hydra bench stack. Replaces the old external team-gm / FA-cute envs and
the deleted `cute-env/` + `te-env/` folders. Use `pixi run`/`pixi run --frozen`.

- **Always `--frozen`** for run/bench: a bare `pixi install`/`pixi run` re-pins
  the cu13 TE core (TE's sdist metadata bug), which the 575 driver can't load.
  After any real `pixi install`, run `pixi run fix-te-cu12` to restore the cu12
  core. (cu13 = `libcublas.so.13: cannot open`.)
- **Runtime/matplotlib need the env's libstdc++:**
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (system gcc lacks
  `CXXABI_1.3.15`).
- **Build needs no GPU** (`srun` without `--gres`); **compute nodes are offline**
  so resolve/download runs on the login node wrapped in `timeout` (download-only,
  no compile), then build on a compute node.

## Layout / benchmarking

See `README.md` and `docs/benchmarks.md`. One bench entry point only:
`benchmarks/runners/bench.py` (Hydra) +
`benchmarks/modules/<module>/configs/bench.yaml`, launched via
`tests/run_bench.sbatch`. Do not add new standalone bench scripts.

If a kernel is not yet wired into `benchmarks/runners/bench.py`, any temporary
local probe must stay untracked until it is promoted into
`benchmarks/kernels/<kernel>/` or `benchmarks/modules/<module>/`, and it must
still follow the **team-gm harness style**:

- same kind of baseline comparisons,
- same parseable stdout format,
- same capture flow (`.out` log first, then render `.md` + `.png`),
- same compute-node execution discipline.

Do **not** treat one-off Python snippets or hand-timed loops as benchmark
results. Those are debug probes only.
