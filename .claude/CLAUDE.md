# Agent instructions — miniworld-engine

<!-- Loaded automatically by Claude Code from .claude/CLAUDE.md. Other agents: read this file first. -->

GPU kernel library for MiniWorld / AF3-style ops: hand-written CUDA where it exists, a Triton
fallback everywhere else, a PyTorch reference for each op. Layout and quickstart: `README.md`.
Docs index: `docs/README.md`.

## Hard rules

1. **Never run real work on a login node.** No `python`, `pytest`, `ruff`, `pixi install`,
   `import torch`, builds, benchmarks or recursive scans outside this repo. Even a one-line
   import check goes through Slurm. Only trivial shell ops (`ls`, `cat`, `git`, editing files)
   run locally. How to reach a compute node on each GPU cluster: `docs/gpus/<gpu>.md`.
2. **Always pass `--mem`** to `srun`/`sbatch`; without it the job requests the node's full RAM.
3. **Never `git checkout`/`git restore` a file that has uncommitted work.** Stash, or revert by
   hand. Built-but-uncommitted kernels have been lost this way.
4. **Commit or push only when asked.** Work on a branch; the default branch is `main`.
5. **Don't delete code while restructuring.** Move it, and keep history reachable.

## Environment

One pixi env at the repo root (`[tool.pixi]` in `pyproject.toml`, materialised to `.pixi/`):
torch 2.13 + cu129, triton 3.7, Transformer Engine, cuequivariance 0.12, FlashAttention-4.

- `pixi install` needs no GPU; run it on a CPU allocation, then `pixi run fix-te-cu12` and
  `miniworld-engine dev install-flash --arch <sm>`. Use `pixi run --frozen` afterwards.
- At runtime: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
- Per-GPU JIT caches: set `MINIWORLD_ENGINE_JIT_ROOT` per checkout/GPU so builds don't collide.
- `dev derive` needs `MINIWORLD_COMPILE_WRAP=disable`.

## Where things go

- New kernels: `src/miniworld_engine/kernels/<family>/{cuda,triton}/` + `reference.py` +
  `interface.py`; modules only connect kernels (`src/miniworld_engine/modules/<op>/`).
- Benchmarks: only `benchmarks/runners/bench.py` + `benchmarks/modules/<module>/configs/bench.yaml`.
  One-off probes stay untracked under `.bench/` (ignored). Timings are CUDA-graph or compiled —
  never eager for final numbers.
- Per-GPU completion status (which op is finished for which shapes): `docs/gpus/<gpu>.md` (section "Completion status").
  The **judgement** column is filled by the maintainer, not by an agent.
- Dated measurements and audits: `docs/records/`, never edited after the fact.
- Local scratch: `.bench/` (ignored). Removed research lives in git (see
  `docs/records/experiments-archive.md`).
