# GPUs

One page per GPU: how to run on it, then **which operation is finished for which shapes**.
The maintainer makes that call; the pages record it.

| GPU | arch | cluster / partition | page | summary |
|---|---|---|---|---|
| H100 80GB HBM3 | sm90 | cssb, `h100` partition | [h100/h100.md](h100/h100.md) · [module dispatch](h100/dispatch.md) | hand CUDA for TriMul, Transition (n=4), TriAttn training, OPM/PWA, token DiT; Triton elsewhere |
| B200 | sm100 | cssb (same cluster as H100/A100) | [b200/b200.md](b200/b200.md) | Triton only in this repo (CUDA work is happening elsewhere) |
| A100 80GB PCIe | sm80 | cssb (same cluster as H100/B200) | [a100/a100.md](a100/a100.md) | Triton only |
| RTX A6000 / A5000 | sm86 | `cssb-master`, `gpu` partition | [ampere-workstation/ampere-workstation.md](ampere-workstation/ampere-workstation.md) | Triton only (no completion table) |

## Common to all


- Nothing runs on a login node (see `.claude/CLAUDE.md`). Always pass `--mem`.
- Install the pixi env on a CPU allocation of the cluster you will run on
  (`pixi install`, `pixi run fix-te-cu12`, `miniworld-engine dev install-flash --arch <sm>`).
  Conda prefixes are absolute paths: an env built in one checkout cannot be moved to another.
- Give each GPU/checkout its own JIT cache (`MINIWORLD_ENGINE_JIT_ROOT`); prebuilt
  TriangleAttention `.so` files are tied to the torch/python they were built with and the
  CUDA path silently falls back to Triton when they do not match
  (rebuild: see [h100/h100.md](h100/h100.md)).
- Autotune caches are keyed by GPU name **and** toolchain (`env_identity`): a cache built
  under another torch/triton is ignored. Check with `miniworld-engine dev cache-status --gpu <name>`.

## Layout and completion tables

```
docs/gpus/<gpu>/
  <gpu>.md              environment + module-level completion tables
  dispatch.md           module dispatch contracts (H100)
  <module>/
    <module>.md         kernel-level tables, kernel-flow figures, measurements
    figures/            <module>_<variant>.json spec -> *_<name>.svg -> *.png
```

Tables: columns are shapes (`(Length, Dimension)` for bf16-only TriMul / TriAttn, otherwise
`(Length, Dimension, dtype)`); rows are **implementation** (agent: the backend default dispatch
runs, or 미구현), **성능 확인** (maintainer only, ✓ / ✗) and **cache build** (agent, ✓ / ✗).
Kernel tables cover CUDA / Triton kernels; PyTorch and cuBLAS steps appear only in the figures.
Measurements compare PyTorch compiled / cuEquivariance / Anthropic / ours, CUDA-graph or compiled
timing only (`benchmarks/cautions.md`). Shapes come from the model shape registry
(`src/miniworld_engine/kernels/registry/registry_module.csv`). Full template:
[template.md](template.md).

H100 uses this format. The A100 and B200 pages still carry the previous op-per-row table
(its **judgement** column is the maintainer's) until they are converted.
