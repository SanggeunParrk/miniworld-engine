# Working on each GPU

How to reach, set up and run this repo on each GPU. What is finished per GPU is in
[docs/status/](../status/README.md).

| GPU | arch | cluster / partition | page |
|---|---|---|---|
| H100 80GB HBM3 | sm90 | cssb cluster, `h100` partition | [h100.md](h100.md) · [module dispatch](h100-dispatch.md) |
| B200 | sm100 | cssb cluster (same as H100/A100) | [b200.md](b200.md) |
| A100 80GB PCIe | sm80 | cssb cluster (same as H100/B200) | [a100.md](a100.md) |
| RTX A6000 / A5000 | sm86 | `cssb-master` cluster, `gpu` partition | [ampere-workstation.md](ampere-workstation.md) |

Common to all:

- Nothing runs on a login node (see `AGENTS.md`). Always pass `--mem`.
- Install the pixi env on a CPU allocation of the cluster you will run on
  (`pixi install`, `pixi run fix-te-cu12`, `miniworld-engine dev install-flash --arch <sm>`).
  Conda prefixes are absolute paths: an env built in one checkout cannot be moved to another.
- Give each GPU/checkout its own JIT cache (`MINIWORLD_ENGINE_JIT_ROOT`); prebuilt
  TriangleAttention `.so` files are tied to the torch/python they were built with and the
  CUDA path silently falls back to Triton when they do not match
  (rebuild: `.bench/v220-env/triattn_rebuild.py`, see [h100.md](h100.md)).
- Autotune caches are keyed by GPU name **and** toolchain (`env_identity`): a cache built
  under another torch/triton is ignored. Check with `miniworld-engine dev cache-status --gpu <name>`.
