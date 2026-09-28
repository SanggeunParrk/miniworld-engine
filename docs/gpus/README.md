# GPUs

One page per GPU: how to run on it, then **which operation is finished for which shapes**.
The maintainer makes that call; the pages record it.

| GPU | arch | cluster / partition | page | summary |
|---|---|---|---|---|
| H100 80GB HBM3 | sm90 | cssb, `h100` partition | [h100.md](h100.md) · [module dispatch](h100-dispatch.md) | hand CUDA for TriMul, Transition (n=4), TriAttn training, OPM/PWA, token DiT; Triton elsewhere |
| B200 | sm100 | cssb (same cluster as H100/A100) | [b200.md](b200.md) | Triton only in this repo (CUDA work is happening elsewhere) |
| A100 80GB PCIe | sm80 | cssb (same cluster as H100/B200) | [a100.md](a100.md) | Triton only |
| RTX A6000 / A5000 | sm86 | `cssb-master`, `gpu` partition | [ampere-workstation.md](ampere-workstation.md) | Triton only (no completion table) |

## Common to all


- Nothing runs on a login node (see `.claude/CLAUDE.md`). Always pass `--mem`.
- Install the pixi env on a CPU allocation of the cluster you will run on
  (`pixi install`, `pixi run fix-te-cu12`, `miniworld-engine dev install-flash --arch <sm>`).
  Conda prefixes are absolute paths: an env built in one checkout cannot be moved to another.
- Give each GPU/checkout its own JIT cache (`MINIWORLD_ENGINE_JIT_ROOT`); prebuilt
  TriangleAttention `.so` files are tied to the torch/python they were built with and the
  CUDA path silently falls back to Triton when they do not match
  (rebuild: see [h100.md](h100.md)).
- Autotune caches are keyed by GPU name **and** toolchain (`env_identity`): a cache built
  under another torch/triton is ignored. Check with `miniworld-engine dev cache-status --gpu <name>`.

## Completion tables


| column | who fills it | meaning |
|---|---|---|
| op, mode, shapes | agent | the operation, `inference` / `training` (fwd+bwd), and the exact shape range the row covers: widths `D`, lengths `L`, batch, dtype, direction |
| implementation | agent | what the default `engine_backend="auto"` dispatch runs for these shapes on this GPU (`CUDA <module>`, `Triton`, …), verified by counting entry-point calls |
| vs Triton / vs baseline | agent | measured speedup, CUDA-graph or compiled timing only, with the baseline named |
| evidence | agent | link to the record (`docs/records/...`) or test that backs the numbers |
| **judgement** | **maintainer** | `done` · `partial` · `not started` · `won't do` — left blank by agents |
| note / date | maintainer | why, and when it was judged |

Rules:

- Agents may add rows and update the factual columns when dispatch or a measurement changes,
  but never write the **judgement** column.
- A row covers one contiguous shape range with one implementation. When the implementation
  differs inside a range (e.g. uni TriMul training: CUDA at D128, Triton at D64/256/384),
  split the row.
- Numbers without a GPU, a baseline and a timing mode are not allowed (see
  `benchmarks/cautions.md`).
- Shapes come from the model shape policy (`src/miniworld_engine/kernels/registry/registry_module.csv`,
  `docs/kernels/autotune-training-shape-policy.md`); training lengths are L384/L768.
