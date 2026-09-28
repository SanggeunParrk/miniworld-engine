# Completion status per GPU

One page per GPU says, for each operation and shape range, **whether the work on that GPU is
finished**. The maintainer makes that call; these pages record it.

| GPU | arch | page | summary |
|---|---|---|---|
| H100 80GB HBM3 | sm90 | [h100.md](h100.md) | hand CUDA for TriMul, Transition (n=4), TriAttn training, OPM/PWA, token DiT; Triton elsewhere |
| B200 | sm100 | [b200.md](b200.md) | Triton only in this repo (CUDA work is happening elsewhere) |
| A100 80GB PCIe | sm80 | [a100.md](a100.md) | Triton only |

## How a row is written

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
  `docs/benchmarks/cautions.md`).
- Shapes come from the model shape policy (`src/miniworld_engine/kernels/registry/registry_module.csv`,
  `docs/autotune/training-shape-policy.md`); training lengths are L384/L768.
