# A5000 and A6000 model cache rebuild

Status: **building; not yet certified complete**.

Both builds use the 107-row model registry, covering 8,242 inference/training calls and 1,331 required kernel keys. Existing valid cache entries and completed shards are reused. The default module plan adds no diagnostic per-op sweep.

| GPU | Node | Job | GPUs | Usable keys before build | Missing keys | Selected module calls |
|---|---|---:|---:|---:|---:|---:|
| A5000 | gpu02 | 1707734 | 6 | 649 | 682 | 319 |
| A6000 | gpu04 | 1707792 | 3 | 682 | 649 | 301 |

The previous broad per-op jobs 1706622 and 1706623 were stopped. Their completed shards were merged before planning the replacement builds. Historical shards and pre-build cache snapshots remain local. A6000's first replacement stopped before tuning because dispatch files changed during verification; the final source was then re-derived successfully (8,242 calls, zero errors).

Final dispatch identity: `981ac7988e7671a60c707676f4d2121ee5822ae52611d284c6f857bdef53832e`.

Four RMSNorm cache files contained 284 obsolete implementation-profile keys outside the current model plan. Complete originals were archived locally before removing those unusable keys; 78 valid entries per A5000 file and 100 per A6000 file remain. Required model keys were not removed, and no validity stamp was rewritten to bless an old measurement.

Publication is gated on successful exits from both builds, zero missing required keys on each GPU, Ruff, ty with the project Python environment, and the full CPU test suite. A failed or incomplete build must not be described as complete. Logs, shards, backups and publication receipts are under the ignored `.bench/model-cache-20260916/` directory.

This rebuild preserves token/atom bucket rules and tile/warp/stage candidate grids. It does not constitute a new end-to-end latency or structural-accuracy benchmark.
