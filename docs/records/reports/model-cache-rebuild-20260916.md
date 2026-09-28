# A5000 and A6000 model cache rebuild

Status (2026-09-17): **A6000 fully covered; A5000 published with nine explicitly recorded keys unbuilt because their parent training workloads exceeded GPU memory.** This is not a claim that every workload fits or that both original builds exited successfully.

| GPU | Original job | Recovery job | Required keys | Usable keys | Unbuilt due to parent OOM |
|---|---:|---:|---:|---:|---:|
| A5000 | 1707734 | 1709340 | 1,331 | 1,322 | 9 |
| A6000 | 1707792 | 1709339 | 1,331 | 1,331 | 0 |

The original A5000 run finished 311 calls and failed eight (seven OOM, one illegal-memory access); A6000 finished 300 and failed one (OOM). Both exited 1, and the strict automatic publisher correctly stopped without pushing. The exact failed A6000 projected-attention unit (dims4, L768, training) passed on an isolated GPU. The A5000 TriMul recovery also passed and filled the remaining non-OOM cache key. Completed shards were preserved.

## A5000 capacity exclusions

The seven OOM parent calls are augmented-attention training at L4096 and L8192 with BF16 and FP32 attention cores (four calls), plus projected-attention training at dims4/L640, dims3/L768 and dims4/L768 (three calls). They leave seven `augmented_attention_bwd_split_triton` keys and two `augmented_attention_bwd_reduce_triton` keys unbuilt. Each missing key was matched against the exact failed parent unit in the derived plan. These are not assertions that the individual kernels are unsupported.

The nine keys remain in the required plan; they were not removed, substituted with a smaller augmentation workload, or marked valid without measurements. Exact keys, unit labels, OOM allocation messages and log paths are in [the machine-readable qualification](model-cache-rebuild-20260917.json). GPU-memory exhaustion is an accepted capacity exclusion for this publication.

## Illegal-memory incident checks

The original A5000 TriMul failure involved `gated_projection_bwd_gate_dropres_triton` at L512, width384, BLOCK_M1=128, BLOCK_K=256, two warps and one stage. The exact raw candidate passed compute-sanitizer and sampled numerical checks (job1709337). Three original-module forward/backward runs with that forced candidate also passed compute-sanitizer with zero errors (job1709338). The original cache-build unit then passed normally (job1709340). **The original fault was not reproduced; its root cause is not established.** No candidate was removed to conceal the incident.

## Plan and validation

The unchanged plan covers 107 model-shape rows, 8,242 inference/training calls and 49 kernels. Dispatch source identity: `981ac7988e7671a60c707676f4d2121ee5822ae52611d284c6f857bdef53832e`. Token/atom buckets and tile/warp/stage grids are unchanged. Existing valid entries were reused. Four RMSNorm cache files had 284 obsolete implementation-profile keys outside the current plan; their originals were archived locally before removing those unusable keys, preserving valid entries. No validity stamp was relabeled.

Final CPU-node verification passed Ruff, ty, and pytest (3,343 passed, 35 skipped; GPU-marked tests excluded). Fresh coverage found no invalid required entries and no unexplained missing keys. This qualification does not constitute new end-to-end latency or structural-accuracy validation. Logs, shards, backups and receipts remain under the ignored `.bench/model-cache-20260916/` and `.bench/model-cache-20260917/` directories.
