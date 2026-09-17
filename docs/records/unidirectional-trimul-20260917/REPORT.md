# Single-direction Triton TriMul optimization — 2026-09-17

## Implemented

- Outgoing and incoming BF16 training reuse the bidirectional F567 output projection + gate + dropout/residual kernel.
- B9+B10 reuse the dual input-gradient GEMM; B11+B12 reuse input LayerNorm backward + residual gradient.
- Input masking, saved forward tensors, parameter gradients, LayerNorm epsilons and dropout semantics are preserved.
- FP32 module execution retains its existing PyTorch fallback. The existing inference route is unchanged.
- Full-grid tuning exposed gate-dot shared-memory/layout errors. The shared F567 now transposes the gate GEMM only for tall output tiles (M tile > N tile); square/wide tiles retain the native orientation. No candidate blacklist or fixed tile was added.

## Measurement

One actual `TriangleMultiplication` module via `benchmarks.runners.bench.bench_module_triangle_multiplication`: H100 80GB, B=1, d_pair=d_hidden=128, BF16 mixed precision, mask probability 0.2, forward+backward without optimizer. Fixed-shape `torch.compile(dynamic=False)`. Old single-direction function comes from merged main `4d28918d16730370b57762acb518a725daaaa6b9`; new function comes from this worktree. Each pair uses the same GPU and 12 alternating rounds with 100 ms timer samples; table values are medians. Compilation/autotuning is outside timing.

All three new kernel cache lookups are asserted to hit the completed targeted full-grid tunes. Other existing kernels use the same cache/fallback policy in both arms. This measures the complete module, not a full MiniWorld training step. Concurrent jobs use separate GPUs.

## Dropout 0.25, CUDA graph disabled

| L | Direction | Before (ms) | After (ms) | Speedup | Time reduction |
|---:|---|---:|---:|---:|---:|
| 128 | outgoing | 1.683816 | 1.683240 | 1.0003x | 0.03% |
| 128 | incoming | 1.719016 | 1.745840 | 0.9846x | -1.56% |
| 384 | outgoing | 1.741488 | 1.712048 | 1.0172x | 1.69% |
| 384 | incoming | 1.680976 | 1.624520 | 1.0348x | 3.36% |
| 768 | outgoing | 4.295216 | 3.847960 | 1.1162x | 10.41% |
| 768 | incoming | 4.291496 | 3.849224 | 1.1149x | 10.31% |

## Dropout 0, manual CUDA graph

| L | Direction | Before (ms) | After (ms) | Speedup | Time reduction |
|---:|---|---:|---:|---:|---:|
| 128 | outgoing | 0.181216 | 0.170096 | 1.0654x | 6.14% |
| 128 | incoming | 0.181328 | 0.169952 | 1.0669x | 6.27% |
| 384 | outgoing | 1.146928 | 1.043840 | 1.0988x | 8.99% |
| 384 | incoming | 1.146424 | 1.042200 | 1.1000x | 9.09% |
| 768 | outgoing | 4.231664 | 3.818992 | 1.1081x | 9.75% |
| 768 | incoming | 4.231920 | 3.819376 | 1.1080x | 9.75% |

## L384 PyTorch comparison (dropout 0.25, graph disabled)

| Direction | PyTorch (ms) | New Triton (ms) | PyTorch / Triton |
|---|---:|---:|---:|
| outgoing | 2.467536 | 1.712048 | 1.4413x |
| incoming | 2.474368 | 1.624520 | 1.5231x |

## Validation and delivery

- 69 regression cases passed: F567 tiling/strides/tails and single-direction outputs/input/parameter gradients with masking and dropout; BF16 fast path and FP32 fallback.
- All 1,728 shape-pruned F567 schedules passed a numerical check on 512 rows at L384, d128. This samples rows; it is not exhaustive validation of every model shape.
- Previously failing square tile 128x128x32, one warp, two stages: compute-sanitizer memcheck reports zero errors.
- Module harness output and gradient checks passed for every table entry against its FP32 reference. A separate installed-package run (`installed-L384.json`, package paths recorded) confirmed all three new cache hits and 1.0214x outgoing speedup at L384.
- Source and targeted cache patches are applied to the MiniWorld cu128 installation and persisted in both MiniWorld/team-gm patch stacks. Full stack reapplication is idempotent.
- Targeted tuning covers the three new kernels at L128/384/768, d128. It does not claim completion of the separate global cache build or every bidirectional shape. The L128 and L768 F567 sweeps respectively excluded one and two configurations that hit the compiler time budget; L384 timed all 1,728 eligible configurations.
- The F567 source change invalidates old F567 source identities, including bidirectional KP256 cache entries. Those shapes use the normal runtime tuning fallback until rebuilt with this source. The independent global build uses its immutable older main snapshot; its F567 cache must not be represented as a tune for this revision.
- The pre-existing four-GPU training and two-GPU global Triton cache job were not restarted or changed. Already-running Python processes keep their loaded implementation.
- Source commit: `3dd92561c46a755e010195b94b8a9db6d769571f` on `perf/unidirectional-trimul-fusion`.

## Raw measurements

- [outgoing-L128-tuned.json](outgoing-L128-tuned.json)
- [outgoing-L128-graph-tuned.json](outgoing-L128-graph-tuned.json)
- [incoming-L128-tuned.json](incoming-L128-tuned.json)
- [incoming-L128-graph-tuned.json](incoming-L128-graph-tuned.json)
- [outgoing-L384-tuned.json](outgoing-L384-tuned.json)
- [outgoing-L384-graph-tuned.json](outgoing-L384-graph-tuned.json)
- [incoming-L384-tuned.json](incoming-L384-tuned.json)
- [incoming-L384-graph-tuned.json](incoming-L384-graph-tuned.json)
- [outgoing-L768-tuned.json](outgoing-L768-tuned.json)
- [outgoing-L768-graph-tuned.json](outgoing-L768-graph-tuned.json)
- [incoming-L768-tuned.json](incoming-L768-tuned.json)
- [incoming-L768-graph-tuned.json](incoming-L768-graph-tuned.json)

Other evidence: `tuning-progress.json`, `grid-validation.json`, `f567-strict-tests.log`, `sanitizer-strict.log`, `cache-published.json`.
