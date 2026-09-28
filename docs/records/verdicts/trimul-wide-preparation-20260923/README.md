# Wide TriMul preparation reuse — 2026-09-23

Bidirectional H100 training at D64/256/384/512, L384/768. D128 keeps its existing path. Single-direction wide kernels are outside this change.

## Changes

- Weight packing: 3 calls → 1 in forward, retained for backward.
- FP32 mask conversion: 2 → 1, owned by each forward.
- D512 separate input LN: 2 → 1.
- Corrected real/fake x_n shape mismatch for compile.
- No parameter/optimizer layout migration. Extra retained memory: 16*D² + 4*L² bytes; maximum 6.25 MiB.

## Timing

H100 node01, BF16, residual and fixed 25% dropout mask; RNG excluded. Direct training API, static fullgraph compile + CUDA graph. Median of 8 alternating rounds × 40 replays. Not the complete nn.Module/optimizer step.

| D | L | Before ms | After ms | Time reduction |
|---:|---:|---:|---:|---:|
| 64 | 384 | 1.3641 | 1.3541 | +0.73% |
| 64 | 768 | 5.0183 | 5.0286 | -0.21% |
| 256 | 384 | 5.4418 | 5.4667 | -0.46% |
| 256 | 768 | 22.0254 | 22.0351 | -0.04% |
| 384 | 384 | 9.5835 | 9.5975 | -0.15% |
| 384 | 768 | 39.4668 | 39.4464 | +0.05% |
| 512 | 384 | 14.6995 | 14.5532 | +1.00% |
| 512 | 768 | 60.9349 | 60.3142 | +1.02% |

D512 improves about 1%; other widths have small mixed changes, including regressions. No broad speedup is established. Baseline snapshots are preserved; the baseline loader applies only a copy-free x_n.reshape_as(x) to fix old compile metadata.

## Validation

- Job 16286: ten width/length oracle tests passed before hitting the old baseline compile defect.
- Job 16290: eight before/after cases passed, including independent forward ownership, compiled CUDA graph and changed live inputs/weights/masks. All relative L2 errors <2e-6.
- Job 16291: four permanent preparation-count, FP32 mask ownership and fullgraph gradient tests passed.
- pack count=1 at every width; explicit normalize count=1 only D512.
- Evidence: results.json, check.py, job logs; permanent tests: tests/integrations/test_trimul_wide_preparation_gpu.py.
