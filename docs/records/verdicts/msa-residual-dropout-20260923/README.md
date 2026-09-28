# OPM residual / PWA backward dropout fusion — 2026-09-23

Default H100 auto paths, B1, S1024, MSA64/pair128, BF16. OPM includes pair residual; PWA includes residual and training dropout0.15 with RNG. Static compile + CUDA graph, 7×50 replay medians, same GPU before/after. These are paired new measurements, not subtraction from the earlier Claude eager table.

| Module | L | Mode | Before ms | After ms | Speedup |
|---|---:|---|---:|---:|---:|
| opm | 384 | inference | 0.711 | 0.685 | 1.038× |
| opm | 384 | training | 2.006 | 1.997 | 1.005× |
| opm | 768 | inference | 2.597 | 2.523 | 1.030× |
| opm | 768 | training | 7.461 | 7.405 | 1.007× |
| pwa | 384 | inference | 0.402 | 0.405 | 0.992× |
| pwa | 384 | training | 1.586 | 1.527 | 1.039× |
| pwa | 768 | inference | 1.304 | 1.301 | 1.002× |
| pwa | 768 | training | 4.112 | 3.987 | 1.031× |

## Changes

- OPM: add the BF16 pair residual in the final CUDA epilogue before its TMA store. Round the projected update first, then add the residual, preserving separate-op BF16 numerics. Residual gradient is identity, without an extra CUDA pass. Nonstandard residual dtype/broadcast shapes retain the general add.
- OPM inference: a separate opaque entry point omits LN-statistics allocation/stores; no training activations are returned. Weight conversion stays live per invocation so weight updates remain valid.
- PWA forward already fused dropout application and residual. Backward now masks/scales TMA-loaded dres tiles in shared memory before both du and dWo read them; removes the materialized [S,N,64] gradient and its elementwise HBM passes. Input residual gradient continues to use the original unmasked dres. The small dropout RNG/mask creation remains in the forward.
- OPM has no dropout in the model; none was introduced. GEMM tiling, producer/consumer roles and dW reduction structure remain unchanged.

## Validation

17 H100 tests passed, including L384/768 bit-exact fused vs separate residual outputs/all gradients and bit-exact PWA masked vs materialized-gradient glue outputs. Fullgraph compilation passed. OPM memcheck: 0 errors. PWA shared-memory racecheck: 0 errors/0 warnings. No mutation of residual or incoming gradient.

cuobjdump: OPM epilogue 102 registers before/after; PWA glue 168 before/after, unchanged shared-memory layout, stack/local memory 0. Native tile/config space is unchanged. This is not a new exhaustive tuning sweep or SoL measurement.

OPM training changes below 1% are small and should not be promoted as a robust training speedup. PWA inference code was unchanged; its sub-1% fluctuations are measurement variation.

[Machine-readable results](results.csv) · [Source and validation manifest](manifest.json) · [Binary resource usage](resources.json)
