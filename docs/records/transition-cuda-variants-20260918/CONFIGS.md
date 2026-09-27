# Selected native configurations

Core winners: bounded sweep at L384. Forward and gate backward are selected independently. The shared native LN uses the independent reduction sweep. L768 reuses these configurations.

| Variant | D | Direction | BK | BN | BO | M groups | Output groups | Stages | Min blocks | Core ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| streamed_k | 128 | forward | 64 | 128 | 64 | 1 | 1 | 2 | 1 | 0.1737 |
| streamed_k | 128 | gate backward | 64 | 32 | 128 | 1 | 1 | 3 | 1 | 0.4008 |
| streamed_k | 256 | forward | 64 | 64 | 64 | 1 | 1 | 3 | 1 | 0.6235 |
| streamed_k | 256 | gate backward | 64 | 64 | 64 | 1 | 1 | 3 | 1 | 1.0012 |
| streamed_k | 384 | forward | 64 | 64 | 64 | 1 | 1 | 2 | 1 | 1.7663 |
| streamed_k | 384 | gate backward | 64 | 64 | 64 | 1 | 2 | 3 | 1 | 1.8892 |
| streamed_k | 512 | forward | 128 | 64 | 64 | 1 | 2 | 2 | 1 | 3.2347 |
| streamed_k | 512 | gate backward | 64 | 64 | 64 | 1 | 2 | 3 | 1 | 2.8127 |
| full_k | 128 | forward | 128 | 64 | 64 | 1 | 1 | 2 | 1 | 0.1746 |
| full_k | 128 | gate backward | 128 | 32 | 128 | 2 | 1 | 2 | 1 | 0.3531 |
| full_k | 256 | forward | 256 | 64 | 128 | 2 | 1 | 2 | 1 | 0.5797 |
| full_k | 256 | gate backward | 256 | 32 | 128 | 1 | 1 | 2 | 1 | 0.8749 |
| full_k | 384 | forward | 384 | 64 | 64 | 1 | 2 | 1 | 1 | 1.5255 |
| full_k | 384 | gate backward | 384 | 32 | 64 | 1 | 2 | 1 | 1 | 2.0514 |
| full_k | 512 | forward | 512 | 32 | 64 | 1 | 2 | 2 | 1 | 3.2058 |
| full_k | 512 | gate backward | 512 | 32 | 64 | 2 | 2 | 1 | 1 | 3.5192 |

BM = 64 × M groups. Gate backward internally uses one output group; BO/output-group values in historical forward-seed records do not create different backward work. `candidates(backward=True)` canonicalizes these redundant axes. Core milliseconds exclude LN and common cuBLAS; use RESULTS.md for complete module times.

## Candidate axes

- Streamed BK: 64/128/256 below D. Full BK: D or next power of two; exact BK384 is validated for native CUDA.
- BN: 32/64/128; BO: 64/128; M/output groups: 1/2; pipeline stages: 1/2/3; launch-bound minimum blocks: 1/2.
- Native WGMMA fixes the row atom at 64; Triton BM16/32 layouts are not equivalent native WGMMA candidates. Unsupported shared-memory configurations remain explicit exclusions.
- The measured seed set uses minimum blocks=1. Public candidates include 2, but that axis has not been exhaustively timed. See tuning JSON for each measured/excluded/failed configuration.

## Shared LN

| D | Warps | Waves | Vector bytes | Reduction threads | Channels per reduction CTA |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 8 | 8 | 256 | 1 |
| 256 | 4 | 8 | 8 | 128 | 1 |
| 384 | 8 | 2 | 8 | 256 | 4 |
| 512 | 4 | 4 | 16 | 256 | 4 |
