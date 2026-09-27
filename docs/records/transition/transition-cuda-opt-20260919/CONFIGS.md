# Measured configurations

Bounded search, not an exhaustive optimum certificate. Public configuration axes and fusion boundaries are unchanged. Forward and backward configurations are independent. Minimum blocks is 1 in these selections. M = L*L. L384-selected configs are reused at L768.

| D | Schedule | Direction | BK | BN | BO | M groups | N groups | Stages |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 128 | full_k | forward | 128 | 64 | 64 | 1 | 1 | 2 |
| 128 | streamed_k | forward | 64 | 128 | 64 | 1 | 1 | 2 |
| 256 | full_k | forward | 256 | 64 | 128 | 2 | 1 | 2 |
| 256 | streamed_k | forward | 64 | 64 | 64 | 1 | 1 | 3 |
| 384 | full_k | forward | 384 | 64 | 64 | 1 | 2 | 1 |
| 384 | streamed_k | forward | 128 | 64 | 64 | 1 | 2 | 2 |
| 512 | full_k | forward | 512 | 32 | 64 | 1 | 2 | 2 |
| 512 | streamed_k | forward | 128 | 64 | 64 | 1 | 2 | 2 |
| 128 | full_k | backward | 128 | 32 | 128 | 2 | 1 | 2 |
| 128 | streamed_k | backward | 64 | 32 | 128 | 1 | 1 | 3 |
| 256 | full_k | backward | 256 | 32 | 128 | 1 | 1 | 2 |
| 256 | streamed_k | backward | 64 | 64 | 64 | 1 | 1 | 3 |
| 384 | full_k | backward | 384 | 32 | 64 | 1 | 2 | 1 |
| 384 | streamed_k | backward | 64 | 64 | 64 | 1 | 2 | 3 |
| 512 | full_k | backward | 512 | 32 | 64 | 2 | 2 | 1 |
| 512 | streamed_k | backward | 64 | 64 | 64 | 1 | 2 | 3 |

M groups set BM=64*M groups. Backward always has one output group; its N-groups value is ignored. The current row-only grid has no GROUP_M tile-order axis. LayerNorm settings are unchanged from `transition-cuda-variants-20260918/norm-selections.json`.
