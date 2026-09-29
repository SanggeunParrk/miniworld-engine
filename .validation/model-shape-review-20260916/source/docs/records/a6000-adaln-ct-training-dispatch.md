# A6000 AdaLN / ConditionedTransition training attribution — 2026-09-11

The main AdaLN training deficit comes from its **training forward dispatch**, not its backward.
The width-only `_pick_fwd(nx)` chooses fused Triton at nx=768. On this measured A6000 shape,
that choice is slower than the existing cuBLAS + Triton epilogue path. Inference uses a different
implementation and a different augmentation count; this experiment changes no inference path.

All rows below use the same physical A6000, L384, A48, hidden/condition widths 768/384,
BF16 mixed precision, TF32 off, depth1, actual torch.compile, CUDA Graph disabled, and
forward+backward without optimizer. These two modules have no dropout. Calls use the existing
native module benchmark functions and unchanged timer. Each row is a median of three samples
within its own process; they are not three independent process repetitions. No cache misses or
tuning were permitted. Cached tile files and production source were unchanged.

| Training configuration | AdaLN ms | ConditionedTransition ms |
|---|---:|---:|
| PyTorch | 1.211 | 6.060 |
| MiniWorld current default | 1.613 | 6.204 |
| MiniWorld, only AdaLN training forward forced to existing cuBLAS path | 1.279 | 5.887 |
| PyTorch AdaLN + MiniWorld transition tail | — | 5.767 |
| MiniWorld AdaLN + PyTorch transition tail | — | 6.502 |

The last three rows are diagnostic overrides in separate processes, not changes to production
defaults. They identify dispatch as a concrete performance problem and do not qualify a new
production dispatch policy across shapes/GPUs. No new numerical qualification of the overrides
was performed in this attribution run.

Separate ten-step profiler runs measured the following CUDA work inside each compiled forward
and backward. These instrumented numbers are not the unprofiled wall-clock timings above.

| AdaLN GPU work | PyTorch ms | MiniWorld ms |
|---|---:|---:|
| Training forward | 0.443 | 0.814 |
| Backward | 0.749 | 0.784 |

MiniWorld's `_adaln_fwd_gate_kernel` alone costs approximately 0.720 ms per step. The alternative
cuBLAS projection costs approximately 0.241 ms and its `_epilogue_train_kernel` 0.198 ms.
Thus fusing the projection with normalization/gating is slower for this workload despite
using fewer launches. The retained backward and other costs explain why the AdaLN cuBLAS
variant still trails compiled PyTorch slightly.

ConditionedTransition contains AdaLN. Replacing only that component improves its complete
forward+backward from 6.204 to 5.767 ms. Replacing only the transition tail with PyTorch makes
it slower (6.502 ms). This controlled component swap, rather than subtracting independently
measured module timings, identifies AdaLN as the source of its net disadvantage here.

## SWA clarification

Both benchmark implementations use the same FlashAttention backend (FA2 on A6000) for sliding
window attention, and the same QKV/output linear projections and mask/window semantics.
MiniWorld supplies Triton Q/K RMSNorm, 3D RoPE and sigmoid-times-output kernels around it;
PyTorch supplies equivalent torch operations compiled by Inductor. It is not a separately
implemented MiniWorld attention core.

Git records RMSNorm and gate wiring on 2026-09-01 (`a6cd560f`, `86bd40db`), and RoPE wiring on
2026-09-02 (`5245c3e4`). The subsequent independent-PyTorch-baseline fix added the implementation
branch so the PyTorch-labelled benchmark stopped calling those MiniWorld kernels. FA2 graph
packing and the FA4 API correction are shared wrapper changes, not a different attention
algorithm for the two benchmark columns.

[Full timings, profiler groups, inputs, compile evidence and provenance](a6000-adaln-ct-training-dispatch.json).
Raw traces: `/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-adaln-ct-training-review/`.
