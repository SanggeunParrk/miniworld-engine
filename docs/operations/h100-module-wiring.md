# H100 module dispatch after 2.0.0

`implementation="miniworld"` with the default `engine_backend="auto"` selects the
packaged H100 implementations below. No research-directory import, external
TriMul payload, or MSA opt-in environment variable is required. Configure the
policy before constructing/compiling models.

| Module | Inference | Training | Native contract |
|---|---|---|---|
| Bidirectional TriMul | Anthropic-derived native K1 → two cuBLAS contractions → K3/residual | Selected CUDA forward + B1–B4 + four cuBLAS contractions + B7–B12 | BF16, batch 1; training L384/768, direction hidden=D, D64/128/256/384/512; D128 requires a full 132-SM H100 |
| Single-direction TriMul | Packaged K1 → contraction → K3/residual | Native K1/K3, fused CUDA B1, two cuBLAS gradient contractions, streamed producer-consumer B7 | Training: BF16, batch 1, D=hidden=128, L384/768, full 132-SM H100; both outgoing and incoming. Inference: width/hidden pair in the native tile table |
| Transition | Existing residual-fused hand-CUDA path | Same forward with native backward | BF16, n=4, D64/128/256/384/512 and each kernel's resource guards |
| OuterProductMean | Packaged OPM with residual in the CUDA epilogue, no LN statistics saved | Residual-fused forward and native backward; residual gradient is passed through | BF16, batch 1, MSA64/hidden32/pair128, L multiple of 64, MSA depth multiple of 256; normalization before projection; no interchain masking |
| MSAPairWeightedAveraging | Packaged forward without training saves | Packaged forward/backward, residual and row dropout | BF16, batch 1, MSA64/pair128, 8 heads × 32, L multiple of 128, even MSA depth |
| Token DiT | Fused inference row kernels, attention/gate and GEMM path | Existing general autograd route | BF16/FP32, batch 1, single768/condition384/pair128, 16 heads, expansion1536, L multiple of 128, shared sample conditioning, no QK norm |

Unsupported contracts retain the general implementation. `auto` selects these
connected bidirectional CUDA training ports at all five TriMul widths; it does **not** assert
that all widths are faster. D128 has the selected optimization; additional
D64/256/384/512 performance tuning remains deferred. Inference width/hidden
coverage follows the packaged K1/K3 table and differs from training coverage.

## Retained values and execution

TriMul training retains input `x_n`, left/right, `tri`, and packed front weights.
Bidirectional D128 additionally retains output-LN statistics; single-direction
D128 recomputes output-LN statistics in B1. Projection/gate intermediates are recomputed in backward. Every forward
owns its saved tensors, and original parameters participate in PyTorch's
saved-tensor version checking. Parameter packing is live on every call, including
CUDA graph replay. There is no process-global activation or weight snapshot.

The module entry points have opaque operators with shape-only implementations,
so supported paths work under `torch.compile(fullgraph=True)`. MSA dropout depends
on `module.training`, including a training-mode call under `torch.no_grad()`.
CUDA sources and includes ship in the wheel; nvcc builds into the user cache,
never into the installed source tree. Warm up each shape before CUDA capture.

OPM has no dropout in the model. Its BF16 pair residual is added after rounding
the projected update, preserving the separate add's numerics. Other residual
dtypes or broadcast shapes retain the general add. PWA applies dropout only to
the update: its forward residual/dropout was already fused; backward now masks
the TMA-loaded gradient tiles in shared memory before both input and weight
gradient GEMMs consume them. The residual gradient stays unmasked. This removes
the separate `[S,N,64]` masked-gradient allocation and HBM write/read.

## Policy and tuning

The existing explicit `engine_backend="triton"` comparison option remains an
opt-out of these automatic paths, including residual-fused Transition.
`MINIWORLD_OPM_TRAIN=0`, `MINIWORLD_PWA_TRAIN=0` and
`MINIWORLD_PWA_INFER=0` disable their respective integrations.
`trimul_h100_training_widths` can restrict the connected CUDA training widths;
its default includes all five widths above.

This change connects already-selected implementations. It is **not** a complete
retune or rebuild of all native/Triton caches. Native TriMul ships measured tile
selections; Token DiT retains its local first-use GEMM selection. General Triton
paths keep the engine's cache system. Derived build plans model these native
composites without invoking nvcc on fake tensors.

The research histories remain available under `experiments/`; source hashes for
the selected CUDA bodies are in
`kernels/trimul_inproj/cuda/h100_sources/PROVENANCE.json` within the package.
Anthropic attribution and Apache-2.0 terms apply as described in
[third-party notices](../../THIRD_PARTY_NOTICES.md).

## Validation

[Validation record](../../verdicts/h100-module-wiring.json): 46 H100 test
executions passed, including a separate wheel installation on CUDA device 1,
all ten TriMul training width/length combinations, masked/dropout gradients,
live-weight CUDA graph replay, token DiT inference graphs, and MSA repeated
compiled calls without recompilation, including noncontiguous inputs. Source/config provenance hashes were
verified. The record distinguishes the initial CPU audit failures from the
successful checks after fixing their static analysis.

## Production-module performance audit

The [four-backend H100 comparison](../../verdicts/version-compare-20260923/README.md)
records the actual `miniworld` / `auto` module paths with masks, residuals and
training dropout, alongside PyTorch, cuEquivariance and the literal v1.0.0 tag.
The audit found that Transition's availability probe attempted to trace the
extension builder under `torch.compile(fullgraph=True)`. It now defers the build
to the opaque runtime launch under Dynamo as well as FakeTensor dispatch. Four
CPU regression checks cover both narrow and wide availability probes; the GPU
comparison records the selected forward and backward kernel names.

## D128 TriMul parameter layout and optimizer resume

The four front matrices of a `miniworld` bidirectional D128/H128 module retain
legacy parameter names and `[out, in]` shapes, with column-major storage. Native
backward reads their transposes as views and returns gradients in the same
storage order. Forward's 256 KiB packed-weight tensor is saved per invocation
and reused by backward. No weights are cached across training steps. Output
projection/gate layouts remain unchanged; their necessary conversions remain.
Existing row-major weights (including `load_state_dict(assign=True)` and external
weights-as-arguments callers) still execute correctly through preparation.

Model checkpoints preserve names, shapes and values. Ordinary `load_state_dict`
copies into the module's new strides. **Old optimizer moments need a one-time
layout migration**, especially with fused AdamW:

```python
from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
align_optimizer_state_layout_(optimizer)  # before capturing any training graph
```

This only copies same-shape state tensors whose strides differ from their
parameter; it preserves values and scalar step counters. Fresh optimizers
already initialize state with the right layout. See the
[paired preparation benchmark](../../verdicts/trimul-preparation-20260923/README.md).

## Single-direction training

Both outgoing and incoming use the native route automatically under the contract
above. A single invocation performs one forward contraction and two backward
contractions; it never computes the other direction or normalizes across 2D.
K1 saves affine `x_n`; K3 fuses output LN, projection, gate, dropout and residual.
B1 recomputes output LN/projection/gate on chip and emits dTri, dGate and parameter
gradients. B7 reuses the bidirectional producer-consumer design with eight
32-channel producers (H=128), four 16-KiB derivative planes and eight consumers
per group. Its bounded ring buffer is not a full L² activation cache.

The input/output LN derivatives are separately checked against FP64 using the
actual incoming BF16 gradients, including zero affine scales. Both directions
also cover static compile, CUDA graph replay with changed weights/masks,
outstanding-forward ownership and fully dropped updates.

Selected schedules live in `h100_sources/single_output/selection.json`.
The measured local search covers 6 K1 tiles, 6 K3 schedules, 4 B1 CTA counts and
10 B7 producer/consumer/ring combinations per length. This is a bounded search,
not a claim of exhaustive configuration tuning or 90% speed-of-light.
D64/256/384/512 **single-direction training** retains the general path.
See [measurements and validation](../../verdicts/trimul-single-20260923/README.md).

## Preparation reuse at D64/256/384/512

Bidirectional training now packs front weights once in forward and retains the
pack and converted FP32 mask for its own backward. This replaces three pack
calls per forward+backward with one. D512's separate input LN now runs once,
rather than both during plan construction and during forward. Existing weight
strides and optimizer-state layouts at these widths are unchanged; the D128
layout migration above does not apply to this extension.

The wide forward also returns `x_n` with the declared batch dimension. Previously
its real 3D tensor disagreed with the opaque operator's 4D fake metadata and
could fail compiled execution. Revised opaque operator names keep old Inductor
cache artifacts from reusing the previous saved-tensor contract.

Extra retained memory: BF16 packed weights of `16*D²` bytes plus the FP32 pair
mask of `4*L²` bytes (at most 6.25 MiB for D512/L768). This replaces recomputation
within a step; weights remain live across optimizer steps and CUDA graph replay.
See [wide preparation checks and timings](../../verdicts/trimul-wide-preparation-20260923/README.md).
