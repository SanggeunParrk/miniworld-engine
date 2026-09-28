# H100 module dispatch (2.2.0)

`implementation="miniworld"` with the default `engine_backend="auto"` selects the
packaged H100 implementations below. No research-directory import, external
TriMul payload, or MSA opt-in environment variable is required. Configure the
policy before constructing/compiling models.

| Module | Inference | Training | Native contract |
|---|---|---|---|
| Bidirectional TriMul | Anthropic-derived native K1 → two cuBLAS contractions → K3/residual | D128: selected CUDA forward + B1–B4 + four cuBLAS contractions + B7–B12. D256/384/512: flattened port of the qualified large-width research plans (below) | BF16, batch 1; training L384/768, direction hidden=D, D128/256/384/512 (D64 trains on Triton); D128 requires a full 132-SM H100 |
| Single-direction TriMul | Packaged K1 → contraction → K3/residual | Native K1/K3, fused CUDA B1, two cuBLAS gradient contractions, streamed producer-consumer B7 | Training: BF16, batch 1, D=hidden=128, L384/768, full 132-SM H100; both outgoing and incoming. Inference: width/hidden pair in the native tile table |
| Transition | Existing residual-fused hand-CUDA path | Same forward with native backward | BF16, n=4, D64/128/256/384/512 and each kernel's resource guards |
| OuterProductMean | Packaged OPM with residual in the CUDA epilogue, no LN statistics saved | Residual-fused forward and native backward; residual gradient is passed through | BF16, batch 1, MSA64/hidden32/pair128, L multiple of 64, MSA depth multiple of 256; normalization before projection; no interchain masking |
| MSAPairWeightedAveraging | Packaged forward without training saves | Packaged forward/backward, residual and row dropout | BF16, batch 1, MSA64/pair128, 8 heads × 32, L multiple of 128, even MSA depth |
| Token DiT | Fused inference row kernels, attention/gate and GEMM path | Existing general autograd route | BF16/FP32, batch 1, single768/condition384/pair128, 16 heads, expansion1536, L multiple of 128, shared sample conditioning, no QK norm |

Unsupported contracts retain the general implementation. `auto` selects the
bidirectional CUDA training path at D64/128/256/384/512 (D64: its own fused path
`h100_d64_training`, about 2.0x Triton); each is faster than the Triton path under
CUDA-graph replay. Inference: the packaged K1/K3 table (D64/128, uni D256/384), the
wide bidirectional K1/K3 `h100_wide_inference` (D256/384/512, 1.89-2.04x) and the
single-direction D512 path `h100_uni_wide_inference` (1.80x). Single-direction training
stays D128-only; other widths train on Triton. Per-shape status: `docs/gpus/h100.md`.

## Retained values and execution

D128 TriMul training retains input `x_n`, left/right, `tri`, and packed front weights.
Bidirectional D128 additionally retains output-LN statistics; single-direction
D128 recomputes output-LN statistics in B1; D128 projection/gate intermediates are
recomputed in backward. The D256/384/512 saved set is listed in the wide section below. Every forward
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
its default is `(64, 128, 256, 384, 512)`.

This change connects already-selected implementations. It is **not** a complete
retune or rebuild of all native/Triton caches. Native TriMul ships measured tile
selections; Token DiT retains its local first-use GEMM selection. General Triton
paths keep the engine's cache system. Derived build plans model these native
composites without invoking nvcc on fake tensors.

The research histories remain available under `experiments/`; source hashes for
the selected CUDA bodies are in
`kernels/trimul_inproj/cuda/h100_sources/PROVENANCE.json` within the package.
Anthropic attribution and Apache-2.0 terms apply as described in
[third-party notices](../../licenses/THIRD_PARTY_NOTICES.md).

## Validation

Validation record (`archive/docs-20260928:docs/records/verdicts/h100-module-wiring.json`): 46 H100 test
executions passed, including a separate wheel installation on CUDA device 1,
all ten TriMul training width/length combinations, masked/dropout gradients,
live-weight CUDA graph replay, token DiT inference graphs, and MSA repeated
compiled calls without recompilation, including noncontiguous inputs. Source/config provenance hashes were
verified. The record distinguishes the initial CPU audit failures from the
successful checks after fixing their static analysis.

## Production-module performance audit

The four-backend H100 comparison (`archive/docs-20260928:docs/records/verdicts/version-compare-20260923/`)
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
paired preparation benchmark (`archive/docs-20260928:docs/records/verdicts/trimul-preparation-20260923/`).

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
See the measurements and validation at `archive/docs-20260928:docs/records/verdicts/trimul-single-20260923/`.

## Wide bidirectional training (D256/384/512)

`kernels/trimul_inproj/cuda/h100_wide_training.py` is a flattened port of the
qualified research selections recorded in
`wip/main-20260928:experiments/trimul_large_d_vast` (September 27): D256 `d256_pool_checkpoint`,
D384 `wide_checkpoint23`, D512 `wide_checkpoint24`, and D512/L384 with the 4-way
input-weight split. The research checkpoint chains, runtime text patching and
`quack`/CuTe imports are gone: each of the 29 kernels is a frozen `.cu` file under
`h100_sources/wide_train` (the final generated text of its chain, hashes in
`PROVENANCE.json`). The D256 chain constructed a CuTe/quack GEMM that the selected
path never launched; the port has no CuTe/quack/cutlass-DSL dependency.

Stages per call (grids, shared memory and tensor maps match the research plans):

- Forward: packed front K1 (D256 also writes input-LN stats; D384/D512 also save
  channel-major projection pre-activations; D512 normalizes `x` first), two
  contraction GEMMs, output LN (D512: packed normalization plus exact-scalar patch
  records), cuBLASLt projection and gate GEMMs, gate/dropout/residual epilogue.
- Backward: D512 patch application, fallback and row-compacted re-projection;
  output-gate kernel (L384 also zeroes the LN affine sums); dN GEMM; dWproj/dWgate
  (cuBLASLt, split FP32 with a native reduction at D512/L768, torch.mm where the plan
  used it); output-LN backward (cooperative at L768); contraction plus
  gate/projection gradients (D256: native contraction + register-budget or bulk-mask
  source; D384/D512: fused contraction/GP); split-K input-weight partials; one dX GEMM;
  input-LN backward with the ordered partial reductions. D256/L384 overlaps
  dWproj/dWgate with the source kernel and D512/L768 overlaps the joint input dW with
  the dX GEMM on a side stream (graph-capturable fork/join).
- D256/L384's source kernel uses `setmaxnreg` 32/208; its cubin's `.nv.info` launch
  register count is lowered to 120 (two resident CTAs) only after a control-flow walk
  of the SASS (`cuobjdump`) proves every register operand fits each allocation state.
  Without `cuobjdump` or on a failed check the original cubin runs (one CTA/SM).

Saved per forward (owned by that call): `ab`, `tri`, `x_n`, the normalized output-LN
rows, projection and gate products, output-LN mean/rstd; plus D256 packed front
weights, or D384/D512 channel-major pre-activations `[8D, L²]` (plus D512 patch
records). Measured forward-retained memory (saves plus output) equals the Triton
path's at D384/D512 (D512/L768 11.26 vs 11.27 GiB) and is 40% lower at D256
(L768 3.38 vs 5.64 GiB); the removed port retained about 4.5 GiB at D512/L768.
Peak fwd+bwd step memory is within -7%/+10% of Triton (D512/L768 20.7 vs 19.4 GiB).

**cuBLASLt algorithms.** The research plans froze heuristic indices and asserted the
algorithm words under cuBLASLt 12.8.4. torch 2.13+cu129 loads cuBLASLt 12.9.1, whose
lists differ and whose algorithm word 4 is runtime metadata, so those assertions
could not pass unchanged. `h100_sources/wide_train/lt_selection.json` keeps each
frozen algorithm; at first use the port selects the heuristic whose other seven words
match exactly and otherwise uses cuBLASLt's first heuristic with a one-time
`RuntimeWarning` (currently only D256/L384 dWgate). The compacted D512 re-projection
reuses the projection's algorithm after `cublasLtMatmulAlgoCheck`. Validation under
12.9.1: with the research plans' indices both implementations produce bitwise-equal
output, dX and weight gradients; all launched kernels have identical SASS. Reselection
check: the matched frozen configurations are 0.4–3.6% faster end to end than taking
cuBLASLt's first heuristic for every GEMM (paired graph replay, all six cells), so they
remain selected. `h100_wide_training.LT_SELECTION` switches the policy for such checks.

**Measurements (H100 SXM, torch 2.13.0+cu129, module fwd+bwd, B1, BF16).**
Paired CUDA-graph replay medians, one process per GPU:

| D | L | Triton ms | CUDA ms | Speedup | Vast H100 research (Triton / native ms) |
|---:|---:|---:|---:|---:|---|
| 256 | 384 | 3.742 | 2.592 | 1.44x | 3.680 / 2.600 |
| 256 | 768 | 16.640 | 10.584 | 1.57x | 18.286 / 10.441 |
| 384 | 384 | 6.637 | 4.563 | 1.45x | 6.483 / 4.504 |
| 384 | 768 | 29.162 | 18.938 | 1.54x | 28.499 / 18.656 |
| 512 | 384 | 10.089 | 6.927 | 1.46x | 9.845 / 6.858 |
| 512 | 768 | 45.765 | 28.278 | 1.62x | 44.529 / 28.023 |
| 64 (old port) | 384 | 0.828 | 1.354 | 0.61x | removed |
| 64 (old port) | 768 | 3.042 | 4.984 | 0.61x | removed |

Eager medians track graph replay within 1–3% at D256–D512 (for example D512/L768
28.94 vs 43.89 ms). The previous wide port measured 0.69–0.75x of Triton. D64's
old port was 1.2x faster than Triton only in eager L384 (1.37 vs 1.66 ms, launch
overhead) and slower in every other mode.

Validation: strict gradients of every leaf against an FP32 PyTorch reference with
random LN affine parameters, a 15% masked pair mask and 25% dropout scale (errors
equal to or below Triton's, e.g. dX 0.0040 vs 0.0043 relative); changed-input CUDA
graph replay after mutating input, every weight, LN affine, mask, dropout scale and
upstream gradient (outputs and weight gradients bitwise; LN affine gradients within
7e-7 because their sums use float atomics); `torch.compile(fullgraph=True)`;
outstanding forwards with independent saves; FP32 caller masks.
