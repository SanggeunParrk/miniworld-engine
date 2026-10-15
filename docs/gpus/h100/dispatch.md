# H100 module dispatch (2.2.0)

`implementation="miniworld"` with the default `engine_backend="auto"` selects the
packaged H100 implementations below. No research-directory import, external
TriMul payload, or MSA opt-in environment variable is required. Configure the
policy before constructing/compiling models.

| Module | Inference | Training | Native contract |
|---|---|---|---|
| Bidirectional TriMul | Anthropic-derived native K1 → two cuBLAS contractions → K3/residual; D256/384/512 at every L: B200-style port (`h100_b200style`), L384 / L768 D512 L384 keep the preserved K1/K3 | D128: selected CUDA forward + B1–B4 + four cuBLAS contractions + B7–B12. D256/384/512: flattened port of the qualified large-width research plans (below), B200-style port at the other lengths | BF16, batch 1, L a multiple of 128 up to 1024; direction hidden=D, D64/128/256/384/512; D128 requires a full 132-SM H100 |
| Single-direction TriMul | D64/128: packaged K1 → contraction → K3/residual; D256: the same K1/K3 tiles at every L except 384 (B200-style port at L384); D384/512: B200-style port; D512 L384/768 may use `h100_uni_wide_inference` | D64/128: native K1/K3, fused CUDA B1, two cuBLAS gradient contractions, streamed producer-consumer B7. D256/384/512: the wide path `h100_uni_wide_training` (front LN → K1 → cuBLAS contraction → LN → projection / gate GEMMs → output epilogue; backward gate, dn GEMM, output-LN backward, `contract_gp`, weight GEMMs, input-LN backward) | BF16, batch 1, D=hidden=64/128/256/384/512, L a multiple of 128 up to 1024 (training: wide path L384/768 and the B200-style port elsewhere); both outgoing and incoming; D128 requires a full 132-SM H100 |
| Transition | Existing residual-fused hand-CUDA path | Same forward with native backward | BF16, n=4, D64/128/256/384/512 and each kernel's resource guards |
| OuterProductMean | Packaged OPM with residual in the CUDA epilogue, no LN statistics saved | Residual-fused forward and native backward; residual gradient is passed through | BF16, batch 1, hidden32; (MSA, pair) 64/128 (MiniWorld), 128/256 (Protenix-v2, ESMFold2) or 128/384 (OpenDDE); any L and MSA depth (padded to 64 / 256 with masked rows); either normalization order; no interchain masking |
| MSAPairWeightedAveraging | Packaged forward without training saves; N ≥ 512: cuBLAS contraction + `pwa_gateout` | Packaged forward/backward, residual and row dropout; N ≥ 512: same split forward; dv on cuBLAS, read tile by tile by `dgv_bwd`; no pair-LayerNorm beta (the module has none) | BF16, batch 1. MiniWorld: MSA64/pair128, 8 heads × 32, L multiple of 128, even MSA depth. Wide (`pwa_wide`): MSA 64 or 128, pair 128 / 256 / 384, 8 heads × 8 or 16 (ESMFold2, Protenix-v2, OpenDDE), L ≤ 1024 (padded to 16), any depth |
| Token DiT | Fused step (`integrations/token_dit.py`): CUDA rows, CUDA gated core (bf16 `attn_fwd.cu` GATED, fp32 TF32 `attn_tf32.cu` with a v^T GEMM and a key-permuted pair bias), cuBLAS GEMMs; capture-scoped weight-pack and pair-bias caches | One opaque fwd / bwd per block (`integrations/token_dit_train.py`): CUDA rows (bf16 or fp32-operand build), cuBLAS GEMMs, bf16 `attn_fwd` + `attn_dqb`/`attn_dkv` or fp32 TF32 `attn_tf32` fwd + `dqb` + `dkv` | BF16/FP32, batch 1, single768/condition384/pair128, 16 heads, expansion1536, L multiple of 128, QK norm on or off, key mask [1, L]; inference: shared or per-sample conditioning ([token_dit/token_dit.md](token_dit/token_dit.md)) |
| Local atom DiT (AF3 windowed, cross-attention) | `integrations/local_dit_h100.py`: hoisted conditioning tables and windowed pair bias, `pre90` (AdaLN 1 + q / gate + K / V AdaLN + k / v), `mma.sync` window attention, `post90` (gate, out-projection, AdaLN 2, SwiGLU, residuals); wgmma row kernels, TF32 or bf16 | One opaque fwd / bwd per block: row kernels `cond90` `pre90` `post90` / `tr_bwd90` `post_bwd90` `pre_bwd90` `cond_bwd90` (wgmma, register A), `mma.sync` attention fwd / dq / dkv, cuBLAS weight gradients, `finalize90` (gradient packing) | fp32 activations + fp32 parameters (no autocast, TF32) or bf16 activations (fp32 master or bf16 parameters, autocast allowed); B = 1, d128 / cond 128 / pair 16, 4 x 32 heads, n = 2, any A, any N (padded to 128 inside), key mask [1, N]; `settings.local_dit_h100_mma{,_fp32,_bf16}`, `local_dit_h100_pdl`, `local_dit_hoist` ([local_dit/local_dit.md](local_dit/local_dit.md)); few 64-row tiles: weight packs in one launch (`pack90`), `cond90` / `pre90` / `tr_bwd90` / `post_bwd90` / `cond_bwd90` split over CTAs (`grid.y`), shared conditioning: four samples per CTA group and `cond_bwd_sh90` |
| Bidirectional TriMul | Anthropic-derived native K1 → two cuBLAS contractions → K3/residual | D128: selected CUDA forward + B1–B4 + four cuBLAS contractions + B7–B12. D256/384/512: flattened port of the qualified large-width research plans (below) | BF16, batch 1; training L384/768, direction hidden=D, D128/256/384/512 (D64 trains on Triton); D128 requires a full 132-SM H100 |
| Single-direction TriMul | Packaged K1 → contraction → K3/residual | Native K1/K3, fused CUDA B1, two cuBLAS gradient contractions, streamed producer-consumer B7 | Training: BF16, batch 1, D=hidden=128, L384/768, full 132-SM H100; both outgoing and incoming. Inference: width/hidden pair in the native tile table |
| Transition | Residual-fused hand-CUDA path (tuned fused kernels, else the generic wgmma kernels) | Same forward saving xn / rstd / c1, native backward (dW: cuBLAS) | BF16 or FP32 on TF32 (needs `set_float32_matmul_precision("high")` or `allow_tf32`); tuned: n=4 D64/128/256/384/512 and n=2 D64/128/256 with rows a multiple of 128; generic: any D multiple of 64 up to 1024, hidden a multiple of 128, any row count |
| OuterProductMean | Packaged OPM with residual in the CUDA epilogue, no LN statistics saved | Residual-fused forward and native backward; residual gradient is passed through | BF16, batch 1, MSA64/hidden32/pair128, L multiple of 64, MSA depth multiple of 256; normalization before projection; no interchain masking |
| MSAPairWeightedAveraging | Packaged forward without training saves | Packaged forward/backward, residual and row dropout | BF16, batch 1, MSA64/pair128, 8 heads × 32, L multiple of 128, even MSA depth |
| Token DiT | Fused inference row kernels, attention/gate and GEMM path | Existing general autograd route | BF16/FP32, batch 1, single768/condition384/pair128, 16 heads, expansion1536, L multiple of 128, shared sample conditioning, no QK norm |

Unsupported contracts retain the general implementation. `auto` selects the
bidirectional CUDA training path at D64/128/256/384/512 (D64: its own fused path
`h100_d64_training`, about 2.0x Triton); each is faster than the Triton path under
CUDA-graph replay. Inference: the packaged K1/K3 table (D64/128, uni D256/384), the
wide bidirectional K1/K3 `h100_wide_inference` (D256/384/512, 1.89-2.04x) and the
single-direction D512 path `h100_uni_wide_inference` (1.80x). Single-direction training
serves D64/128/256/384/512 in both directions (D512 incoming was on the Triton fallback until 2026-10-15). Per-shape status: `docs/gpus/h100/h100.md`; comparison matrix: `docs/records/trimul_h100_comparison_20261015.md`.

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
`settings.opm_train=0`, `settings.pwa_train=0` and
`settings.pwa_infer=0` disable their respective integrations.
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
[third-party notices](../../../licenses/THIRD_PARTY_NOTICES.md).

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
D256/384/512 **single-direction training** runs the wide path (`h100_uni_wide_training`) described in the next section.
See the measurements and validation at `archive/docs-20260928:docs/records/verdicts/trimul-single-20260923/`.

## Single-direction wide path (D256/384/512), 2026-10-15

`h100_uni_wide_training` serves outgoing and incoming (`GP_DIR` 1 / 2 in `contract_gp`; the contraction picks the transposed operand).
D512 front: `ln_in_d512.cu` writes `x_n` (one warp per row, fp32 two-pass statistics), then `front_pre_ss.cu` (derived from the frozen
`front_pre_d384.cu`) reads the normalised operand and issues wgmma with A and B from shared memory (no 128-register A fragment set);
tile (1, 64, 4, 4, 1). A 2-CTA multicast variant (`front_pre_mc.cu`) and an N-split two-warpgroup variant (`front_pre_ns.cu`) exist behind
`_MC` / `_NS` (default off: 3.14 / 3.19 ms against 3.18 ms for the default). Weight pack, LN-affine fold and input LN of the wide paths
use the CUDA aux kernels (`pack_into`, `fold_into`, `normalize_into`) whenever they run eagerly; the Triton versions only serve compiled
regions. A kernel-name scan of D256/384/512 at L384/768 (36 cells) finds no Triton kernel.

Inference selection (`_legacy_native_faster`): single direction D256 takes the tmn K1/K3 tiles at every L except 384; bidirectional
keeps the preserved K1/K3 at (D384, L768) and (D512, L384).

## Wide bidirectional training (D256/384/512)

`kernels/trimul_inproj/cuda/h100_wide_training.py` is a flattened port of the
qualified research selections recorded in
`archive/wip-main-20260928:experiments/trimul_large_d_vast` (September 27): D256 `d256_pool_checkpoint`,
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


## H100 FP32 master parameters (2026-10-05)

The qualified BF16-mixed paths keep FP32 master parameters and unrounded FP32
matrix gradients with BF16 compute. Weight casts happen inside the autograd
functions, run afresh on every CUDA graph replay, and original parameters remain
in saved-tensor version checks. Native narrow reducers store FP32 directly; wide
cuBLAS gradients avoid a BF16 round trip. See [master_weights.md](master_weights.md)
for the per-shape full F+B latency table, dispatch and validation evidence, and
scope of the less-than-2-percent mixed overhead requirement.

## AttentionPairBias (2026-10-13)

`modules/attention_pair_bias` -> `integrations/attention_pair_bias_b200.py` (the B200 file; sm_90 branches inside). Served when:
`implementation=miniworld`, engine backend not `triton`, bf16 `single` / `pair`, B = 1, contiguous, (heads, d_single) in
{(8, 384), (12, 384), (16, 384), (24, 384), (16, 512)}, d_pair 128, no QK-norm, key mask `[1, L]` or none, capability 9.0 or 10.0;
inference `L % 16 == 0`, training `L % 128 == 0` (H100). Anything else takes the module path (Triton / PyTorch); on H100 with
`implementation=miniworld` a call outside the contract raises instead of falling back silently.

| switch (`settings`) | default | meaning |
|---|---|---|
| `apb_h100_train` | on | native CUDA training; off keeps the module (Triton) path as an explicit baseline |
| `apb_h100_exact_dwb` | **off** | recompute the `to_bias.weight` gradient in fp32 from bf16-rounded inputs (needs CUTLASS 4.2 headers via `CUTLASS_PATH`); 1.7e-2 -> 0.8e-2 on that one gradient, 2.4x the step |

Saved tensors (training): xa, (mean, rstd), q|k|v|g, bias, O fp32, LSE, og, Wf, packed W q|k|v|g, bf16 Wo, bf16 Wb; the original
parameters are kept for autograd version checks. Parameter gradients are fp32 and are cast to the parameter dtype only when the
parameter is not fp32. Weight packs are cached per parameter version and scoped to the CUDA-graph capture (`kernels/_capture.py`).
