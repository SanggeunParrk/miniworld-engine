# B7–B12 H100 optimization (2026-09-21)

Selected development adapter: `trimul_b7_nextrow_20260921/gate_policy.py:Training`,
via `runs/trimul_training_current.py`. Production dispatch is separate.
Scope: L384/L768, C128, bidirectional H256, BF16, dropout 25%, pair mask and residual.
B1 remains v51. Keep input affine BF16 x_n, original BF16 tri, output FP32 mean/rstd.
**SoL90 has not been demonstrated.**

## What changed

Two CUDA launches, with no extra HBM activation tensors:

- dW: 256 CTAs, one producer + one consumer warp group, 112 KiB shared,
  two CTAs/SM. Four TMA input slots, two GLU shared buffers. Pair row tiles:
  projection/gate GEMM for row1 overlaps GLU derivative for row0; accumulate
  both rows in one K128 dW group. Keep 16 splits and 2 accumulation slices.
- dX/LN/residual: 264 CTAs, one producer + one consumer warp group,
  112 KiB shared, two CTAs/SM. One compute group handles all 128 channels.
  Dedicated producer streams weights and inputs. L768 prefetches next-row
  x_n and first gate tile during current LN/residual; L384 disables that prefetch.
- Reuse raw LN fragments and mask registers. Remove identical x_n shared stores.
- Partial dW scratch stays 16.78 MB. Partial input-LN scratch changes from
  135.17 to 270.34 kB because there are 264 rather than 132 CTAs.
- All other HBM input/output payloads are unchanged. Projection/gate are
  recomputed separately in dW and dX; no dConcat or dx_n HBM tensor.

## Same-run full-module comparison

Baseline = development state immediately BEFORE this B7 work: B1 v51 plus
old split saved-xn B7. It is NOT Anthropic inference and NOT historical saved-all B7.
Job14143, node01 H100, 600 alternating CUDA-graph samples per scope:

| Scope (us) | L384 old → selected | Speedup | L768 old → selected | Speedup |
| --- | ---: | ---: | ---: | ---: |
| B7–B12 | 544.928 → 395.584 | 1.378× | 2154.080 → 1486.816 | 1.449× |
| Entire backward | 911.712 → 758.848 | 1.201× | 3728.592 → 3023.456 | 1.233× |
| Forward + backward | 1200.352 → 1048.768 | 1.145× | 4980.608 → 4265.904 | 1.168× |

Scopes are measured separately, not summed. Both directions, live weight packing
and all eleven gradients included; optimizer, RNG generation, compilation and CPU
launch overhead excluded. B1 and forward math/selection unchanged.

## Validation

Full-module initial and mutated input/weight/dy/dropout/mask tests pass.
Forward plus nine gradients are bit-exact to baseline. Input LN dgamma/dbeta
change summation order with 264 CTAs; relative L2 stays below 5e-6 (observed
about 3–4.3e-7), the existing input-LN comparison limit. Graph replay equals eager.
The separate strict BF16 cross-reference limitation below remains unresolved.

Job14144: both dW and dX at BOTH lengths pass memcheck (0 errors) and racecheck
(0 hazards), three input states × two replays; counters return to zero.
ptxas: no register spills, 128 static registers/thread, runtime producer/consumer
redistribution 32/224. Both kernels have two resident CTAs/SM.

## NCU checkpoint (job14145)

These are profiled durations, separate from the alternating CUDA-event benchmark.

| L | dW / dX us | tensor-pipe elapsed activity dW / dX | DRAM active cycles dW / dX |
| --- | ---: | ---: | ---: |
| 384 | 171.872 / 227.296 | 48.25% / 38.46% | 38.68% / 45.43% |
| 768 | 599.872 / 784.512 | 52.95% / 43.21% | 41.28% / 51.94% |

These metrics are NOT whole-algorithm SoL. L768 source-PC analysis finds
27786/30437 dW long-scoreboard samples at the producer waiting for a released
shared slot (`dw_pair.inc:6`). This is not evidence of saturated HBM.
The largest dX barrier sample site is the common kernel role boundary;
producer/idle warp waits must not be misread as compute-critical stalls.
Deduplicate source PCs per named kernel, not across both kernels.

## Accepted and rejected experiments

- Accepted: one compute WG dX plus dedicated TMA producer, two CTAs/SM.
- Accepted: paired dW with four TMA slots and two GLU buffers.
- Accepted for L768 only: next-row x_n + first gate-weight/gradient prefetch.
- Rejected: deeper buffers alone; reduced GLU unrolling; BF16 conversion pairing.
- Rejected: dW full streaming/delayed WGMMA waits or extra GLU overlap: slower.
- Rejected: shared x_n across two consumers: correct, roughly 4% slower.
- Rejected: native N128 GP variants that failed correctness. Never timed as valid.
- Earlier one-WG prototype had a shared-memory race; explicit LN read/store
  synchronization fixed it. Only fully sanitized successor is selected.

Next target remains SoL90. A stage-level lower bound and further compute/transfer
scheduling are needed; observed utilization does not justify declaring completion.

## Accuracy investigation

Both lengths, initial and mutated inputs:
- Saved input x_n equals the independent saved-activation path bit for bit.
- Recomputed BF16 projection/gate preactivations equal saved values bit for bit.
- The actual BF16 GLU derivatives consumed by dW equal reference dConcat bit for bit.
- Remaining disagreement is from GEMM accumulation / final BF16 rounding,
  not a changed activation or derivative equation.

L768 mutated dWL differs from the historical BF16 cuBLAS reference by
0.053318% for identical incoming gradients (full pipeline: 0.055569%).
The old 0.05% cross-reference check still fails; it has not been hidden or relaxed.
However, FP64 GEMM followed by BF16 rounding gives:

| L768 dWL | CUDA vs FP64-rounded truth | historical reference vs truth |
| --- | ---: | ---: |
| Initial | 0.018366% | 0.051741% |
| Mutated | 0.020176% | 0.056992% |

In these cases CUDA is closer to the independent FP64 calculation. This
prevents treating reference disagreement alone as proof of a CUDA bug.
All-four-dW FP64 verification completed for both input states. L768 initial dWR
has CUDA / reference rounded-truth relative errors of 0.055680% / 0.059545%;
therefore the evidence does NOT show that every dW passes a strict 0.05%
FP64-rounded-output comparison. BF16 rounding near a midpoint magnifies
small accumulation differences; no accuracy threshold was relaxed. This experiment does not yet
establish general production accuracy for all parameters/seeds/shapes.

