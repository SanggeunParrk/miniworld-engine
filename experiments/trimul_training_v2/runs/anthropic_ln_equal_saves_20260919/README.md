# Equal-save input LayerNorm fusion A/B

## Question

Does separating input LN improve training speed when the fused implementation also saves
exactly the same normalized input and LN statistics?

2026-09-19, node02 H100. B1, L384/768, C128/H256, BF16 activations and weights,
FP32 LN parameters/statistics, pair mask and 25% row dropout, residual enabled.

## Controlled implementation

Both paths are derived from Anthropic native v5 f4f62fa (Apache-2.0). Upstream vendored
files and production dispatch are unchanged. `front.cu` extends the existing training
front derivative; upstream K1 already has SAVE/EMITX support.

- Fused: K1 normalizes input, emits BF16 x_n and FP32 mean/rstd, computes projection/gate,
  and saves the same interleaved preactivations and contraction inputs as split.
- Split: standalone Anthropic LN emits x_n and the same statistics; the same K1 body
  consumes x_n with input normalization disabled. LN output storage is done once.
- Both: K3 consumes saved x_n, fuses output LN/projection/gate/dropout/residual,
  saves output LN values/statistics and projection/gate with the existing training
  rounding policy. This is different from the earlier inference-only rounding experiment.
- Both use exactly the existing Triton/cuBLAS back-half backward and input-LN/residual
  backward, with matching saved tensor shapes, strides and dtypes.
- K3 does NOT recompute input LN in either path. The fused path can reuse its saved x_n;
  forcing it to recompute would confound this question.

Weights are prepacked outside timing. Fixed common dropout scale; RNG generation excluded.
No optimizer, CPU/autograd dispatch or model-wide training step timing. Manual calls use
existing backward implementation without modifying its algorithm.

## Tuning and timing

Both modified K1 variants search all 58 valid training front configs: BI/BJ, weight-ring
slots, K chunks, single/double accumulator schedule. Shared memory legality follows the
existing saved-front candidate validator. Four standalone LN affine/store schedules.
Each candidate is checked against fused reference saves before timing; best four are
remeasured with alternating runs. This is the existing config space, not a claim of
exhaustive hardware optimization. Registers/start offset remain existing defaults.

CUDA Graph timing, 80 replays x 20 alternating rounds, medians and individual rounds kept.
Front, full forward, isolated backward and directly captured forward+backward timed separately.
Full training is measured directly, not inferred by adding isolated stage timings.
Unchanged shared backward uses existing cache where available and a fixed 3-candidate miss
fallback elsewhere; it is not newly exhaustively tuned. Relative A/B uses the same kernels.

## Validation

check.py: L64/72 forward/save parity, gradient comparison, existing independent engine path
reference. Selected-config validation and sanitizer evidence are recorded separately.

## Reproduction (allocated node02 GPU only)

bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_equal_saves_20260919/check.py
bash runs/anthropic_adoption_20260919/env.sh python runs/anthropic_ln_equal_saves_20260919/bench.py

Artifacts: results.json, tune-*.json, hashed cubins/ptxas logs in build/.

## Result

| L | Stage | Fused us | Split us |
|---|---|---:|---:|
| 384 | front | 202.84 | 218.19 |
| 384 | fwd | 434.63 | 451.18 |
| 384 | bwd | 1086.64 | 1086.22 |
| 384 | train | 1523.60 | 1538.22 |
| 768 | front | 815.58 | 847.42 |
| 768 | fwd | 1742.07 | 1749.74 |
| 768 | bwd | 4339.08 | 4333.38 |
| 768 | train | 6068.41 | 6061.93 |

Fused total training is ~1% faster at L384; L768 totals are effectively equal (~0.1% difference). No demonstrated speed advantage for separating input LN under identical saves. Selected configs passed L64/72 validation, memcheck 0 errors, racecheck 0 hazards.
