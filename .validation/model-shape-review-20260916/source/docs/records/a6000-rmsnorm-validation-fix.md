# A6000 RMSNorm validation correction

Completed 2026-09-10. This closes the RMSNorm precision qualification in [the L384 cache repair record](a6000-l384-cache-built.md). The defect was in numerical validation: BF16 rounding of the reference and a forward threshold below the representable output precision. Production RMSNorm kernels were unchanged.

## Cause and correction

The old checker passed BF16 inputs and BF16 leaves to the reference. Although its formula promoted arithmetic to FP32, it rounded the output and autograd leaf gradients back to BF16. Two nearby FP32 values could fall on opposite sides of a BF16 rounding boundary and disagree by a full BF16 step. The checker now passes FP32 copies of the same exact input/weight values and the same upstream gradient; reference outputs and gradients remain FP32.

Promotion alone does not make the former forward threshold valid. BF16 round-to-nearest has a normal-value relative rounding bound of 2^-8 (about 0.390625%). The independent regression example x=[1, 0.9921875], eps=1e-5 exceeds the old 0.32% global relative bound even when the FP64 formula is correctly rounded to BF16. The forward BF16 threshold is therefore 0.40%. Backward BF16 remains 0.40%, forward FP32 remains 8e-7, and backward FP32 remains 4e-6. No other registry row changed.

## A6000 validation

GPU UUID `40e6429b-fc40-3f79-7b81-ffc1a1bd6482`, Slurm job `1672723`. Reused all five stored candidates for each of the two previously qualified cache entries at [589824,32], with seeds 0/1 and contiguous/column-major inputs: 40 candidate/input checks. The exact same kernel outputs were compared against the old rounded reference and the corrected unrounded reference.

| Kernel | Old-reference maximum error | Corrected-reference maximum error | Applied BF16 band |
|---|---:|---:|---:|
| rmsnorm_fwd_triton | 0.3472% | 0.3332% | 0.40% |
| rmsnorm_bwd_triton | 0.4425% | 0.2330% | 0.40% |

All 40 also passed an independent FP64 per-element check: error <= half a BF16 interval + 2e-6 times the arithmetic scale. Backward scale includes sum of absolute products to account for cancellation. There were zero violating elements and no nonfinite outputs. Saved forward rstd and the unweighted backward weight-gradient buffer were checked too. This is stronger evidence than merely increasing a scalar tolerance.

- 16 GPU regression cases ran the actual registry checkers, covering BF16/FP32, affine/non-affine, input and weight gradients, contiguous/column-major layouts, ragged dimensions, and covering/tiled reductions.
- 23 CPU tests passed; one existing blank-band parametrization was intentionally skipped. Tests preserve reference dtype, demonstrate why FP32 leaves are necessary, reject injected output/gradient errors, and independently reproduce the old threshold problem.
- Kernel source and A6000 tile/dispatch caches were unchanged. No kernel rebuild, retuning or performance remeasurement was required. Prior steady-state timings still describe the same production kernel code.

Checker source snapshot: `e26b6b50c6b7aaa3c0cf08afda2b6361d9abc408a84831a891d259f6fd700298`. [Candidate results and validation provenance](a6000-rmsnorm-validation-fix.json).
