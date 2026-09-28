# A6000 module training with dropout 0.25

Completed 2026-09-10. The native module benchmark now defaults to dropout 0.25 for dropout-bearing training modules. The earlier dropout=0 training tables remain historical diagnostics, not the default workload.

## Scope and behavior

- Triangle multiplication: outgoing, incoming, alternating stacks and bidirectional all receive the resolved probability for every implementation, including the DTv1 residual wrapper.
- Triangle attention receives the same probability. Its starting/ending broadcast axes are unchanged. The current native timing target measures starting attention; the numerical tests cover both axes.
- Transition, adaptive LayerNorm, conditioned transition, augmented attention and DiT/SWA variants have no module dropout and keep 0. Pairformer already defaults to 0.25. No dropout was added to an architecture that has none.
- Inference uses dropout 0. Explicit nonzero inference/dropout-free-target probabilities are rejected. Explicit 0 remains available for diagnostic training runs.
- Every timed training forward constructs a fresh dropout mask in the production module. Timing includes RNG, scale application, residual and forward+backward; it excludes compile/warmup and optimizer steps.
- Trimul accuracy calls temporarily share identical per-layer scales, rounded in the measured dtype. The production generators are restored before timing warmup. Equal seeds alone would not align compiled BF16 and FP32 reference random streams.
- Triangle attention output projection used to be initialized to zero, which hid the effect of dropout and zeroed upstream gradients. Its benchmark now uses reproducible nonzero linear weights.
- CSV rows and sidecars record the resolved dropout probability. Plotting rejects mixed dropout conditions and explicitly labels old missing values as unrecorded.

## Native timing results

One gpu04 allocation, NVIDIA RTX A6000 UUID `40e6429b-fc40-3f79-7b81-ffc1a1bd6482`, Slurm job `1672700`. L384, d_pair128, depth1, BF16 activations/trunk with FP32 norm affine, mask probability 0.125, dropout 0.25, TF32 disabled. Actual torch.compile, CUDA Graph OFF, A48 (pair modules have no augmentation axis). Three fresh CLI processes per target, with implementation order rotated; values below are median milliseconds.

| Operation | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| outgoing | 14.1112 | 4.7401 | 4.3807 |
| incoming | 14.1041 | 4.7288 | 4.4022 |
| bidir | 26.9855 | 8.8105 | 7.6575 |
| triangle_attention | 13.8076 | 8.3845 | 6.4671 |

Bidirectional cuEquivariance is the equivalent vendor-primitive composition with shared output normalization. The timer and production kernels were not replaced. All 36 raw CSV rows and sidecars were reread, hashed and checked against actual compilation, dtype, graph and dropout metadata.

## Validation and cache status

- 160 CPU module/measurement/vendor contract tests and 58 reporting tests passed. These include actual timed-closure RNG calls, new outputs on successive steps, effective probability in each implementation, YAML default resolution and mixed-condition plot rejection.
- 10 CUDA tests passed: MiniWorld and cuEquivariance, outgoing/incoming/bidirectional and starting/ending attention. With shared dropout scales, outputs, input gradients and every parameter gradient are compared against the FP32 PyTorch reference (relative Frobenius threshold 0.03). After removing the validation override, compiled forward+backward must generate different outputs on successive calls.
- Across the native trimul rows, maximum output relative Frobenius error was 0.003932; maximum input-gradient error was 0.005188. Native triangle-attention timing rows have no per-row reference metric; their dropout numerical validation is in the separate CUDA tests at L128.
- Kernel source and A6000 tile/dispatch caches were unchanged from the completed mask-fusion cache refresh. Benchmarks stop on a cache miss; no full-grid tuning or new kernel-cache build was needed.
- Initial setup checks caught a missing metric argument in a new CPU test and a missing explicit dropout key in two YAML files; both were corrected before the accepted runs. The successful run provenance is retained below.

Source hash: `f631140cd495cc09340346979ce0d105bafd24ecbc9464f919dae32c505c73f3`. [Full runs, raw CSV paths and hashes](a6000-training-dropout025.json).
