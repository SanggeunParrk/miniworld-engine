# A6000 mask fusion for single-direction and bidirectional trimul

Completed 2026-09-10. Pair masking now happens in the common projection stores and projection-gradient preparation kernel for the A6000 Triton paths. Both outgoing/incoming and bidirectional use it, in inference and training.
Current source hash: `618b0127d7fa55b8a64a876332c9e725e7920bdde375958c1bacc0144e921f1b`. Before source hash: `6df41ce4c744c85f51927609d783209c66950f22d2e8af1624b48191ec808b22`. [Full results and provenance](a6000-trimul-mask-fusion.json).

## What changed

- `bidir_front_triton` accepts an optional pair mask. The front kernel masks left/right before storing them. Input normalization, output gating, and saved preactivation values remain unmasked.
- `front_bwd_dW` forwards the mask to `_dconcat_kernel`, which applies it while loading left/right gradients. The existing separate large-tensor multiplications are removed from both callers.
- The fused operations retain the original dtype rounding boundary. None remains the unmasked specialization. Fake/custom-op signatures include the optional mask, preserving compiled graph support.
- Bidirectional still concatenates the directional contraction gradients; that necessary materialization was not removed by masking fusion. Earlier Inductor builds could combine its cat and mask into one kernel, so eliminating the mask does not imply eliminating the cat copy.

## Controlled native module timings

All six comparisons ran in one allocation on A6000 UUID `e1054d59-b429-c2a3-2b82-dad9e3512734`. L384, d_pair128, depth1, BF16 inputs/trunk weights and FP32 norm affine, mask probability .125, dropout 0, TF32 disabled. Both versions use actual compile; inference uses manual CUDA Graph ON and training uses Graph OFF (forward+backward, no optimizer). Augmentation resolves to A5/A48, though pair operations have no augmentation axis.
The existing native benchmark CLI and timer ran three fresh processes per version/task, with before/after order alternated and implementation order rotated. The original three Python files and original two caches were loaded for the before runs through a checked package overlay; their full source identity was recomputed with those file contents and matched the earlier source hash. After runs use the repository files. Compilation and graph setup are outside timing. Units are median ms.

| Mode | Operation | MiniWorld before | MiniWorld after | Reduction | cuEquivariance |
|---|---|---:|---:|---:|---:|
| inference | outgoing | 1.0588 | 0.8591 | 18.86% | 1.1428 |
| inference | incoming | 1.0424 | 0.8448 | 18.96% | 1.1274 |
| inference | bidir | 2.0920 | 1.6783 | 19.77% | 2.7607 |
| training | outgoing | 4.8046 | 4.3878 | 8.67% | 4.6413 |
| training | incoming | 4.8015 | 4.3889 | 8.59% | 4.6484 |
| training | bidir | 8.0998 | 7.7041 | 4.89% | 8.7813 |

Bidirectional cuEquivariance is the equivalent vendor-primitive composition with shared output normalization, not a native fused bidirectional API. These timings compare the two implementations of that same module.

## Numerical and execution validation

- 36 new CUDA regression cases: outgoing/incoming/bidirectional at L128 and L384; training with None, all-valid, holes and all-invalid masks; inference with holes and all-invalid masks. Outputs, input gradient and every parameter gradient are checked against FP32 PyTorch with the same representable weights. Non-norm linears are randomized, including gate/output weights.
- All-invalid cases, with output-normalization bias at its zero initialization, explicitly check that output equals the residual input and that the input gradient equals the upstream gradient. All parameter gradients, including output-normalization bias, still match the reference. This is not a promise to zero the final update for arbitrary trained affine parameters; masking applies to contraction inputs, while output gating and the residual stay unmasked.
- Four prior CUDA reference/vendor tests also passed: 40 CUDA tests total. The 109 CPU compile/benchmark-contract tests passed separately. The initial combined run exposed CPU-only SWA fixtures attempting CUDA queries on CPU devices; the mask numerical tests passed, and the CPU/GPU suites were rerun in their intended environments.
- All 54 native CSV rows were reread and matched their stored hashes, sidecars, actual compilation evidence and graph policy. The source identity remained fixed during comparisons.
- Final training traces for outgoing and bidirectional contain no former `bitwise_and_mul` mask kernels; mask construction and bidirectional concatenation may still execute. The profiles are diagnostics, not replacements for the native timing table.

## Cache handling

Only two A6000 cache files changed. Their existing 96 keys and 469 stored candidates were revalidated and retimed for both masked and unmasked kernel specializations; masking results matched the separate-multiply form exactly. No keys were dropped or added. Source identities were updated only after validation.
This was a remeasurement of prior top-K candidates, not a full-grid search or proof of global optimality for the new kernels. Entries are ordered by the worse of masked/unmasked diagnostic times. Old full-grid coverage stamps were cleared because they belonged to the old kernel source; the refresh metadata records parent hashes, source identity, job, GPU and scope. Other A6000 tile caches and dispatch caches are unchanged.
Other GPU architectures were not benchmarked in this change. L128 correctness was checked; small-L tuning remains deferred.
