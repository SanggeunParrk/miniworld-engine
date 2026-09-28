# D128/256: matched Triton versus hand-CUDA b2b — 2026-09-18

## Conclusion

On node02 H100, **hand-CUDA remains faster with the same width, fusion boundaries,
saved operands and backward**. Module inference speedup over the newly tuned
Triton b2b is 1.14x at D128 and 1.36–1.38x at D256. Complete training speedup is
1.03x and 1.07–1.09x respectively because backward is shared.

This establishes an implementation difference for these tested kernels, not a
general CUDA-versus-Triton language limit. It also does not demonstrate that a
D384/512 CUDA b2b would win; those larger-width designs face different constraints.

## Controlled comparison

`comparison.py::MatchedB2B` gives both implementations the same:

- Flattening, BF16 affine/weight casts, epsilon and Triton stats pass.
- **LN → two expand projections → SwiGLU → squeeze → residual in one b2b kernel**,
  with stats separate. Both round squeeze to BF16 before residual addition.
- Full-K normalized operand computed once and reused through the hidden loop.
  Triton is restricted to BK=D here; the wide-D K-streaming input rereads are absent.
- Training saves BF16 xn in the b2b launch. Inference omits that save.
- The exact same `_fused_bwd` invocation, settings, tensor dtypes and saved-tensor
  layout. Both parameter gradients retain the same FP32 affine contract.

Only the b2b forward launcher changes. This Triton candidate is the experimental
`wide_b2b.py` kernel with D128/256 enabled as control cases; it is neither the
production Triton split nor the older legacy b2b's unrounded residual epilogue.
The production split is measured separately as a third reference arm.

## Module times

Official `bench_module_transition`, B1, `[1,L,L,D]`, expansion4, one layer, BF16,
deterministic nonzero squeeze, residual included. Static compile (`dynamic=False`,
one observed graph), manual CUDA graph. Training is forward+backward, no optimizer.
There is no dropout in general Transition. Each cell is the median of two
independent captures with reversed backend order; all measurements use node02.

[Table including the production split](timings.md) · [raw samples and NCU metrics](summary.json).

| D | L | Inference Triton / CUDA ms | CUDA speedup | Training Triton / CUDA ms | CUDA speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 0.1818 / 0.1597 | 1.139x | 0.9654 / 0.9385 | 1.029x |
| 128 | 768 | 0.6491 / 0.5670 | 1.145x | 3.6280 / 3.5243 | 1.029x |
| 256 | 384 | 0.7144 / 0.5245 | 1.362x | 2.4465 / 2.2533 | 1.086x |
| 256 | 768 | 2.6919 / 1.9573 | 1.375x | 9.5125 / 8.9029 | 1.068x |

## Config search

Both save modes were tuned independently at L384; L768 uses those selected configs.
Triton: BM16/32/64/128, BN32/64/128, BK=D, warps4/8, stages1/2/3. Configs whose
output accumulators alone exceed 255 registers/thread are excluded. This gives
72 configs per mode at D128, 63 per mode at D256. Of the 144/126 attempts,
136/98 completed numerical validation and timing; other configs failed resources
or compilation and are recorded in `tune-D*.json`. All completed comparisons to
the native forward had zero measured relative error, including saved xn when used.

The native declared grid was also measured: four configs per mode at D128 and two
at D256, all passing exact output checks. Selected configs are explicitly passed
to both launchers; this comparison does not depend on native cache-miss defaults.

| D | Triton BM / BN / BK / warps / stages | CUDA CTA rows / BN / K / warpgroups / stages |
|---:|---|---|
| 128 | 64 / 64 / 128 / 4 / 2 | 128 / 128 / 128 / 2 / 2 |
| 256 | 128 / 32 / 256 / 8 / 3 | 128 / 64 / 256 / 2 / 1 |

Both inference and training selected the same config within each implementation.
Triton search covers the listed grid, not all possible Triton implementations.
The split baseline's cache misses use a heuristic subset of 24 configs.

## Compiled code and NCU

Both selected implementations use **WGMMA**, and their disassembly contains HGMMA.
CUDA SASS also contains **UTMALDG** for the weight TMA transfers; Triton SASS has
LDGSTS/cp.async and no TMA loads. Both use register-sourced squeeze operands;
the difference is not that Triton must spill h to HBM.

| D | Backend | Registers/thread | Compiler spill slots | NCU SM throughput | Active warp occupancy |
|---:|---|---:|---:|---:|---:|
| 128 | Triton | 242 | 0 | 39.64% | 12.30% |
| 128 | CUDA | 255 | 0 local bytes | 45.62% | 12.18% |
| 256 | Triton | 255 | 10 (inference) | 35.64% | 12.46% |
| 256 | CUDA | 242 | 0 local bytes | 50.66% | 12.49% |

Triton D256 training has six compiler spill slots. Native `cuobjdump` resource
reports show zero stack/local storage for both saved and non-saved specializations.
The native dump contains both specializations, so raw opcode counts across that
file must not be compared directly to the single-specialization Triton dump.

The similar occupancy rules out explaining this matched result merely by fewer
active warps in Triton. Concrete differences include:

- Native weights arrive through explicit TMA and are shared by the two consumer
  warpgroups. D128 native handles twice as many rows per CTA as the selected Triton.
- Native uses twice the hidden-channel chunk width in both widths: half as many
  hidden-loop iterations (4 vs 8 at D128, 16 vs 32 at D256).
- Native D256 avoids the selected Triton kernel's local-memory spill and uses less
  shared memory. Its higher SM throughput is consistent with better scheduling.

These observations explain plausible contributors. There was no isolated TMA
on/off ablation, so the measured speedup cannot be attributed entirely to TMA.
NCU profiling is one inference kernel at L384, separate from the module graph
timings. Counter-derived time is not substituted for benchmark latency.

Actual PTX/TTGIR/CUBIN/SASS, CUDA resource dumps, and NCU reports are retained under
`/home/psk6950/MiniWorld/runs/transition_small_b2b_20260918`.
Raw NCU CSV and metadata are copied alongside this record.

## Validation, scope and reproduction

- `validation-D*.json`: both widths' output and all six gradients had zero measured
  difference on the direct comparison case, including a zero gamma element.
- Zero squeeze gave exact y=x and dx=dy, with zero affine gradients.
- All **48** module benchmark rows passed finite-value and CUDA graph replay checks.
- No production dispatch, installed package, remote repository or cache was published.
  Only the experimental Triton width guard was extended for these control cases.
- Node02 allocation 13276 was used exclusively for this work and released afterward.

Scripts and environment are archived here. Run GPU scripts inside a fresh node02
allocation: `tune.py`, `tune_cuda.py`, `validate.py`, then `measure.py --width D
--length L`. `profile_kernel.py` brackets its single target launch with
cudaProfilerStart/Stop for `ncu --profile-from-start off`. Set PYTHONNOUSERSITE=1
for the NCU parent to avoid embedded-Python user-site conflicts.
Source hashes are in `sources.json`.
