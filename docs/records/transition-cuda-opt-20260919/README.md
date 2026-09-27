# Transition CUDA: compiled-code-guided IO optimization

Date: 2026-09-19. node02 H100; D128/256/384/512; L384/768.

## Final outcome and validation

The selected integrated kernels have no local spill loads/stores in their actual
SASS (16 selected direction/configuration combinations). Backward contains three
UTMASTG instructions instead of per-thread global vector stores. The public
configuration axes and the two fusion schedules are unchanged.

Compared with the previous best CUDA schedule, complete-module training improves
by 1.08–1.16x. Against the best Triton path, D256 improves by about 1.04x, while
D128/384/512 remain slower. A general 15% win over Triton for the whole module has
**not** been reached. See [RESULTS.md](RESULTS.md) for every shape and both schedules.

- 24 boundary-shape cases: D128/256/384/512, both schedules, M65/129/257;
  independent PyTorch formula, output and all six gradients, exact zero-squeeze
  residual/gradient identity.
- Eight static-compile + manual CUDA Graph training combinations passed.
- Eight old/new complete-operation comparisons had bitwise-identical output
  and all six gradients on the tested M129 inputs.
- Production memcheck: 16 selected configurations, zero errors.
- Production racecheck: 16 selected configurations, zero errors or warnings.
- Module timing: eight D/L shapes, five arms, two modes, two captures per arm;
  projection parameters are BF16 and LN affine parameters are FP32. The benchmark
  reference remains the unmodified PyTorch module.

The FP32-expanded independent formula and the ordinary BF16 PyTorch module have
expected rounding differences from the fused kernels. They are distinct from
the zero-difference old/new comparison. These tests cover the measured selections,
not every valid configuration in the search space.

## Implementation

The two saved-xn native schedules retain their fusion boundaries, configuration
axes, BF16 rounding boundaries, FP32 LayerNorm affine parameters, and residual.
Forward does not materialize the expanded activation in HBM. Backward retains
h + stacked dA/dB, the four cuBLAS GEMMs, and native LN/residual backward.

- Replace generic shared-memory pointer IO with explicit PTX shared loads/stores.
  Pack adjacent BF16 pairs for the WGMMA accumulator layout. Final forward output
  still rounds to BF16 before the residual addition.
- Remove CTA barriers where the per-thread TMA wait or completed WGMMA already
  establishes the dependency. Keep barriers for shared-H publication/reuse.
- Backward emits h, dA and dB with three TMA stores. A proxy fence and CTA barrier
  publish the shared tiles; TMA read completion and a CTA barrier protect reuse.
- Retune D384 streamed-K forward to BK128 and two output groups. Shapes are
  represented by measured configurations, not shape-specific kernel branches.

Public `Transition(..., implementation="cuda", cuda_variant=...)` remains the
explicit experimental entry. The two variants are not automatically promoted
into the existing H100 auto dispatch or build-all registry.

## Evidence and comparison scope

Actual Triton cubins were disassembled, with their PTX reassembled by Triton's
bundled ptxas 12.8.93 as a separate diagnostic. Native CUDA uses NVCC/ptxas 12.9.
The native tuning logs retain ptxas register/spill diagnostics. These compiler
versions are different and are recorded rather than assumed equivalent.

Triton already uses WGMMA. The examined Triton cubins do not use TMA loads;
most use cp.async/LDGSTS, while large full-K stage-1 kernels use synchronous
loads. Native CUDA uses explicit TMA and WGMMA. These instruction choices alone
do not establish a performance advantage.

D256 full-K forward prototype: registers 255 -> 241, and local spill loads/stores
were eliminated. NCU SM throughput increased from 44.53% to 56.56%. The identical
NCU replay measured 672.67 -> 533.25 us; these profiling times must not be mixed
with CUDA-graph benchmark latency. Both retain approximately 12.5% occupancy.

A same-format comparison against large-D Triton full-K is not a comparison
against the fastest Triton path. The module comparison uses full-K Triton for
D128/256 and split Triton for D384/512. Kernel-only and complete-module results
are reported separately.

## Rejected or unpromoted experiments

- Small independent hidden-tile CTAs increased setup and memory traffic costs.
- Direct packed global stores from the WGMMA layout were poorly coalesced.
- Combining weight pipelines and overlapping squeeze with expand did not win.
- Explicit shared IO without TMA output stores regressed some backward shapes.
- Delayed TMA-store waits have mixed small gains; the synchronized version is
  the integrated implementation in this record.
- Scalar register-H generation did not resolve wide forward performance.

Raw experiments, cubins/SASS/PTX, ptxas logs, and NCU reports:
`/home/psk6950/MiniWorld/runs/transition_cuda_opt_20260919/`.

Final measurements and validation are collected by `final_worker.sh` using the
existing `bench_module_transition` fixture, static compilation, manual CUDA
Graph capture, nonzero squeeze weights, and both inference and training.
