# Transition CUDA streamed-K and full-K (2026-09-18)

Implementation and measurements for the two explicit development variants. SM90a,
D128/256/384/512, expansion 4, BF16 activations/projection weights and FP32 LayerNorm
parameters. Residual is part of both forward and backward. Experiments use node02.

## Delivered and measured

Both native schedules implement forward, all backward gradients and residual,
including explicit `Transition` module selection. **18 GPU tests passed**;
unfiltered Compute Sanitizer ran **10 numerical/module tests, 0 errors**.
Related import/dispatch checks: **47 passed, 1 skipped**. Source hashes and logs
are included. The sampled configuration sweep and all D128/256/384/512 ×
L384/768 module/direction measurements are complete.

- [PyTorch / prior split / current Triton / new CUDA forward comparison](FORWARD_COMPARISON.md)
- [Both native variants, full training and direction timings](RESULTS.md)
- [Selected configurations and search limits](CONFIGS.md)

The new CUDA variants do **not** beat the current Triton choices across these
shapes. Full-K native improves substantially over large-D full-K Triton, but
that Triton full-K is itself slower than the retained split. The new variants
remain explicit experiments; faster legacy paths are not replaced.

## Implementation

- `Transition(implementation="cuda", cuda_variant=..., cuda_forward_config=...,
  cuda_backward_config=..., cuda_norm_config=...)` selects the new native module.
  `kernels/transition/cuda/variants.py:transition` is the lower-level autograd entry.
  It accepts `variant`, independent `forward_config`/`backward_config`, and
  `norm_config`. Production automatic dispatch is separate; these experiments do
  not change the installed training package.
- `transition_variants_kernel.cu`: C++ CuTe/CUTLASS primitives, compiled by nvcc to
  SM90a. Explicit TMA input/weight loads and WGMMA in both GEMM-based kernels.
  `streamed_k` walks input K in BK chunks within each hidden tile. `full_k` loads
  the normalized input into shared memory once before the hidden loop.
- Forward fuses expand A/B, SwiGLU, squeeze and residual. Only row tiles are in
  the launch grid. Output warpgroups partition the output accumulator; expand is
  computed once per row tile. For one output group, h is converted directly to the RS WGMMA register
  layout reused from the existing native b2b implementation. With multiple
  output groups, h is handed over in shared memory. It never reaches HBM in
  forward. Where BK >= BM, the shared h tile reuses the last consumed A-weight
  stage after its WGMMA completes. Ws TMA boxes are split by BO to respect descriptor limits.
- Backward recomputes a/b from saved xn, loads dh using TMA, and emits BF16 h and
  stacked dAB. Shared-memory exchange produces coalesced 128-bit global stores.
  The common four cuBLAS GEMMs remain: dh, dWs, dWab, dxn. `[Wa;Wb]` still uses
  `torch.cat`; that cost is included in complete training measurements.
- `transition_variant_norm.cu`: shared native LN forward and backward, FP32
  affine and statistics. Backward fuses the rounded LN dx with residual dy.
  Persistent vectorized rows accumulate parameter gradients privately, followed
  by a separate reduction. Its channels-per-CTA tile is configurable (1/4/16/32)
  to distribute small-D reductions over more SMs without adding a kernel. The residual branch does not enter dgamma/dbeta.
- Forward rounding is `BF16(BF16(h @ Ws.T) + x)` with BF16 h. Backward similarly
  adds residual after rounding the LN derivative to BF16. Gamma may be zero.

## Configuration axes

`bk`, `bn`, `bo`, `mgroups`, `ngroups`, `stages`, `min_blocks`. BM=64*mgroups.
BO/ngroups partition forward output; backward has no squeeze and uses one output
warpgroup per row group. Its public candidate list canonicalizes BO=64. Both
kernels have a one-dimensional row grid, so GROUP_M is inapplicable.

The public candidate generator exposes a broader search space than the bounded
first sweep. Hardware resource exclusions and compilation/numerical failures are
retained in the sweep logs. A sampled winner is not an exhaustive optimum.

## Reproduction and evidence

Work directory: `/home/psk6950/MiniWorld/runs/transition_cuda_variants_20260918`.
All compilation, GPU checks, sanitizers and benchmarks run inside the node02
Slurm allocation; the login node is used for source/docs only.

- `configs.py`, `prebuild.py`, `tune.py`: bounded candidates, native resources,
  independent forward/gate winners and a runtime LN configuration sweep. `tune_norm.py` independently selects
  the final five-axis LN configuration shared by both variants;
  `norm-selections.json` takes precedence over the initial LN entries in core
  sweep files.
- `tests/numerics/test_transition_cuda_variants_gpu.py`: output/six gradients,
  independent PyTorch formulation, zero gamma, residual identity, tail rows,
  static fullgraph compile and complete training CUDA Graph replay.
- `measure_module.py`: repository `bench_module_transition` harness, static
  compilation, manual graph, identical nonzero squeeze and FP32 LN affine.
- `breakdown.py`: training forward, retained-graph backward, full autograd.grad
  step, inference, peak allocated memory and isolated common cuBLAS operations.
  These timings use autograd.grad rather than module `.grad` accumulation.
- `profile_gate.py`, `profile.sh`, `audit_sass.py`: NCU and emitted instruction
  evidence. NCU replay durations are diagnostic, not benchmark timings.
- `initial/`, `vector_store_only/` and `tma_ss/`: earlier implementations/results retained
  to make the optimization history auditable. Their timings are not final results.

## Explicit module use

The normal Triton/automatic dispatch is unchanged. The CUDA module requires an
explicit variant and both configurations; it does not silently choose an
unmeasured default. An `engine_backend="triton"` setting intentionally rejects
explicit CUDA, so use `engine_backend="auto"` for this option.

```python
import json
from pathlib import Path
from miniworld_engine import settings
from miniworld_engine.modules.transition.module import Transition

settings.configure(engine_backend="auto")
record = Path("docs/records/transition-cuda-variants-20260918")
d, variant = 256, "full_k"
r = json.loads((record / f"tune-{variant}-D{d}.json").read_text())
n = json.loads((record / "norm-selections.json").read_text())[str(d)]
model = Transition(
    d, implementation="cuda", cuda_variant=variant,
    cuda_forward_config=r["best_forward"]["config"],
    cuda_backward_config=r["best_backward"]["config"],
    cuda_norm_config=tuple(n["best_norm"]["config"]),
).cuda().bfloat16()  # LN affine stays FP32
```

This is an explicit experimental implementation. `build all` registration and
portable autotune-cache selection are not implemented for these new variants.
The recorded sweep uses a bounded set of candidates, not the whole public grid.
Module and direction measurements are recorded in [RESULTS.md](RESULTS.md).

## Measurement limits

- CUDA uses a bounded native seed sweep plus large-D MG2/S1 candidates. The
  full public candidate grid, including every `min_blocks` choice, was not run.
- Triton segmented forward compares five configurations per schedule/width.
  Untuned Triton split/gate/LN cache misses use the repository's heuristic cap
  of 24 candidates. These are reproducible sampled baselines, not a claim of
  globally optimal Triton or CUDA.
- Native core configurations are selected at L384 and reused at L768. The
  shared native LN has a separate 96-candidate-per-width screening with its
  finalists retimed. Its tuning record predates the final host initialization
  fix; the GPU algorithm is identical, and all final module timings use the fix.
- The repository module harness measures static compiled forward with manual
  CUDA Graphs and backward gradient accumulation. The separate direction
  fixture uses fullgraph compile and `autograd.grad`; `donated_buffer=False`
  is required to replay retained-graph backward. Its full-step/peak numbers
  should not be substituted for the module harness results.
- Raw tuning checks all sampled native configurations on tail-row inputs.
  Module/direction runs also check the selected paths at the measured large
  shapes. Memcheck covers the test suite's representative configurations,
  not every combination in the public grid.
- C++ CuTe/CUTLASS is used inside the nvcc CUDA source. These are native CUDA
  extensions, distinct from the Python CuTe DSL implementations.


## Native kernel initialization fix

Initial unfiltered memcheck runs reported a `cuKernelGetFunction` invalid-handle
query inside the CUDA runtime on the first native LN launch. A standalone CUDA
write kernel and the TMA/WGMMA core were clean; an isolated native LN reproduced
the report. Eager module loading, a CUDA 12.9 runtime preload, and explicit-only
API reporting did not resolve it.

A checked `cudaFuncSetAttribute` initialization before the first LN kernel launch
removed the report in the isolated reproduction without suppressions. The shared
native LN now initializes every entry this way. Attribute values are constant
across configurations, so later calls cannot shrink the allowance of a captured
reduction kernel. This is a host initialization change; the tuned GPU reduction
algorithm is unchanged. Final full-suite and unfiltered memcheck results are
recorded separately from the earlier diagnostic logs.

## Remaining performance work

Selected D384 full-K forward uses 255 registers/thread and reaches 12.46%
occupancy, 35.20% SM throughput and 19.34% eligible scheduler cycles in NCU.
Its sampled warp stalls include CTA barriers (36.7%) and L1TEX dependencies
(34.5%). Selected D512 full-K forward also uses 255 registers/thread and
reaches 29.18% SM throughput; gate backward reaches 20.56%. These are profiler
observations, not benchmark times or an established performance ceiling.
SASS local loads/stores remain in several selected wide configurations.

The current implementation still synchronizes the CTA and waits for each
WGMMA batch within its loops. Reducing live accumulator/operand state and
overlapping producer/consumer work are plausible next experiments supported
by these observations; their gains have not been demonstrated. Backward's
row-tile/hidden-loop scheduling also needs work to match the Triton gate.

Full public-grid tuning, portable cache/build-all registration and automatic
dispatch promotion remain separate work. Reproduction scripts under `repro/`
are exact archived runners; their original run-directory paths are preserved.
