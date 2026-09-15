# Hopper native configuration audit

Updated 2026-09-12. Scope: maintained SM90 execution paths and the shared CUDA
LayerNorm they call. Historical `notes/` prototypes and SM100 implementations
are outside this audit.

Performance configuration belongs to `autotune/cute_config.py` or
`autotune/hopper_cuda_config.py`, with explicit configuration arguments at the
launchers. Kernel bodies no longer choose a hand-tuned block, cluster or stage
count. A cache miss uses the first declared candidate; it is a default, not a
claim that the configuration has been measured on the current device.

| Path | Configurable performance parameters | Structural constraints |
| --- | --- | --- |
| LayerNormLinear M1 | tile M/N, cluster M/N, pingpong/cooperative, scheduler, swizzle | Custom epilogue does not implement swapped A/B |
| LayerNormLinear M2 | tile M/N, cluster M/N, pingpong/cooperative, swizzle; live `lnl_ws` setting | Static scheduling; stats handshake belongs to this implementation |
| Transition SwiGLU / gate backward | tile M/N, clusters, pingpong/cooperative, scheduler, swizzle | Gate/up interleave; unswapped A/B |
| dgrad / dAB + LN backward | tile M, cluster M | Full output-width reduction, one MMA atom in N, pingpong register limits |
| TM2 dual GEMM | tile M / consumer warpgroup count | K_SW128 descriptor, m64 WGMMA atoms, all input buffers resident in shared memory |
| CUDA transition B2B | BN, stages, minimum resident blocks | Two cooperating consumers and the m64n128 squeeze atom |
| CUDA transition expand / gate backward | BN, KT, stages, consumer warpgroups, minimum resident blocks | WGMMA instruction tiles and shared-memory/layout assertions |
| CUDA LayerNorm forward | block threads | Warp-sized, power-of-two block |
| CUDA LayerNorm backward | warps, waves, reduction block, vector transaction, minimum resident blocks | Vector alignment and maximum supported register width; scalar fallback for other widths |
| Triton helpers in CuTe pipelines | Existing registry CSV grids | Reduction/packing math |
| quack GEMM/GLU and normalization adapters | Upstream library configuration/tuning | Library-owned kernels |

Warp size 32, WGMMA warpgroup size 128, instruction shapes, descriptor alignment,
barrier identities, mathematical coefficients and dtype sizes are not arbitrary
performance constants. They remain explicit. The configuration policy necessarily
contains numeric candidate values; moving a single constant to another file would
not constitute a search space.

## Build behavior

`native.choose_config` measures the declared native candidates when
`settings.run_autotune` is enabled. Normal execution only reads a cache or uses
the declared default. Explicit configs bypass selection. Unsupported fields are
rejected rather than silently discarded.

Native timings and searched candidates enter `capture` and the ordinary unit
shards. The existing single publisher merges shards; timing workers do not write
to the committed cache. Failed candidates appear in logs and coverage, nonfinite
timings cannot win, and an entirely failed round raises. Completed unit shards
use the builder's existing resume mechanism. In-process repeat calls reuse a
winner for the exact workload. Native precompilation now uses isolated CPU
subprocesses before acquiring the GPU timing lock. `--compile-jobs` bounds this
pool as well as Triton compilation; each native subprocess hides CUDA devices
and limits its compiler and numerical-library thread counts to one. Crashes and
timeouts reject individual candidates, with their requests and logs retained.
CuTe persistent objects and CUDA extensions share the launcher's cache and
compile ABI. TM2 still compiles in the timing process because its callable cache
is process-local; its candidates can nevertheless be compiled in the CPU audit.

Native cache keys include exact tensor shapes, strides, dtypes and relevant
options. Reads validate the declared build revision and native source/environment
identity and intersect with the live candidate space. `build all` explicitly
includes native drivers even when a module reaches their default; Triton
derivation alone does not prove native
configuration coverage. A Triton `--config-dir` does not exclude native drivers.

The historical M1/M2 `_tuned.py` winner tables were removed. M2 compilation was
repaired for quack 0.5: internally produced shared-memory statistics must survive
None-argument filtering and count toward the epilogue shared-memory budget; the
C-input pipeline now uses quack's current transaction-byte interface. `lnl_ws`
is read per launch and is part of the compiled variant's key. Production dispatch
still selects M1 until M2 is numerically qualified on a GPU.

The unreferenced, unregistered ConditionedTransition TF32 `b2b_fwd.cu` prototype
was removed from the runtime source tree. Its fixed launch and unmasked tail
were not a supported execution path; its history remains in Git.

## Behavior corrections

Hopper trimul normalizes the original input without a mask, applies pair masks
to left/right projection outputs, and gives the output gate the unmasked
normalized input. This includes the separate inference entry point. Input and
output LayerNorm epsilons remain separate. Main module training uses the module's
original parameters. The back-half honors its supplied LayerNormLinear config.

TM2 pads partial M/K/N instruction tiles and crops the result. Its compile cache
includes the device and architecture. CUDA LayerNorm no longer reads
`LNBWD_WAVES` or permanently caches the first device's SM count. Its vectorized
backward rejects unsupported alignments; the wrapper supplies a scalar fallback
whose row-scale and affine gradients match the mathematical reference.

## Qualification

CPU unit tests exercise actual configuration selection/capture, failure handling,
exact workload keys, build-plan inclusion, mask gradients, and epilogue setup.
CPU-only nvcc/CuTe compilation checks are in `scripts/check-hopper-*-compile.py`
and `scripts/check-hopper-config-variants.py`. They do not launch GPU kernels.

GPU numerical comparisons, synchronization/race checks and measured winning
configs for this revision remain pending. CPU compilation cannot establish those
properties. No GPU job was submitted for this audit.

## Second review (2026-09-12)

The follow-up review found integration defects that the initial selection tests
did not cover. Eight launchers recorded historical operation aliases instead of
their registry names, and the shard publisher only resolved Triton identities.
Native measurements now use registry names throughout selection, capture,
publication and runtime lookup. Cache status handles native policies without
registering nonexistent Triton CSVs. The resume generation includes native CUDA
source and configuration policy, and one cached shape cannot skip all remaining
native shapes through the builder's coarse op/dtype shortcut.

The CUDA LayerNorm backward driver now uses the requested feature width instead
of always driving 128. The M2 driver covers both input layouts and presence/absence
of its gate input. M2 rejects ignored architecture fields and includes its debug
output in selection keys and benchmark calls. Dispatch keys include the addmm
addend and the actual gate/backward operands' layouts and dtypes.

The shared trimul gate backward launcher now materializes broadcast/transposed
gradients, projections, gates and dropout scales before its row-major Triton
kernel. Its fake outputs declare the same contiguous layout. This fixes the
out-of-bounds risk from zero-stride gradients such as `output.sum().backward()`.

Regression tests exercise actual native capture -> shard -> filtered merge ->
runtime lookup for all twelve registered native operations, native staleness and
resume invalidation, requested driver widths, and gradient layout handling.
`scripts/check-hopper-family-compile.py` additionally lowers the maintained CuTe
families using fake tensors without launching GPU work.

That compilation audit reproduced a compiler abort at TM2 output tile N=24.
MMA's N%8 rule alone was insufficient for this output store layout. The launcher
now pads N to the bf16 SW32/STSM alignment of 16 elements and rejects output tiles
outside the WGMMA N<=256 instruction range before lowering. Logical ragged output
widths are cropped after the padded launch; this is a layout constraint rather
than an autotune tile preference.

Second-review validation: 131 distinct CPU tests passed (126 in the combined
suite, then 39 native tests including five additional TM2 cases); fifteen CuTe
family compilation variants passed after the TM2 fix. The latter include M1
K/M-major, cooperative and pingpong SwiGLU/gate backward, two LN backward widths,
and TM2 output widths 16/32/48/128. These are compile results, not GPU numerical
or race-check results.

## CPU follow-up (2026-09-12)

The maintained candidate matrix compiled successfully without a GPU: 320 CuTe
variants and 259 CUDA extension variants. This covers declared tile/cluster/
stage grids, M2 gate/layout/live-workspace branches, supported CUDA transition
widths, LN launch bounds and padded TM2 widths. It is not an exhaustive numerical
test over tensor shapes or dtypes. Eighteen invalid CUDA output-shuffle layouts
found during compilation are now excluded by structural constraints.

The CPU worker tests verify the exact runtime compiler signatures, concurrent
execution, crash/timeout isolation and failed-candidate capture without launch.
Forty-eight trimul autograd reference cases and 72 real Triton helper body cases
(CPU interpreter, FP32) also passed. Runtime GPU arithmetic remains unqualified.

Use the project environment with `PYTHONNOUSERSITE=1` and `PYTHONPATH="$PWD/src"`
on a CPU compute node. The audit entry points are
`scripts/check-hopper-candidate-matrix.py --backend all --jobs 32 --output <dir>`
and `scripts/check-hopper-triton-helpers-cpu.py`. Matrix logs/results are in
`.scratch/hopper-cpu-cute-final/` and `.scratch/hopper-cpu-cuda-final/`.

The final full CPU suite passed 2,818 tests (32 skipped, 222 GPU tests deselected).
Ruff passed. Project-environment type checking retains two unresolved optional
FA2 imports in the attention module because this environment installs FA4; it
reported no other diagnostics. Detailed evidence and environment requirements
are recorded in `docs/records/h100-build-preparation.md`.

The subsequent cache lifecycle review passed 2,825 CPU tests and thirteen real
cross-process object-cache reuse checks with compilation disabled in the reader.
It added build-revision validation to runtime lookup and aligned native timing
with Triton's paired benchmark-budget policy, including restoring the original
cache-eviction provider when that policy is disabled. These checks exercise
cache/control behavior, not GPU kernel execution.
