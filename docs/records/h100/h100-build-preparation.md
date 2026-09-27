# H100 build preparation — 2026-09-11

The requested allocation is one gpu-4farm node, four H100 GPUs and 112 Slurm CPUs.
The builder runs one unit per GPU with 28 compile workers per unit. Slurm's
`sbatch --test-only` accepted this resource request. This is not 112 guaranteed
physical cores: the nodes expose 224 logical CPUs across 112 physical cores.

## Correctness changes

Hopper trimul used a masked LayerNorm output as both the front projection input
and the output-gate input. The reference masks only left/right contraction
operands. The Hopper training fronts now mask those operands after projection;
the shared front backward applies the same pair mask before computing gradients.
The output gate receives the unmasked normalized input. The inference wrappers
follow the same rule and retain the original tensor rank for shape attribution.
Input and output LayerNorm epsilon values are forwarded independently in the
Hopper module training path.

Single-direction Hopper module training now passes its original parameters through
differentiable views instead of creating first-forward parameter copies. The
whole-op facade selects Hopper implementations instead of SM100-only backends.
The bidirectional inference environment switch selecting the SM100 implementation
has been removed; architecture dispatch determines this choice.

## Execution and qualification

`scripts/slurm/prepare-h100.sh` installs missing native dependencies without
upgrading the locked torch/Triton/quack/CUTLASS versions, restores the CUDA 12 TE
core, and runs CPU regression checks. It runs on a CPU compute node.

`scripts/slurm/build-h100.sh` requires that preparation to succeed. On the GPU
node it checks native dependencies, then runs mask/gradient, compiled dropout and
registered trimul/Transition/LayerNorm numerical checks. A failed check stops the
job before tuning. Only after these checks does it execute `build all grid`.
Shards, compiler artifacts and test XML remain in `.scratch/h100-build` for
inspection/resume. A killed build may require explicit orphan-claim reclamation
once no other worker uses the same shard directory.

GPU qualification and full-cache completion are not claimed by this source change.
The scripts operate on this checkout; do not edit runtime sources while a build
is running against it.

## Remaining native tuning coverage

The six CuTe resolver families `transition_swiglu_fwd`, `transition_gate_bwd`,
`dab_lnbwd`, `dgrad_lnbwd`, `layernorm_linear_m1` and `tm2_dual_fwd` still lack a
builder call to `sweep_and_cache`. A selected module can execute their native
kernels, but that is not an exhaustive native configuration search. This change
fixes Hopper correctness/routing and gates the existing build with numerical
checks; it does not certify or implement those six additional tuning searches.

## Submission evidence

- CPU environment installation job: 266951, successful.
- Native dependency preparation and CPU regression job: 266956, successful;
  lint passed and 50 CPU tests passed.
- CPU import probe: FlashAttention CuTe, quack, CUTLASS and both Hopper trimul
  training modules imported successfully; cuBLASDx include paths resolved.
- GPU validation/build job: 266961, submitted with four H100 GPUs, 112 CPUs,
  896 GiB requested memory and a 24-hour limit. At the post-submission check it
  was pending for resources, with preparation dependencies satisfied.

## 2026-09-12: CPU-only native configuration audit

The previous GPU job 266961 failed its kernel validation at M2's missing `mRstd`
epilogue entry; `build all` did not start. No replacement GPU job was submitted.

The Hopper code now routes performance settings through native candidate policies
and explicit launcher config arguments. Removed the baked M1/M2 winner table and
the unreferenced ConditionedTransition TF32 prototype. Added native measurement
capture, failure reporting, source/environment checks, exact layout/shape keys,
and native driver inclusion in the `build all` complement pass. CUDA Ninja receives
the worker's `--compile-jobs` budget. At this initial audit, CuTe candidate
compilation remained serial; the CPU follow-up below supersedes that limitation.

Further behavioral fixes cover the separate trimul inference mask path, distinct
input/output epsilons, legacy back-half config dictionaries, TM2 tail padding and
its device-specific compile cache, and the CUDA LayerNorm environment/SM-count
latches and unsupported vector-width fallback.

Validation on CPU compute nodes:

- Related combined regression selection: 84 passed.
- Final native-specific selection, including config forwarding and the CUDA-only
  environment without CuTe: 13 passed (11 overlap with the combined selection;
  86 distinct tests total).
- Eight default Hopper transition CUDA specializations compiled successfully.
- Two nondefault single-warpgroup CUDA variants and two CUDA LayerNorm launch-bound
  variants compiled successfully.
- Six M2 gate/layout/pingpong variants compiled successfully, plus explicit
  `lnl_ws=False` and `lnl_ws=True` compilation.
- Lint and `git diff --check` clean after formatting corrections.

Logs: `.scratch/hopper-final-cpu-tests.log`, `.scratch/hopper-native-final.log`,
`.scratch/hopper-cuda-compile.log`, `.scratch/hopper-config-variants.log`,
`.scratch/hopper-fused-compile-matrix.log`, `.scratch/hopper-lint.log`.

GPU numerics, race/synchronization checks and timing-based config qualification
are still outstanding for this revision. CPU compilation does not certify them.
See `docs/kernels/cute-autotune-and-config-pinning.md` for the current configuration
and invariant inventory.

## 2026-09-12: complete CPU follow-up

Native CuTe/CUDA candidates now precompile in isolated CPU subprocesses before
the GPU timing lock. The existing `--compile-jobs 28` setting can therefore use
28 compiler workers per GPU build worker (four GPU workers on the planned
112-CPU allocation). Each subprocess has CUDA devices hidden and one compiler
thread. Failed/aborted/timed-out candidates retain logs and cannot kill their
parent tuner. TM2 remains an exception: its runtime callable cache is local to
the process, so only the standalone CPU audit parallelizes its compilation.

The audit fixed an actual GPU-free nvcc failure: `--list-gpu-arch` omits `90a`
while accepting that target, so filtering removed the explicit Hopper target
and accidentally invoked PyTorch's GPU architecture autodetection. It also
found eighteen CUDA candidate combinations whose shared output shuffle did not
fit. The candidate policy now enforces the same layout constraints as CUDA's
static assertions. Registry names/stream coverage, build-grid expectations and
the generated documentation counts were reconciled through the full CPU suite.

New tests compare real trimul forward/backward orchestration against independent
PyTorch autograd for 48 mask/dropout/direction/length cases. GPU primitives are
substituted by CPU reference functions in these tests. Separately, the actual
Triton dconcat, gate-element backward and normalized-input recomputation bodies
run in Triton's CPU interpreter for 72 FP32 tail/mask/layout cases. The harness
locally adapts Triton's scalar-index conversion to the installed NumPy version.
Neither method validates BF16 GPU instructions or synchronization.

Reproduction entry points: `scripts/check-hopper-candidate-matrix.py` and
`scripts/check-hopper-triton-helpers-cpu.py`. Run these on CPU compute nodes.
The candidate matrix saves each request, compiler log and result independently,
plus a JSON summary. The build job shares its persistent Quack and CUDA-extension
cache paths with this audit. No GPU job was submitted during the follow-up.

Final evidence for the current revision:

| CPU check | Result | Evidence under `.scratch/` |
| --- | --- | --- |
| Full `pytest tests/ -m 'not gpu'` | 2,818 passed; 32 skipped; 222 GPU tests deselected | `hopper-cpu-full-suite-verified.log` |
| CuTe candidate matrix | 320/320 compiled | `hopper-cpu-cute-final/summary.json` |
| CUDA candidate matrix after layout filtering | 259/259 compiled | `hopper-cpu-cuda-final/summary.json` |
| Actual Triton helper bodies, CPU interpreter | 72/72 passed | `hopper-cpu-interpreter-final.log` |
| Ruff: src, tests, benchmarks, new audit scripts | Passed | `hopper-cpu-lint-verified.log` |
| ty with the project Python environment | Two unresolved optional FA2 imports; no other diagnostics | `hopper-cpu-types-verified.log` |

The two type diagnostics are `flash_attn.flash_attn_interface` and
`flash_attn.bert_padding` in `modules/swa_atom_attention/module.py`: the current
environment supplies FA4 but not FA2. They remain visible; no dependency or
type-check suppression was added to conceal them. The combined check job exits
nonzero because of these diagnostics even though pytest and Ruff passed.

An intermediate full-suite run failed because its harness explicitly set
`CUDA_VISIBLE_DEVICES=''`, conflicting with subprocess unit tests that simulate
GPU index zero. The final CPU-node run leaves this variable unset; native compile
workers still hide devices themselves. The interpreter also requires
`PYTHONNOUSERSITE=1` to avoid the incompatible user-site Triton installation.

GPU numerical accuracy (including BF16), races/synchronization and performance
ranking still require an allocated H100. CPU results do not certify those checks.

## 2026-09-12: final cache lifecycle review

Three additional CPU-reproducible gaps were corrected:

1. The native runtime reader now rejects a changed `build_rev`, matching the
   status/planning verdict. Previously it checked source/environment but could
   reuse a cache after the declared measurement revision changed. In-process
   winners also include the revision in their keys.
2. Native timing now uses the same paired `bench_clear_mb` / `bench_rep_ms`
   policy as Triton, preserving median selection. It no longer silently fixes
   the warmup/repetition budget at 25/100 ms. A changed budget invalidates the
   in-process winner.
3. Disabling or incompletely configuring that paired budget restores the
   original driver's cache-eviction provider. Previously its installed provider
   could remain active with a live zero-MB setting, disabling eviction.

The new `scripts/check-hopper-cache-reuse-cpu.py` warms thirteen real CuTe
compile contracts across six persistent-cache families (including all eight
M2 layout/gate/workspace branches). It then starts a fresh Python process for
each, with `cute.compile` replaced by a failure. Success requires an actual
disk-object cache hit; recompilation cannot make this check pass. TM2 is
explicitly excluded because its runtime cache remains process-local.

Final-run evidence is in `.scratch/hopper-last-review-verified-tests.log`,
`.scratch/hopper-last-review-verified-lint.log`,
`.scratch/hopper-last-review-verified-types.log` and
`.scratch/hopper-last-cache-reuse-verified/reuse-summary.json`.

Results: 2,825 CPU tests passed, 32 skipped and 222 GPU tests deselected;
13/13 cold-warm CuTe contracts loaded in fresh processes with compilation
disabled. Ruff and `git diff --check` passed. Type checking still reports only
the same two unavailable optional FA2 imports documented above. No GPU was
allocated or used for this review; GPU numerical/race/performance qualification
is still pending.

## 2026-09-12: cache fault-injection verification

Twenty-eight additional CPU cases exercise actual cache files with truncated
JSON, non-object roots, malformed entry maps/records, absent or nonpositive/
nonfinite timings, and failed native measurement rounds. The first run exposed
22 failing variants of two missing protections: malformed cache data could
raise during lookup, and unusable stored measurements could still be selected.

The loader now treats non-object JSON roots as a miss. Native selection rejects
unmeasured/invalid timings, skips malformed candidate records and retains a valid
runner-up; malformed entry maps fall back to the declared default. Legacy
non-native selection still allows its optional timing field but rejects an
explicitly invalid one. Invalid native measurements during a build remain
searched failures, with no winner or successful-entry claim in the shard.

Evidence: `.scratch/hopper-cache-fault-before.log` records the pre-fix failures;
the final full suite, lint and type diagnostics are in
`.scratch/hopper-cache-fault-full-tests.log`,
`.scratch/hopper-cache-fault-full-lint.log` and
`.scratch/hopper-cache-fault-types.log`. Object reuse and interpreter checks use
`.scratch/hopper-cache-fault-reuse/` and
`.scratch/hopper-cache-fault-interpreter.log`. These remain CPU-only checks.

Final results: 2,853 CPU tests passed (including all 28 new fault cases), 32
skipped, 222 GPU tests deselected. All 13 object-cache reuse checks and 72 FP32
Triton interpreter cases passed. Ruff and `git diff --check` passed; type checking
retains only the two previously documented missing optional FA2 imports. No GPU
job was submitted. H100 execution accuracy, synchronization and timing still
require separate GPU qualification.

## 2026-09-12 17:29 KST: build all submitted

On explicit user request, submitted job **267166** (`mw-h100-build`) to
`gpu-4farm`: one node, four H100 GPUs, 112 Slurm CPUs, 896G memory and a 24-hour
limit. The script runs GPU numerical checks first, then `build all grid` with
28 compile workers per GPU, one unit per GPU and resume enabled.

Submission was confirmed as `PENDING (Resources)`. Slurm's initial projected
start was 19:44:35 KST on node27; this is a scheduler estimate, not a reservation
or guaranteed start. Output/error log:
`.scratch/h100-build/build-267166.log`. The job uses the current checkout and
shared compiler caches. Submission does not establish GPU qualification or
successful build completion.
