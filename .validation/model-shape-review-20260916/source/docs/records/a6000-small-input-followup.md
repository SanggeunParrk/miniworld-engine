# A6000 small-input performance follow-up

Recorded 2026-09-10. **Deferred at the user's request.** This record does not implement
a launch-path change or authorize a new tile sweep.

## Observed problem

At L=128, MiniWorld and cuEquivariance TriangleMultiplication / TriangleAttention
training steps were slower than compiled PyTorch despite shorter profiled GPU kernel
time sums. This supports investigating host launch/dispatcher overhead. It does not
establish that every small-input module has the same bottleneck.

The isolated diagnostic used RTX A6000 on gpu04, BF16, actual `torch.compile`,
CUDA Graph **OFF**, forward + backward with fresh gradient resets and no optimizer.
Training augmentation was 48; triangle operations have no augmentation axis and
their pair input was `[1, 128, 128, 128]`. Each target's three implementations shared
one allocated GPU. Direct-loop times are medians of three 20-step repetitions.

| Target | Implementation | Direct loop ms/step | Profile GPU kernel sum ms/step |
|---|---|---:|---:|
| TriangleMultiplication | PyTorch | 0.8600 | 0.8266 |
| TriangleMultiplication | cuEquivariance | 1.9745 | 0.6062 |
| TriangleMultiplication | MiniWorld | 1.9667 | 0.5537 |
| TriangleAttention | PyTorch | 1.0917 | 0.8801 |
| TriangleAttention | cuEquivariance | 1.5353 | 0.7021 |
| TriangleAttention | MiniWorld | 1.5599 | 0.5923 |

These are diagnostics, not the repeated native CLI result table. The diagnostic
uses the official target setup and measured callable but bypasses CLI-only setup
such as the memoized launch recorder. Graph-break counters were empty and vendor
GPU execution was confirmed. Absolute timings varied between runs.

**Do not subtract profiled GPU sums from separately measured step times to report
CPU overhead.** Profiling changes execution overhead. A launch gap can occur when
the GPU exhausts queued work while the CPU prepares subsequent calls; it need not
be an explicit wait in the code.

## Work to resume later

1. Inspect one small pure-Triton region first: `gate_elem_bwd_ew` and its consumers
   in `src/miniworld_engine/kernels/trimul_inproj/triton/unidirectional.py`.
   Consider compiler-visible direct Triton calls, or `torch.library.triton_op`
   with `wrap_triton` when registration is needed. The current `custom_op` body is
   opaque to the compiler. This proposal has no measured speedup yet and does not
   imply automatic kernel fusion.
2. Move remaining invariant device/config decisions out of repeated launch paths
   only where measurements show they remain. Some decisions already use
   `device_constant`; do not duplicate that work.
3. Consider fusing compatible small kernels to reduce actual GPU launches.
   Grouping launches into one custom op alone does not reduce GPU launch count;
   placing traceable matrix operations inside it can also hide compiler work.

Keep input and parameter gradients, saved intermediates, layout/alias correctness,
and execution verification intact. Previous diagnostic A/B changes to the compile
witness (0.18%) and alias checks (0.72%) did not establish meaningful improvements;
removing the duplicate outer gradient reset did not help.

## Acceptance conditions

- Compare identical workloads at L=128 and L=384 on the same A6000, sequentially,
  using native benchmark CLI repetitions and actual compilation evidence.
- Preserve inference augmentation **5 / Graph ON** and training augmentation
  **48 / Graph OFF**. Do not enable training graphs to hide this bottleneck.
- Freeze existing tile choices and cache contents for a launch-path A/B; report
  any numerical or full-gradient regression. Separate later tile tuning from it.
- Compare PyTorch and cuEquivariance where equivalent implementations exist;
  require a repeatable whole-step improvement, not only a shorter kernel sum.

Evidence at recording time: sibling workspace
`../team-gm/mw-cueq-training-isolated/summary.{md,json}` and raw traces;
`../team-gm/mw-trimul-training-profile-v2/summary.{md,json}`;
`../team-gm/mw-augmentation-bench/results.{md,json}` (native L128 matrix);
`../team-gm/mw-l384-modules/results.{md,json}` (native L384 matrix, six unmeasured cells).
These external artifacts are host-local and are not shipped with this repository.

Source snapshot: `4fa017e5636bb2a393ebb1dfc3a7231de48bfe1644c76b35bb767d375bf2e8b3`.
API rationale: [PyTorch user-defined Triton kernel tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html).
