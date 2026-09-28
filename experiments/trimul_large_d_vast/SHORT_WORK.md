# Short large-D follow-up

User requests fixing the smaller Triton speedup at L384, D256/384/512.
No production dispatch changes are made before complete qualification.

## Current comparison baseline: PyTorch

The user requested replacing the engine Triton comparator with PyTorch after
the LN dispatch diagnosis. `compare_pytorch.py` uses the frozen fixture's pure
PyTorch `ref` (F.layer_norm/F.linear/einsum), fullgraph static torch.compile as
the main comparison and eager as a separate reference measurement. Both use
manual CUDA Graph capture; compilation and CPU launch overhead are excluded.
Native kernels remain selected_vast.make_plan (including D512/L384 split4), or
the installed specialized D128 path. Each comparison is paired in one process.
Normal and changed-input/weight/dy/mask/dropout graph checks use the existing
independent-reference tolerances, not relaxed native-candidate qualification.

Remote output `/workspace/vast-results/trimul-pytorch-v1`; local logs
`.bench/pytorch-compare-gpu0.log` and `.bench/pytorch-compare-gpu1.log`.
Initial handles 71011/2857 completed D128/D384 then stopped on failed optional
retained-forward BWD graph checks at D256/D512 L384. Matching capture streams in
v2 did not fix D256 (handle 65637); do not claim that diagnostic is qualified.
The requested full F+B graph rebuilds forward state on each replay and passes
all checks. Isolated v3 defaulted to full scope; handles 92968/32307 completed
D256/D512 at both lengths, compiled and eager. All 16 full F+B records passed;
see PYTORCH_LATENCY_TABLE.md. Raw v1/v2 failures are retained, not counted as passes.
There are no remaining jobs from these queues. Check for other sessions before
using GPUs or modifying shared source.
The older Triton records below remain historical evidence, not the new baseline.

## Earlier short-large-D experiments

- Same-runtime Triton comparison: `compare_short.py`. Each scope is paired via
  manual CUDA Graph replay; Triton uses static compilation, heuristic-24 tuning,
  live cloned tensors and full gradients. Raw results: `/workspace/vast-results/trimul-short-large-d`.
- `partition_front.py`: output-channel CTA partitioning. D512 is bitwise on all
  saved tensors but regresses full time by 1-6%; reject. Initial D384 variant
  deadlocked because its start-offset credit used the offset channel index.
  Only that verified experiment process was terminated. Do not use this variant.
- `front_n128.py`: m64n128 WGMMA replaces two m64n64 instructions/groups while
  retaining K order and the two original gate/save epilogues. Bitwise saves,
  but tested full-workload speedups are 0.87-0.99x; rejected.
- `short_weights.py`: jointly tune split count and Lt algorithm using strict
  final weight gradients, not requiring bitwise-identical FP32 partials. Split2
  fails strict gradients. D512 split4/index0 reproduces the already qualified
  roughly 1% win; D384 best pilot is 1.007x, not qualified or selected.
- `gp_lut.py`: BF16 sigmoid lookup tables pass all 65,536 input-bit patterns and
  exact GP checks. Shared/global variants regress full time (0.88-0.93x).
- `input_whole_tma.py`: combine per-channel input-LN TMA transactions without
  arithmetic changes; strict checks pass, full speedups 1.000-1.003x. Not selected.
- `compiler_schedule.py`: CUDA12.8 register-usage levels 0/2/8/10; exact saves,
  no spills, no meaningful end-to-end gain (best about 1.0016x). Not selected.
- `triton_saved_front.py`: untested D512-only alternative front. Do not select.

Completed isolated copies: `trimul-short-v2`, `trimul-short-n128-v1`,
`trimul-short-weights-v1`, `trimul-short-v3`, `trimul-short-lut-v1`,
`trimul-short-input-v1`, `trimul-short-schedule-v1` under `/workspace/experiments`.
Never edit a copy while its jobs run. All runs hold the usual source/GPU locks.

The user's current question is why the relative speedup falls at short lengths
and why D128 is different. Historical full timings scale L384->L768 as follows:
Triton/native: D128 4.00/4.00, D256 4.92/4.01, D384 4.52/4.17,
D512 4.56/4.07. This does not establish launch overhead or a hardware bottleneck.
D128 uses specialized B1/B7, including a fixed-size global ring in B7; wide
checkpoints materialize full intermediates and call separate GEMMs/LN stages.
The ring is global memory, not an assertion that all data avoids HBM.

`profile_length_scaling.py` measures same-runtime paired CUDA Graph workloads
and separately records 3-replay CUPTI activity. Native baseline is the frozen
wide checkpoint, or installed specialized D128. Native activity can be perturbed
by profiling; use paired graph times for speedup, not the sum of activity events.
Isolated source: `/workspace/experiments/trimul-length-profile-v1`.
Results: `/workspace/vast-results/trimul-short-large-d/length-profile-v1`.
Managed handles GPU0=52751 (D256 then D128, each L384/L768), GPU1=82185
(D512 L384/L768) completed successfully. Local logs `.bench/length-profile-gpu[01].log`.
No NCU is being attempted.

**Diagnosis confirmed:** Triton output LN switches from atomic to canonical
persistent at M>=300000. D128 output LN grows 0.094->0.449 ms, D256
0.223->4.176 ms, D512 0.792->6.247 ms. Native wide full times grow almost 4x.
The high long-length relative speedup is substantially due to this Triton cliff.
See LENGTH_SCALING.md for measurements, actual configs and source pointers.

`ablate_output_ln.py` then forced only the Triton LN path to atomic at L768:
full default/atomic D128 6.357/6.258 ms, D256 18.227/14.922 ms, D512
43.390/40.155 ms. This is diagnostic-only, NOT an equivalent selected candidate:
different reduction/arithmetic ordering fails the unchanged strict dX/affine
gates. Both first-run failures and second-run `strict_equivalent=false` are
retained. Isolated source `/workspace/experiments/trimul-ln-ablation-v2`;
results `trimul-short-large-d/ln-ablation-v2`; handles 18193/90899 completed.
There are no remaining jobs from this follow-up. Check for other sessions before
using a GPU. No production code or default was changed by this diagnosis.

Same-runtime direct L384 comparisons from `compare_short.py` gave full speedups
D256 1.417, D384 1.439, D512 split4 1.436. These differ from the historical
1.35/1.39/1.35 baselines due to runtime/tuning/measurement and are NOT new
optimization wins. The follow-up pilots above have not fixed the short-shape
deficit. CANDIDATE.json and selected_vast.py retain only the earlier qualified
D512 split4 change; production dispatch is unchanged.
