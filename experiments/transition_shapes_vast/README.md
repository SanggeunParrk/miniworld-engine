# Transition pair-shape expansion, Vast H100, 2026-09-27

Scope: BF16 B1, pair L384/768, D128/256/384/512, expansion 4, residual
included, no dropout, all parameter and input gradients. TriMul work is closed.
The installed Transition already has native CUDA implementations for all eight
cells; this work qualifies the complete matrix and compares the existing wide
save-h option. Performance baseline is actual `Transition(implementation='pytorch')`
with `torch.compile`, not engine Triton. Both sides use fullgraph compilation and
manual CUDA graphs of fresh forward + backward, with no retained-forward BWD.

Follow root AGENTS.md and docs/operations/vast-h100.md for locks and sync.
Source is local-authoritative. Do not change shared source while jobs run.
GPU0 queue: D128/384. GPU1 queue: D256/512. Results are isolated under
`/workspace/vast-results/transition-shapes-v1/gpu0` and `gpu1`.

`bench.py` uses randomized nonzero squeeze weights and LN affine parameters,
checks actual native module dispatch, existing full-gradient numerical gates
against engine Triton (correctness only), eager/compiled graph agreement,
changed-state replay, source hashes, and interleaved paired timings. It compares
save-h against recomputation using the existing strict acceptance thresholds:
bitwise except LN atomics (<1e-5) and changed dWs rounding (<2e-3).
`--sanitize` runs actual full-shape native F+B and changed-state graph checks
without the unrelated compiler/reference work. No global save-h default change
is implied: retaining h adds M * 4D * 2 bytes of live activation per layer.

Example (after explicit source push):

```sh
scripts/vast-sync.sh run 0 bash experiments/trimul_large_d_vast/env.sh bash experiments/transition_shapes_vast/run.sh /workspace/vast-results/transition-shapes-next/gpu0 128 384
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/env.sh bash experiments/transition_shapes_vast/run.sh /workspace/vast-results/transition-shapes-next/gpu1 256 512
scripts/vast-sync.sh pull
```

Do not treat incomplete JSON or failed assertions as qualified results. Use a
fresh result directory for a new source revision; preserve the archived v1/v2
and failed diagnostic records used by this report.

The regular benchmark width list in
`benchmarks/modules/transition/configs/bench.yaml` now also includes D384.
The complete 2 x 4 training matrix is explicit in `run.sh`; ordinary benchmark
inference lengths and runtime kernel defaults are unchanged.

Generate the table only from the completed benchmark directory (exclude sanitizer
JSON, which intentionally contains no latency measurements):

```sh
python3 experiments/transition_shapes_vast/report.py .bench/vast-20260927/results/transition-shapes-v1 --output experiments/transition_shapes_vast/RESULTS.md
```

Sanitizer queue after all source pushes:

```sh
scripts/vast-sync.sh run 0 bash experiments/trimul_large_d_vast/env.sh bash experiments/transition_shapes_vast/sanitize.sh /workspace/vast-results/transition-sanitize-next/gpu0 128 384
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/env.sh bash experiments/transition_shapes_vast/sanitize.sh /workspace/vast-results/transition-sanitize-next/gpu1 256 512
```

This checks all eight shapes with memcheck and every width at L768 with
racecheck/synccheck, including both save-h states at D384/512. Exit code 86 means
a sanitizer error. Memcheck/synccheck and D128 racecheck use the full correctness
and changed-state graph protocol. Wider racechecks use one complete F+B per
save-h state (`race_once.py`), avoiding repeated identical expensive instrumented
WGMMA passes; their graph replay is checked separately. Completed memcheck and
synccheck records with matching harness hashes and zero-error logs are reused.
D384/512 racecheck is filtered to `kns=wide_` (all handwritten forward, squeeze,
gate and no-h gate kernels); vendor GEMMs and unchanged Triton LN autotuning are
excluded from race instrumentation. Full F+B still executes, and memcheck and
synccheck remain unfiltered. Earlier unfiltered wide racecheck runs were stopped
for runtime, without a reported hazard, and retained as incomplete `race-once`
logs. They are not counted as passes. Filtered full-module logs use `race-native`.
Those filtered full-module runs subsequently failed with `Internal Sanitizer
Error: The Sanitizer failed to handle a hardware exception`, followed by
`CUBLAS_STATUS_EXECUTION_FAILED` at the first weight-gradient GEMM. D384 reproduced
this with both 2026.1.1 and 2025.4.1; D512 with 2026.1.1. These are retained failed
checks, not passed native-kernel hazard reports.

`race_kernels.py` isolates the wide forward, squeeze, gate and no-h gate kernels
at L768, with explicit synchronization after each launch group. It checks finite
outputs and bitwise equality of dAB with and without h materialization. Its
selected logs are `race-isolated-v2`. This is narrower than full-module racecheck:
vendor GEMMs and Triton LN are not race-instrumented. Full-module memcheck,
synccheck, gradients and changed-state graphs remain separate required checks.
Use the isolated 2026.1.1 sanitizer; see the
operations document for why the older bundled version is not the reference.

Existing small-shape FP32-oracle, dispatch, numerical and compile test records:
`/workspace/vast-results/transition-tests-gpu0.xml` (D128/384 + compile/cache tests)
and `transition-tests-gpu1.xml` (D256/512). These are complementary to the full
pair-shaped graph/gradient checks, not substitutes for them.

## D128 input-slot race discovered during expansion

Initial full L768 race-instrumented graph validation failed with dX relative
error 0.18650 and LN dgamma 0.02572, while all other outputs/gradients agreed
exactly. Racecheck itself reported zero hazards. The failed record is retained
under `transition-sanitize/gpu0/racecheck`; it is not a passed qualification.
The initial GPU1 repeated racecheck was explicitly stopped when the protocol
was reduced to one full F+B per wide variant; that incomplete log is retained.

`input_role` in the D128 backward released its input slot from each warpgroup
leader without waiting for the other warps' scalar x/dy epilogue reads. WGMMA
retirement earlier in the tile does not order those later scalar reads. The
fix adds a warpgroup barrier immediately before release. The extension receives
the distinct `_input_barrier_v1` cache identity. A persistent-refill graph
regression test was added to `test_transition_fused_sm90a_gpu.py`.

After the fix, D128 test/compile results are in `transition-tests-d128-fixed.xml`;
fresh paired PyTorch timings are in `transition-shapes-v2/gpu0`. New sanitizer
logs for D128/384 are under `transition-sanitize-fixed/gpu0`. Wide D256/512
continues in `transition-sanitize/gpu1`. The six wide timing records remain v1
because their kernels did not change. To render the final selected table:

```sh
python3 experiments/transition_shapes_vast/report.py .bench/vast-20260927/results/transition-shapes-v1 --d128-fixed .bench/vast-20260927/results/transition-shapes-v2 --output experiments/transition_shapes_vast/RESULTS.md
```

`compare_d128_fix.py` loads the frozen pre-fix extension by its original cache
identity and compares it directly to the installed corrected extension, with
bitwise full-output/all-gradient checks and 150 interleaved timing samples.
Both lengths passed bitwise equality. The paired full F+B medians were
0.532800 -> 0.533488 ms at L384 (+0.13%) and 2.153200 -> 2.149200 ms at L768
(-0.19%). These are direct eager-native graph A/B results; the main PyTorch
table uses compiled modules and its own paired timing run.

The v1/v2 kernel manifests differ only in `cuda/fused_sm90a.py` and
`cuda/transition_fused_bwd_sm90a_kernel.cu`. No wide CUDA/Triton source or benchmark
math changed. The D128 availability gate rejects the other widths before its
extension loader; the wide measurements therefore retain their original identity.

`audit.py` verifies every selected benchmark/source hash, all 16 selected
sanitizer records and logs, the three pytest XML files, and the two paired
pre/post-fix records. It retains the failed mixed-racecheck logs and records the
wide racecheck scope explicitly in `VALIDATION.json`. Run it after result pull:

```sh
python3 experiments/transition_shapes_vast/audit.py .bench/vast-20260927/results --output experiments/transition_shapes_vast/VALIDATION.json
```

Final audit passed: 8 benchmark cells; 8 full-shape memchecks; 4 unfiltered
L768 syncchecks; 4 L768 racechecks with the explicit scope above; 22 + 16 original
pytest cases and 13 post-fix D128/compile cases; 2 bitwise pre/post-fix paired
comparisons. All selected sanitizer logs report zero errors/hazards. D384/512
full-module racecheck remains unqualified because of the retained internal errors;
the qualified racecheck scope there is the isolated handwritten CUDA kernels.
