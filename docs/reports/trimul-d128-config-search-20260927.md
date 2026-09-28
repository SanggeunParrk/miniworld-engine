# D128 TriMul config search on Vast H100 — 2026-09-27

## Result

The first 51 configurations at each length did not establish a meaningful full
forward+backward speedup. The best measured differences were 0.025% at L384
and 0.191% at L768, both from B1's dTri store tile=128. These small one-run
advantages do not justify changing the selected defaults.

| Workload | Initial baseline self-pair | Best nonbaseline pair | Ratio |
|---|---:|---:|---:|
| D128/L384 | 1.0108 / 1.0126 ms | 1.0194 / 1.0192 ms | 1.00025x |
| D128/L768 | 3.9717 / 3.9705 ms | 4.0018 / 3.9942 ms | 1.00191x |

Each pair is same-process alternating baseline/candidate. Do not compare
baseline times across separate pairs as if they were a speedup. Fixed live
mask/drop scales are supplied; a full public CUDA autograd graph executes
forward and all backward gradients, including packing/layout operations.
Fresh stochastic dropout generation and the outer model are not timed.

## Final repeat of the initial winner

The dTri store128 candidate was rerun independently with 300 alternating
samples per baseline/candidate, strict gradients, changed-input graph,
independent reference, zero gamma/mask/dropout cases and activity profiling.
All numerical checks passed, but the apparent first-round gain reversed:

| Length | Baseline full F+B | Candidate full F+B | Speedup |
|---|---:|---:|---:|
| L384 | 1.02262 ms | 1.02328 ms | 0.99936x |
| L768 | 4.06504 ms | 4.07248 ms | 0.99817x |

Thus there is no reproducible D128 speedup in this search. The packaged defaults
remain unchanged. Managed repeat handle 51123 finished successfully at both
lengths (`repeat-status.json`). No experiment from this session remains running.
No sanitizer qualification was launched because no beneficial candidate was
selected; no candidate is promoted or described as fully qualified.

## What was searched

Baseline: current `h100_training.bidirectional_trimul`, BF16 activations/weights,
FP32 affine, batch=1, D128, L384/768. Forward was preserved. B1 uses its packaged
per-length JSON. B7 defaults: 10 consumers, ring depth 12, 10 logical groups,
mode 52. Both GPUs are full 132-SM H100s.

Per length, the 51 initial entries comprise one baseline, 31 alternative B7
configurations, and 19 B1 changes. B7 explores consumers {6,8,10,12}, ring depth
{4,8,12,16}, and the device-feasible maximum group count or two fewer groups.
B1 changes CTA count, affine/parameter unroll, dTri store tile, TMA priority and
gate demotion, one axis at a time. Every initial entry passed strict complete
output/gradient and changed-input CUDA Graph comparison before timing.

Thresholds versus baseline: dx relative-L2 <2e-5, affine <5e-6, others <5e-4.
Changed graph inputs include x, projection/gate weights, dy, mask, drop scale,
and one output-gamma component set to zero. Three alternating blocks provide
90 measurements for each baseline and candidate.

Additional isolated runs compare B7 mode53/54/55/60, ring depths 10/14/20,
and a B1 joint config (affine unroll2, parameter unroll8, dTri tile128).
These add zero-gamma/mask/dropout fixtures, independent PyTorch reference
and CUDA activity traces, with longer 300-sample pairs. Independent-reference
relative-L2 thresholds are 0.005 output / 0.01 gradients for its different
BF16 rounding contract; baseline strict thresholds still apply.

## Completed isolated follow-ups

All 16 isolated executions have finished: 10 valid timed rows, 4 strict
numerical rejections (mode54/55 at both lengths), and 2 timeouts (mode60).
Each valid row also passed independent-reference and all three zero edge cases.
Mode53 speedups were 0.9996x / 0.9989x; B1 joint 0.9998x / 0.9985x.
Ring-depth alternatives were also slower at both lengths. CUDA activity traces
of the joint candidate contain `b1_fused` and `b7_joint` exactly once per F+B.

The timeouts retain incomplete JSON by design; `isolated-status.json` records
that those processes were terminated and are no longer running. The separate
mode48 stalled run is retained as an owner-terminated failed trial.

## Config-space defects exposed

- The host derives 8/16/32-KiB ring chunks from the mode, but packaged
  `consumer_compute` hardcodes eight phases and two 16-KiB slots. Mode48
  (32 KiB) stalled at candidate capture and was terminated after checking its
  exact process identity. Mode60 (8 KiB) timed out in an isolated process.
  These are unsupported schedules of this packaged body, not measured losses.
- Mode54/55 enables gate-first scheduling. At both lengths they fail strict gradients; at L384:
  dx relative-L2 is about 0.249, and input-affine gradients are also incorrect.
  They cannot be accepted as valid configs merely because compilation succeeds.
- Some historical defines, including B7_ROLLING/B7_LNPREFETCH, have no consumer
  in the packaged CUDA body. Mode still affects other real fields; counting
  all flag combinations as independent implemented schedules is misleading.
- The local next-version experiment runner now rejects incompatible chunk modes
  before launching. This is an experiment guard, not a production-source change.

The unsafe-mode observations do not affect the selected mode52 implementation.
No production source/default was changed. No candidate is promoted or claimed
sanitizer-qualified; the timing search alone is not a safety qualification.

## Reproduction and evidence

Directory: `experiments/trimul_d128_config_20260927/`.
`executed_search_v1.py` preserves the exact script used remotely; its SHA256
is checked against the result record. `search.py` is a later local runner
with artifact tracking, extra register-axis support and chunk rejection; that
new version was not used for the initial or isolated measurements.

Initial GPU0/L384 and GPU1/L768 managed handles: 80812 / 86091.
Follow-up mode48 handle: 14041 (own process terminated, exit 143).
Isolated follow-ups run sequentially on GPU0, handle 59407, with 60-second
per-candidate subprocess limits. Another session's GPU1 sanitizer queue held
the shared source lock; the attempted source push failed and was not bypassed.
Existing remote code consumed new experiment-specific JSON inputs instead.
All GPU execution remains within `scripts/vast-sync.sh run` locks.

Remote results: `/workspace/vast-results/trimul-d128-config-20260927/`.
Local mirror: `.bench/vast-20260927/results/trimul-d128-config-20260927/`.
The initial `search-L384.json` / `search-L768.json` contain every configuration,
strict errors, changed-input graph errors, all timing samples and source hashes.
`isolated-status.json` records subprocess outcomes, which must be checked in
addition to partial result JSONs. `summary.json` is generated by `summarize.py`
and ranks only rows whose recorded numerical/graph/reference/edge checks pass.

This is bounded schedule exploration, not a global optimum certificate.
Producer-register variants and a full B1 Cartesian search remain unmeasured.
