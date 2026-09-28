# TriMul CUDA config-space audit — 2026-09-27

The current hand-written CUDA training path has selected, historically tuned
schedules, but no unified exhaustive config space or per-workload search ledger.
It is not equivalent to the large declared Triton grids. This does **not** mean
that every Triton grid was exhaustively measured, or that increasing CUDA's grid
will improve complete training time.

## Scope and dispatch

- `integrations/trimul_h100.py` serves BF16/B1/L384,768, FP32 affine,
  widths 64/128/256/384/512 on SM90; the D128 special path requires 132 SMs.
- Bidirectional D128 enters `cuda/h100_training.py`: selected K1/K3,
  `b1/configs.json`, and B7 defaults (`clusters=10`, `mode=52`). B7 also fixes
  consumers=10, rings=12, hardware cluster=2 and producer registers=32.
- Other bidirectional widths enter `cuda/h100_width.py`, using
  `wide/selection.json`, fixed `tuning(D)` and `gp_config(D)`. These do not call
  `native.choose_config`. `h100_wide_forward.py` is a separate explicit plan;
  its selection JSON is not the automatic training front selection.
- Single-direction training has its own D128/L384,768 plan and selection.
  It must not inherit bidirectional coverage claims.
- CUDA inference *does* use `native.choose_config`, with the Cartesian product
  of precompiled K1/K3 variant lists. Matched-width D64/128/256/384 have
  18/30/9/3 table combinations respectively. These are selected variants, not
  all feasible tile/ring/schedule configurations. Other hidden widths differ;
  the inventory records every BF16 table entry.
- The three CuTe parity kernels share raw Triton grids (front=864,
  F567=3072, dual BWD=1152), then reject unsupported physical configurations.
  They are opt-in (`trimul_sm90_kernels` defaults to empty), and their H100
  manifest rows say `untested`. They do not establish the default hand-CUDA
  training path's coverage. Manifest status alone is not a fresh execution test.
- Native derivation wraps training in fake contracts, without a config selector;
  only the inference wrapper invokes `_select_config`. A successful generic
  build/derive cannot certify exhaustive training B1/B7 exploration.

## Retained historical evidence

`inventory.py` reads the actual CSVs, table declarations, generator functions and
retained JSONs without importing CUDA. `inventory.json` records counts, observed
axes, selections and source SHA256 values.

| Family | Retained evidence | Interpretation |
|---|---|---|
| Native K1 D64/256/384/512 | 16/21/7/3 candidate records **at each of L384 and L768**, in `trimul_widths_20260922/native-D*.json` | Substantial complete enumeration of that small generator, not a global legal domain |
| Native K1 D128 historical width probe | One retained config per length in `native-D128-nosave.json` | Does not represent all later D128 specialized research |
| Wide groups × weight splits | D64/256/384/512: 3/6/3/6 rows at L384 | Limited Cartesian sweep |
| Standalone GP | 7/8/2/3 rows at L384 | `tune_gp.py` skips `(1,128)` and slots >4 except retained front; not hardware proofs |
| Fused GP | 4/3/4/3 rows at L384 | Fixes groups/splits, varies SK and a narrow slots set |
| Wide register/shared variants | 2 register rows and 1 shared row for D384/D512 | Local follow-up experiments |
| D128 B1 gate demotion | `tune.py` scans levels 0,1,2,3 around earlier selected baseline | One-axis staged optimization, not joint search |
| D128 B7 follow-up | Three `tune128.json` records | Local search, not exhaustive ring/consumer/cluster exploration |

These are specifically inspected retained files, not a claim that no other
historical experiments exist. L768 correctness/paired results do not substitute
for an L768 BWD config sweep. The historical widths report itself explicitly
says all possible configurations were not explored; its Triton comparator could
use `autotune_miss_cap=24`.

The native K1 generator varies only `(BI,BJ)={(1,64),(2,64),(1,128)}`,
`SK={1,2,4,6,8}`, `slots={2,4,6,8}`, and one/two minimum resident CTAs.
Resource rules prune these. The CUDA template admits more tile shapes (up to
four consumer warpgroups), and divisible SK=3 for D384 is absent. Such omissions
need feasibility checks and target measurements; neither universal inferiority
nor a speedup is established by their absence.

## Vast pilot: current installed wide training K1

`sweep.py` keeps the production source/defaults untouched and instantiates its
current `h100_width.Training` plan explicitly. D256 runs on GPU0/L384 and
GPU1/L768. Both use the lock-aware helper. Each length tests all 21 existing
K1 candidates plus 10 host-feasible odd-slot candidates, keeping all other
stages unchanged. This is a **bounded K1 expansion**, not global tuning.

Checks: exact saved K1 outputs, all output/gradient relative-L2 checks against
the current baseline (dx <2e-5; affine <5e-6; others <5e-4), actual CUDA Graph
replay after changing input, weight, upstream gradient, mask and drop scale;
three alternating timing blocks, 45 samples per baseline/candidate. Timings are
complete reusable-plan F+B, including packing and all stages, not public-module
Python/autograd overhead. Baseline CUDA activity traces and source/artifact hashes
are retained. No counters or SOL claim is made.

Result directory: `.bench/vast-20260927/results/trimul-config-audit-20260927/`.
Remote directory: `/workspace/vast-results/trimul-config-audit-20260927/`.
Launcher logs: `/workspace/vast-results/trimul-config-audit-L384.log` and
`trimul-config-audit-L768.log`. Managed local execution handles were 87951 and
18930. Commands:

```sh
scripts/vast-sync.sh push experiments/trimul_config_audit_20260927/sweep.py
scripts/vast-sync.sh run 0 python experiments/trimul_config_audit_20260927/sweep.py --length 384
scripts/vast-sync.sh run 1 python experiments/trimul_config_audit_20260927/sweep.py --length 768
scripts/vast-sync.sh pull
```

This pilot's numerical gate is baseline-relative. It does not replace a new
independent-reference/sanitizer qualification before any promotion. No config
is installed by the script.

## Completed pilot results

| D256 length | Existing + added | Measured / compile rejected | Baseline F+B self-pair | Best added candidate speedup |
|---|---:|---:|---:|---:|
| L384, GPU0 | 21 + 10 | 28 / 3 | 5.472 ms | 0.9947x |
| L768, GPU1 | 21 + 10 | 28 / 3 | 22.000 ms | 0.9956x |

All 28 measured candidates per length passed the saved-value, full-output/gradient
and changed-input graph checks. The best added tuple was `(1,128,5,2,1)` in both
cases: it was slightly slower. The best existing alternative at L768 was only
0.14% ahead; L384's top result was the baseline itself (0.09% self-pair variation).
No meaningful speedup or default change is supported.

All three compile rejections per length were slots=7. The CUDA body requires
`(NBAR + 1) * 8 <= SMEM_BAR`; the host sizing helper does not model that extra
barrier. This does not invalidate the current even-slot grid, but shows why
expanding it requires more than its current shared-memory size check.

Both activity traces contain `width_k1`, `width_forward`, `width_b1`, `width_b7`
and the cuBLAS contractions. All sources and candidate cubins have recorded
SHA256 identities. `pilot-summary.json` includes result hashes. The historical
D256 K1 config sets were also checked for exact equality with the current 21
candidate generator at both lengths.

These timings belong to the packaged wide plan, which is older/slower than the
separate qualified large-D research checkpoint. They must not be presented as
that checkpoint's performance or as a regression of that checkpoint.

## Priorities after this audit

1. Define separate versioned spaces for inference, D128 specialized training,
   wide training and single-direction training; retain existing measured defaults.
2. Enumerate meaningful physical schedules, with explicit compiler/layout/occupancy
   rejection reasons. For K1, include missing SK/tile/ring cases; mirror the extra
   barrier invariant before trying odd rings. Avoid forcing Triton axis names onto
   structurally different CUDA kernels.
3. Give B1/B7 independent candidate/status histories keyed by source, workload,
   GPU/runtime, dtype, direction, save/recompute and timing regime. Distinguish
   invalid, compile-failed, numerically rejected, measured and unmeasured cases.
4. Re-run shortlisted stages with full F+B at both training lengths. Require
   independent reference, changed-input graph, sanitizer and installed dispatch
   checks before changing defaults. Preserve separate latest research baselines;
   this pilot measures the currently packaged wide path, not the faster standalone
   D256 checkpoint in the concurrent large-D experiment.
