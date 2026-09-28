# Vast large-D TriMul: bounded first optimization pass

D512/L384 has a small measured improvement: input-weight gradients use **4 FP32
splits instead of 8**, with cuBLASLt index 0. Across three paired repetitions on
each H100, full F+B time fell **0.3–1.3%** and backward time fell **0.7–2.0%**.
This is an explicit research selection, not an automatic production-module change.
No large or universal speedup is claimed.

## Paired complete-workload measurements

Milliseconds; CUDA Graph replay, same live tensors, alternating baseline/candidate
order, 75 samples per repetition. The baseline is the latest qualified standalone
checkpoint24, not the older installed engine wide path. Absolute timings drifted
with sustained workload; use the paired comparisons, not unrelated endpoints.

| GPU | Repeat | Baseline full | Split4 full | Baseline BWD | Split4 BWD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 7.066 | 7.001 | 4.471 | 4.414 |
| 0 | 1 | 7.190 | 7.166 | 4.506 | 4.476 |
| 0 | 2 | 7.226 | 7.195 | 4.516 | 4.483 |
| 1 | 0 | 6.996 | 6.905 | 4.453 | 4.364 |
| 1 | 1 | 7.104 | 7.035 | 4.441 | 4.405 |
| 1 | 2 | 7.181 | 7.113 | 4.466 | 4.433 |

The active FP32 partial payload falls from 64 MiB to 32 MiB. The backing workspace
allocation is retained, so this is not a claim of halving peak memory. Forward,
GP and dX arithmetic are unchanged. D256, D384 and D512/L768 retain their previous
selections; do not apply the shorter split reduction to long shapes.

## Validation

- Original five-fixture strict validation passed: normal, changed inputs/weights/
  mask/dropout, zero gamma, zero mask and zero dropout. Forward/dX matched the
  strict native reference; worst input-weight relative L2 was about 4.50e-4
  against the unchanged 5e-4 limit. Affine limits remain 5e-6.
- Changed-input CUDA Graph replay and independent PyTorch output/full-gradient
  checks passed. The independent BF16 mathematical oracle uses the original
  validator's separate 0.005 output / 0.01 gradient thresholds.
- `selected_vast.py` was actually exercised: split count 4, Lt index 0, finite
  full results and changed-input graph error below 5e-6. Its source hash is in
  `selected-entry-check.json`.
- Compute Sanitizer **2026.1.1.0** passed full-workload memcheck, racecheck and
  synccheck with zero errors, hazards and warnings. No vendor-kernel filter was
  used. The baseline split8 path also passed this version's racecheck.

The initial 2025.4.1 tool reported 22 hazards in cuBLAS `nvjet_*` kernels. Those
logs are retained under `*.2025-4-1.log`, not discarded or counted as passes.
NVIDIA documents Hopper warpgroup racecheck false-positive fixes in the
[2026.1 release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html).
The newer tool is isolated at `/workspace/tools/sanitizer132`; the compiler,
Torch, driver and shared cuBLAS installation were not upgraded for this fix.
The unchanged split8 baseline also reported 21 hazards with 2025.4.1, but zero
with 2026.1.1.0 (`baseline8-racecheck-{old,new}.log`). This supports a tool-version
issue rather than a split4-specific regression; the old logs remain available.

## Reproduced baseline matrix

All six shapes passed the historical five-fixture strict/graph/independent
validation in the Vast runtime. These are standalone graph medians from the
baseline pilot; they are not substituted for the paired candidate comparisons.

| D | L | Full F+B ms | BWD ms |
| ---: | ---: | ---: | ---: |
| 256 | 384 | 2.608 | 1.771 |
| 256 | 768 | 10.826 | 7.498 |
| 384 | 384 | 4.567 | 2.889 |
| 384 | 768 | 19.269 | 12.507 |
| 512 | 384 | 6.956 | 4.460 |
| 512 | 768 | 28.583 | 19.239 |

D256 uses the pool checkpoint, D384 checkpoint23 (checkpoint24 leaves D384
unchanged), and D512 checkpoint24. CUDA 12.8.93 compiles the native kernels.
Frozen Lt algorithm identities require the wheel's cuBLASLt 12.8.4; all identity
assertions remain active. Python package metadata alone did not identify the
loaded Conda 12.8.5 library, so wrappers/explicit paths pin the tested library.

## Rejected candidates and limits

- Exact-output Lt retuning: no improvement of complete F+B.
- Three output-weight overlap phases: neutral or slower.
- Fourteen saved-forward tile configurations: bitwise saves/strict outputs, but
  slower complete workloads.
- CUDA 13.1 recompilation of unchanged kernels: runtime errors, rejected. The
  original compiler/cubins were preserved; fresh CUDA probes and original strict
  validation passed afterwards.
- Input-weight splits 6/8/12/16: smaller gains or neutral; select only split4 at
  D512/L384. Numerical margin is limited, so broader shapes/dtypes are unqualified.

NCU/performance counters and SOL were not measured, per the user's instruction.
No production dispatch/autograd registration was changed. Use the explicit plan
factory in `selected_vast.py`; historical research sources are hash-frozen in the
separate snapshot described in README.md.

## Reproduce and hand off

From the local checkout (the helper owns source/GPU locks):

```sh
scripts/vast-sync.sh run 0 bash experiments/trimul_large_d_vast/env.sh python experiments/trimul_large_d_vast/check_selected.py
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/env.sh python experiments/trimul_large_d_vast/repeat_input_split.py --width 512 --length 384
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/env.sh python experiments/trimul_large_d_vast/qualify_input_split.py
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/sanitize.sh
scripts/vast-sync.sh pull
python3 experiments/trimul_large_d_vast/summarize.py
```

Remote raw records: `/workspace/vast-results/trimul-large-d/`.
Local mirror: `.bench/vast-20260927/results/trimul-large-d/`.
`summary.json` checks qualification markers, source hashes and actual selected
entry execution. `CANDIDATE.json` records the final selection and scope.
