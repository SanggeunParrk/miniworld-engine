# D128 B1/B7 config search on Vast H100

Baseline is the current installed D128 bidirectional CUDA autograd path,
BF16 input/weights, FP32 affine, B1/L384 and L768. Forward unchanged.
Baseline B1 comes directly from `h100_sources/b1/configs.json`; B7 is
consumers=10, rings=12, groups=10, mode=52.

Initial search: B7 consumers {6,8,10,12} x rings {4,8,12,16} x
{maximum cooperative groups, maximum-2}. Maximum=264/(16+consumers),
rounded down and checked against actual device cluster occupancy before launch.
B1 varies CTA count {66,96,120,132}, affine/parameter unroll {1,2,4,8},
dTri store {16,32,64,128}, cache priority {0,1,2,3,6}, gate demotion {0,1,2,3}
one axis at a time around the baseline. This is not an exhaustive joint sweep.

Host B7 variants are separate in-memory modules with exact literal replacements.
The production source and dispatch defaults are not edited. B1 configuration
is injected into a process-local constructor. CUDA Graphs capture the real
`h100_training` autograd entrypoint including packing, conversions, all stages,
and gradient layout conversions. Fixed dropout scales remain live graph inputs;
this is not a fresh-RNG dropout module benchmark.

Every timed candidate requires full-output/gradient baseline-relative checks:
dx <2e-5, affine <5e-6, others <5e-4; and graph replay with changed input,
weights, dy, mask, dropout and zero output gamma component. Timing uses three
alternating blocks of 30 samples for each baseline/candidate.

Remote logs: `/workspace/vast-results/d128-config-search-L384.log` and
`d128-config-search-L768.log`. Results:
`/workspace/vast-results/trimul-d128-config-20260927/`, mirrored locally by
`scripts/vast-sync.sh pull`. Initial managed session handles: 80812, 86091.
All GPU jobs go through `scripts/vast-sync.sh run`; push only between jobs.

Qualification of a shortlist adds full zero gamma/mask/dropout edge cases,
independent PyTorch math reference (different BF16 rounding, 0.005 output /
0.01 gradient relative-L2 tolerance), source and cubin hashes, CUDA activity
trace, longer paired repeats, and compute-sanitizer. Baseline-relative strict
checks remain required; independent-reference tolerance does not replace them.
No production promotion is implied by an experimental timing win.

Follow-up execution notes:

- The attempted updated-script push returned rsync code 12 because the other
  session held source.lock for its GPU1 sanitizer queue. No lock was bypassed.
- Follow-up inputs were generated under this experiment's result directory and
  consumed by the already-synchronized runner. `executed_search_v1.py` is the
  preserved exact remote file (SHA256 checked against both first-round JSONs).
- Own mode48 process was terminated after a stalled capture (managed handle
  14041, exit 143). Isolated follow-ups use handle 59407 and a 60-second timeout
  per subprocess; `isolated-status.json` records termination separately from
  partial JSON. Successful process exit is not a numeric pass: inspect rows.
- A later local runner adds a 16-KiB chunk guard, source/cubin tracking and an
  experimental register axis. It has not been measured. Do not attribute those
  changes to earlier result files.

Final outcome: no reproducible speedup. Initial store128 gains reversed in
300-sample repeats: L384 1.02262 -> 1.02328 ms; L768 4.06504 -> 4.07248 ms.
Repeat handle 51123 completed both lengths. Defaults unchanged; no promotion.
See `docs/reports/trimul-d128-config-search-20260927.md`.
