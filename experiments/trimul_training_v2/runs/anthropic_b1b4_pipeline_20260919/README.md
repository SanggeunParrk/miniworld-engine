# TriMul B1-B4: shared-input CUDA backward

**Target not reached:** at least 1.7x versus the existing Triton/cuBLAS B1-B4 region.
The selected new implementation is `dual.cu` / `dual.py`, with `count=132, part=2`.
It is an experimental integration exercised in the complete backward benchmark;
production dispatch and defaults are not changed.

## Attribution

Anthropic `uplifting-biomolecular-modeling` commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, native v5 (Apache-2.0), supplies the
TMA/mbarrier, WGMMA, ldmatrix/stmatrix, shared descriptors and BF16 fragment primitives.
We retain the upstream code and attribution. This new backward schedule is Miniworld's
extension, not an Anthropic backward kernel. See the preceding CUDA/dW source survey
for the separately studied CUTLASS, DeepGEMM, FBGEMM and Liger implementations.

## Selected schedule

- H100 SM90a, B=1, C=128, packed H=256, BF16, L384/768; dropout probability 0.25.
- 132 persistent CTAs, 256 threads (two warpgroups) each. A CTA consumes consecutive
  64-row tiles and computes both dW matrices plus its dX/output-LN gradients.
- One B1 pass forms BF16 `dp` and `dg`; 128-bit mask loads and `dg` stores.
  `dg` is also written to HBM for the later input-gate backward.
- Six SS WGMMA m64n128 output tiles, three per warpgroup, retain FP32 dW accumulators
  across all rows assigned to the CTA. This avoids the earlier per-dW-tile rereads.
- dnorm = BF16(dp @ Wp), then output-LN row reductions in the GEMM fragment layout.
  A channel-contiguous epilogue emits dtri using TMA and accumulates dgamma/dbeta.
- The next tile's dy, gate, x_n and normalized-tri TMA transfers (80 KiB) overlap
  the current tile's LN epilogue. Projection and raw tri load after their buffers are free.
- The original forward save policy and BF16 intermediate rounding are preserved.
- FP32 dW and LN partials are published to global workspace. A cooperative grid
  barrier then lets all CTAs reduce disjoint output elements. Final dW is rounded to
  BF16 once; LN parameter gradients are FP32. This is **one CUDA launch**, but the
  small global partial-gradient workspace still exists. It is not an HBM-free algorithm.
- `part=1` is a diagnostic two-launch variant with a separate parallel reduction.
  `part=2` must use `cuLaunchCooperativeKernel`, never an ordinary oversubscribed launch.

## Measurement

`final-results.json` is the authoritative final paired run. H100 on node02 only;
CUDA Graph 80 replays x 20 alternating rounds; same saved activations and same
row-dropout mask. Allocation, weight packing and RNG are outside timing. Baseline is
Triton gate/LN backward plus cuBLAS GEMMs. It is not the older slower CUDA prototype.
The full backward differs only at B1-B4.

The full-backward run encountered a common-tail `trimul_input_dual_bwd_triton` cache
miss and used the configured three-candidate fallback, identically in both variants.
Thus its absolute total is not evidence of a fully retuned remainder of backward.

| L | Baseline B1-B4 us | New single CUDA kernel us | Speedup | Baseline whole bwd us | New whole bwd us |
|---|---:|---:|---:|---:|---:|
| 384 | 295.26 | 250.65 | 1.178x | 1084.91 | 1043.84 |
| 768 | 1178.59 | 868.99 | 1.356x | 4307.44 | 4035.99 |

For 1.7x, the B1-B4 times must fall below 173.68 / 693.29 us respectively.
This remaining gap is not presented as a hardware limit.

## Validation and limitations

- `validation.json`: L64/72/384/768, both reduction modes; dropout=0 gives exactly
  zero region outputs; dropout=1 and changed dy work through captured CUDA graphs;
  repeated replays reset cooperative counters.
- `final-results.json`: all six region outputs and all eleven full-backward outputs
  compared with the existing implementation. Max relative L2 in the final large-shape
  run is 0.000286; max across the additional live-input tests is 0.000427.
  These are not bitwise-identical dW results because FP32 summation order changes.
- `memcheck.log`: small-shape/graph cases. `memcheck-persistent.log` and
  `racecheck-persistent.log`: L72 with four CTAs, exercising 20/21 tiles per CTA,
  prefetch reuse and multiple invocations. See actual tool summaries before claiming pass.
- This plan owns input pointers, prepacked Wp, output buffers and scratch. It is
  single-stream/non-reentrant. Construct independent plans for concurrent invocations.
- Only the stated BF16 C128/H256 layout is implemented. The runner rejects L<64,
  a row count not divisible by 64, and non-SM90 devices. General engine dispatch,
  shape coverage and persistent autotune-cache integration remain work.

## Why earlier attempts were rejected

| Attempt | Finding |
|---|---|
| MN-major SS WGMMA replacing transposes | Correct; by itself slower than the previous RS schedule |
| Dedicated TMA/B1 producer, wider N128 tiles | Correct; more registers and lower occupancy erased reuse gains |
| Four-warpgroups, one CTA owns all dW+dX | Register spill: NCU ~282 MB local reads and writes each at L384 |
| Last-arriver dW reduction | Too few CTAs doing the final sum; ~200 us avoidable tail in that design |
| Split role CTAs + two-stage dWp | Spill-free but common resource footprint and barriers still slow |
| Two warpgroups share all work | Selected basis; no register spills |
| Next-tile raw-input TMA prefetch | Retained; hides transfers under the LN epilogue |
| Vector B1 mask loads / dg stores | Retained; substantial improvement, especially L768 |
| Wider vector LN epilogue | Rejected: slower than the channel-contiguous pair epilogue |
| Overlap dW MMA with dgrad setup | No material repeatable improvement; conservative ordering retained |
| Prefetch dropout mask into L1 | No improvement in this measurement |

## Reproduce (inside an allocated node02 GPU job)

```bash
bash runs/anthropic_adoption_20260919/env.sh python -B runs/anthropic_b1b4_pipeline_20260919/final_bench.py
bash runs/anthropic_adoption_20260919/env.sh python -B runs/anthropic_b1b4_pipeline_20260919/validate_dual.py
bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool memcheck --kernel-name kns=dual_b1b4 --error-exitcode 3 python -B runs/anthropic_b1b4_pipeline_20260919/sanitize_dual.py
bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool racecheck --kernel-name kns=dual_b1b4 --error-exitcode 3 python -B runs/anthropic_b1b4_pipeline_20260919/sanitize_dual.py
```

Snapshots and failed variants are retained as experiments, not dispatch choices.
