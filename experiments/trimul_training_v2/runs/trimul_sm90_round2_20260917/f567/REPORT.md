# F567 H100 round 2 — 2026-09-17

Production checkpoint is now dropout TMA + initial projection L2 prefetch, built on committed `0f2d455b` sigmoid-overlap kernel. Source and dedicated tests are committed by the parent as `8c7d8b39`. No fusion, independent GEMM, rounding, saved-buffer, dropout/residual, CSV axis, or default backend change.

## Changes retained

1. Stage the dropout scale through TMA into a **fourth otherwise-unused retired operand tile**. Reuse gate stage-zero barrier only after all MMA reads retire. Projection output staging overlaps that transfer. Chunk epilogue uses LDSM instead of partially utilized global LSU loads. Shared allocation does not increase. When capacity or row/N alignment does not permit this, preserve original vector/scalar loads.
2. Prefetch the initial `min(num_stages, projection_iterations)` projection input tiles into L2 while gate compute runs. This is a `cute.prefetch` hint on the existing TMA descriptor, with no new shared writes or barrier transactions. Original actual TMA transfer and waits remain unchanged.

Both changes are geometry/config derived; no length-dependent implementation choice or new tuning axis. GROUP_M tuning is retained in normal CSV/native config selection.

## Correctness and safety

Both dropout-only and final initial-prefetch variants passed **35 memcheck tests and 35 racecheck tests**, zero errors/hazards. Cases include noncontiguous aligned weights, M/N/K tails, 4/8 warps, odd/even gate stage-zero arrival counts, all three possible fourth-tile placements, capacity/alignment fallbacks, zero/nonzero dropout and full-range sigmoid overflow/subnormal behavior. Main L384/L768 outputs and both saved buffers match Triton bitwise.

Promoted source has engine Ruff checks and `git diff --check` passing. Parent completed the production35-case regression and compiled whole-module output/all-gradient checks atL384/L768 (max relativeL2 1.415e-6). See `../module/final-evidence.json`.

## Fair performance evidence

`prefetch-final-results.json` records three independent allocation/seeds (123/345/678), five rotated rounds each, comparing committed checkpoint, dropout-only, candidate and four Triton candidates. All dimensions: M=L², KP256, KG128, N128, BF16. CUDA graph timings include the actual kernel, not NCU replay time.

Strongest Triton remains BM128/BN64/BK64/G4/warps8/stages3. Candidate initial manifest BM64/BN64/BK64/G4/warps4/stages2.

| L | Committed CuTe | Dropout only | New CuTe | Strongest Triton | Speedup |
|---|---:|---:|---:|---:|---:|
|384|95.258 µs|93.483 µs|93.356 µs|105.750 µs|1.133×|
|768|362.263 µs|360.578 µs|356.603 µs|408.339 µs|1.145×|

Values are medians across independent allocation medians. These **do not meet 1.15× consistently**. Earlier strongest Triton observations near104/402µs reinforce that the threshold must not be inferred from a favorable single round. Final GROUP_M1 confirmation is recorded separately; source is identical.

## Current-kernel NCU evidence

Fresh full-set NCU reports (`current-L*.ncu-rep`, `prefetch-L*.ncu-rep`) use one warm kernel, cache-control none. NCU replay durations are diagnostic, not benchmark claims.

|Metric|L384 old→new|L768 old→new|
|---|---|---|
|Registers/thread|84→92|163→92|
|Shared bytes|41,984→41,984|66,560→41,984|
|Achieved occupancy|30.34→30.45%|18.50→30.66%|
|LSU global load sectors|2,359,296→0|9,437,184→0|
|L2 throughput|82.19→86.17%|77.39→92.41%|
|DRAM throughput|77.28→78.80%|84.60→86.23%|
|DRAM read MB|151.26→151.58|604.92→605.63|
|DRAM write MB|112.99→112.94|451.82→452.01|

L768 old uses its prior BN128 winner; new uses BN64. Therefore the L768 occupancy/resource change combines config selection with implementation. TMA replaces poorly utilized LSU reads; **it does not eliminate semantic dropout traffic or materially reduce DRAM bytes**. Actual cubins contain HGMMA and TMA, with zero spill/local allocation. Current source-PC sampling showed dominant long-scoreboard samples at gate/projection barrier waits; dropout was the exact source of excessive global sectors, not the majority of all long-scoreboard samples.

## Bounded rejected experiments

- Continued/staged projection prefetch: L38496.66µs vs initial93.71, rejected; too many hints did not help.
- Small-ring GROUP_M loop (original shared storage, no resident weights): exact but G2/G4 L38499.89/109.92µs vs93.31/93.28; L768365.59/374.57 vs353.24/353.68. Rejected. CPU independent review verified7200 barrier-phase combinations and exhaustive partial-group tile coverage. This experiment still reloads weights each iteration; no weight-traffic reduction claim.
- Prior-round multicast/full-resident-weight variants were not repeated.

Artifacts stay under this directory. Benchmark harnesses now import the frozen `committed_checkpoint.py` so re-running after production promotion preserves the historical comparison.

## Final GROUP_M1 confirmation and manifest

Five rotated rounds on each of two further independent allocations (seeds901/234), including exact matching Triton G1 and all prior strong baselines, confirm a small consistent G1 improvement over G4:

|L|New G1 per-allocation medians|G4 controls|Strongest Triton|Speedups|
|---|---|---|---|---|
|384|93.389 /93.073 µs|93.564 /93.367|105.058 /105.961|1.12495 /1.13848×|
|768|355.470 /355.942 µs|355.651 /356.419|409.011 /409.202|1.15062 /1.14963×|

Final config both lengths: **BM64/BN64/BK64/G1/warps4/stages2**. This is an explicit benchmark/tuning manifest, not a hardcoded runtime dispatch. Strongest Triton remains **BM128/BN64/BK64/G4/warps8/stages3**. `final-results.json` records full dicts and aggregate samples. L384 clearly misses15%; L768 straddles15%, and earlier stronger Triton observations mean it is not robustly above the threshold. All agent GPU work is complete; production source is frozen for parent integration.
