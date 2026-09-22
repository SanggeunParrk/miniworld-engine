# L384 B7–B12 optimization with the single-kernel restriction removed

2026-09-22, node01 H100. No L768 runs, full training validation, engine dispatch change, commit or push.

## Result

The best tested candidate is a small improvement, not the requested 252 us result. Eight alternating graph-timing rounds show 1.41–1.65% lower median latency for the selected candidate in every round. Absolute clocks/latencies drift across rounds, so use the matched comparisons below, not earlier-job absolute times.

| Same-process implementation | Median us | Baseline / candidate |
|---|---:|---:|
| split | 420.128 | 1.0000x |
| reduce_one_slice | 416.576 | 1.0085x |
| parallel_ln_reduce | 413.952 | 1.0149x |

Job 14900. Five mutated input cases, eager and repeated graph replay, NaN-poisoned scratch and reset counters pass the unchanged relative-L2 limits: dX 2e-5; four dW matrices 5e-4 each; dgamma/dbeta 5e-6 each. dX remains bit-exact in these cases; reduction order changes dW and LN parameter gradients. Raw results, per-round timings and source hashes: [audit-L384.json](audit-L384.json).

Separate isolated role timings are also recorded, but their sum is not the matched combined latency: they run later, have different cache state, and clock drift is visible.

## Selected code

[Selected factory](selected.py), wrapping `../trimul_b7_split_lnreduce_20260922/role_plan.py`.

- Keep separate dW and dX+LN+residual CUDA launches, with the existing TMA/WGMMA loops.
- Keep saved x_n. Recompute projection/gate independently in the two paths as before.
- dW accumulates its full assigned row interval in FP32 registers before storing a partial once, rather than twice. Its partial buffer falls from 16 MiB to 8 MiB.
- Final dW reduction uses four independent FP32 sums instead of one serial sum.
- Distribute the final LN parameter-gradient reduction over eight CTAs; eight lanes collaborate on each parameter. Previously one CTA read all row partials serially per parameter.
- Use 256 dX CTAs for 2304 row tiles, nine tiles per CTA.

This selection is local and experimental. The factory rejects unvalidated L values. This work did not change engine dispatch.

## NCU

Profiles run separately from the main paired timings. Baseline dW: 170.528 us; one-slice reduction dW: 164.960 us. HBM reads fall 199.126 → 190.931 MB and writes 24.076 → 16.240 MB, consistent with removing one 8 MiB partial write/read. This accounts for a modest reduction, not a large algorithmic speedup. Raw [NCU summary](ncu-summary.json), CSV and ncu-rep files are alongside this report.

Source-level attribution matters: much of the old dW long-scoreboard sample count belongs to the producer waiting for a free shared-memory ring slot, not scalar mask loads. Much of the dX barrier count belongs to idle producer threads or the final reduction barrier. These aggregate stall percentages must not be interpreted as the fraction of wall time removable by optimizing mask loads or barriers. Neither Tensor activity nor the largest utilization percentage is a measured SoL ratio.

## Explored alternatives

26 configurations completed numerical checks and timing; full records are in [exploration.json](exploration.json). Several other settings were rejected at compilation/launch, recorded below. Times across different jobs are exploratory, not one common absolute series.

| Approach | Observation |
|---|---|
| Async mask load | Reusing otherwise-unused shared gamma storage preserves residency, but the gain is small. Adding 512 B dynamic shared memory crossed a residency limit and larger cooperative grids were rejected. |
| Separate reduction + concurrent streams | Correct, but little/no overall gain; sharing the same SM resources does not automatically overlap useful work. |
| Packed BF16 conversions / dp reuse | Correct within the tested limits, without a convincing speed gain. dp reuse changes FP32 multiplication order and was not selected. |
| Two GP operations in flight | 32/224 producer/consumer register setting spills. 24/232 compiles without spills and passes, but is slower. |
| N128 WGMMA | dX-only conversion slows down. Converting both dX and projection, or projection alone, does not give a robust gain. |
| Hidden-32 dX / 72 KiB shared | Two CTAs per SM are slower. Targeting three CTAs creates spills and WGMMA serialization, rejected. |
| Fewer dW partials / four-way sum | Small repeatable improvement; included in selection. |
| Parallel LN parameter-gradient reduction | Further small improvement; included in selection. |
| End-only TMA atomic dW | Correct, but initialization/synchronization costs offset the smaller final reduction. Not selected. |

The single-CTA reuse / multi-tile atomic experiments from the previous turn remain separate: [prior report](../trimul_b7_accum_analysis_20260922/README.md).

## Limits

252 us and 35% faster than split have not been achieved. This search does not establish a hardware ceiling. The surviving change reduces partial-buffer traffic and reduction overhead; it does not eliminate duplicated projection/gate recomputation between dX and dW. No global claim about training speed is supported by this isolated fixture.

## Final sanitizer

Job 14905: selected `parallel_ln_reduce` candidate passed compute-sanitizer memcheck and racecheck (both return code 0; 0 errors, 0 warnings). Filters include the B7 dW/dX kernels, and also cover matching reference kernels during fixture setup. See [sanitizer-L384-parallel.json](sanitizer-L384-parallel.json). The initial audit/sanitizer launcher jobs failed on a filename/filter syntax typo before testing candidates; jobs 14900/14901/14905 reran the corrected harness successfully.

## Benchmark-condition correction

The previous ~384 us split result and this report’s 420 us result use the same cubins. A subsequent controlled reproduction found 386.8 us at sampled median 1980 MHz with the old slow-candidate mixture, versus 421.8 us at 1785 MHz with the fast-candidate mixture. These are different sustained operating conditions; 420 us is not a replacement for the earlier baseline. See [timing correction](../trimul_b7_timing_drift_20260922/README.md).
