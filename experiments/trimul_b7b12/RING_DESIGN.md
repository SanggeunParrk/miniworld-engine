# Ring transfer experiment (2026-09-20)

## Current qualified ring checkpoint (2026-09-20 08:55 UTC)

`front_ring96_cache3.cu` / `RingPlan(count=264,splits=20)` is now validated: six shape/dropout cases,24 changing-input/weight comparisons, six mem/race/sync checks atL64/384 and independent unfiltered initcheck atL64 all pass. It computes B7 once per tile. The96-tile12MiB ring uses release/acquire generation handoff; TMA ring traffic has evict-last priority, original preactivation/upstream reads have evict-first priority. Ring memory is global memory intended to remain in L2, not guaranteed on-chip storage.

600sample paired L768:2365.632→1429.136µs (1.655×). Whole backward with identical B1–B6:4338.912→3405.008µs (1.274×). Source and evidence: `records/paired-ring96cache3-L768.json`, `records/full-ring96cache3-L768.json`, `records/ring96cache3-production-check.json`, `records/ring96cache3-sanitizers.json`, `records/ring96cache3-initcheck-unfiltered-L64.log`.

NCU:1.444704ms, L2 throughput89.257%, SM38.431%, observed DRAM2.648GB. This is an implementation bottleneck diagnostic, not proof of algorithmic SoL90. One launch and the forward/save contract remain unchanged. Production dispatch remains unchanged.

Further tuning remains experimental. The published checkpoint retains the simpler fully verified source.
