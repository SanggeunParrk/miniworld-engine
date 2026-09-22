# Ring transfer experiment (2026-09-20)

## Current qualified ring checkpoint (2026-09-20 08:55 UTC)

`front_ring96_cache3.cu` / `RingPlan(count=264,splits=20)` is now validated: six shape/dropout cases,24 changing-input/weight comparisons, six mem/race/sync checks atL64/384 and independent unfiltered initcheck atL64 all pass. It computes B7 once per tile. The96-tile12MiB ring uses release/acquire generation handoff; TMA ring traffic has evict-last priority, original preactivation/upstream reads have evict-first priority. Ring memory is global memory intended to remain in L2, not guaranteed on-chip storage.

600sample paired L768:2365.632→1429.136µs (1.655×). Whole backward with identical B1–B6:4338.912→3405.008µs (1.274×). Source and evidence: `paired-ring96cache3-L768.json`, `full-ring96cache3-L768.json`, `ring96cache3-production-check.json`, `ring96cache3-sanitizers.json`, `ring96cache3-initcheck-unfiltered-L64.log`.

NCU:1.444704ms, L2 throughput89.257%, SM38.431%, observed DRAM2.648GB. This is an implementation bottleneck diagnostic, not proof of algorithmic SoL90. One launch and the forward/save contract remain unchanged. Production dispatch remains unchanged.

Follow-up candidates: dx store overlaps next gate, acquire-load final polling, a dynamic dX tile queue, and128-row weight TMA. Full queue/weight candidate passed24 replay comparisons; full sanitizer qualification is pending. 600sample L7681426.576µs does not establish a material win over the simpler1429.136µs ring. Smaller32/48/64/80-tile rings, single dW accumulation segment, and reuse-cache hints did not reliably improve the best selection. Preserve the simpler fully verified checkpoint.

## Earlier design record


Experimental, not selected or production-wired. Fused forward saves are unchanged.

The dW CTAs compute B7 once, then TMA-store the rounded G/P derivatives to a bounded global ring. dX CTAs wait for all eight H64 producer groups and TMA-load these same derivatives, eliminating their GLU recomputation and second full preactivation/upstream read. The ring is intended to remain in L2; it is not guaranteed zero HBM traffic. Window128 uses16MiB; window256 uses32MiB.

Each slot has eight producer generation counters and one consumer-free generation counter. Release/acquire GPU memory operations publish completed TMA writes and protect slot reuse. The TMA store's full completion (not just shared-read completion) is observed before publication. Counter clearing is parallel after the end-of-work grid barrier, with final completion fencing before counters0/1 reset.

The pipeline version overlaps a tile's TMA stores with the next tile's GLU. A shared stage is not overwritten before the previous store completed. All original BF16 boundaries and dX contraction order are preserved. dW uses the existing partial accumulation splits.

Initial checks passed L64/384/768. Window256/sp16 was1990us atL768. Pipeline+role tuning improved window128/sp20 to1569us in a multi-candidate sweep. Further tuning is active. These timings are not the qualified performance table. No comprehensive ring replay or sanitizer claim yet.

The PTX async-completion and proxy semantics used here are documented in https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-data-movement-async-proxy.
