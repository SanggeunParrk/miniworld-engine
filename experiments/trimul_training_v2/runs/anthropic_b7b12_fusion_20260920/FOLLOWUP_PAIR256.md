# Two-row input-gradient experiment, 2026-09-20

Selected checkpoint and production wiring are unchanged. Goal remains active:
the frozen-baseline 1.7x target and a defensible algorithmic SoL90 are unmet.
Source checkpoint is published to miniworld-engine/main at `3f1c7a77`, with
L2 interpretation/profiling documentation at `789665af`. Remote main was
verified at the latter commit during this work.

## Implementation

`make_ring_pair256.py` incorporates `ring_pair256_input.cuh` into the selected
ring source. Each 256-thread dX CTA processes two adjacent 64-row tiles, one
N128 WGMMA warpgroup per tile, sharing projection and gate weight TMA transfers.
Two 32 KiB projection stages and a 32 KiB BF16 gate/dx_n scratch fit in the same
112 KiB dynamic shared memory. B7, contraction order, separate B9 rounding and
BF16 dx_n/LN/residual boundaries are preserved. No global full-size derivative
tensor or weight concatenation was added.

Bounded GLU unrolling eliminates the initial compiler spills. All executed
candidates have zero ptxas stack/spills; base uses125 registers. The alternative
LN mapping uses128. No spill-check bypass was used.

## Experiments

All timings below are L768, dropout25%,600 samples with rotated order and
shared workspace addresses. These candidates have initial strict numerical
checks, not the selected checkpoint's full replay/sanitizer qualification.

| Experiment | Result |
|---|---|
| Initial96-row-ring source | Selected1442.512us, pair1585.696us |
| Role splits17..23, ring96 | Best split22:1579.408us |
| LN loop unroll1/2/4/8 | 1578.112/1577.040/1576.944/1601.104us; no recovery |
| Ring96/128/160/192/224, split22 | 1589.248/1466.144/1506.032/1553.648/1618.832us |
| Fine ring104/112/120/128/136/144 | Minimum128; larger ring locality costs remain |
| Ring128 role splits18..25 | Best valid split22; split25 rejected for dWL relativeL2=0.000503660 >0.0005 |
| Existing C64-per-WG LN mapping | No improvement over N128 mapping |
| First LN-row TMA overlapped with final front WGMMA | Slower:1475.888us |

**Compare each source at its own tuned role count.** The old source with
split22 in the fine-window sweep is slower than its selected split20 and must
not be used to claim a speedup. `bench_shared_sources.py --source-splits` now
supports distinct role counts while retaining shared output, partial, counter
and ring allocation addresses.

Final fair comparison (`ring-pair256-tuned-L768.json`): selected ring96/split20
1456.512us, pair128/split22 1471.344us, pair C64-LN1468.672us, pair unroll4
1464.736us. The follow-up LN-prefetch comparison gives selected1453.632us,
pair unroll4 1463.264us, prefetch1475.888us. **No candidate is promoted.**

## NCU evidence

`pair256-ncu-summary.json` records source/CSV hashes and metric units. Full
reports and raw CSV remain local. Separate warm, unlocked-clock profiles:

| Metric | Selected ring96 | Pair ring96 | Pair ring128 |
|---|---:|---:|---:|
| Duration ms |1.444704|1.568448|1.433920|
| L2 sectors |328288189|293861679|294596115|
| DRAM read GB |2.435185|2.436087|2.450819|
| DRAM write MB |212.627200|206.887168|366.412032|
| SM throughput % |38.431|35.778|39.152|
| Executed warp instructions |515838776|529067906|527735020|

Pair ring96 reduces L2 sector volume by10.5% but is slower. Expanding the ring
reduces CTA barrier stalls from35.0% to31.1% of cycles between issue events,
while increasing DRAM writes. Pair128's isolated NCU time is not a speedup
claim: the stronger paired comparisons above still favor the selected source.
These observations show reuse/synchronization/locality tradeoffs, not a proof
of a global optimum or SoL90.

## Earlier diagnostics from the same allocation

`ring-role-stages-L768.json`: instrumented selected dW/dX median role durations
1420.048/1425.728us; final grid waits14.464/8.976us. `ring-wait-timing-L768.json`
measures about200.592/193.024us per-CTA accumulated ring waits, including timer
overhead and dX ready-barrier cost; concurrent waits cannot be summed into an
end-to-end removable cost.

Mask TMA, three-warpgroup split, group/side ring credits, relaxed-load acquire
polling and oversized rings did not improve the selected source. Earlier logs
and generators are retained locally. Pure acquire fence was also tested;
there was no stable gain. Do not repeat the same parameter sweeps as new work.

Next work should attack repeated operands or the remaining issue/synchronization
cost with an explicit resource model. LN unroll, ring size and role counts for
this pair256 design have already been measured. Training and Claude's separate
B1-B4 allocation were not modified. Allocation13346 is released after this
checkpoint rather than holding idle GPUs.
