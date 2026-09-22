# B7–B12 continued optimization target

User target on 2026-09-20: at least 1.7× over the existing Triton/cuBLAS B7–B12 baseline, continuing toward independently supported SoL90. The earlier 1.146×/1.151× implementation is a checkpoint, not completion.

Fixed shapes: L384/768, B1/C128, left/right H256, BF16, dropout25%; preserve fused-input-LN forward and existing saved values. Keep numerical limits unchanged. One CUDA launch including parameter reductions. node02 only; leave Claude's B1–B4 and existing training jobs alone.

| Shape | Published baseline | 1.7× latency target |
|---|---:|---:|
| L384 | 615.008 µs | <=361.769 µs |
| L768 | 2361.056 µs | <=1388.857 µs |

Use fresh paired measurement as well as this frozen checkpoint. Do not substitute whole-backward speedup for this B7–B12 target or claim a whole-model result.

## Findings after the checkpoint

Instrumented CTA roles finish at almost the same time (~1.98 ms each at L768), so one-sided optimization is insufficient. Sampled steady-state tiles in the instrumented kernel:

- dW tile: ~3.424 µs, of which GLU ~2.464 µs and WGMMA wait/computation ~0.672 µs.
- dX tile: ~16.32 µs; front ~10.816 µs, gate load ~0.864 µs, LN load ~1.12 µs, LN reductions/output ~2.592 µs.
- These are diagnostic samples with instrumentation, not a replacement for whole-kernel paired timing.
- Gamma caching helps ~1%. Simple unrolling, packed-mask arithmetic and an initial GLU/MMA overlap attempt did not provide the required gain.

## New architecture under evaluation

`front_warp.cu` + `warp_plan.py`: one 384-thread CTA, two producer warpgroups for TMA/GLU and one WGMMA/epilogue warpgroup. Dynamic register split 128/128/240; persistent queues with ready/empty barriers. Hidden tile64 instead of128. dW uses a five-stage TMA ring; dX retains four GLU chunks to preserve Lg→Lp→Rg→Rp accumulation order. One dX warpgroup owns all128 channels, avoiding a cross-warpgroup input-LN reduction. The forward/save contract is unchanged.

First L64 check passed with no ptxas spills or stack usage. This does not yet qualify training shapes or establish final performance. Keep the published selected implementation until strict replay, sanitizer and timing checks qualify a replacement.

## SoL requirement

NCU's largest throughput percentage alone is not a proof of SoL90. Model compulsory memory traffic, tensor FLOPs, scalar/SFU issue and dependency constraints; compare against appropriate measured/hardware ceilings and the actual selected kernel profile. `ceiling_probe.py` measures copy/GEMM diagnostics on the same GPU but those alone do not define the fused kernel's limit. Report a defensible limit and achieved fraction, or explicitly state that SoL90 remains unproved.

## 06:15 UTC checkpoint (not a replacement selection)

Further attempts preserved fixed numerical tolerances. Most warp-specialized 384/512-thread designs were slower (L768 about2.25–3.1ms). A128-row tile alone spilled; bounded loops removed spills but did not improve throughput. Interleaving G/P contraction to reduce staging changed accumulation order and failed the LN-parameter tolerance, so it was rejected. Full-table and shared bounded sigmoid tables were slower. Standalone GLU production/materialization was too expensive to justify a global queue without further changes.

The useful new candidate is `front_twocta_kindwg.cu` with `WarpPlan(count=264,splits=13)`:256threads,114688dynamic shared bytes,128registers, two resident CTAs/SM, hidden tile64. The two dW warpgroups own gate/projection gradients separately and each uses N128 WGMMA. This halves dW MMA instruction count versus two N64 fragments. dX preserves Lg→Lp→Rg→Rp K order. BF16x2 mask multiply and the upstream approximate-reciprocal sigmoid reduce pointwise instructions. A derivative-FMA variation failed strict L384 accuracy and was rejected.

- Initial paired L768 kind-warpgroup candidate:1732.224µs against2343.936µs baseline; repeated comparison1713.504µs against2336.928µs.
- Prior `front_twocta_maskrcp` candidate:453.808µs L384 /1814.304µs L768 in an initial paired comparison. It passed6fixed264-CTA/one-launch cases (24comparisons: L64/384/768,dropout0/.25,changed inputs and weights). The132-CTA/two-launch control spilled and was refused before launch; this is not a qualified general configuration.
- New full NCU kind-warpgroup profile:1.71ms, DRAM76.18%, L2throughput79.40%, SM48.15%, no reported stack/spills. DRAM traffic has increased to about4.36GB, versus about2.59GB for the slower selected source. Faster dW advances ahead of dX and may lose shared-input L2reuse. This is a cache-locality hypothesis under test, not a proven attribution.
- Role/SM trace found8SMs with two dWCTAs,88with one of each,36with two dXCTAs. dW median1455.84µs/max1630.112; dX median1651.648/max1690.688.

`front_window{256,512,1024,2048}` tests progress barriers at bounded row-tile windows. It preserves each CTA's exact tile sequence, accumulation order and saves; the intent is to limit dW/dX drift and retain shared inputs in L2. It adds one reusable progress counter, reset at completion. L64 initial checks passed; training-shape measurements are pending. These variants are one cooperative launch and never relax full-residency requirements.

No new candidate is promoted until replay, sanitizer, paired timing and full-backward integration checks complete. Neither1.7× nor SoL90 is achieved yet.

## Current qualified experimental checkpoint (continued target work)

`front_twocta_kindwg.cu`, `WarpPlan(count=264,splits=13,part=2)`: two CTAs/SM, 256 threads/CTA, 128 registers, 112 KiB dynamic shared. The dW warpgroups own gate/projection separately and use N128 WGMMA. Partial buffers: dW 13 MiB, LN 160 KiB. Production dispatch and the older `Plan` default are unchanged.

| Scope | L | Paired baseline µs | New µs | Speedup |
|---|---:|---:|---:|---:|
| B7–B12 |384|617.344|446.208|1.384×|
| B7–B12 |768|2364.384|1730.160|1.367×|
| Full backward, same B1–B6 |384|1087.728|917.104|1.186×|
| Full backward, same B1–B6 |768|4355.200|3741.952|1.164×|

600 graph samples/path. `paired-kindwg-L*.json`, `full-kindwg-L*.json`, `qualify_kindwg.py`. Fixed original tolerances passed six cases/24 comparisons (`kindwg-production-check.json`); six scoped memcheck/racecheck/synccheck runs passed (`kindwg-sanitizers.json`). Independent host fixture unfiltered initcheck passed (`kindwg-initcheck-unfiltered-L64.log`). The filtered initcheck produced counter-initialization reports; the unfiltered run includes the GPU initialization kernel and reports zero errors. Preserve both logs and the earlier whole-fixture limitation.

NCU L768: ~1.71ms, DRAM76.18%, L279.40%, SM48.15%; zero stack/spills. Raw report: `twocta-full-L768.ncu-rep`. **At least1.7× and defensible SoL90 remain active, unachieved targets.** This is the strongest fully checked checkpoint, not the finish. New weight-residency/loop/layout experiments are not selected. The historical checkpoint below remains reproducible and is the unchanged `Plan` default.


## Latest fully checked checkpoint: prefetch

`front_kindprefetch.cu` / `WarpPlan(count=264,splits=13)` overlaps next-row gate TMA with current LN, and current X/residual TMA with the last projection contraction. LN is relocated to shared offset65536; gate occupies bytes0..49152, so the transfers do not collide. Gate and LN now have independent transaction barriers. All arithmetic and stored values are unchanged.

600 alternating samples/path: L384 baseline615.872→384.896µs (1.600×), L7682364.512→1532.160µs (1.543×). Whole backward, identical B1–B6:1090.448→860.416µs and4349.584→3526.976µs. Six shape/dropout cases with24changing-input/weight comparisons passed. Six scoped memcheck/racecheck/synccheck checks plus unfiltered isolated initcheck passed. See `kindprefetch-*` artifacts. NCU L7681.54ms, DRAM78.27%, L279.66%, SM46.62%. Neither1.7× nor SoL90 is yet reached.

Rejected or non-selected branches: register-resident projection source fails repeated tiles; reload control correct but dX-only2652µs. Cache eviction hints did not help. Cluster DSM source computes B7 once, with unchanged saves, but measured2794µs; double-buffered pipeline2441µs; moving LN out of the next DMA destination2079µs. A512thread cluster source is correct on initial checks but slower than the256thread variant. Raw DSM copy/handshake probe2.133TB/s. These cluster experiments are not qualified selections.

Current experiment: `front_onewg_lowreg` uses128threads, three CTAs/SM,72KiBshared,168registers, one WG owns all128channels. dW retains two N128 accumulators; descriptor/store offsets are computed inside PTX to eliminate spills. L64 initial check passed. Training-shape measurements are pending.

## Paired BF16 LN checkpoint

Fully checked candidate: `front_prefetch_lnpair`,264CTA/sp13. 600sample paired B7–B12:617.456→370.992µs atL384 and2365.056→1477.856µs atL768. Sixcases/24comparisons+seven sanitizer checks passed. Goal1.7×/SoL90 remains active. Ring transfer prototypes are not qualified or selected. See `RING_DESIGN.md`.

## Latest validated checkpoint: shape-specific one-launch paths

| Scope | L | Frozen baseline µs | Fresh paired baseline µs | CUDA µs | Fresh speedup |
|---|---:|---:|---:|---:|---:|
| B7–B12 |384|615.008|617.216|364.848|1.692×|
| B7–B12 |768|2361.056|2365.632|1429.136|1.655×|
| Full backward, identical B1–B6 |384|—|1092.304|843.104|1.296×|
| Full backward, identical B1–B6 |768|—|4338.912|3405.008|1.274×|

L384: `front_prefetch_lnpair_storepipe` / `WarpPlan(count=264,splits=13)`. Prior dx TMA store overlaps next gate contraction; store drains before shared buffer reuse. L768: `front_ring96_cache3` / `RingPlan(count=264,splits=20)`. B7 computed only in dW role, rounded derivatives passed through12MiB global ring with L2 priority. See `RING_DESIGN.md` and `front-b7b12-ring-checkpoint.svg`.

Both paths passed6cases/24replay comparisons,6mem/race/sync checks and independent unfiltered initcheck. The limits remain dx2e-5,dW5e-4,LNparams5e-6. Full-backward11outputs pass. Evidence labels `storepipe-*`, `ring96cache3-*`; timing files `paired-lnpair_storepipe-L384.json`, `full-lnpair_storepipe-L384.json`, `paired-ring96cache3-L768.json`, `full-ring96cache3-L768.json`. 600alternating samples per path. NCU ring1.444704ms,L2 89.257%,SM38.431%,DRAM2.648GB: implementation bottleneck diagnostic, not SoL90 proof. Production dispatch and old Plan default remain unchanged.

Follow-up queue plus128-row weight-TMA source `front_ring96_queue_wtma128` also passed24replay comparisons and7sanitizer checks. Its L7681426.576µs is too close to1429.136µs to establish a useful improvement; keep the simpler source. One-segment accumulation, more aggressive cache hints, and small rings did not show stable gains. `RingPlan.bind` now retains its ring allocation while rebinding inputs, keeping graph-captured addresses stable.

**At least1.7× in both training buckets and independently supported SoL90 remain active, unachieved targets.** The frozen target remains361.769/1388.857µs. The fixed baseline still uses the existing input-dual Triton heuristic subset on cache miss; do not claim an exhaustive-retuned Triton comparison. HTML/SVG publication records live in `site-publication.json`.

