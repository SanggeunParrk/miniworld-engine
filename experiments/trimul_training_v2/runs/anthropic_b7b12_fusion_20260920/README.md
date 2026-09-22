# B7–B12: one-launch bidirectional TriMul training backward

**Status (2026-09-20): implemented and connected in the experimental full-backward adapter. Production engine dispatch is unchanged.**

The current fused-input-LN forward and its saves are preserved. Saving `x_n`, mean and inverse std does not require a separate LN kernel. This work covers B7–B12; B1–B4 remains a separate task owned by Claude.

## Latest validated checkpoint: shape-specific one-launch paths

### Direct forward + backward measurement

`bench_train_total.py`, node02 H100, BF16, C128/H256, dropout25%,
600 alternating CUDA Graph samples per path. Each captured call executes
the fused-input-LN Anthropic-derived forward and produces fresh saves for
its complete backward. Only B7–B12 changes; B1–B6 stays identical.
These are directly measured totals, not sums of isolated stage timings.

| L | Existing total µs | B7–B12 CUDA total µs | Speedup | Time reduction |
|---|---:|---:|---:|---:|
|384|1530.240|1273.920|1.201×|16.75%|
|768|6068.656|5146.272|1.179×|15.20%|

Forward outputs match exactly; all11 gradient outputs pass the fixed
per-output tolerances before capture and after replay. Evidence:
`train-total-qualified-L384.json`, `train-total-qualified-L768.json`.
Optimizer, RNG, CPU/autograd dispatch and weight packing are excluded.
This does not incorporate the separately developed B1–B4 CUDA replacement
and is not a full MiniWorld model training-step benchmark. The existing
baseline still uses the documented3-config input-dual fallback.

### Region and backward measurements

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

## Historical checkpoints

## Latest fully checked checkpoint: paired BF16 LN access

`front_prefetch_lnpair.cu`, `WarpPlan(count=264,splits=13,part=2)`, [wiring SVG](front-b7b12-lnpair.svg). This retains the previous TMA overlap and packs residual loads and dx shared stores into BF16 pairs. Arithmetic, rounding boundaries and forward saves are unchanged.

| Scope | L | Paired baseline µs | New µs | Speedup |
|---|---:|---:|---:|---:|
| B7–B12 |384|617.456|370.992|1.664×|
| B7–B12 |768|2365.056|1477.856|1.600×|
| Whole backward, same B1–B6 |384|1092.000|851.072|1.283×|
| Whole backward, same B1–B6 |768|4358.688|3502.192|1.245×|

600 samples/path: `paired-lnpair-L*.json`, `full-lnpair-L*.json`. Six cases/24 replay comparisons passed (`lnpair-production-check.json`), six mem/race/sync checks passed (`lnpair-sanitizers.json`), and unfiltered independent initcheck passed (`lnpair-initcheck-unfiltered-L64.log`). NCU:1.463ms, DRAM76.36%, L282.15%, SM47.98%; stack/spills0. `lnpair-ncu-summary.json` records3.745GB observed DRAM traffic and2.560TB/s. Its86.4% ratio to the historical copy probe is only diagnostic: different access patterns and non-compulsory traffic prevent a SoL90 claim.

**Neither1.7× at both training buckets nor independently supported SoL90 is achieved.** Production and the older `Plan` default remain unchanged. The frozen target latencies remain361.769µs /1388.857µs. All sections below are historical checkpoints.

## Previous checkpoint: overlapped TMA

`front_kindprefetch.cu`, `WarpPlan(count=264,splits=13,part=2)`, [wiring SVG](front-b7b12-prefetch.svg). Next gate loads during current LN; current X/residual loads during the last projection GEMM. Shared regions and independent barriers prevent overwrites.

| Scope | L | Paired baseline µs | New µs | Speedup |
|---|---:|---:|---:|---:|
| B7–B12 |384|615.872|384.896|1.600×|
| B7–B12 |768|2364.512|1532.160|1.543×|
| Whole backward, same B1–B6 |384|1090.448|860.416|1.267×|
| Whole backward, same B1–B6 |768|4349.584|3526.976|1.233×|

600 samples/path; `paired-kindprefetch-L*.json`, `full-kindprefetch-L*.json`. Six cases/24replay comparisons and seven scoped sanitizer checks passed. NCU L768:1.54ms, DRAM78.27%, L279.66%, SM46.62%; stack/spills0. Neither1.7× nor SoL90 is achieved. Production and the old `Plan` default remain unchanged. This supersedes the current-checkpoint figures below; those sections retain the historical evidence.

## Earlier qualified experimental checkpoint (continued target work)

`front_twocta_kindwg.cu`, `WarpPlan(count=264,splits=13,part=2)`: two CTAs/SM, 256 threads/CTA, 128 registers, 112 KiB dynamic shared. The dW warpgroups own gate/projection separately and use N128 WGMMA. Partial buffers: dW 13 MiB, LN 160 KiB. Production dispatch and the older `Plan` default are unchanged.

| Scope | L | Paired baseline µs | New µs | Speedup |
|---|---:|---:|---:|---:|
| B7–B12 |384|617.344|446.208|1.384×|
| B7–B12 |768|2364.384|1730.160|1.367×|
| Full backward, same B1–B6 |384|1087.728|917.104|1.186×|
| Full backward, same B1–B6 |768|4355.200|3741.952|1.164×|

600 graph samples/path. `paired-kindwg-L*.json`, `full-kindwg-L*.json`, `qualify_kindwg.py`. Fixed original tolerances passed six cases/24 comparisons (`kindwg-production-check.json`); six scoped memcheck/racecheck/synccheck runs passed (`kindwg-sanitizers.json`). Independent host fixture unfiltered initcheck passed (`kindwg-initcheck-unfiltered-L64.log`). The filtered initcheck produced counter-initialization reports; the unfiltered run includes the GPU initialization kernel and reports zero errors. Preserve both logs and the earlier whole-fixture limitation.

NCU L768: ~1.71ms, DRAM76.18%, L279.40%, SM48.15%; zero stack/spills. Raw report: `twocta-full-L768.ncu-rep`. **At least1.7× and defensible SoL90 remain active, unachieved targets.** This was the strongest fully checked checkpoint at that stage, not the finish. New weight-residency/loop/layout experiments are not selected. The historical checkpoint below remains reproducible and is the unchanged `Plan` default.

## Origin and scope

We previously developed inference kernels, but Anthropic's published implementation achieved better results. This work builds on that implementation and adds training support. It reuses Anthropic native v5 TMA/WGMMA primitives and shared-memory layouts; it is not a claim of superior independent inference development.

Vendored upstream: [anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e), revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. Preserve upstream Apache-2.0 license and notices. PTX synchronization semantics follow the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

Target: node02 H100 80GB, BF16, B=1, C=128, left/right width=256 each, L=384/768. L64 is a validation shape. Dropout=0 and 0.25 checked. This is a training-backward comparison against the current Triton/cuBLAS training path, **not** a speed comparison against an Anthropic inference kernel.

## Actual wiring

See [front-b7b12.svg](front-b7b12.svg).

One cooperative CUDA launch, 132 CTAs, 256 threads/CTA:

- **60 dW CTAs:** B7 mask + GLU backward → B8 WGMMA → FP32 partials for four input weight gradients.
- **72 dX CTAs:** B9 output-gate WGMMA → B7 recomputation + B10 front WGMMA → input LN backward B11 + residual B12 → TMA store of dx.
- Grid completion barrier → reduction of dW and LN-parameter partials in that same launch. Two counters reset before a subsequent launch.

No global `d_concat` or `dx_n` tensor; no weight concatenation or four final transpose copies. B7 is computed once by each role. Small global partial buffers remain: dW 7.5 MiB + LN 72 KiB at 132 CTAs. This is not a zero-HBM-intermediate design. One Plan workspace permits one in-flight invocation; do not share it across concurrently executing streams/graphs.

Direct TMA descriptors read the existing four weight tensors, including in-place weight updates. B10 preserves reference K order: Lg256 → Lp256 → Rg256 → Rp256. The pair-mask multiply, gate dgrad, dx_n, and LN-gradient-before-residual BF16 rounding boundaries are retained.

## Selected implementation

| File | Purpose |
|---|---|
| `front_roundonly.cu` | Selected 132-CTA kernel, 60 dW + 72 dX, 15 dW splits |
| `front_rounding.cu` | 66-CTA validation alternative; two dW accumulation segments per physical CTA |
| `front_mn_primitives.cuh`, `front_primitives.cuh` | Explicit WGMMA and native TMA helpers |
| `front_plan.py` | `Plan(source='front_selected')`, compile/load, tensor maps, buffers, cooperative launch |
| `integrate_front.py` | Full-backward adapter, identical B1–B6 reference prefix |
| `check_front.py` | Fixed numerical limits and changing-input/weight graph replay |
| `selected-build.json` | Exact selected cubin SHA-256 and compiler command |

The default is one launch (`part=2`). `part=1` is a two-launch control used for validation, not the selected production proposal. The 66-CTA configuration is tested for reduced residency but is not the chosen performance configuration. Other `front_*` sources preserve experiments; their presence does not mean they are selected.

## Measured performance

H100 node02; identical saved forward values; dropout 25%; alternating CUDA graph replays, 3 blocks × 200 samples per path; median of 600 samples. Includes partial reductions and output writes. Allocation/compilation/host tensor-map binding are outside graph replay timing. Full backward retains the same B1–B6 reference prefix on both sides.

| Scope | L | Triton/cuBLAS (µs) | One-launch B7–B12 (µs) | Speedup | Time reduction |
|---|---:|---:|---:|---:|---:|
| B7–B12 | 384 | 615.008 | 536.448 | 1.146× | 12.77% |
| B7–B12 | 768 | 2361.056 | 2051.424 | 1.151× | 13.11% |
| Full backward | 384 | 1090.064 | 1008.944 | 1.080× | 7.44% |
| Full backward | 768 | 4310.848 | 4022.800 | 1.072× | 6.68% |

Evidence: `paired-selected-L{384,768}.json`, `full-selected-L{384,768}.json`. Baseline input-dual kernel reports a cache miss and uses its existing 3-of-648 heuristic subset; this is not a newly exhaustive-tuned Triton comparison. No whole-model/optimizer-step claim is made. Claude's newer B1–B4 implementation is not combined into these full-backward numbers.

## Correctness

`selected-validation.json`: **24 configurations / 96 comparisons**, assembled from concrete-source test results:

- L64/384/768 × dropout off/on × 66/132 CTAs × one/two-launch control.
- Initial result, two captured replays changing input gradients/mask/LN saves/gamma, and a replay after in-place weight updates.
- Seven outputs and counter reset checked on every comparison.
- Fixed relative-L2 limits, set before tests: dx 2e-5, dW 5e-4, dgamma/dbeta 5e-6.
- Worst observed: dx 8.59e-6; dW 3.91e-4; LN parameters 2.01e-6.
- Full-backward adapter: all 11 gradient outputs checked at both training shapes.

Two actual issues were found and corrected, without relaxing the limits:

1. NVCC contracted `dy * gamma - correction` into FMA, unlike the reference's separately rounded multiply. Explicit `__fmul_rn` preserves that boundary.
2. The 66-CTA / seven-split dW control accumulated too long in FP32. Two logical accumulation segments reduce error. The 132-CTA / 15-split selected path passes without that extra segmentation and is faster.

`front_direct` timings preceded these expanded checks. Use only the selected timings above for a qualified result.

## Sanitizers

Eight scoped checks passed (`selected-sanitizers.json`): memcheck/racecheck/synccheck at L64 and L384, memcheck of the 66-CTA two-launch control at L384, and an isolated L64 initcheck with host-initialized inputs. Explicit TMA/WGMMA checks remain enabled.

The original whole-fixture initcheck is **not a clean result**. It reported uninitialized-read/API-copy diagnostics in setup and `_gate_elem_bwd_ew_kernel`, before this new B7–B12 kernel, and exceeded the diagnostic time bound. Filtered and unfiltered logs are retained. The independent host-initialized fixture runs the selected B7–B12 twice and passes initcheck; this does not certify the upstream forward/B1–B6 setup. That upstream diagnostic remains outside this change and should be investigated separately before claiming end-to-end initcheck coverage.

## NCU and generated instructions

Selected warm profiles: `selected-warm-L{384,768}.ncu-rep/.csv`, summarized with units in `selected-ncu-L{384,768}.json`. Cache control and clock control disabled; profile values are diagnostic, not an independently proved roofline limit.

| NCU metric | L384 | L768 |
|---|---:|---:|
| Profile duration | 538.656 µs | 2.050912 ms |
| DRAM throughput / sustained peak | 36.37% | 37.63% |
| SM throughput / sustained peak | 45.22% | 47.29% |
| Tensor-pipe active / sustained peak | 15.75% | 16.52% |
| Global memory read | 608.244 MB | 2.425 GB |
| Global memory write | 48.494 MB | 161.849 MB |
| Local-memory load/store sectors | 0 / 0 | 0 / 0 |

ptxas: 243 registers, 0-byte stack, zero spills; 229376 bytes dynamic + 1024 bytes static shared memory. `selected.sass` contains explicit TMA (`UTMALDG`) and warp-group MMA (`HGMMA`), with no LDL/STL. Initial dynamic-index barrier-phase arrays generated local accesses despite zero reported spills; the selected source uses explicit parity and removes them. Channel-major WGMMA operands avoid the initial expensive shared transpose.

**This is not SoL90.** Further work should investigate dW/dX role balance, overlap of TMA with WGMMA and the GLU instruction/barrier cost, using measured critical paths. A single kernel by itself does not guarantee roofline performance.

## Reproduce (inside a node02 H100 allocation)

```bash
bash runs/anthropic_adoption_20260919/env.sh python -u -B \
  runs/anthropic_b7b12_fusion_20260920/check_front.py \
  --source front_selected --output replay-check.json

bash runs/anthropic_adoption_20260919/env.sh python -u -B \
  runs/anthropic_b7b12_fusion_20260920/compare_front.py \
  --length 384 --sources front_selected --output repeat-L384.json

bash runs/anthropic_adoption_20260919/env.sh python -u -B \
  runs/anthropic_b7b12_fusion_20260920/integrate_front.py \
  --length 768 --output full-repeat-L768.json
```

Do not compile, benchmark or profile on the login node. Scripts require this checkout's existing Anthropic adoption environment and equal-saves forward sources. New CUDA source and raw reports remain local; the status site receives the HTML summary and SVG wiring only.

## Follow-up experiment audit (2026-09-20 09:34 UTC)

Fresh600sample recheck of the selected sources: L384614.752→364.624µs (~1.686×), L7682365.408→1436.048µs (~1.647×). This supports approximately1.69×/1.65×, not a1.7× claim. Files `paired-storepipe_recheck-L384.json`, `paired-ringcache3_recheck-L768.json`; the earlier published paired datasets remain intact.

Two-CTA TMA multicast (full cluster barrier and thread0 handshake),512-thread local two-tile reuse, wide transposed weight reduction, LN-stat prefetch, mask-register prefetch, smaller/consumer-matched ring windows, consumed-ring eviction hints, and asynchronous two-group WGMMA were tried. None has yet established a validated improvement over the selection. See `MULTICAST_DESIGN.md`, `PAIR_CTA_DESIGN.md`, named sweep JSON/log artifacts. Nonzero ptxas stack/spills continue to be rejected before execution. No engine push occurred; Sites summary-only publication is recorded in `site-publication.json`.

Final candidate comparison for this follow-up: native full-CTA `__syncthreads` versus the original named barrier had small improvements in initial sweeps; combining it with removed store barriers or acquire polling did not helpL384. Ring112 + native CTA sync + evict-first on consumed ring data ranked best in a nine-candidate sweep(~1451µs versus1492µs control in that sweep). Independent600sample qualification is required; these sweep times are not a replacement for the published paired table.
