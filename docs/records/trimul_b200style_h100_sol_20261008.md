# TriMul H100: measured fusion improvements and SOL audit — 2026-10-08

The explicit B200-style path now avoids reading a unit dropout scale during inference and uses TMA to save D512 projection/gate activations during training. The paired full-module inference comparison improves by **1.127688x geometric mean** across 12 cells. D512 complete forward+backward improves by **1.010382–1.024552x** across four cells. This is an improvement over the preceding project core, not an all-shape win over vendor implementations. The experiment flag still defaults to zero.

## Scope and selected source

The wide path accepts batch 1, BF16 input, D256/384/512, lengths 128 through 1024 in increments of 128, and outgoing/incoming/bidirectional contraction. BF16 and FP32 master weights receive correctness coverage; the timings below use BF16 linear weights and FP32 normalization parameters. `MINIWORLD_TRIMUL_H100_B200STYLE=1` is required. D64/D128 inference length expansion to 896/1024 also requires this flag. The narrow training expansion tested separately is **not installed** by this record.

Job 27928 installed exactly six files, recorded in `installed-combined-manifest.json` in the evidence directory. They are the wide autograd/launcher module, the new training TMA launcher and CUDA source, the ctypes driver enum correction, the integration, and the test file. Job 27931 revalidated 43 live-checkout tests. Archive job 27938 independently verified that all six live file hashes still match the selected snapshot. The default backend selection was not promoted.

The following changes preserve the folded B200 wide computation and its BF16 rounding points:

- In inference, a missing dropout scale remains a null pointer through the opaque CUDA call. K3 already implements a null scale as multiplicative one, so it avoids repeated global loads and the unit-scale preparation launch. Nine direct checks establish bitwise equality to an explicit all-ones scale with FP32 master weights, all three widths and directions.
- D512 training K3 saves BF16 projection and gate values through TMA. It waits for the output store to finish reading the shared output buffer, stages projection values there, stores them, waits again, and reuses the buffer for gate values. No additional global intermediate is introduced. The same transformation regressed inference by about 1.2% in job 27907, so inference retains its previous K3 implementation.
- Weight conversion, returning gradients in their master parameter dtype, and dropout generation now stay within native CUDA boundaries. This removes the Inductor Triton conversion/dropout glue from the audited wide execution. The arithmetic conversions themselves still occur. The existing handwritten CUDA Philox dropout implementation is reused.
- The ctypes launch-ex attribute is corrected from enum value 3 to 6, as specified by the installed CUDA 12.9 header for programmatic stream serialization. This fixes the forced-ctypes narrow inference error observed in the initial inventory. The normal `cuda_bindings` driver already used the correct enum. This does not add missing cooperative-launch helpers to the ctypes driver.

## Paired complete-module timings

Job 27915 uses the unchanged official `benchmarks/runners/bench.py`, compiled execution and manual CUDA Graphs. Both graphs have identical seeded inputs and parameters, batch 1, mask probability 0.1, dropout zero for inference and 0.25 for training. Ten warmup pairs precede 90 alternating AB/BA CUDA-event pairs. Training is complete forward+backward. The baseline is a frozen copy of the preceding opt5 **core under the common current native integration**; both sides receive the new common CUDA dropout glue. Thus these ratios do not isolate the integration improvement or compare the entire old installation against the new one.

| Mode | Cells | Geometric mean | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Inference, D256/384/512 | 12 | 1.127688x | 1.077174x | 1.183026x |
| Training, D256/384/512 | 12 | 1.006981x | 0.994153x | 1.024552x |

Cells are outgoing/bidirectional × L384/768 × the listed widths. All 12 inference cells improve. D384 bidirectional L384 training is 0.994153x, a 0.59% regression in this measurement; there is no all-cell training win claim. The D512 training rows targeted by TMA saving are:

| Direction | L | Prior core ms | Selected ms | Prior/selected |
|---|---:|---:|---:|---:|
| outgoing | 384 | 4.904800 | 4.787264 | 1.024552x |
| outgoing | 768 | 19.202145 | 18.849632 | 1.018701x |
| bidirectional | 384 | 8.208448 | 8.101920 | 1.013149x |
| bidirectional | 768 | 32.965569 | 32.626846 | 1.010382x |

The complete 24-row table, raw samples, dispatch profiles and official runner validation metadata are retained in `paired-table.md` and `comparison-27915.json`. No confidence claim across independent jobs is made.

## Fresh NCU evidence

Local Nsight Compute successfully collected hardware counters on H100 80GB. The historical counter-permission failure from a different environment does not apply here. Reports contain kernel replay timings; they are **not** full-module paired benchmarks. SM clocks vary approximately 1.43–1.48 GHz between reports. Six unavailable C2C-link metrics do not affect the reported SM/DRAM metrics.

| Kernel and fixture | Replay duration µs | SM throughput % | DRAM throughput % | Compute-memory throughput % |
|---|---:|---:|---:|---:|
| D512 L384 bidi, old training K3 | 994.784 | 29.10 | 44.87 | 50.45 |
| D512 L384 bidi, TMA-save training K3 | 908.320 | 31.99 | 48.00 | 48.00 |
| D512 L384 bidi, inference K3 with unit scale | 733.568 | 40.19 | 44.81 | 44.81 |
| D512 L384 bidi, inference K3 with null scale | 573.536 | 52.08 | 57.11 | 57.11 |
| D256 L384 outgoing, input K1 | 188.576 | 54.24 | 32.72 | 39.22 |
| D256 L384 outgoing, output K3, null scale | 150.496 | 33.26 | 42.62 | 42.62 |

The first training K3 report shows 80.85% cycles without an eligible warp and 52.2% long-scoreboard stalls, plus roughly half-used global load/store sectors. Shared-memory load conflicts account for 12.71% excessive wavefronts. One CTA per SM and about 14% active-warp occupancy limit latency hiding. This supports attacking accesses and scheduling; it does **not** establish an unavoidable fusion performance limit. The TMA-save and null-scale experiments independently test two concrete sources of memory latency, and the paired full-module measurements establish their practical effect.

The original full F+B profile also records approximately 89.7–93.1% SM throughput for large cuBLAS gradient GEMMs, 80.8% DRAM throughput for output-normalization backward and 84.9% for input-normalization backward. These already-efficient components receive lower priority. Fused fronts/output remain below the 70–80% stopping range: **SOL90 is not achieved**. These are NCU throughput counters, not a claim that the complete workload reaches an arithmetic roofline.

## B200 activation-lifetime deviation: saved normalized input

The preceding opt5 round changed D512 from recomputing input normalization during backward to retaining its BF16 normalized input from forward. The rest of front recomputation remains: the backward front recomputes gate/projection preactivations, fuses their derivatives, and emits the packed gradient consumed by the parameter/input GEMMs. This remains a documented deviation from the original B200 activation storage policy.

The old backward front read raw input, recomputed statistics/normalization, and emitted `xn` for the weight-gradient GEMM. The selected forward front already computes normalized input for projection; it writes that value once and retains it. Backward reads the saved value and avoids the normalization/reformatting work. The old `xn` write is moved rather than removed, so nominal HBM bytes alone do not establish a win. The lifetime costs exactly `2*L*L*D` bytes: 576 MiB at D512 L768 and 1024 MiB at L1024.

The isolated prior cache4 experiment, job 27850, measured 1.010269x complete training at D512 L768 bidirectional. The final opt5 paired training result was 1.018512x across four cells, with all gradients, graph changes and three sanitizers validated. Its measured single-module peak was unchanged outgoing and +0.5 MiB bidirectional, while forward-retained activations increased by 576 MiB. Multiple outstanding forwards can accumulate that extra lifetime; multilayer peak remains unmeasured.

Fresh NCU comparison at D512 L384 bidirectional records the previous recomputing backward K1 at 1853.856 µs / 44.78% SM throughput and the selected saved-input backward K1 at 1726.976 µs / 48.71%. Corresponding forward K1 durations are 1235.968 and 1229.824 µs. These reports also differ in register-cache policy, so they support the observed location of the gain but do not isolate the lifetime change. The isolated paired experiment supplies that evidence. Keeping this deviation is justified for the explicit single-module candidate by the measured full-F+B improvement and existing correctness/sanitizer evidence, with the retention cost explicitly preserved. See the earlier immutable [opt5 record](trimul_b200style_h100_opt5_20261008.md) for the original source/resource/memory measurements.

## Validation and remaining scope

- Job 27915: all 58 tests pass, including independent FP32 output/every-gradient comparisons, actual module use, BF16/FP32 master weights, changed-input/weight/mask/dropout/dy CUDA Graph replay, and null-scale equality. Thresholds remain output relative error <0.006 and each gradient <0.01.
- Another 18 independent output/all-gradient cases pass at D256/384/512 × L896/1024 × all three directions.
- Memcheck: 43 tests, zero errors. The initial pytest racecheck was stopped because its reference calculations made it slow; it is not counted as a pass. Replacement job 27924 exercises five native full-F+B/changed-graph scenarios under racecheck: zero hazards/errors/warnings. Synccheck: 43 tests, zero errors. No sanitizer filtering or suppression was added.
- Job 27894 profiles both a compiled function invocation and captured training graph at D512 L384 bidirectional, with no Triton kernels, changed dropout replay and all-parameter gradient auditing. The wider inventory separately audits actual graph/compiled kernel names. Missing trace data is not treated as proof of native dispatch.
- The completed inference inventory, job 27909, predates null-scale/TMA-save selection. It contains 120 native successful cells and 80 supported Anthropic cells; 40 Anthropic cases are unsupported. The unchanged upstream payload supports D64/128 all directions and D256/384 unidirectional; it does not support D512 or D256/384 bidirectional. Unsupported results are not wins and no repaired/hybrid baseline is substituted.
- That older inventory has 24 Anthropic losses: all 16 D256 unidirectional cells, plus eight D384 short-length cells. It identifies priorities but is not a fresh vendor timing of this installed revision. All-shape paired vendor comparison and the narrow training B200-fusion implementation remain open.

D256 experiments are preserved: an output reorder and three SS input configurations all lost; five of six RS schedule/configuration probes lost or regressed a cell. The sixth (two accumulators, schedule zero) improves four initial inference pairs by 1.027009x geometric mean but is **not installed or qualified by this record**. Neither compilation success nor launch reduction is counted as a speedup.

Evidence is frozen in `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_sol_20261008/`: raw NCU reports/CSV/text, paired samples, complete selected CUDA source, official artifacts, sanitizer logs, preserved failed inventories, reproduction scripts and SHA256 manifest. Original scratch: `/home/psk6950/miniworld-engine-scratch/trimul-sol-20261008/`. No commit/push or maintainer performance-status change was made.
