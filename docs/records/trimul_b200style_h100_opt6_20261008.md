# TriMul H100 opt6 — vendor wins across the measured grid, 2026-10-08

The explicit CUDA candidate wins all **80 supported Anthropic inference comparisons** and all **120 cuEquivariance training comparisons** in the grid below. D256 inference now clears Anthropic after two further input/output changes. D128 training uses the preserved fused B1/B7 path at every tested length, replacing the much slower generic path used by the first inventory. The candidate is installed; the default B200-style flag remains zero. **SOL90 is not achieved.**

The grid is batch 1, BF16 activations and linear weights, FP32 LayerNorm parameters, hidden width equal to pair width per direction, D64/128/256/384/512, L128/256/384/512/640/768/896/1024, and outgoing/incoming/bidirectional. Training includes complete forward+backward with dropout 0.25; inference has no dropout. Both use mask probability 0.1 and one module layer. FP32 master parameters receive correctness coverage, not the performance claim above. Other batches, hidden widths, dtypes and arbitrary lengths are outside this result.

## Selected changes

- D256/D384 K1 reads the four original row-major projection weights through separate TMA descriptors. It constructs pair-mask factors directly from the bool token mask. This removes weight packing and square-mask preparation from forward while keeping input LayerNorm, gated projections and plane stores fused. Other weight layouts and mask representations retain the previous preparation path. D512 retains its SS front.
- D256 forward uses the two-accumulator RS schedule `(2,64,4,4,1,0)`. Backward retains its prior schedule. On outgoing/bidirectional L384/768, the combined input change improves complete inference by 1.044533x geometric mean for D256 and 1.016840x for D384 over the preceding installed core. Training improvements are smaller; D384 outgoing L768 measures 0.998410x and is not reported as a win over that core. Incoming L128/1024 checks and timings are also retained.
- Inference with no dropout scale gets a dedicated K3 compilation with activation/statistic saving and scale-load branches removed at compile time. Explicit-scale inference and training retain their respective kernels. This improves all 12 initial complete-module inference pairs by 1.014864–1.059381x over the direct-input/RS revision. The unit-scale and null-scale results are bitwise equal in the tested cases.
- D128's existing H100 B1/B7 implementations now accept all eight lengths. The selected L384 configuration is reused through L384, and the L768 configuration above it. This is a validated extension of fixed configurations, not an exhaustive tune. D64's native preparation accepts the two additional lengths. These paths require the explicit native flags below.
- The preceding round's CUDA-only casts/dropout glue and D512 training TMA activation stores remain selected. See the immutable [SOL audit](trimul_b200style_h100_sol_20261008.md) for those changes and the retained-normalized-input analysis.

This uses the B200 wide folding identities and the small-width B1/B7 grouping. It is not a claim of identical storage and scheduling on the two architectures. In particular, D512 retains normalized input as documented in opt5/SOL. The existing D64 H100 implementation recomputes normalization from saved statistics and places the output-gate input-gradient GEMM in B1, whereas B200 small stores normalized input and combines that GEMM in B7. Its current vendor/correctness results are validated here; an isolated comparison of those D64 storage/placement policies remains open. No claim that every B200 deviation has been exhausted is made.

## Vendor comparisons

All timings come from the unchanged official `benchmarks/runners/bench.py`. Inference uses manual CUDA Graphs, with compilation enabled for our module and the official uncompiled native Anthropic wrapper. Training uses compilation and manual graphs on both sides. The upstream Anthropic payload is the unchanged CUDA 12.9 rebuild, without an overlay or repaired comparator. cuEquivariance is version 0.12.

The main protocol constructs both complete graphs, asserts identical initial input/parameter tensors, warms ten pairs, and records 90 alternating AB/BA CUDA-event pairs on the same allocated H100. The reported ratio is vendor median time divided by candidate median time. Native dispatch is checked on both graph replay and a separate compiled-function invocation. All audited candidate traces contain CUDA/cuBLAS kernels and **zero Triton kernels**. Baseline or reference Triton kernels are not relabeled as candidate execution.

| D | Inference cells | Inference geometric mean vs Anthropic | Range |
|---:|---:|---:|---:|
| 64 | 24/24 wins | 1.158396x | 1.070801–1.345638x |
| 128 | 24/24 wins | 1.142299x | 1.056133–1.246187x |
| 256 | 16/16 wins | 1.056975x | 1.032036–1.155903x |
| 384 | 16/16 wins | 1.248402x | 1.229426–1.283041x |

Anthropic does not support the remaining **40** cells: D512 all directions and D256/D384 bidirectional. Those cells are not counted as vendor wins. Our native inference and correctness coverage includes them. D256/D384 vendor comparisons are outgoing/incoming only; D64/D128 include all three directions. Jobs 27963 and 27969 supply these 80 final pairs.

| D | Training cells | Geometric mean vs cuEquivariance | Range |
|---:|---:|---:|---:|
| 64 | 24/24 wins | 2.674828x | 1.871016–2.930207x |
| 128 | 24/24 wins | 1.940531x | 1.526219–2.305100x |
| 256 | 24/24 wins | 1.864280x | 1.125554–2.651173x |
| 384 | 24/24 wins | 2.159519x | 1.321739–2.601202x |
| 512 | 24/24 wins | 2.405769x | 1.322454–3.017292x |

There are **119 alternating graph pairs**, from the successful rows of jobs 27977 and 27990. The last D512 L1024 bidirectional cell uses a separate, explicitly different protocol because retaining both graphs plus capture-time working memory exceeds 80GB. Job 27980 runs the unchanged official measurement in four fresh processes on one GPU, in vendor/ours/ours/vendor order:

| Block | Implementation | Full F+B ms |
|---:|---|---:|
| 1 | cuEquivariance | 118.636963 |
| 2 | candidate | 58.587681 |
| 3 | candidate | 59.701344 |
| 4 | cuEquivariance | 118.215393 |

The ratio of block means is **2.002319x**. Even the fastest vendor block divided by the slowest candidate block is **1.980113x**. This is not a 90-pair estimate or an independent-job confidence interval. The D512 table includes this one ratio of block means; the remaining 23 D512 cells use alternating graph medians. The ABBA processes use the official fixed-seed setup and matching arguments; cross-process tensor equality was not separately asserted. Both native blocks pass the dispatch and all-parameter gradient audit.

The initial generic D128 training inventory had zero wins in 24 cells. The B1/B7 extension is independently checked rather than treating that generic result as a limit on fusion. No vendor implementation or correctness threshold was changed to produce these results.

## Correctness, sanitizers and installed state

- Direct-input/RS candidate, job 27946: 58 tests plus 18 full output/every-gradient cases at L896/1024; memcheck 43 tests, racecheck five full-F+B/changed-graph scenarios, synccheck 43 tests, all with zero sanitizer errors/hazards.
- Final wide inference specialization, job 27963: 67 tests, including nine changed-input/weight/affine/mask inference graph cases. Memcheck and synccheck each pass 18 inference/changed-graph cases; racecheck passes nine inference cases with zero hazards. Explicit unit-scale equality is retained.
- Expanded D128, job 27966: 15 independent tests covering every gradient, BF16/FP32 master parameters, L1024, changed graph values and outstanding-forward ownership. Memcheck/synccheck each pass nine cases. Three full native F+B/changed-graph race scenarios pass with zero hazards; zero dropout gives exact residual output/input gradient and zero parameter gradients.
- Expanded D64, job 27976: the corresponding 15 checks and all three sanitizer tools pass. The full training inventory also audits all parameter gradients and actual compiled/graph dispatch across 120 cells. Output relative error remains <0.006 and each gradient <0.01 against the FP32 oracle; thresholds were not relaxed.
- Live checkout, job 27981: 52 wide tests, nine D128 tests and nine D64 tests pass after installation, followed by 18 L896/1024 inference cases against FP32 plus bitwise agreement with explicit-unit-scale inference. The final test-only cleanup adds proper CPU/Hopper skip guards and import formatting; job 27995 confirms 15 CPU skips.
- Job 27993 verifies SHA256 equality for all 14 installed files, including the prior round's three supporting files. The installed manifest is `installed-final-manifest.json`. The candidate core used for timing matches the selected snapshot; the final test-only cleanup is recorded separately. No default dispatch promotion, commit or push occurred.

## NCU and remaining work

Fresh NCU full reports from job 27963, D256 L384 outgoing inference:

| Kernel | Replay duration µs | SM throughput % | DRAM throughput % | Compute-memory throughput % |
|---|---:|---:|---:|---:|
| direct-input K1, two-accumulator RS | 182.944 | 57.31 | 33.62 | 40.78 |
| specialized K3 | 128.128 | 40.13 | 50.07 | 50.07 |

The previous D256 report recorded K1 at 188.576µs / 54.24% SM and K3 at 150.496µs / 33.26% SM. These are replayed kernel counters from separate profiling runs, not paired full-module timings. Six unavailable C2C metrics do not affect these counters. The fused kernels remain below the 70–80% stopping range and SOL90 remains open. Prior large cuBLAS gradient GEMMs already measured approximately 90% SM throughput.

Six output tile/stage configurations failed to improve the complete workload overall. A later M64/one-consumer-warpgroup design intended to allow two CTAs per SM passes 18 inference/graph checks, but all four D256 full-module pairs regress (0.938548–0.961155x, job 27985). It is not installed and did not receive sanitizer or NCU qualification. More resident blocks and fewer preparation launches are not by themselves performance results.

Failures are preserved: the first static specialization rejected a mismatched D512 source anchor; the first expanded bidirectional D128 sweep hit an old host length guard; the first small-width test incorrectly differentiated mask preparation and was corrected to match the integration's `no_grad` boundary. Large-pair attempts hit capture/profiling OOM, and one post-OOM trace ended with an unspecified launch failure. The measurement orchestrator now profiles graph replay first, frees graph pools before profiling the compiled function, and releases previous-cell references. Successful rows are retained; failed attempts are not counted as validated timings. The official runner and timed arithmetic were unchanged.

## Reproduction and evidence

The complete measured native training candidate uses these explicit settings; inference only needs the B200-style flag:

```bash
export MINIWORLD_TRIMUL_H100_B200STYLE=1
export MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS=1
export MINIWORLD_TRIMUL_H100_PRESERVE_D128=1
export MINIWORLD_TRIMUL_H100_TRAIN_CUDA=1
export MINIWORLD_TRIMUL_H100_TRAIN_FUSED_PREP=1
export MINIWORLD_TRIMUL_H100_CUDA_AUX=1
export TRIMUL_NATIVE_DRIVER=cuda_bindings
```

Run GPU work through Slurm `h100/cssb/normal_h100`, one H100 per job with an explicit memory request. The cooperative D128 helpers require the `cuda_bindings` driver; the forced ctypes inference enum correction is not a general cooperative-driver implementation.

Evidence is frozen in `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt6_20261008/`. `vendor-summary.csv` has all 200 comparable inference/training rows with the timing method attached. Raw alternating samples, ABBA audits, official artifacts, source/manifests, successful and failed logs, NCU reports and reproduction scripts are retained. Archive job 27996 records 2,720 evidence files and verifies all 14 installed hashes. Frozen benchmark snapshots remain under `/home/psk6950/miniworld-engine-scratch/trimul-sol-20261008/`.
