# TriMul H100 opt7 — K3 lifetime and producer-statistics probes, 2026-10-12

Two further CUDA K3 designs were implemented and checked on H100. Neither is selected for installation. Reusing gate operands for the residual regresses all four NCU cases. Moving output-LayerNorm statistics to producer warps gives a small D256/H256 replay-duration reduction and regresses the other three cases. These are **isolated K3 diagnostics**, not new full-module speedups. **SOL90 remains unmet.**

The installed opt6 files remain unchanged: all 14 hashes match `installed-final-manifest.json`. The live base `k3.cu` and `tmn_kernels.cuh` also match the frozen opt6 snapshot. The previous [full inference/training vendor comparison](trimul_b200style_h100_opt6_20261008.md) is the latest complete-workload evidence; its unsupported Anthropic cells and explicit dispatch settings still apply.

## Environment and measurement scope

Both H100 nodes had rebooted and advertised IDLE, but their `/home` directories were empty. CPU preparation job 27999 failed before entering the script, with signal 53 and no job log. A direct Slurm probe on each node confirmed that the checkout and Pixi environment were absent. The dependent GPU job 28000 was cancelled because its dependency could never succeed. A later node02 check at 09:19 KST still found no shared home.

To make progress without changing system mounts, a small source bundle was streamed through Slurm to `/tmp/psk6950-trimul-opt7-oHb0VkQI` on node02. All compilation, CUDA execution, sanitizers, parsing and archive generation ran on allocated compute resources. No Python, builds or benchmarks ran on the login node. GPU jobs used `h100/cssb/normal_h100`, one H100 each, explicit memory, and at most two concurrent GPU jobs. The native probes use CUDA 12.9 and an H100 80GB HBM3 with driver 575.51.03.

The standalone executable includes the same inference-only folded K3 source used by opt6 and a candidate. It constructs matching TMA descriptors, gives both kernels identical BF16 X/triangle/folded-weight tensors and FP32 folding vectors/input statistics, and compares every output bit. A graph containing both kernels is replayed three times per input case. Cases redraw all inputs, zero X, zero the triangle, and zero projection weights. The grid is D256/D384, H=D/2D, L128/384: **40 cases per variant**. This is not an independent FP32 oracle or full TriMul forward/backward qualification.

NCU runs one full counter capture per kernel at L384 after warm-up, with baseline and candidate in the same process. NCU uses kernel replay; the numbers below are not alternating CUDA-event medians or confidence intervals. Measured SM clocks vary between runs. There is no new Anthropic or cuEquivariance comparison. The unchanged official `benchmarks/runners/bench.py` remains the full-workload timing path.

## Residual reuse: safe, but slower

The original K3 loads X for the gate projection, then loads its residual tile into the output staging buffer. `k3reuse.cu` directs the relevant gate-operand chunks into that buffer immediately. The epilogue uses those same bytes, eliminating the separate residual TMA loads. The buffer layout becomes column-chunk/warpgroup/token, matching the gate descriptor. Two consumer warpgroups release the buffer only after their output TMA stores finish reading it; the producer waits before refilling it.

This keeps folding, matrix products, BF16 rounding and the epilogue unchanged. It also adds a dependency between the next gate input and the preceding output store. The logical reduction is one BF16 `[M,D]` read, but that does not establish an HBM reduction.

| D / H, L384 | Original K3 µs | Reuse K3 µs | Original / reuse SM % | Original / reuse DRAM reads, MB |
|---|---:|---:|---:|---:|
| 256 / 256 | 131.424 | 144.544 | 40.62 / 34.96 | 153.27 / 152.61 |
| 256 / 512 | 168.320 | 182.016 | 46.36 / 41.46 | 229.32 / 229.07 |
| 384 / 384 | 227.296 | 261.344 | 51.81 / 44.06 | 269.10 / 289.77 |
| 384 / 768 | 312.448 | 349.504 | 55.87 / 49.68 | 574.92 / 629.64 |

The D256 DRAM reads barely change. This supports the interpretation that the removed residual loads were largely served from cache. Eligible warps per scheduler fall from 0.566 to 0.483 for D256/H256; added serialization is consistent with that loss. D384 actually reads more DRAM in these captures. The counters do not isolate the precise cache-eviction cause, so a complete causal cache model is not claimed. Reducing source-level loads did not improve this implementation.

Job 28006 passes all 40 bitwise/graph cases with zero spills. Job 28007 passes memcheck, racecheck and synccheck for all four D/H cases at L128. Job 28026 adds all three tools at L256 for D256/H256 and D384/H768, exercising multiple persistent tiles per CTA. All 18 sanitizer invocations pass with zero errors or hazards. Job 28008 supplies the four NCU captures.

## Producer statistics: corrected and qualified as a native probe

The second design keeps the original X/residual pipeline. Two otherwise idle producer warps calculate output-LayerNorm statistics while consumer warpgroups issue WGMMA. Each producer lane retains the same four partial sum/square-sum chains as the original consumer layout; the final four-way combination keeps its original order. Consumers snapshot the four statistics for their accumulator rows before the producer can reuse the shared buffer. Fusion boundaries, global intermediates and the folded formula stay the same.

The first version passes all 40 numerical cases but spills 24 bytes per thread. SASS places the local loads/stores after the producer's register reduction. Reallocating producer/consumer registers from 40/232 to **56/224** removes spills for all four D/H builds while preserving the same 64,512-register CTA allocation. Numerical checks alone were insufficient:

- `stats56`, job 28015: memcheck passes, but the first racecheck reports 32 displayed errors on shared statistics publication. Its NCU captures are retained as unqualified diagnostics, not as a safe optimization result.
- `statsbar`, job 28021: full-participant named barriers remove racecheck hazards in all four cases, but synccheck reports 5,728 errors for the legacy aligned `bar.sync` used from separate producer/consumer branches. This version is rejected.
- `statssync`, job 28025: the new cross-branch handoff uses `barrier.sync` without `.aligned`, with 160 participants per producer-warp/consumer-warpgroup pair. This follows the conditional-execution distinction in the [NVIDIA PTX barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier). The existing common header is not changed. All 40 bitwise/graph cases and all 12 L128 sanitizer invocations pass. Job 28027 adds six L256 persistent-tile sanitizer invocations; all pass. The final four builds have zero stack/spill bytes.

The final producer-statistics counters are:

| D / H, L384 | Original K3 µs | Statistics K3 µs | Original / candidate SM % | Candidate DRAM % |
|---|---:|---:|---:|---:|
| 256 / 256 | 130.304 | 127.776 | 40.85 / 40.81 | 50.27 |
| 256 / 512 | 171.072 | 176.608 | 46.66 / 44.47 | 49.65 |
| 384 / 384 | 227.392 | 236.608 | 51.84 / 49.67 | 46.85 |
| 384 / 768 | 315.936 | 336.448 | 56.36 / 51.97 | 60.53 |

The roughly 2% D256/H256 replay-duration reduction is a narrow hypothesis for later full-module measurement. It is not a demonstrated module speedup. The other cases regress; the new transfer barriers and concentration of statistics work into two warps do not deliver a general gain. Neither SM nor DRAM throughput reaches the requested 70–80% stopping range in these final candidate captures.

## Evidence and next gate

Evidence is frozen in `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt7_20261012/`: 155 files listed in `SHA256.json`, including generated CUDA sources, source-generation scripts, raw successful and failed logs, all 12 two-kernel NCU reports, and `ncu-comparison.csv`. `native-summary.json` records the limited qualification scope and marks installation false. Failed intermediate variants remain identifiable. An initial executable build with an overly broad architecture target also remains in the logs; the successful builds explicitly target `compute_90a/sm_90a`.

`prepare_full_statssync.py` is ready to create an isolated complete-module snapshot after `/home` is restored. It first checks all 14 live opt6 hashes and the archived CUDA-source hash, retains a separate native reference, and adds the required 32 bytes to shared-memory sizing. `qual_full_statssync.sh` then runs the existing wide suite and the unchanged official runner through the existing paired observer. These two full-module steps have **not run**. Their Python syntax was checked on a CPU allocation; that does not qualify the generated full-module path.

No dispatch promotion, source installation, commit or push occurred. Whole-module correctness and paired inference timing for a new candidate remain pending shared-environment access. The goal also still includes SOL work and the previously documented B200 storage/placement comparisons; this round does not close them.
