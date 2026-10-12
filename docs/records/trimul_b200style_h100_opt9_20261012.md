# TriMul H100 opt9 — B200 small-width saved-XN policy, 2026-10-12

Four D64 CUDA implementations of B200's normalized-input save were built and measured against the installed **opt8** implementation. None qualifies for selection. The final version matches B200 small's saved normalization set—BF16 affine XN plus output-LayerNorm statistics—and recomputes input statistics in B7. It loses **11/12 complete F+B comparisons**, with ratios of **0.988347–1.001513x**. The one positive row is L128 bidirectional, **1.001513x**; that single 0.15% difference is not treated as an established improvement.

The installed opt8 implementation remains unchanged. Its six selected files were returned from the live checkout and matched against `d64-selected-manifest.json`. No experimental dispatch was promoted, and no commit or push occurred. **SOL90 remains unmet.** These results support retaining statistics/recomputation for the tested D64 implementation; they do not establish a limit on future fusion or cover every possible XN production schedule.

The baseline and preceding qualification are documented in [opt8](trimul_b200style_h100_opt8_20261012.md). This round measures native optimization candidates, not a fresh Anthropic/cuEquivariance comparison. Inference and other widths receive no new performance claim here.

All performance rows use batch 1, D64, hidden width 64 per direction, BF16 activations/linear weights, FP32 LayerNorm parameters, dropout 0.25, mask probability 0.1 and one module layer. Each variant covers outgoing, incoming and bidirectional modules at **L128/384/768/1024**. The unchanged official `benchmarks/runners/bench.py` captures compiled manual CUDA Graphs. The observer verifies matching input/parameter state and measures **90 alternating event pairs** per cell. Ratios are baseline median divided by candidate median, not confidence intervals. Dropout regimes match; independent graph replays are not claimed to use identical random draws.

Both measured graphs and the candidate compiled invocation are profiled for actual kernel names and absence of Triton kernels. All 48 successful paired cells meet these dispatch assertions. The runner SHA256 is `9c414e84c20b74a3c94f16d6928066a0910e0ce306ea8d4d15dbc43463f9181a`. Work used one allocated H100 per GPU job on node02, `h100/cssb/normal_h100`, with explicit memory requests and isolated source snapshots. The existing staged environment was reused.

| Variant | XN save | B7 XN consumer | Input statistics | Faster cells | Opt8 / candidate F+B |
|---|---|---|---|---:|---:|
| `savedxn` | TMA | Copy staged XN into persistent operand tile | Saved | 0/12 | 0.978790–0.999093x |
| `savedxn-stg` | Direct register stores | Same copy | Saved | 0/12 | 0.966127–0.987765x |
| `savedxn-ring` | Direct register stores | Consume ring tile directly | Saved | 0/12 | 0.976134–0.996413x |
| `savedxn-nostats-v2` | TMA | Consume ring tile directly | Recomputed in B7 | 1/12 | 0.988347–1.001513x |

The first three variants were measured in jobs **28131, 28138 and 28142**. The final successful compiled comparison is **28162**. The final row combines returning to the faster TMA store with removing input-statistics saves; its improvement over the direct-store ring control is not attributed solely to statistics recomputation.

B1 now consumes the forward's rounded affine XN. This removes reconstruction of input xhat, input-affine folding into Wg, gate-bias correction, and the persistent `dGout` row sums previously used to unfold dWg. Output-LayerNorm folding is retained. B1 still writes BF16 dGout, and B7 still accumulates its gate GEMM with the input-projection gradient in FP32, as selected in opt8.

B7 initially loaded saved XN into its ring, copied it into the existing persistent XN tile, and overlapped a raw-X reload with GEMMs. The ring variant removes that copy: WGMMA reads XN directly from the ready ring slot, while the old persistent allocation holds raw X for LayerNorm backward. Existing tile-lifetime barriers and an additional TMA completion barrier protect the buffers. Shared-memory allocation does not increase. The final variant also removes input-statistics writes/reads and computes mean/rstd from raw X. This matches the B200 small saved-normalization policy, while H100 scheduling, output folding and partial-gradient reduction remain architecture-specific.

There is a concrete storage cost. At L1024, saving BF16 `[L²,64]` XN adds **128MiB**. Removing the two FP32 input-statistics arrays recovers **8MiB**, leaving **120MiB** of additional retained activation storage. Counting logical transfers per token in the final variant gives:

| Changed stage | Additional bytes/token relative to opt8 |
|---|---:|
| K3: save XN, omit input statistics | +120 |
| B1: read XN instead of X, omit input statistics | −8 |
| B7: read XN and raw X, omit input statistics | +120 |
| Total changed logical traffic | **+232** |

Those are source-level bytes, not measured HBM traffic. Cache behavior changes the physical traffic.

NCU kernel replay at L384 confirms reduced B1 work but higher K3 cost. The final variant's captures, with each baseline/candidate pair in the same process, are:

| Direction / stage | Opt8 µs | Final candidate µs | Candidate SM throughput | Candidate DRAM throughput |
|---|---:|---:|---:|---:|
| Outgoing K3 | 44.288 | 49.056 | 38.82% | 35.56% |
| Outgoing B1 | 50.336 | 44.480 | 36.16% | 54.35% |
| Outgoing B7 | 87.072 | 85.184 | 36.09% | 45.55% |
| Bidirectional K3 | 49.216 | 54.368 | 41.34% | 43.78% |
| Bidirectional B1 | 63.648 | 60.832 | 34.19% | 58.41% |
| Bidirectional B7 | 125.408 | 121.984 | 36.12% | 41.48% |

Candidate B1 has zero measured local load/store sectors; the baseline bidirectional B1 has about 73,728 local-load sectors. B7 DRAM reads increase from **96.24 to 113.92 MB** outgoing and **134.02 to 151.71 MB** bidirectional. Direct register stores proved slower than TMA stores. Removing the shared copy recovers some of that loss, but the complete workload still fails the selection criterion.

Summing individually replayed kernel durations would misleadingly suggest a small gain. Therefore job **28167** also profiles the actual official-runner graphs with `--graph-profiling graph`, preserving dependencies and data reuse between their kernels. At L384 the final candidate has:

| Complete graph | Opt8 DRAM reads + writes | Candidate DRAM reads + writes | Increase |
|---|---:|---:|---:|
| Outgoing | 462.866 MB | 499.769 MB | **36.903 MB / 7.97%** |
| Bidirectional | 786.917 MB | 822.278 MB | **35.360 MB / 4.49%** |

This directly confirms the additional traffic in the complete graph and supports the storage-cost explanation. It does not assign every microsecond of the paired regression to a particular transfer. NCU's instrumented graph durations, including separately captured RNG scalar-write kernels, differ substantially from the unprofiled timings and are **not** used as benchmark results. The speed decision uses the 90-pair measurements above. Kernel-level utilization remains below the requested 70–80% stopping range, as well as SOL90.

Validation retains the existing FP32-oracle limits: output relative error below 0.006 and every gradient below 0.01. Each of the four CUDA variants passes the expanded 15-case suite: both master-weight types, all directions, L1024, changed input/weight/mask/dropout/dy graph replay, zero-mask behavior and outstanding-forward ownership. The direct-store, ring and final no-input-statistics variants each pass **48 full-grid correctness cases**, covering all eight lengths from L128 through L1024 and both master-weight types. Maximum output error is **0.00360320** and maximum gradient error is **0.00718474**. No tolerance was relaxed.

Memcheck, racecheck and synccheck pass for all three directions at L256 for all four variants, including persistent ring reuse and changed graph values. The no-statistics namespace repair repeats the 15-case suite and all three sanitizers, giving **45 direction/tool combinations including that repeat**, all with zero errors/hazards.

The initial no-statistics compiled attempt, job **28148**, failed with a two-versus-four-row saved-statistics shape assertion. Its CUDA correctness and sanitizer passes did not qualify the compiled path. Reusing the earlier custom-op name allowed the old saved-shape contract to be reused; the corrected version gives this contract a unique `trimul_h100_savedxn_nostats_*` opaque namespace. The original failed snapshot, manifest and runner artifacts are preserved. The CUDA source is unchanged by that repair; job 28162 reruns graph/ownership checks and sanitizers and successfully measures all 12 compiled cells. No failed timing is counted as a successful row.

Frozen evidence is in `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt9_20261012/`: **525 files listed in `SHA256.json`**, 48 paired cells with raw samples and dispatch profiles, 48 individual-kernel NCU captures, four whole-graph captures plus their eight RNG scalar-write captures, ten NCU reports, numerical errors, sanitizer logs, source variants, B200 reference sources, compiler logs, generators, the failed namespace attempt and official runner artifacts. Source manifests were checked before archiving, and all 525 returned files passed hash verification. `results/d64-savedxn-summary.json` provides the machine-readable verdict and traffic totals.

The saved-XN policy comparison left open in opt8 now has a tested exception for this D64 implementation. Further optimization remains open: eliminating the existing B1 spill without introducing the additional saved tensor, or changing where XN is produced, would be separate candidates requiring the same complete-workload validation.
