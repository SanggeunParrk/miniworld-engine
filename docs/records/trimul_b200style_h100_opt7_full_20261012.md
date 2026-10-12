# TriMul H100 opt7 full-module follow-up, 2026-10-12

The producer-statistics K3 candidate has now passed the existing full-module correctness suite, but **does not provide a material inference speedup**. Keep installed opt6. This closes the full-module measurement gap in the earlier [opt7 native-probe record](trimul_b200style_h100_opt7_20261012.md); it does not change that historical record or install the candidate.

## Environment and qualification

The compute nodes still lacked the shared `/home` mount. The existing Pixi environment and source checkout were streamed through Slurm into a private node02 `/tmp` directory. This uses the same PyTorch 2.13.0+cu129 environment without package installation or changes to system mounts. Compilation, imports, tests, measurement and archive creation ran on compute allocations with explicit memory requests, `cssb/normal_h100`, and one H100 per GPU job.

All 14 opt6 source hashes were checked before creating the isolated snapshot. Only the no-save/no-dropout-scale D256/D384 inference output launcher selects `mw_b200style_k3_statssync`; D512 and training retain their existing paths. The candidate adds 32 shared-memory bytes. A separately named opaque reference preserves the original static K3. No Triton kernel appears in either measured graph or the candidate compiled invocation.

Job 28042 passes **67 tests** in `test_trimul_h100_b200style_gpu.py`, covering output/gradient correctness, module dispatch, changed graph inputs/weights/mask/dropout/dy, and inference without an explicit scale. Its subsequent benchmark invocation initially failed because the wrapper omitted the snapshot path export; that failure remains in the evidence. Correcting the wrapper resumed measurement without changing candidate CUDA or tests. The earlier final native candidate already passed 40 bitwise/changed-input graph cases and 18 sanitizer invocations; see opt7 for their narrower scope and raw evidence.

## Paired full-module inference

Job **28044**, GPU `GPU-7e9246a1-17c3-399f-5de8-5528ecae81b6`, captures both implementations through unchanged `benchmarks/runners/bench.py`, then measures 90 alternating CUDA-event pairs. Inputs and parameter tensors are asserted identical. The scope is batch 1, BF16 activations/linear weights, FP32 LayerNorm parameters, hidden width D per direction, dropout zero, mask probability 0.1, one module layer, `compile=true`, and manual CUDA Graphs.

The observer profiles each **captured graph** and the candidate compiled invocation after timing. It asserts that the baseline launches `mw_b200style_k3_static`, the candidate launches `mw_b200style_k3_statssync`, and all three profiles contain no Triton kernels. These names prove the measured graphs differ at the intended dispatch point. The official runner SHA256 is `9c414e84c20b74a3c94f16d6928066a0910e0ce306ea8d4d15dbc43463f9181a`.

| Direction | D | L | Opt6 ms | Statistics candidate ms | Opt6 / candidate |
|---|---:|---:|---:|---:|---:|
| Outgoing | 256 | 384 | 0.352768 | 0.352768 | 1.000000 |
| Outgoing | 256 | 768 | 1.475520 | 1.474560 | 1.000651 |
| Outgoing | 384 | 384 | 0.663968 | 0.669600 | 0.991589 |
| Outgoing | 384 | 768 | 2.710272 | 2.746688 | 0.986742 |
| Bidirectional | 256 | 384 | 0.610688 | 0.610656 | 1.000052 |
| Bidirectional | 256 | 768 | 2.590304 | 2.603808 | 0.994814 |
| Bidirectional | 384 | 384 | 1.165600 | 1.184544 | 0.984007 |
| Bidirectional | 384 | 768 | 4.923456 | 4.987680 | 0.987124 |

The isolated NCU D256/H256 duration reduction did not survive as a useful module gain. D256 outgoing is effectively tied; D384 and the longer bidirectional cases regress. No further vendor sweep is warranted for this candidate. The [opt6 vendor comparison](trimul_b200style_h100_opt6_20261008.md) remains the latest installed-path evidence, including its unsupported Anthropic cells and explicit native flags. SOL90 remains unmet; no new utilization claim is inferred from these CUDA-event timings.

## Evidence

`runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt7_full_20261012/` contains 75 files listed in `SHA256.json`: raw pairs, graph and compiled dispatch profiles, official runner artifacts, source/launchers, tests, setup and observer scripts, manifests, successful logs and the initial path failure. `summary.json` records installation as false. The private staged environment remains available for subsequent D64 B200 storage/gradient-placement experiments.

No production source installation, default dispatch promotion, commit or push occurred in this follow-up.
