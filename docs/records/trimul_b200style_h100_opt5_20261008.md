# TriMul H100 B200-style opt5 — 2026-10-08

D512 명시적 실험 경로의 입력 융합과 학습 중 정규화 입력 재사용을 개선했다. 기본 선택은 `MINIWORLD_TRIMUL_H100_B200STYLE=0`으로 유지한다. D256/D384, contraction, output K3, backward GEMM과 정규화 수식은 그대로다.

## Selected implementation

- The D512 input front retains seven of eight64-channel raw-input chunks in registers during normalization, up from four. Producer/consumer register roles change40/232 to24/240, retaining the64,512-register CTA budget. Nominal shared input reads per128-token tile drop256KiB to160KiB; normalized shared writes remain128KiB. This is shared-memory traffic, not HBM traffic.
- Training emits BF16 normalized input `xn` from the fused forward front. Backward loads this normalized input directly, recomputes the same gate/projection preactivations, and uses the saved `xn` for weight-gradient GEMM. The old path normalized raw input again and emitted `xn` during backward. The write is moved to forward, not duplicated. Inference keeps its no-`xn` storage path with the larger register cache.
- This deliberately changes the input activation lifetime relative to the preceding recomputation path: the normalized input is now retained between forward and backward. It is an optimized B200-style variant, not a claim of identical B200 activation storage policy. Output folding and all BF16 rounding points remain unchanged. Full tests require training and inference outputs to match exactly.
- Selected kernels are `mw_b200style_k1ssregtune_n128t128s4p0w1c7` for inference, `mw_b200style_k1sssave_n128t128s4p0w1c7` for training forward, and `mw_b200style_k1sssavewb_n128t128s4p0w1c7` for training backward. Resource checks give STACK0/local instructions0, REG168 and dynamic shared229,888B plus static1,024B. H512/H1024 training forward reaches SASS R237; normalized-input backward reaches R146.

The production code changes are the existing selector/autograd module plus two CUDA sources and two Python launchers. The duplicated exploratory module and opt4 reference module remain in scratch/evidence only. Neither the integration flag default nor the project-default backend is changed.

## Paired full module performance

Authoritative comparison job27869 uses frozen `snapshot-final`, actual selected dispatch, and a preserved opt4 reference module. Reference opaque-op names are made unique so both graphs coexist; its algorithms/configuration remain opt4. Both sides use flag1, batch1, BF16 activations and linear weights, FP32 normalization, identical seeded input/parameters, mask probability0.1, and dropout0/0.25 for inference/training. The unchanged official runner captures `torch.compile` plus manual CUDA Graph complete module steps. Ten warmup pairs precede ninety alternating AB/BA event pairs. Training includes complete forward+backward.

| Mode | Direction | L | Opt4 ms | Opt5 ms | Opt4/opt5 |
|---|---|---:|---:|---:|---:|
| inference | outgoing | 384 | 1.298880 | 1.280896 | 1.014040x |
| inference | outgoing | 768 | 5.146464 | 5.064928 | 1.016098x |
| inference | bidirectional | 384 | 2.136832 | 2.114624 | 1.010502x |
| inference | bidirectional | 768 | 8.580512 | 8.496416 | 1.009898x |
| training | outgoing | 384 | 4.920704 | 4.808608 | 1.023311x |
| training | outgoing | 768 | 19.304577 | 18.924416 | 1.020088x |
| training | bidirectional | 384 | 8.181024 | 8.074816 | 1.013153x |
| training | bidirectional | 768 | 33.048191 | 32.479038 | 1.017524x |

Geometric mean across four cells per mode: inference **1.012631x**, training **1.018512x**. All eight within-job95% bootstrap intervals for paired mean time differences are positive. The10,000 paired resamples do not model serial correlation or variation between jobs. Earlier combined-prototype job27865 also improved all eight cells; the final selected result above is authoritative.

Separate paired project-default comparison, job27873, D512 L768 bidirectional:

| Mode | Default ms | Opt5 explicit path ms | Default/opt5 |
|---|---:|---:|---:|
| inference | 7.863936 | 8.784128 | 0.895244x |
| training | 28.605888 | 34.187809 | 0.836728x |

The project default remains faster. Absolute times across different jobs are not a matched three-way comparison. This round does not remeasure vendor kernels, SOL90, batch>1, incoming-wide performance, or FP32-master timing.

## Memory cost

The saved normalized tensor is `2*L*L*D` bytes:144MiB at D512 L384 and576MiB at L768. Forward-retained activation bytes increase by exactly this amount. The former backward already allocated `xn`, so the single-module F+B peak need not increase by the retained amount.

Native allocator probe job27871, D512 L768, BF16 linear weights, warmup followed by reset peak statistics. The following values exclude the existing fixture/input allocation. These are native eager memory observations, not performance timings or CUDA Graph/multilayer memory guarantees.

| Direction | Path | Retained after forward MiB | Full F+B peak extra MiB |
|---|---|---:|---:|
| outgoing | opt4 | 3465.008 | 10405.152 |
| outgoing | opt5 | 4041.008 | 10405.152 |
| bidirectional | opt4 | 5193.008 | 16181.660 |
| bidirectional | opt5 | 5769.008 | 16182.160 |

Outgoing full F+B peak is unchanged; bidirectional rises0.5MiB in this probe. Multiple outstanding forward activations can accumulate the extra retained tensors; no multilayer peak claim is made.

## Validation and live installation

- Saved-input prototype job27849, pipeline prototype27854, register-cache7 prototype27857, normalized-gate prototype27861, and combined prototype27864 each pass all49 existing tests. Final actual selected job27870 also passes49 tests, including independent FP32 output/every-gradient checks, L384/768 full shapes, BF16 and FP32 master weights, zero masks, changed input/weight/mask/dropout/dy graph replay and actual module dispatch. Thresholds remain output relative<0.006 and every gradient relative<0.01.
- Final job27870:memcheck25 tests/errors0; racecheck five full native F+B/changed-graph scenarios/hazards0/errors0/warnings0; synccheck25 tests/errors0. No filters/suppressions. The earlier combined prototype also passed these three tools in job27867.
- Native full-F+B and inference CUDA profiles in job27871 assert the intended three kernel names for outgoing and bidirectional. Official inference profiles also record the new front. Post-capture training profiles can omit forward kernel names, so those sparse traces are not treated as complete dispatch evidence. Eager probes establish dispatch/memory only; all reported speed ratios are compiled manual-graph full steps.
- Live checkout jobs27876/27877:25 native/edge/changed-graph tests and12 actual-module all-gradient tests passed after copying exactly five selected files. All23 listed source files match frozen selected source bytes. The live family digest can differ due to retained unrelated experimental sources; this is not claimed to be whole-family hash equality.
- Ruff formatting/checks and `git diff --check` pass. No commits or pushes. D512 flag1 receives the new path; the flag still defaults to0.

## Other experiments preserved

| Candidate | Result and disposition |
|---|---|
| Save normalized input, original cache4 | job27850:inference0.995003x, training1.010269x at D512 L768 bidi. Training saving reused in final design; inference uses the separate no-save cache7 front. |
| Alternate two accumulators across consecutive input-front output blocks |40/232 register roles spill16B;24/240 roles remove spills. Job27854:49pass. Job27855:inference0.625321x, training0.869249x. Rejected. |
| Cache5/6/7/8 with24/240 roles | CPU job27856:5/6/7 have zero spills,8 has24B stack/24 local instructions. Only cache7 receives GPU qualification/timing; no exhaustive tuning claim. Job27858:inference1.008520x, training1.001594x. |
| Reuse normalized input in output gate, remove gate folding | Preserves training/inference equality by using the same normalized-gate calculation in both modes. Job27861:49pass. Job27862:inference0.970339x, training1.014367x. Rejected due to inference regression. |

Combined setup job27863 stopped on a source-anchor assertion before GPU execution; corrected job27866 produced the tested source. All failed and slower candidates are archived. No correctness thresholds, source-stability guards, or baseline implementations were relaxed to obtain the result.

Evidence: `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt5_20261008/` contains raw paired samples, logs, official runner artifacts, confidence intervals, memory/dispatch evidence, candidate sources, reproduction scripts, and manifests. Scratch root: `/home/psk6950/miniworld-engine-scratch/trimul-b200style-opt5-20261008/`. Earlier dated reports remain unchanged.
