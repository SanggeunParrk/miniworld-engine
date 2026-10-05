# H100 FP32 master weights: qualified implementation (2026-10-05)

Installed and verified on one NVIDIA H100 80GB HBM3, SM90, 132 SM, using Slurm `h100` / `normal_h100`. All 56 measured configurations have BF16-mixed overhead below 2%. These measurements include 52 configurations of modified module families and four unchanged OPM/TokenDiT controls.

BF16-mixed means FP32 master parameters, BF16 activation/projection operands, and FP32 matrix gradients computed without a BF16 rounding intermediate. FP32 weights are freshly packed on each call and CUDA Graph replay. Original parameters remain saved for autograd version checks.

## Measurement method

The complete forward pass and gradients for every input and parameter are timed; optimizer updates are excluded. Both modes use the same state, including pinned normalization parameters. Timing uses CUDA Graph replay, 90 alternating paired samples, five replays per CUDA-event measurement, and medians. Each listed configuration independently satisfies `mixed_ms / bf16_ms < 1.02`; negative overhead means mixed is faster. Native CUDA defaults are retained. Portable Triton cold dispatch uses `autotune_miss_cap=1`, not exhaustive autotuning. Four empty profiler captures from job 21477 are repaired in profile-only job 21490; the original 90-sample latency results remain unchanged. Actual CUDA/Triton dispatch profiles, dtype inventories, unrounded matrix-gradient checks, and all raw timing samples are retained in [the measurement artifact](../../../benchmarks/modules/master_weights/results/h100/master_weights_20261005.json).

## Implementation

- Transition packs three master matrices and the squeeze transpose in one Triton launch. Native narrow reducers store FP32 matrix gradients directly; wide cuBLAS outputs retain FP32 precision.
- TriMul packs inside autograd and retains raw master parameters. Typed FP32 native reducers cover bidirectional D64/D128/D256/D384/D512 and incoming/outgoing D64/D128/D256/D384.
- TriangleAttention D128 retains the native fused QG attention path with a typed FP32 four-weight reducer. Wide TriangleAttention and APB use BF16 forward/input-gradient GEMMs and FP32 matrix-gradient GEMMs.
- PWA and LocalDiT batch casts. AdaLN and ConditionedTransition retain FP32 matrix gradients and bias reductions. LocalDiT explicitly preserves its FP32 attention arithmetic under BF16 autocast.
- SWADiT packs five block matrices together. Hoisted modulation and its backward remain FP32 through opaque compiler boundaries; this prevents AOTAutograd from rounding the modulation weight gradient to BF16.
- Bidirectional and single-direction TriMul D128 B7 producers drain final empty barriers before reusing ring-buffer storage. PWA glue and dGV backward use separate completion barriers for each double-buffer slot and drain both before exit. These synchronization changes were required by synccheck.

## Full forward + backward latency

| Workload / length / width | BF16 (ms) | BF16-mixed (ms) | Mixed overhead | BF16 / mixed |
|---|---:|---:|---:|---:|
| apb_L384_D384 | 0.523731 | 0.527718 | +0.761% | 0.9924x |
| apb_L768_D384 | 1.525526 | 1.524288 | -0.081% | 1.0008x |
| local_dit_L384_D128_A4 | 1.122778 | 1.128528 | +0.512% | 0.9949x |
| local_dit_L768_D128_A4 | 1.225536 | 1.235107 | +0.781% | 0.9923x |
| opm_L384_MSA64_pair128 | 0.970890 | 0.966458 | -0.456% | 1.0046x |
| opm_L768_MSA64_pair128 | 3.451776 | 3.439750 | -0.348% | 1.0035x |
| pwa_L384_MSA64_pair128 | 0.513046 | 0.516058 | +0.587% | 0.9942x |
| pwa_L768_MSA64_pair128 | 1.242128 | 1.246758 | +0.373% | 0.9963x |
| swa_dit_L384_D128_A4 | 0.173523 | 0.165770 | -4.468% | 1.0468x |
| swa_dit_L768_D128_A4 | 0.216374 | 0.203706 | -5.855% | 1.0622x |
| token_dit_L384_D768_A4 | 0.785530 | 0.740714 | -5.705% | 1.0605x |
| token_dit_L768_D768_A4 | 1.560064 | 1.510691 | -3.165% | 1.0327x |
| transition_L384_D128 | 0.560378 | 0.560154 | -0.040% | 1.0004x |
| transition_L384_D256 | 1.858458 | 1.871811 | +0.719% | 0.9929x |
| transition_L384_D384 | 3.927312 | 3.935059 | +0.197% | 0.9980x |
| transition_L384_D512 | 6.284467 | 6.337693 | +0.847% | 0.9916x |
| transition_L384_D64 | 0.237245 | 0.237933 | +0.290% | 0.9971x |
| transition_L768_D128 | 2.197024 | 2.199299 | +0.104% | 0.9990x |
| transition_L768_D256 | 7.309981 | 7.307927 | -0.028% | 1.0003x |
| transition_L768_D384 | 15.317519 | 15.386852 | +0.453% | 0.9955x |
| transition_L768_D512 | 24.457677 | 24.530576 | +0.298% | 0.9970x |
| transition_L768_D64 | 0.833619 | 0.834496 | +0.105% | 0.9989x |
| triangle_attention_L384_D128 | 1.246547 | 1.245194 | -0.109% | 1.0011x |
| triangle_attention_L384_D256 | 3.714278 | 3.742541 | +0.761% | 0.9924x |
| triangle_attention_L384_D384 | 6.030829 | 6.017418 | -0.222% | 1.0022x |
| triangle_attention_L384_D512 | 7.131072 | 7.126570 | -0.063% | 1.0006x |
| triangle_attention_L768_D128 | 6.812534 | 6.809245 | -0.048% | 1.0005x |
| triangle_attention_L768_D256 | 20.313815 | 20.351674 | +0.186% | 0.9981x |
| triangle_attention_L768_D384 | 31.255220 | 31.304381 | +0.157% | 0.9984x |
| triangle_attention_L768_D512 | 35.054988 | 35.000752 | -0.155% | 1.0015x |
| trimul_bi_L384_D128 | 1.031434 | 1.039574 | +0.789% | 0.9922x |
| trimul_bi_L384_D256 | 2.692323 | 2.716070 | +0.882% | 0.9913x |
| trimul_bi_L384_D384 | 4.726563 | 4.759142 | +0.689% | 0.9932x |
| trimul_bi_L384_D512 | 7.206035 | 7.199760 | -0.087% | 1.0009x |
| trimul_bi_L384_D64 | 0.421174 | 0.422678 | +0.357% | 0.9964x |
| trimul_bi_L768_D128 | 4.172621 | 4.186419 | +0.331% | 0.9967x |
| trimul_bi_L768_D256 | 11.306054 | 11.466698 | +1.421% | 0.9860x |
| trimul_bi_L768_D384 | 19.961984 | 19.965220 | +0.016% | 0.9998x |
| trimul_bi_L768_D512 | 29.790633 | 29.779027 | -0.039% | 1.0004x |
| trimul_bi_L768_D64 | 1.649648 | 1.652160 | +0.152% | 0.9985x |
| trimul_incoming_L384_D128 | 0.782867 | 0.785187 | +0.296% | 0.9970x |
| trimul_incoming_L384_D256 | 1.761901 | 1.778160 | +0.923% | 0.9909x |
| trimul_incoming_L384_D384 | 2.888179 | 2.894829 | +0.230% | 0.9977x |
| trimul_incoming_L384_D64 | 0.275248 | 0.276544 | +0.471% | 0.9953x |
| trimul_incoming_L768_D128 | 2.881258 | 2.883862 | +0.090% | 0.9991x |
| trimul_incoming_L768_D256 | 6.924150 | 6.943290 | +0.276% | 0.9972x |
| trimul_incoming_L768_D384 | 11.883846 | 11.893661 | +0.083% | 0.9992x |
| trimul_incoming_L768_D64 | 1.005216 | 1.006822 | +0.160% | 0.9984x |
| trimul_outgoing_L384_D128 | 0.782320 | 0.786070 | +0.479% | 0.9952x |
| trimul_outgoing_L384_D256 | 1.764307 | 1.779914 | +0.885% | 0.9912x |
| trimul_outgoing_L384_D384 | 2.892410 | 2.900582 | +0.283% | 0.9972x |
| trimul_outgoing_L384_D64 | 0.273834 | 0.275053 | +0.445% | 0.9956x |
| trimul_outgoing_L768_D128 | 2.901862 | 2.902442 | +0.020% | 0.9998x |
| trimul_outgoing_L768_D256 | 6.895312 | 6.913923 | +0.270% | 0.9973x |
| trimul_outgoing_L768_D384 | 11.892730 | 11.913834 | +0.177% | 0.9982x |
| trimul_outgoing_L768_D64 | 1.010099 | 1.011293 | +0.118% | 0.9988x |

## Correctness and sanitizer evidence

- Installed-root job **21477** verifies source/binary hashes and the actual root import path, then passes all **37** new precision/compile/graph tests before timing. Its 35-minute allocation expires during the unchanged OPM build after 24 completed timing rows; job **21488** resumes only the remaining measurements. These cover eager training/inference, `torch.compile(fullgraph=True)`, changed inputs/weights/dy/masks during live CUDA Graph replay, BF16 and FP32 gradient dtypes, nonzero unrounded matrix gradients, parameter version errors, and TriMul mask/dropout fixtures.
- Independent FP64 GEMMs of saved BF16 operands and FP64 reductions check relevant FP32 gradients. This is an operand-level oracle, not a full-model FP64 reference. Compiler-sensitive normalization gradients are checked against the measured BF16 compiler discrepancy; existing test tolerances were not changed.
- Candidate job **21472** passes all 37 new tests. Final PWA source changes are additionally checked by job **21475**. The broader existing GPU regression run **21453** reports **82 passed, 4 skipped, 2 failed**. Those two PWA dMSA gradient failures are preexisting: frozen baseline job **21462** reproduces exactly `0.0052400678396224976` and `0.005299858748912811` relative error against the original `0.005` limit. They are not counted as passing.
- **memcheck:** all 52 modified configurations, full L384/L768 F+B, job **21451**; final changed SingleD128 and PWA sources rechecked by **21472/21475**, errors zero.
- **racecheck:** 26 focused configurations (Transition/TriMul/TriangleAttention widths at most 128, plus APB/LocalDiT/PWA/SWADiT), job **21453**; final changed SingleD128 and PWA sources rechecked by **21472/21475**, errors zero. Wide global FP32 store variants do not have full racecheck coverage.
- **synccheck:** all 52 modified configuration variants, job **21453**, supplemented by **21475** for all single-direction widths and final PWA. Transition uses 1024 rows for racecheck/synccheck; memcheck used the full measured row counts. Final applicable checks report zero errors.

Qualification is restricted to these L384/L768 shapes and test fixtures. It does not establish arbitrary batches, heads, widths, lengths, dropout performance, optimizer performance, or SOL90. OPM and TokenDiT are unchanged controls; they were remeasured but their preexisting sanitizer behavior is outside this patch.

## Rebuilding the H100 native weight reducer

The locally installed `triattn_wgrad_shared_z_master.so` is ignored by Git. Its hash and qualified source hashes are recorded in the measurement artifact and tracked native manifests. On a Slurm allocation with the project environment and CUTLASS 4.2:

```bash
CUTLASS_PATH=/path/to/cutlass-4.2 python -m miniworld_engine.kernels.triangle_attention.cuda.build_wgrad
```

Do not compile CUDA extensions on the login node. Existing native ABI binaries are retained. Raw experiment logs are in `experiments/h100_master_20261005`; job IDs above identify the qualification chain.
