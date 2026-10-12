# TriMul H100 inference with matched FP32 masters — 2026-10-12

The installed opt12 candidate wins **80/80 supported Anthropic inference comparisons** with identical FP32 master parameters. Speedup ranges from **1.014943x to 1.265719x**. All **120 native cells** pass the independent FP32 output check and execute CUDA/cuBLAS without Triton kernels. The remaining **40 Anthropic cells are unsupported**, not wins. This is a fresh parameter-matched inference audit; it does not introduce another installed optimization or establish SOL90.

## Workload and comparison

Batch 1, one layer, BF16 activations, FP32 parameters including all projection weights, hidden width equal to pair width per direction, dropout 0, mask probability 0.1. Widths are D64/128/256/384/512; lengths are L128/256/384/512/640/768/896/1024; directions are outgoing, incoming and bidirectional. No claim is made for other batches, hidden widths, dtypes or arbitrary lengths.

The unchanged official `benchmarks/runners/bench.py` creates, validates and measures both complete module graphs. The observer retains those graphs, asserts exact equality of the input, all ten parameter tensors and token mask, warms ten pairs, then records 90 alternating AB/BA CUDA-event pairs on one H100. Ratios are vendor median divided by native median. Native uses actual `torch.compile` and manual CUDA Graphs. The upstream native Anthropic wrapper uses `compile=False` and manual CUDA Graphs; this is explicitly a different wrapper compilation setting, as in the earlier inference comparison. Compilation and initial packing are outside steady-state timing.

The Anthropic comparator is the unmodified-source CUDA 12.9 rebuild, payload version v5 1.2.2, with no overlay. The benchmark records its manifest provenance. Unidirectional calls use upstream `trimul_native.face.serve`. Bidirectional D64/D128 use the engine's established composition of upstream K1, two cuBLAS contractions and upstream K3; upstream has no bidirectional face entry. No repaired vendor kernel or relaxed accuracy threshold is used.

The previous [opt6 inference result](trimul_b200style_h100_opt6_20261008.md) used BF16 projection weights. These measurements provide the missing FP32-master inference evidence after the [master-parameter harness correction](trimul_h100_master_matched_20261012.md). Absolute times across those jobs are not attributed to a new kernel change.

## Results

| D | Supported wins | Geometric mean vs Anthropic | Range |
|---:|---:|---:|---:|
| 64 | 24/24 | 1.125816x | 1.077291–1.238390x |
| 128 | 24/24 | 1.127782x | 1.093931–1.158208x |
| 256 | 16/16 | 1.037969x | 1.014943–1.064416x |
| 384 | 16/16 | 1.231996x | 1.186626–1.265719x |

Anthropic has no matching payload unit for D512 in any direction or for D256/D384 bidirectional. These 40 rows retain `unsupported` status and the original reason. Native-only timings and checks are retained separately. The weakest supported margin is D256 outgoing L128, 0.054608 ms native versus 0.055424 ms Anthropic, or 1.014943x. This is a within-job paired estimate, not an independent-job confidence interval or a universal guarantee.

Across all 120 native cells, maximum output relative Frobenius error is **0.0036836397**. Across 80 supported Anthropic cells it is **0.0036830807**. Both satisfy the unchanged **0.006** limit. The official runner checks finite outputs and graph replay. The observer profiles both each native graph replay and a separate compiled-function invocation; every trace contains zero Triton kernels. Reference-side kernels are outside these native dispatch profiles.

This audit performs no new training or sanitizer qualification because runtime sources are unchanged. Installed correctness, changed-input graph and sanitizer evidence remain in [opt12](trimul_b200style_h100_opt12_20261012.md) and its linked records. SOL90 remains open; this job does not collect NCU counters.

## Source and evidence

Job **28321** supplies two pilot pairs; job **28322** supplies the complete 120-cell audit and all 80 comparison ratios above. Pilot rows are excluded from aggregate counts. Before and after measurement, all **168** guarded live TriMul/runtime/test/runner files match the installed snapshot. Runner SHA256 is `4e7eff89d6501adb07dbb5208c5356f47f408a386375bd7c69780e2d8b336a36`. The explicit B200-style/native flags are unchanged and the default B200-style flag remains zero.

Evidence is in `runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/inference_master_matched_20261012/`: raw samples, dispatch traces, official CSV/config/provenance artifacts, the frozen source snapshot, pristine payload, observer scripts, logs and SHA256 manifests. The archive holds **3,704 hashed files**, including **732 official artifacts**. The initial archive attempt failed because the staged snapshot has no README; the corrected archive preserves that log and omits the nonexistent documentation file. Measurement data and runtime source were not changed by the packaging correction.

The next K3 scheduling experiment runs in a separate snapshot and is not part of this installed-kernel result.
