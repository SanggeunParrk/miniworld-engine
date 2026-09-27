# Native CUDA normalization: 2026-09-26 H100 qualification

## Status

Experimental APIs implemented for LayerNorm and RMSNorm forward/backward, plus a CUDA-normalization + cuBLAS LayerNormLinear composition. No production dispatch/default was replaced. This is not a fully fused native LN/GEMM implementation.

Input scope: FP16/BF16/FP32/FP64, arbitrary nonempty last-axis width, arbitrary leading dimensions, mixed affine dtypes, optional affine parameters, strided and unaligned inputs via guarded vector/scalar paths. FP64 accumulates in FP64; lower types in FP32. CUDA higher-order gradients are explicitly unsupported.

## Validation

- 19072_0 / 19072_1: 216 matrix checks each (432 executions; projection cases overlap between shards). Widths 1,3,31,32,33,63,64,127,128,256,384,512,768,1024,4096,8193. Three layouts, four input dtypes, affine/epsilon/empty cases and graph replay.
- 19078: 28 qualification cases including FP64 gradcheck, fullgraph compile and changed-input/weight replay, constant/low-variance/large-offset inputs; 60 maintained GPU regression tests passed.
- 19078: memcheck / racecheck / synccheck all zero errors; racecheck zero warnings/hazards on the representative sanitizer matrix.
- 19091: all 16 selected performance configurations passed full input/affine-gradient comparisons at benchmark sizes.
- 19097: NCU profiled the actual final forward_vec, backward_fused and affine_finish kernels.
- All these Slurm jobs completed with exit code 0. Tests establish these sampled contracts on H100, not every possible shape or GPU architecture.

## Normalization performance (ms)

PyTorch 2.10.0+cu128, H100, FP32 affine parameters; warmed CUDA-graph replay. Training means forward plus all first-order gradients, excluding optimizer. Median of seven measurements, ten invocations per replay. Contiguous inputs. The PyTorch column is the repository-style FP32-accumulation formula (including low-precision casts), not a claim against every native/compiled PyTorch implementation.

Triton uses the current default miss cap of 24 configs; cache-miss warnings remain. Neither the Triton full config space nor the native config space is fully tuned. Engine is the actual layernorm_kernel dispatch (RMS uses its current Triton entry). Native is the best training config from the explicit threads x rows sweep; its forward is not independently selected. Earlier four-config screening is not the final baseline.

| Op | M | D | Input | PyTorch F+B | Triton F+B | Engine F+B | CUDA F+B | Engine/CUDA | Engine fwd | CUDA fwd | Native config |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| LayerNorm | 8192 | 64 | bfloat16 | 0.0534 | 0.0119 | 0.0126 | 0.0115 | 1.09x | 0.0026 | 0.0036 | cuda_t128_r4 |
| RMSNorm | 8192 | 64 | bfloat16 | 0.0616 | 0.0060 | 0.0065 | 0.0104 | 0.63x | 0.0022 | 0.0033 | cuda_t128_r4 |
| LayerNorm | 147456 | 128 | bfloat16 | 0.6134 | 0.0832 | 0.0831 | 0.1130 | 0.74x | 0.0293 | 0.0439 | cuda_t128_r64 |
| RMSNorm | 147456 | 128 | bfloat16 | 1.2407 | 0.0760 | 0.0761 | 0.1039 | 0.73x | 0.0279 | 0.0395 | cuda_t128_r64 |
| LayerNorm | 147456 | 384 | bfloat16 | 1.2621 | 0.2464 | 0.2748 | 0.2531 | 1.09x | 0.0793 | 0.0860 | cuda_t128_r64 |
| RMSNorm | 147456 | 384 | bfloat16 | 3.1028 | 0.2543 | 0.2560 | 0.2398 | 1.07x | 0.0932 | 0.0844 | cuda_t128_r64 |
| LayerNorm | 8192 | 1024 | bfloat16 | 0.2020 | 0.0448 | 0.0721 | 0.0844 | 0.85x | 0.0100 | 0.0131 | cuda_t128_r16 |
| RMSNorm | 8192 | 1024 | bfloat16 | 0.4964 | 0.0524 | 0.0523 | 0.0651 | 0.80x | 0.0100 | 0.0094 | cuda_t256_r4 |
| LayerNorm | 8192 | 64 | float32 | 0.0434 | 0.0131 | 0.0130 | 0.0117 | 1.11x | 0.0032 | 0.0038 | cuda_t256_r4 |
| RMSNorm | 8192 | 64 | float32 | 0.0517 | 0.0062 | 0.0067 | 0.0105 | 0.64x | 0.0024 | 0.0034 | cuda_t128_r4 |
| LayerNorm | 147456 | 128 | float32 | 0.4203 | 0.1706 | 0.1711 | 0.1562 | 1.10x | 0.0539 | 0.0605 | cuda_t256_r64 |
| RMSNorm | 147456 | 128 | float32 | 1.0090 | 0.1371 | 0.1369 | 0.1530 | 0.89x | 0.0530 | 0.0605 | cuda_t128_r64 |
| LayerNorm | 147456 | 384 | float32 | 0.6929 | 0.3899 | 0.3974 | 0.4446 | 0.89x | 0.1551 | 0.1795 | cuda_t128_r64 |
| RMSNorm | 147456 | 384 | float32 | 2.5285 | 0.5453 | 0.5491 | 0.4142 | 1.33x | 0.2051 | 0.1630 | cuda_t128_r64 |
| LayerNorm | 8192 | 1024 | float32 | 0.1136 | 0.0779 | 0.1154 | 0.1039 | 1.11x | 0.0274 | 0.0285 | cuda_t128_r16 |
| RMSNorm | 8192 | 1024 | float32 | 0.4125 | 0.0942 | 0.0947 | 0.0880 | 1.08x | 0.0341 | 0.0259 | cuda_t128_r4 |

## LayerNormLinear screening (ms)

This compares two compositions (norm + PyTorch/cuBLAS linear), not the existing fused CuTe/Triton LNLinear kernel. The old fused training helper rejected the benchmark 2-D input through its rows_of shape contract; do not turn this composition result into a claimed fused-module speedup. Triton composition uses four-config miss screening here; these results are provisional. Native uses threads=128, rows=4, not a complete per-shape tuning search.

| M | D in | D out | Input | PyTorch F+B | Triton composition F+B | CUDA composition F+B |
|---:|---:|---:|---|---:|---:|---:|
| 8192 | 64 | 64 | bfloat16 | 0.0780 | 0.0424 | 0.0358 |
| 8192 | 64 | 64 | float32 | 0.0777 | 0.0462 | 0.0453 |
| 147456 | 128 | 16 | bfloat16 | 0.6980 | 0.1636 | 0.2583 |
| 147456 | 128 | 16 | float32 | 0.5850 | 0.3089 | 0.3866 |
| 8192 | 384 | 512 | bfloat16 | 0.1449 | 0.0830 | 0.0796 |
| 8192 | 384 | 512 | float32 | 0.3165 | 0.2745 | 0.2894 |
| 1024 | 1024 | 256 | bfloat16 | 0.0505 | 0.0467 | 0.0430 |
| 1024 | 1024 | 256 | float32 | 0.0945 | 0.0963 | 0.1024 |

## Optimization and remaining work

- Reused values in registers; fused dX and partial affine gradients for D<=1024; deterministic final reduction without atomics. Larger widths retain a bounded-register split path.
- Dispersed row traversal and a block per affine output column improved final reduction utilization. Partial workspace is capped at 32 MiB.
- Packed vector forward uses pointer/width alignment guards; scalar fallback handles tails and offsets. NCU at M=147456,D128 BF16 measured scalar v3 forward 73.15 us vs vector v4 50.40 us (profiling runs, not the graph benchmark); final forward DRAM 35.69%, SM 56.04%. Backward remains well below a demonstrated hardware limit.
- Important regressions remain: BF16 D128 training, small RMSNorm, and some wide-dtype cases. No universal CUDA speedup or automatic promotion is claimed.
- Next work: vectorized mixed-dtype backward and small-row parameter-gradient reduction; separate forward/training per-shape tuning; a genuine fused LNLinear implementation and comparison against the actual fused baseline; further architectures and out-of-sample shape tests.

Artifacts: parent MiniWorld runs/norm_cuda_20260926; primary fair-bench.json, selected-config-correctness.json, qualification.json, correctness-*.json, linear-bench.json, and Slurm/sanitizer logs. Intermediate v1/v2/v3 files are development snapshots, not final results.
