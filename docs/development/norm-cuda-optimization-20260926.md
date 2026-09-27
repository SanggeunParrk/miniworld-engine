# Native CUDA norm optimization — 2026-09-26

## Status and comparison scope

LayerNorm, RMSNorm, and LayerNormLinear have native CUDA candidates with automatic
schedule selection inside the explicit `norm_cuda` APIs. Existing engine/module
production dispatch is unchanged. This report supersedes the first native norm
candidate report for performance selection.

**Old CUDA below means the pre-optimization native candidate (v4), not engine
1.0.0 or the whole MiniWorld model.** Its launch configuration was re-tuned for
this comparison. `Engine` means the current installed-source dispatch called by
`layernorm_kernel` / `triton_rmsnorm`, with FP32 affine parameters. Those calls
may use different CUDA/Triton paths by dtype.

H100, CUDA graph replay, median of seven event measurements, ten complete calls
per replay. Training means forward plus all input/affine parameter gradients.
Tables use **milliseconds**. Contiguous inputs are used for timing; copies for
noncontiguous inputs are validated but these tables do not price those copies.
Triton baselines use the normal 24-candidate cache-miss search, **not a fully built
autotune cache**. PyTorch FP32-accumulation formula timings remain in JSON and are
not presented as native PyTorch module latency.

## Implementation

1. Explicit float reciprocal square root and packed loads/stores. Common widths
   specialize the exact register tile; D384/D768 no longer reserve padded columns.
2. D64/D128 use eight threads per row. Affine gradients are combined inside each
   CTA, reducing the partial buffer and the final reduction traffic.
3. D256/D384/D512 reuse CTA partial reduction across row warps when shared memory
   fits. Float64 D512 defaults to 128 threads to respect that footprint.
4. D1024..16384 use CTA-wide row reductions, avoiding large per-warp register
   arrays. Larger/unaligned/irregular cases retain the generic CUDA fallback.
5. LayerNormLinear has a native LN/WMMA forward with packed input/saved-activation
   loads, padded shared memory, shared weight tiles, and output-width-dependent
   warp assignment. Its backward keeps the rounded normalized activation,
   cuBLAS dX/dW, and the optimized native norm backward.
6. Auto LNLinear fusion is restricted to the measured low-precision
   M147456/K128/N16 cell. Other cells use native norm plus cuBLAS. Explicit fused
   calls remain available for experiments; unsupported shapes/dtypes compose.

Float16/BFloat16/Float32 statistics and affine gradients accumulate in Float32;
Float64 accumulates in Float64. Centered variance is preserved. RMS epsilon=None
uses input dtype epsilon. No global floating atomics are used in selected norm
paths. Partial gradient workspace is capped at 32 MiB. First-order gradients,
empty leading dimensions, mixed affine dtypes, and last-axis normalization are
supported. Higher-order CUDA gradients remain unsupported.

## Norm forward + backward

### bfloat16

| Op | M | D | Old CUDA ms | New CUDA ms | Speedup | Engine ms | New / engine speedup | Schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LayerNorm | 8192 | 64 | 0.01156 | 0.00812 | 1.42x | 0.01261 | 1.55x | warp, t256, r1 |
| RMSNorm | 8192 | 64 | 0.01045 | 0.00780 | 1.34x | 0.00605 | 0.78x | warp, t256, r1 |
| LayerNorm | 147456 | 128 | 0.11336 | 0.07989 | 1.42x | 0.08284 | 1.04x | warp, t128, r16 |
| RMSNorm | 147456 | 128 | 0.10427 | 0.08377 | 1.24x | 0.07574 | 0.90x | warp, t256, r24 |
| LayerNorm | 147456 | 384 | 0.25372 | 0.22984 | 1.10x | 0.27542 | 1.20x | warp, t256, r16 |
| RMSNorm | 147456 | 384 | 0.23985 | 0.21057 | 1.14x | 0.25375 | 1.21x | warp, t256, r16 |
| LayerNorm | 8192 | 1024 | 0.08538 | 0.06223 | 1.37x | 0.07200 | 1.16x | cta, t128, r16 |
| RMSNorm | 8192 | 1024 | 0.06483 | 0.05702 | 1.14x | 0.05250 | 0.92x | cta, t128, r16 |
| LayerNorm | 2048 | 4096 | 0.16590 | 0.06633 | 2.50x | n/a | n/a | cta, t256, r4 |
| RMSNorm | 2048 | 4096 | 0.14868 | 0.06064 | 2.45x | n/a | n/a | cta, t256, r4 |
| LayerNorm | 257 | 8193 | 0.12569 | 0.04576 | 2.75x | n/a | n/a | cta, t256, r1 |
| RMSNorm | 257 | 8193 | 0.11852 | 0.03477 | 3.41x | n/a | n/a | cta, t256, r1 |

### float16

| Op | M | D | Old CUDA ms | New CUDA ms | Speedup | Engine ms | New / engine speedup | Schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LayerNorm | 8192 | 64 | 0.01156 | 0.00810 | 1.43x | 0.01175 | 1.45x | warp, t256, r1 |
| RMSNorm | 8192 | 64 | 0.01038 | 0.00777 | 1.34x | 0.00648 | 0.83x | warp, t256, r1 |
| LayerNorm | 147456 | 128 | 0.11282 | 0.09752 | 1.16x | 0.13305 | 1.36x | warp, t128, r16 |
| RMSNorm | 147456 | 128 | 0.10346 | 0.08354 | 1.24x | 0.07522 | 0.90x | warp, t256, r24 |
| LayerNorm | 147456 | 384 | 0.25269 | 0.22956 | 1.10x | 0.28889 | 1.26x | warp, t256, r16 |
| RMSNorm | 147456 | 384 | 0.24092 | 0.21012 | 1.15x | 0.25002 | 1.19x | warp, t256, r16 |
| LayerNorm | 8192 | 1024 | 0.08352 | 0.06215 | 1.34x | 0.07006 | 1.13x | cta, t128, r16 |
| RMSNorm | 8192 | 1024 | 0.06489 | 0.05660 | 1.15x | 0.05216 | 0.92x | cta, t128, r16 |
| LayerNorm | 2048 | 4096 | 0.16564 | 0.06570 | 2.52x | n/a | n/a | cta, t256, r4 |
| RMSNorm | 2048 | 4096 | 0.14829 | 0.06045 | 2.45x | n/a | n/a | cta, t256, r4 |
| LayerNorm | 257 | 8193 | 0.13112 | 0.04412 | 2.97x | n/a | n/a | cta, t256, r1 |
| RMSNorm | 257 | 8193 | 0.12399 | 0.03582 | 3.46x | n/a | n/a | cta, t256, r1 |

### float32

| Op | M | D | Old CUDA ms | New CUDA ms | Speedup | Engine ms | New / engine speedup | Schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LayerNorm | 8192 | 64 | 0.01164 | 0.00848 | 1.37x | 0.01233 | 1.45x | warp, t256, r1 |
| RMSNorm | 8192 | 64 | 0.01047 | 0.00809 | 1.29x | 0.00678 | 0.84x | warp, t256, r1 |
| LayerNorm | 147456 | 128 | 0.15594 | 0.14204 | 1.10x | 0.16759 | 1.18x | warp, t256, r4 |
| RMSNorm | 147456 | 128 | 0.15254 | 0.14118 | 1.08x | 0.13722 | 0.97x | warp, t256, r4 |
| LayerNorm | 147456 | 384 | 0.44453 | 0.39543 | 1.12x | 0.39284 | 0.99x | warp, t256, r16 |
| RMSNorm | 147456 | 384 | 0.41504 | 0.38695 | 1.07x | 0.55219 | 1.43x | warp, t256, r64 |
| LayerNorm | 8192 | 1024 | 0.10455 | 0.07864 | 1.33x | 0.11724 | 1.49x | cta, t128, r16 |
| RMSNorm | 8192 | 1024 | 0.08854 | 0.07765 | 1.14x | 0.09386 | 1.21x | cta, t128, r16 |
| LayerNorm | 2048 | 4096 | 0.20337 | 0.08878 | 2.29x | n/a | n/a | cta, t256, r4 |
| RMSNorm | 2048 | 4096 | 0.23210 | 0.08604 | 2.70x | n/a | n/a | cta, t256, r4 |
| LayerNorm | 257 | 8193 | 0.17992 | 0.04949 | 3.64x | n/a | n/a | cta, t256, r1 |
| RMSNorm | 257 | 8193 | 0.17992 | 0.04126 | 4.36x | n/a | n/a | cta, t256, r1 |

### float64

| Op | M | D | Old CUDA ms | New CUDA ms | Speedup | Engine ms | New / engine speedup | Schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| LayerNorm | 8192 | 64 | 0.01382 | 0.01019 | 1.36x | n/a | n/a | warp, t256, r1 |
| RMSNorm | 8192 | 64 | 0.01209 | 0.00973 | 1.24x | n/a | n/a | warp, t256, r1 |
| LayerNorm | 147456 | 128 | 0.28078 | 0.26661 | 1.05x | n/a | n/a | warp, t256, r4 |
| RMSNorm | 147456 | 128 | 0.27982 | 0.26515 | 1.06x | n/a | n/a | warp, t256, r4 |
| LayerNorm | 147456 | 384 | 0.83908 | 0.84431 | 0.99x | n/a | n/a | warp, t256, r16 |
| RMSNorm | 147456 | 384 | 0.82977 | 0.84343 | 0.98x | n/a | n/a | warp, t256, r64 |
| LayerNorm | 8192 | 1024 | 0.20868 | 0.13022 | 1.60x | n/a | n/a | cta, t128, r16 |
| RMSNorm | 8192 | 1024 | 0.18878 | 0.12743 | 1.48x | n/a | n/a | cta, t128, r16 |
| LayerNorm | 2048 | 4096 | 0.32135 | 0.18774 | 1.71x | n/a | n/a | cta, t256, r4 |
| RMSNorm | 2048 | 4096 | 0.30177 | 0.18163 | 1.66x | n/a | n/a | cta, t256, r4 |
| LayerNorm | 257 | 8193 | 0.33692 | 0.24291 | 1.39x | n/a | n/a | cta, t256, r1 |
| RMSNorm | 257 | 8193 | 0.33512 | 0.23570 | 1.42x | n/a | n/a | cta, t256, r1 |

The engine autotune shape key rejects widths >=4096. Engine Float64 timing is
omitted because its accumulation contract differs. A speedup below 1.0 means
that the new native candidate still loses that comparison. Small RMSNorm remains
launch/reduction sensitive, and several cells still favor the engine.

## LayerNormLinear

Old CUDA is independently re-tuned for the full composed workload. New uses the
public auto API. K=input width, N=projection output width.

| dtype | M | K→N | Old inference ms | New inference ms | Inference speedup | Old training ms | New training ms | Training speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| bfloat16 | 8192 | 64→64 | 0.00668 | 0.00603 | 1.11x | 0.03565 | 0.03227 | 1.11x |
| float32 | 8192 | 64→64 | 0.00883 | 0.00813 | 1.08x | 0.04484 | 0.04138 | 1.08x |
| bfloat16 | 147456 | 128→16 | 0.06090 | 0.03127 | 1.95x | 0.19440 | 0.15728 | 1.24x |
| float32 | 147456 | 128→16 | 0.10951 | 0.10330 | 1.06x | 0.31472 | 0.30154 | 1.04x |
| bfloat16 | 8192 | 384→512 | 0.01410 | 0.01383 | 1.02x | 0.08016 | 0.06898 | 1.16x |
| float32 | 8192 | 384→512 | 0.08185 | 0.08103 | 1.01x | 0.28713 | 0.27517 | 1.04x |
| bfloat16 | 1024 | 1024→256 | 0.00926 | 0.00917 | 1.01x | 0.04306 | 0.03839 | 1.12x |
| float32 | 1024 | 1024→256 | 0.03212 | 0.03135 | 1.02x | 0.10160 | 0.09449 | 1.08x |

The native fused prototype is not universally faster than fused Triton. For the
BF16 M147456/K128/N16 inference cell, the new native path is approximately
0.031 ms; fused Triton is approximately 0.022 ms. Broad CUDA fusion promotion is
therefore not justified. Wider projections generally retain composition because
native WMMA fusion is slower. `linear-v1.json` through `linear-v5.json` preserve
all five iterations, including losses, with actual fused Triton/CuTe forward
measurements. Those fused-forward columns do not include backward or training
activation saves and must not be treated as full training comparisons.

## Validation and issues caught

- Job 19231: 432 shape/dtype/layout executions (216 per shard; some linear cases
  repeated across shards), output and all gradients; changed-input graph replay.
- Job 19269: 28 stress/FP64 gradcheck/compile checks and 93 managed GPU tests pass (final source).
- Jobs 19230/19255: 48 large-cell gradient checks against explicit accumulation
  math, then old/new paired timing. Job 19242: eight full LNLinear gradient and
  timing cells.
- A partial-row-group deadlock was found during broad validation. The eight-lane
  shuffle now uses the exact participating subgroup mask. Failed jobs 19198/19199
  were stopped; the corrected broad matrix passes. Frozen pre-fix candidates are
  research artifacts, not dispatch choices.
- Initial sanitizer runs reported a CUDA API invalid-handle at late module
  loading (`cuKernelGetFunction`), while numerical tests passed. Sanitizer
  initialization now preloads all extensions before autograd/graphs and uses
  `CUDA_MODULE_LOADING=EAGER`; no API-error suppression is used.
- Atomic affine accumulation was slower and rejected. Coalescing final reduction
  without enough CTAs was slower for narrow widths and rejected. Subgroup rows
  alone were insufficient; CTA-local affine reduction was necessary.

Sanitizer and final NCU completion details appear below.
No full-model training promotion, release commit, or remote push is included.

## Artifacts

Workspace: `/home/psk6950/MiniWorld/runs/norm_cuda_20260926/opt`.
`results.json` combines the final tables; `manifest.json` records source hashes
and job IDs. `baseline.cu` freezes the pre-optimization reference. Every rejected
experiment retains its own source/JSON/log. Full baseline formula timings,
error norms, old selected configurations, and individual forward timings remain
in JSON for audit.

## Completed sanitizer and NCU gates

Job 19252 completed with exit code 0. Norm and fused LNLinear each pass memcheck,
racecheck, and synccheck: **0 errors / 0 hazards / 0 warnings** in sanitizer
summaries. The earlier API-only invalid-handle report disappears with extension
preloading and eager CUDA module loading; no error-report suppression was used.

Final NCU jobs 19263_0..2 all completed with exit code 0. These are profiler
measurements, not CUDA-graph benchmark timings. DRAM throughput percentages are
not an application-level SOL claim.

| Workload | Kernel | Profile time ms | DRAM % | L2 % | Registers/thread |
|---|---|---:|---:|---:|---:|
| LN M147456/D128 | forward_vec | 0.02797 | 62.51 | 71.88 | 48 |
| LN M147456/D128 | backward_fused | 0.04672 | 66.70 | 71.86 | 96 |
| LN M147456/D128 | affine_finish | 0.00531 | 3.34 | 23.13 | 18 |
| RMS M147456/D384 | forward_vec | 0.07818 | 79.67 | 79.73 | 39 |
| RMS M147456/D384 | backward_fused | 0.12502 | 78.99 | 77.21 | 80 |
| RMS M147456/D384 | affine_finish | 0.00733 | 7.30 | 40.96 | 18 |
| LN M2048/D4096 | wide_forward | 0.02451 | 22.81 | 40.22 | 62 |
| LN M2048/D4096 | wide_backward | 0.03565 | 40.85 | 55.65 | 97 |
| LN M2048/D4096 | finish | 0.01203 | 41.85 | 45.77 | 25 |

The D128 affine finishing kernel fell from 0.00947 ms in the intermediate
warp-partial design (job 19160) to 0.00531 ms after CTA-local reduction.
Its main backward reached 66.70% of profiled sustained DRAM throughput; this
work does not establish a SOL90 result.
