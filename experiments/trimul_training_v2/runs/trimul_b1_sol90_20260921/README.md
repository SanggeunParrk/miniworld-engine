# B1–B4 SoL optimization: node01, 2026-09-21

User authorized node01 after node02 was fully occupied. Existing training jobs were left running. Each length uses one exclusively allocated H100; paired baseline/candidate measurements run on the same GPU. Baseline is `trimul_b1_tri_opt_20260921`, not Anthropic inference or cuEquivariance.

## Invariants

L384/L768, BF16 C128 / combined H256, bidirectional training, dropout 25%, masks, residual, all 11 gradients. Keep input affine BF16 x_n, original BF16 tri, FP32 output mean/rstd. Do not save output normalized activations. B7 and cuBLAS are unchanged. No current candidate establishes SoL90 or fixes the inherited independent-reference B7 L768 dWL tolerance failure.

## Experiments

1. `derive.py`, `tune.py`: split on-chip dNorm into dead dy/dGate and current input x_n storage; prefetch next 48 KiB input before output-LN derivative arithmetic. Double-buffer output mean/rstd (+512 bytes shared memory). 16 implementation configurations per length.
2. `derive_affine.py`, `tune_affine.py`: distribute output-LN affine reconstruction across both warp-groups, rather than only group 1. 16 combinations per length, including unroll and paired dWproj WGMMA.
3. `derive_store.py`, `tune_store.py`: dTri TMA box width 16/32/64/128 channels, reducing 16 store commands to 8/4/2. Same output layout and arithmetic.
4. `tune_count.py`: CTA counts 132/128/120/96. 132 remained faster. Counts changing the reduction order are checked with relative L2 <= 5e-4; dGate/dTri must remain bit-exact. Count128 requires a longer cached dropout-mask period and more registers; divisibility alone did not improve runtime.
5. `derive_pair.py`, `tune_pair.py`: issue both dNorm WGMMA tiles as one group before consuming either. Same arithmetic; no spills in measured builds.
6. `derive_dnreg.py`, `tune_dnreg.py`: experimental register-resident BF16 dNorm fragments. Initial template-index compile error preserved in `dnreg-compile-error-L*.json`; fixed using explicit constexpr values.

All tuning ratios use the same preceding selected baseline, not ratios multiplied across independent timings. A final full-module benchmark and sanitizer run are required before a new development selection.

## SoL definition and limitations

Do not equate NCU's DRAM utilization with full-kernel SoL. Use separate quantities:

- Optimistic algorithmic streaming roofline: `max(unique_tensor_bytes / bandwidth, dense_GEMM_FLOPs / tensor_peak)`. Large activations exceed L2; unique input/output payload is approximately `1800 * L^2` bytes (BF16 x_n, tri, dy, dGate, dTri, FP32 output mean/rstd). Small parameters, dropout scales, partial gradients and scalar math are omitted, so this is an optimistic model, not an exact attainable lower bound.
- Fixed-schedule traffic roofline: use measured DRAM read+write bytes. This includes rereading x_n/dGate in the separate dWgate phase and partial-gradient traffic. It rewards redundant traffic if mistaken for an algorithmic efficiency metric; it must be labeled separately.
- Dense GEMM work is `262144 * L^2` FLOPs: gate recomputation, projection recomputation, dNorm, dWproj and dWgate. Pointwise math, shared-memory traffic, instruction issue limits and dependencies are not included in this simplified bound.

[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) list SXM bandwidth 3.35 TB/s and BF16 1979 TFLOP/s **with sparsity**, hence 989.5 TFLOP/s dense. `calibration-node01.json` independently measures a saturated >L2 two-read/one-write streaming workload: 3.105 TB/s. This empirical reference is not a hard hardware upper bound. Dense cuBLAS calibration varied with clock/power, so it is recorded rather than treated as an attainable peak for small GEMMs.

At the first split-dNorm checkpoint, ordinary B1 medians were 213.744 us / 717.280 us. NCU measured DRAM throughput 53.39% / 58.39%, versus 50.77% / 55.63% for the preceding implementation. These are **not SoL90**. The optimistic unique-payload memory bounds are 79.23 us / 316.93 us at nominal bandwidth, much shorter than current runtime.

## Stage instrumentation

`derive_stage.py` and `stage.py` add clock64 counters to measure mean/max CTA stage cycles. Instrumentation changes scheduling and adds instructions; use it to locate bottlenecks, not as uninstrumented production latency. Phase A already includes stages 0–3; do not count it twice. The largest stage is dNorm + output-LN derivative + dTri storage, followed by output-LN/projection/gate recomputation; dWgate remains a material separate phase.

## Evidence

- Initial checkpoint: `results-L*.json`, `verification-L*.json`, NCU raw CSV/reports; 11 gradients and mutation/graph checks pass against baseline; memcheck both lengths and racecheck L384 pass.
- Tuning: `*-tune-L*.json`, `*-selected-L*.json`; each compiled cubin has ptxas output and SHA-256 source manifest.
- Bandwidth calibration: `calibration-node01.json`.
- Internal stages: `stage-node01.json`.

The canonical development entry remains unchanged until the final candidate passes full verification. No production-ready or SoL90 claim is made.
