# Native CUDA normalization candidates

Explicit APIs: `cuda_layernorm`, `cuda_rmsnorm`, `cuda_layernorm_linear`.
These are opt-in APIs. Existing engine/module dispatch is unchanged.

## Contract

- Normalize the last axis; arbitrary nonempty width and arbitrary leading dimensions.
- FP16/BF16/FP32 accumulate in FP32; FP64 accumulates in FP64.
- Floating affine parameters may have a different dtype. Parameter casts preserve
  leaf gradient dtypes. Output follows input dtype.
- Noncontiguous input/upstream gradients use contiguous copies. These copies must
  be included when benchmarking a noncontiguous workload. Contiguous views with
  unaligned storage offsets use safe scalar loads or an aligned weight copy.
- RMS `eps=None` uses the input dtype epsilon; explicit/default epsilon is 1e-5.
- Optional affine, empty leading dimensions, first-order gradients, CPU fallback.
- Second-order CUDA differentiation is explicitly unsupported.
- Warm extensions before CUDA graph capture. Opaque operations provide fake
  implementations for `torch.compile(fullgraph=True)`.

## Schedules

`backend="auto"` chooses native schedules; `"warp"` and `"cta"` are explicit
controls. `threads={128,256}` and `rows=1..4096` override the launch defaults.

- D=64/128: eight threads normalize a row. Backward combines affine gradients
  inside the CTA before writing one partial per CTA. Shuffle masks cover exactly
  the eight participating threads, including partial row groups.
- D=256/384/512: one warp per row; aligned backward combines partials across
  warps in the CTA when its shared-memory footprint fits 48 KiB.
- Other widths use scalar-safe/packed warp kernels. Runtime shape tails and
  misaligned inputs remain supported.
- D=1024..16384: a CTA cooperates on a row, avoiding very large per-warp register
  arrays. dX and affine partials share the same input/upstream-gradient load.
- Larger widths retain the bounded-register generic CUDA implementation.

Centered variance avoids cancellation from `E[x²]-E[x]²`. Float uses `rsqrtf`;
FP64 keeps double-precision reciprocal square root. Affine reductions have no
floating global atomics and use at most 32 MiB of partial workspace. Reductions
are deterministic within a configuration, not bitwise identical across layouts.

The defaults are measured schedule heuristics, **not a completed autotune cache**.
See the report for exact measured shapes and comparisons. Hardware validated:
H100. Other GPUs require separate performance qualification.

## LayerNormLinear

`linear_backend="composed"` runs native normalization and cuBLAS through
`F.linear`. `"fused"` requests the experimental native LN/WMMA implementation;
unsupported dtypes/shapes fall back to composition. The direct experimental API
is `norm_cuda.linear.fused_layernorm_linear`.

The fused implementation supports Hopper-or-newer FP16/BF16, input widths
64/128/256/384/512, and output widths divisible by 16. It uses packed loads,
padded shared-memory tiles, and warp assignment over row/output tiles. It
preserves the normalized activation's rounding to input dtype before GEMM.
Training saves x, rounded normalized activation, mean/rstd, and parameter
references; backward uses cuBLAS dX/dW and the native norm backward.

`linear_backend="auto"` uses fusion only for the measured M=147456, K=128,
N=16 low-precision cell; other cells use composition. Fusion helps inference
there, while training is approximately tied with composition. The fused CUDA
prototype does **not** beat the existing fused Triton forward on every cell.
Broad fusion promotion is deliberately withheld.

## Evidence

`tests/numerics/test_norm_cuda_gpu.py` and `test_norm_linear_cuda_gpu.py` cover
mixed dtypes, gradients, tails, views, compile, and graph replay. Development
artifacts and frozen baselines are in the parent workspace under
`runs/norm_cuda_20260926/opt`; the development report records exact job/source
identities, successful gates, rejected candidates, and timing limitations.
