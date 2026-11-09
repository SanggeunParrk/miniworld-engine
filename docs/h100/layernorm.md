# LayerNorm, LayerNormLinear and RMSNorm on H100 (sm90)

Scope.

- `modules.LayerNorm` (`modules/norm.py`): `nn.LayerNorm` with the affine pinned to fp32 (`_Fp32ParamsMixin`, so a bf16 gamma at 1.0
  can still move by one Adam step), computed in at least fp32 and returned in the input dtype. Backends: `pytorch`, `cuda`, `triton`,
  `anthropic` (external baseline, inference only; not described here). Requirements rows: last-dimension widths 64 .. 833 on the pair,
  single, atom and noise streams (including 267, 451, 651, 831, 833, which are not multiples of 8) and 2560 (single, `master` only),
  with and without bias, dtypes `float32` and `master`, eval and train.
- `modules.RMSNorm` (`nn.RMSNorm`, epsilon `effective_eps`: float32's machine epsilon for bf16 / fp16 when `eps=None`). Backends
  `pytorch`, `cuda`, `triton`. It has no requirements row of its own, and no `rmsnorm` kernel is listed in `requirements_kernel.csv`.
- LayerNormLinear (`kernels/layernorm_linear/`): no module class; Triton LayerNorm(+Linear) kernels that other modules' backends call
  (TriMul's strided LN-out, AdaLN, the DiT variants' portable path).
- `AdaptiveLayerNorm`, `IdentityAdaptiveLayerNorm`, `AdaptiveRMSNorm` are PyTorch-only modules.

Judging: [measure](../measure.md), [modules](../modules.md), [kernels](../kernels.md).

## Kernels

| name (`requirements_kernel.csv`) | family, role | what it does |
|---|---|---|
| `layernorm_general_fwd` | layernorm, forward | `ln_fwd_kernel` (`cuda/general/ln_fwd.cu`): any width, TPR threads per row, values in registers, fp32 statistics |
| `layernorm_general_bwd`, `layernorm_general_bwd_reduce` | layernorm, backward | `ln_bwd_kernel` (dx + per-CTA affine partials) and `ln_col_reduce` (fixed-order column sum) |
| `layernorm_pipe_direct_fwd` | layernorm, forward | `direct_fwd` (`cuda/pipe/ln_pipe.cu`): single-chunk rows, non-persistent, a warp issues all loads of its passes then computes |
| `layernorm_pipe_direct2_fwd` | layernorm, forward | `direct2_fwd`: the same for exact-fit rows whose affine has the activation dtype; no per-element dtype branches (40 registers) |
| `layernorm_pipe_bulk_fwd` | layernorm, forward_infer | `bulk_fwd`: inference of exact-fit single-chunk rows, 16 KB tiles moved by one `cp.async.bulk` load and one bulk store each |
| `layernorm_pipe_fwd` | layernorm, forward | `pipe_fwd`: persistent warp-per-row-group forward with a cp.async ring per warp (multi-chunk rows) |
| `layernorm_pipe_bwd`, `layernorm_pipe_reduce_bwd` | layernorm, backward | `pipe_bwd` (dx + one partial row per CTA, plain stores) and `pipe_reduce` (PDL dependent, fixed-order sum, converts to the weight dtype) |
| `layernorm_fwd_saveact` | layernorm, forward | Triton `layer_norm_fwd_fused` (`triton/main.py`); mean / rstd stored only when x, weight or bias requires grad (`ctx.needs_input_grad`) |
| `layernorm_bwd_atomic` | layernorm, backward | Triton `layer_norm_bwd_dx_fused`: persistent (8 programs per SM), dw / db in registers, one `atomic_add` per program |
| `layernorm_bwd_persistent` | layernorm, backward | Triton `_ln_bwd_persistent` (`triton/persistent.py`): persistent grid, one partial row per program, small final reduce, no atomics |
| `layernorm_fwd_saveact_strided`, `layernorm_bwd_atomic_strided` | layernorm_linear, forward_train / backward | `te_style.py`: LN materialise from an arbitrary-stride x (TriMul's m-major `[B, D, L, L]` view) and its atomic backward writing dx in x's strides |
| `layernorm_bwd_split_mmajor` | layernorm_linear, backward | `mmajor_bwd.py`: atomic-free m-major LN backward (persistent, partial rows + reduce, M-contiguous vector loads); small M keeps the atomic kernel |
| `layernorm_linear_fwd` | layernorm_linear, forward_infer | `_lnl_fwd_kernel` (`fused.py`): Y = LN(x) W^T + b in one kernel, the row's K fits one tile |
| `layernorm_linear_fwd_fp32`, `layernorm_linear_bwd_fp32` | layernorm_linear, forward_train / backward | `pair_bias.py`: LN + projection to n_head without materialising the normalised pair, and its backward |

RMSNorm kernels (not in `requirements_kernel.csv`): the general extension built with the mean and bias dropped (`RMS` template flag),
the pipe kernels with `rms=True`, and Triton `rmsnorm_fwd_kernel` / `rmsnorm_bwd_kernel` (`kernels/rmsnorm/triton/main.py`).

## Design: CUDA LayerNorm / RMSNorm

Entry `layernorm_cuda` (`cuda/general/__init__.py`) and `rmsnorm_cuda` (`kernels/rmsnorm/cuda`): on sm_90, rows the pipe kernels take
(`pipe.serves`: row bytes a multiple of 16 in 144 .. 2048, bf16 / fp16 / fp32) go to `cuda/pipe`; every other width goes to the general
kernels. A no-grad call runs the forward without statistics.

General kernels (`ln_common.cuh`): a row of N elements is held by TPR threads (4 .. 1024; TPR <= 32 is a sub-warp group), each thread
keeping CHUNKS vectors of VEC elements in registers (VEC = 16 bytes when N is a multiple of it, else 1: any width). `config(n, bytes)`
picks the fewest idle lanes with at most 32 values per thread and TPR near 128. A CTA is max(TPR, 256) threads and grid-strides over row
groups. Backward: dx per row from the saved mean / rstd; dw, db as per-CTA partials summed by `ln_col_reduce` in a fixed order. Widths
up to 65536 (bf16 / fp16) or 32768 (fp32). Built for sm_80 / 90 / 100 (plain CUDA).

Pipe kernels (`ln_pipe.cu`): a row is `N * itemsize / 16` 16-byte chunks; TPR lanes (16 or 32) hold a row, CH chunks per lane, so a
warp pass covers 32 / TPR rows. The forward picks, by row shape and activation size:

- multi-chunk rows: `pipe_fwd`, persistent (SMs x resident CTAs), each warp streaming its passes through its own cp.async ring
  (stage counts per CH: 4 / 4 / 6 / 3); a lane reads back exactly the bytes it copied, so no barrier is needed.
- single-chunk rows: `direct2_fwd` when the affine has the activation dtype (PDL launch below 8 MB of activation), else `direct_fwd`.
  Under `master` the affine is fp32 and the activation bf16, so `direct_fwd` serves those rows.
- inference of exact-fit single-chunk rows whose affine has the activation dtype, 8 .. 48 MB of activation: `bulk_fwd` (4 tiles in
  flight per CTA).

Training stores (mean, rstd) as one float2 per row. Backward: `pipe_bwd` computes dx from x, dy and the statistics, keeps the affine
gradients in registers per lane, sums them over the CTA's warps and writes one partial row per CTA; `pipe_reduce` (launched with
programmatic dependent launch) adds the rows in a fixed order and converts to the weight dtype. The affine is read in its own dtype
(bf16, fp16 or fp32) and its gradients come back in it.

## Design: Triton LayerNorm (`triton` backend)

`triton_layernorm`: `layer_norm_fwd_fused` tiles rows (`BLOCK_M1`) and the feature axis (`BLOCK_K`). When one tile covers the row, x
stays in registers (mean, then centred variance); otherwise tile moments are combined with Welford's formula and x is re-read to
normalise (E[x^2] - E[x]^2 loses variance on large-offset inputs). Optional per-row scale folded into the epilogue. The saved x keeps its
own dtype. `layer_norm_bwd_dx_fused` is persistent (`SMs x 8` programs, capped at the tile count), software-pipelined over row tiles,
with dw / db accumulated in registers and flushed with one atomic per program.

## LayerNormLinear kernels

- `te_style.py` (training, TriMul's LN-out): `x_normed = LN(x)` by a Triton kernel that reads x at arbitrary strides, then
  `F.linear` (cuBLAS). Backward: `dx_normed = dY W` (cuBLAS), the LN backward writing dx in the input layout, and one wgrad GEMM
  `T = dY^T x_hat` (x_hat recomputed from the saved statistics) from which `dW = gamma * T + outer(db, beta)`, `dgamma = (W * T).sum(0)`,
  `dbeta = db W`. `mmajor_bwd.py` replaces the atomic LN backward on m-major tensors at large M.
- `fused.py` (`layernorm_linear_triton_fwd_t`, used by the DiT variants' portable path): per program, load M rows' full K, fp32 one-pass
  statistics, normalise with gamma / beta, multiply the (BLOCK_M1, K) tile against a (K, BLOCK_N) slice of W; K must fit one block.
- `stats.py` (`layernorm_stats_triton`, used by the AdaLN and Transition Triton kernels): rstd and `c1 = mean * rstd` in one pass.
- `recompute.py`: x_normed / x_hat from saved statistics for the backward.

## Numerics

- Statistics in fp32 in every kernel; output in the input dtype; the affine applied in fp32.
- CUDA (general and pipe): affine gradients without atomics, fixed reduction order: a replay is bit-identical.
- Triton `layer_norm_bwd_dx_fused` and the strided `te_style` backward add dw / db with atomics: the fp32 rounding of the affine
  gradients can differ between replays. `_ln_bwd_persistent` and `mmajor_bwd` are atomic-free.
- RMSNorm: `dx = (w dy - xhat c1) rstd`, `c1 = sum(xhat w dy) / N`; no mean, no bias, no db.
- fp32 GEMMs of the LayerNormLinear paths follow torch's TF32 setting (`te_style._FP32_MATMUL_PRECISION`, default `"high"`).

## Launch configs

`kernels/layernorm/configs/h100_sm90.json`: `layernorm_fwd_saveact_triton`, `layernorm_bwd_atomic_triton`, `layernorm_bwd_split_triton`,
`layernorm_fwd_mmajor_triton`. `kernels/layernorm_linear/configs/h100_sm90.json`: `layernorm_fwd_saveact_strided_triton`,
`layernorm_bwd_atomic_strided_triton`, `layernorm_bwd_split_mmajor_triton`, `layernorm_linear_fwd_triton`,
`layernorm_linear_fwd_fp32_triton`, `layernorm_linear_bwd_fp32_triton`, `layernorm_stats_triton`. `kernels/rmsnorm/configs/h100_sm90.json`:
`rmsnorm_fwd_triton`, `rmsnorm_bwd_triton`. Top 4 per (dtype, shape bucket, flags), rank 1 by default. The CUDA kernels take no table:
`config()` (general) and the compile-time constants of `ln_pipe.cu` (`MWP_*`) fix their schedules.

## Limits

- LayerNorm `cuda` (`_supports_cuda`): CUDA on sm_90; one normalised dimension matching the input width; bf16 / fp16 / fp32; non-empty;
  width within the general builds (<= 65536, fp32 <= 32768).
- LayerNorm `triton`: CUDA; the same shape / dtype checks.
- RMSNorm `cuda` / `triton`: the same checks (the CUDA range check is the general LayerNorm extension's `supported`).
- `layernorm_linear_fwd` needs K to fit one tile (the wrapper keeps eager torch beyond it).

## History

Old notes: `git show HEAD:docs/kernels/layernorm.md` (backward dispatch heuristics and the per-GPU calibration cache, both removed),
`HEAD:docs/kernels/layernorm-linear.md` (the quack / CuTe fused LN+GEMM milestones; the `cute/` path is no longer in the tree),
`HEAD:docs/kernels/rmsnorm-adamod.md` (the adaLN modulate ladder L0 / L2 / L3; `triton_rmsnorm_adamod` still exists but nothing calls it).
