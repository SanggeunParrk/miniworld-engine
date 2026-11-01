# H100 autotune caches for the five Triton kernels registered on 2026-10-31, and the SWA window-attention precision (2026-11-01)

Branch `perf/triton-caches` from `main` 9cf93c27.

## What was built

`miniworld-engine build <op,...> grid --per-op --kernels portable` on H100 (2 GPUs, `--units-per-gpu 2`), merged into
`autotune/data/<op>/NVIDIA H100 80GB HBM3 (sm90).json`:

| kernel | units | cache keys |
|---|---|---|
| gated_projection_fwd_residual_triton | 16 (atom, L 1024-8192, bf16 / fp32) | 8 |
| gated_projection_bwd_residual_triton | 16 | 8 |
| bias_only_attention_layernorm_gemm_triton | 60 (pair, L 128-768, five widths, C rounded to 16-256, N 16; bf16 / fp32) | 48 |
| bias_only_attention_bwd_layernorm_gemm_triton | 20 (training lengths 384 / 768 only, the builder's policy) | 16 |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 4 (edge, 1024-8192 groups) | 4 |

The grid sets were narrowed before the build (residual: 256-4096 elements x 4 / 8 warps; pair bias: 32 / 64 / 128 rows x 4 / 8 warps
x 2 / 3 stages; MPNN: the policy axes x 2 / 4 warps); each contains the default launch, so the build measured the default against the
rest in the same round. The pair-bias driver now launches the production head count (16), which the key folds.

Two code changes make a tuned cache take effect: the early prunes that reproduce the pre-registration launch (`pair_bias._bwd_prune`,
`mpnn inference._prune`) apply only to the DEFAULT config list (identity check against `configs_for`'s live list), so a cache's measured
winners, which the cache reader hands back through the same prune, are not re-pruned to the old rule (the MPNN row-tile validity check
still applies).

## Default launch vs tuned config (H100, CUDA graph, us, min of two interleaved runs; `DEFAULT_LAUNCH=1` = cache reader bypassed)

| kernel | shape | dtype | default | tuned | change |
|---|---|---|---|---|---|
| gated_projection_fwd_residual_triton | 5x1024x128 | bfloat16 | 2.30 | 2.30 | +0.0% |
| gated_projection_bwd_residual_triton | 5x1024x128 | bfloat16 | 2.48 | 2.49 | +0.4% |
| gated_projection_fwd_residual_triton | 5x4096x128 | bfloat16 | 7.21 | 7.21 | +0.0% |
| gated_projection_bwd_residual_triton | 5x4096x128 | bfloat16 | 7.30 | 7.31 | +0.1% |
| gated_projection_fwd_residual_triton | 48x2048x128 | bfloat16 | 34.32 | 34.29 | -0.1% |
| gated_projection_bwd_residual_triton | 48x2048x128 | bfloat16 | 42.92 | 42.88 | -0.1% |
| gated_projection_fwd_residual_triton | 48x8192x128 | bfloat16 | 131.36 | 131.37 | +0.0% |
| gated_projection_bwd_residual_triton | 48x8192x128 | bfloat16 | 164.55 | 164.67 | +0.1% |
| bias_only_attention_layernorm_gemm_triton | 256x256x128 | bfloat16 | 6.77 | 6.76 | -0.1% |
| bias_only_attention_bwd_layernorm_gemm_triton | 256x256x128 | bfloat16 | 21.14 | 22.22 | +5.1% |
| bias_only_attention_layernorm_gemm_triton | 256x256x256 | bfloat16 | 14.10 | 13.64 | -3.3% |
| bias_only_attention_bwd_layernorm_gemm_triton | 256x256x256 | bfloat16 | 49.93 | 50.31 | +0.8% |
| bias_only_attention_layernorm_gemm_triton | 512x512x128 | bfloat16 | 31.61 | 31.44 | -0.5% |
| bias_only_attention_bwd_layernorm_gemm_triton | 512x512x128 | bfloat16 | 71.55 | 70.85 | -1.0% |
| bias_only_attention_layernorm_gemm_triton | 512x512x256 | bfloat16 | 61.91 | 62.12 | +0.3% |
| bias_only_attention_bwd_layernorm_gemm_triton | 512x512x256 | bfloat16 | 152.57 | 152.75 | +0.1% |
| bias_only_attention_layernorm_gemm_triton | 768x768x128 | bfloat16 | 64.72 | 64.80 | +0.1% |
| bias_only_attention_bwd_layernorm_gemm_triton | 768x768x128 | bfloat16 | 145.92 | 141.85 | -2.8% |
| bias_only_attention_layernorm_gemm_triton | 768x768x256 | bfloat16 | 129.43 | 129.46 | +0.0% |
| bias_only_attention_bwd_layernorm_gemm_triton | 768x768x256 | bfloat16 | 320.88 | 236.27 | -26.4% |
| gated_projection_fwd_residual_triton | 5x1024x128 | float32 | 2.51 | 2.51 | +0.0% |
| gated_projection_bwd_residual_triton | 5x1024x128 | float32 | 2.80 | 2.79 | -0.4% |
| gated_projection_fwd_residual_triton | 5x4096x128 | float32 | 12.64 | 12.75 | +0.9% |
| gated_projection_bwd_residual_triton | 5x4096x128 | float32 | 16.91 | 17.09 | +1.1% |
| gated_projection_fwd_residual_triton | 48x2048x128 | float32 | 65.68 | 65.66 | -0.0% |
| gated_projection_bwd_residual_triton | 48x2048x128 | float32 | 83.73 | 83.78 | +0.1% |
| gated_projection_fwd_residual_triton | 48x8192x128 | float32 | 256.49 | 256.50 | +0.0% |
| gated_projection_bwd_residual_triton | 48x8192x128 | float32 | 328.12 | 327.87 | -0.1% |
| bias_only_attention_layernorm_gemm_triton | 256x256x128 | float32 | 16.76 | 16.73 | -0.2% |
| bias_only_attention_bwd_layernorm_gemm_triton | 256x256x128 | float32 | 38.82 | 38.91 | +0.2% |
| bias_only_attention_layernorm_gemm_triton | 256x256x256 | float32 | 45.43 | 45.51 | +0.2% |
| bias_only_attention_bwd_layernorm_gemm_triton | 256x256x256 | float32 | 79.27 | 79.27 | +0.0% |
| bias_only_attention_layernorm_gemm_triton | 512x512x128 | float32 | 61.14 | 61.17 | +0.0% |
| bias_only_attention_bwd_layernorm_gemm_triton | 512x512x128 | float32 | 116.01 | 115.65 | -0.3% |
| bias_only_attention_layernorm_gemm_triton | 512x512x256 | float32 | 161.51 | 161.98 | +0.3% |
| bias_only_attention_bwd_layernorm_gemm_triton | 512x512x256 | float32 | 262.86 | 261.26 | -0.6% |
| bias_only_attention_layernorm_gemm_triton | 768x768x128 | float32 | 132.61 | 132.33 | -0.2% |
| bias_only_attention_bwd_layernorm_gemm_triton | 768x768x128 | float32 | 242.18 | 245.67 | +1.4% |
| bias_only_attention_layernorm_gemm_triton | 768x768x256 | float32 | 368.97 | 371.81 | +0.8% |
| bias_only_attention_bwd_layernorm_gemm_triton | 768x768x256 | float32 | 574.11 | 533.59 | -7.1% |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 1024x48x128 | bfloat16 | 39.87 | 36.03 | -9.6% |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 2048x48x128 | bfloat16 | 70.49 | 66.66 | -5.4% |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 4096x48x128 | bfloat16 | 159.33 | 129.05 | -19.0% |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 16384x48x128 | bfloat16 | 626.98 | 627.61 | +0.1% |

Shapes the cache does not cover still launch the default (and warn once): the pair-bias backward at L 256 / 512 (the build tunes training
kernels at the training lengths 384 / 768 only) and the MPNN fusion above 8192 residue groups (the driver's ladder ends there). The
+5.1% at pair-bias backward 256 x 256 x 128 is such a miss -- the same config both times -- so it measures run-to-run noise of that 20 us
kernel; every covered shape is within +1.4% of the default (noise) or faster, up to -26% (pair-bias backward, 768 x 768 x 256) and -19%
(MPNN, 4096 groups).

## SWA DiT window attention in the fp32 block

An H100 launch spy of the fp32 swa_dit drivers (forward and backward, portable path) saw `_swa_attn_fwd_kernel`, `_swa_attn_bwd_dq_kernel`
and `_swa_attn_bwd_dkv_kernel` launched on bfloat16 operands (with fp32 statistics and int32 lengths) inside the fp32 block, as
`swa_dit/dispatch.py` says. Their registry rows declare bf16, which is what they run at; the routing did not count them for an fp32
call. They are now in `records._RUNS_AT` at bf16: an fp32 `swa_dit` call counts them and asks their bf16 rows.

## Also

`gated_residual` is a `registry_module.csv` checkpoint leaf now (`autotune/checkpoint_cases.py`), so `dev derive` records what
`ops.gated_residual` launches and `build all` can reach it.
