# Portable bias-only DiT, a pair of its own per sample (`bodit_psb`, B > 1), 2026-10-27

Branch `perf/final-psb` (from `integ/ladder-all2`). H100, CUDA-graph replay, `benchmarks/runners/ladder.py` groups `bodit_psb` (bf16) and `bodit_psb_fp32`
(new), `single` [1, 4, L, 768], `pair` [4, L, L, 128], 1 block. Before: run of the base tree (`23009c44`), after: this branch. Times in ms.
The `bodit_psb` group never produced a row before: its override was `+per_sample_pair=true` and the key already exists in `bench.yaml` (hydra error);
fixed (`per_sample_pair=true`), inference added, fp32 group added.

| row | L | portable before | portable after | portable/PyTorch before | after | CUDA/portable before | after |
|---|---|---|---|---|---|---|---|
| inference bf16 | 256 | 0.182 | 0.133 | 0.843 | 0.612 | 1.014 | 1.383 |
| inference bf16 | 512 | 0.476 | 0.287 | 0.932 | 0.562 | 0.993 | 1.649 |
| training bf16 | 256 | 0.579 | 0.425 | 0.944 | 0.691 | 0.815 | 1.113 |
| training bf16 | 512 | 1.485 | 0.881 | 0.984 | 0.584 | 0.775 | 1.310 |
| inference fp32 | 256 | 0.319 | 0.219 | 0.939 | 0.642 | 0.946 | 1.383 |
| inference fp32 | 512 | 0.875 | 0.511 | 0.982 | 0.573 | 0.942 | 1.616 |
| training fp32 | 256 | 1.062 | 0.785 | 0.972 | 0.717 | 0.924 | 1.249 |
| training fp32 | 512 | 2.852 | 1.707 | 0.987 | 0.591 | 0.915 | 1.529 |

Inference rows: `miniworld` has no CUDA path for a per-sample pair (it runs the PyTorch reference), so the ladder reports them as routing gaps, not violations.

## What changed

1. `modules/bias_only_dit/hoist.py`: `pair_bias_all` on the portable family uses the one-pass LayerNorm + projection of `kernels/bias_only_attention/triton/pair_bias.py`
   over the stacked folded weights `[W_b diag(gamma_b)]` (<= 64 outputs: up to 4 blocks of 16 heads). The pair (268 MB at B=4, L=512) is read once forward and
   once backward; before it was PyTorch's LayerNorm (fp32 statistics, read + write) plus a GEMM, and in the backward the LayerNorm backward, two GEMMs and copies
   (at L=512 bf16: LayerNorm forward 1.6 ms + backward 1.0 ms + copies 1.5 ms of a 5.4 ms step). Training, inference, bf16 and fp32.
2. `kernels/bias_only_attention/triton/core_t.py` + `modules/bias_only_dit/module.py`: the feature-major core also for `single` [1, B, L, d], B > 1, with
   `p` [B, H, L, L]: the value / gate GEMM and the sigmoid gate pass are the B = 1 ones (the samples are the columns `b L ..`), `softmax(bias) v` and the two
   backward products are one batched GEMM over the heads per sample on strided views (no permute copy, no concatenation).

## CUDA side (not touched here)

The portable rung is now faster than the hand-CUDA rung in every training row. The CUDA step is dominated by its hoist backward: `hoist_bwd_k`
(`kernels/bias_only_dit/cuda/bo_hoist_bwd.cu`, 1 block, N = 16) takes 422 us at L=512 bf16 (110 us at L=256) against 235 us (68 us) of the Triton `_ln_bias_bwd_kernel`
(floor of the 268 MB pair read + gradient write: ~160 us); fp32: `ln0_bwd_rows` 713 us (181 us) against 404 us (114 us). Also `layernorm128` 48 us / 85 us (fp32) forward.
CUDA training time (ms): bf16 0.473 / 1.154, fp32 0.981 / 2.610 (L = 256 / 512).
