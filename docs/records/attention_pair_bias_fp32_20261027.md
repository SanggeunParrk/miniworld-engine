# AttentionPairBias fp32 (TF32) on H100: hand-CUDA path (2026-10-27)

Branch `perf/final-apb-fp32` (base `integ/ladder-all2` b93a655e). Defect (audit, `perf/final-audit-pair`): AttentionPairBias fp32 on H100 ran the PyTorch reference
(`miniworld` = PyTorch), 1.22 (inference) / 1.26 (training) times slower than the portable Triton rung at 8 x 48, pair 128, L 384 (0.194 / 0.159 / 0.190 ms; 0.562 / 0.445 / 0.559 ms).

## Result (ladder, `--tier full --repeats 3`, Slurm job 36841, H100 SXM, CUDA graph / compiled)

| row | CUDA ms | portable ms | PyTorch ms | CUDA / portable |
|---|---|---|---|---|
| 8x48 pair128 inference L384 | 0.0834 | 0.1595 | 0.1907 | 0.52 |
| 8x48 pair128 inference L768 | 0.1803 | 0.4096 | 0.4760 | 0.44 |
| 8x48 pair128 training L384 | 0.2703 | 0.4672 | 0.5630 | 0.58 |
| 8x48 pair128 training L768 | 0.6706 | 1.1797 | 1.3791 | 0.57 |
| 16x24 pair128 inference L384 | 0.0857 | 0.1605 | 0.1896 | 0.53 |
| 16x24 pair128 training L384 | 0.3039 | 0.4548 | 0.5547 | 0.67 |
| 8x48 pair256 inference L384 | 0.1118 | 0.2328 | 0.2676 | 0.48 |
| 8x48 pair256 training L384 | 0.4252 | 0.6447 | 0.7729 | 0.66 |

Every row: CUDA < portable < PyTorch, no violations.

## What changed

New `kernels/augmented_attention/cuda/apb_tf32.{py,cu}`, wired in `modules/attention_pair_bias/module.py` (`_forward_tf32`), ladder groups `apb_fp32*`, test
`tests/gpu/sm90/test_apb_tf32_gpu.py`; design in `docs/gpus/h100/attention_pair_bias/attention_pair_bias.md` (fp32 section) and `dispatch.md`.
The existing `attn_tf32.cu` forward needed no change (build defines `NHP=8 DHP=48` etc.); the layout the 16 x 24 heads need is a zero-padded 32-wide head.
`pair_bias_tok_cuda` was hard-wired to 16 heads / d_pair 128, so the pair passes are new (8 / 12 / 16 / 24 heads, d_pair 128 on `mma.sync`; d_pair 256 forward on `mma.sync`, backward on the FMA pipe).

## Accuracy finding

First version: attention backward on the token stack's TF32 dqb / dkv kernels. Module-level error vs fp64 over the PyTorch TF32 module: output 1.01, dsingle 1.14, dpair 1.18 but
**to_query.weight 2.41, to_query.bias 2.36, to_key.weight 2.37** (bar 1.5). A hi + lo TF32 split of dS in both kernels (build define, since reverted) changed this to 2.37 only; an fp64 emulation of
the rounding sites (q/k, P, dO/v, dS, each ~2-3e-4 of dq, all together 5.7e-4) showed the error is spread over every TF32-rounded operand of the backward, so no single site could be fixed.
The backward attention therefore runs as exact-fp32 cuBLAS GEMMs around one hand-CUDA softmax-row kernel; all tensors are then 0.92-1.02x PyTorch. The forward keeps the TF32 core (output 1.01x).

## Checks

* GPU: `tests/gpu/sm90/test_apb_tf32_gpu.py` (all cases pass, including L768, 12 x 32, 16 x 24, pair 256, inference, bitwise repeatability, CUDA graph, `torch.compile`, ragged / TF32-off fallbacks).
* CPU (`tests/cpu/registry builder kernels layout`): 2129 passed, 5 failed + 2 errors; the same 5 + 2 (autotune cache staleness, sweep page counts) fail on the base `integ/ladder-all2` (job 36852), none touches this change.
* A probe-level graph profile (8 x 48, L384) puts inference at 76 us: pair pass 34, TF32 core 14, GEMMs 14, rows 8.

## Left

* Training is bound by the pair backward (60 us at 8 x 48, near the 2.7 TB/s copy limit) and five fp32 GEMMs (~45 us); a fused exact-fp32 backward core would save the S / dP / dS round trips.
* The 성능 확인 rows are untouched.
