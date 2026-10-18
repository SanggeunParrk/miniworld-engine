# AttentionPairBias on H100 (sm90)

Kernel-level status of AttentionPairBias (Pairformer single track: LayerNorm(single) -> q / k / v / gate projections, pair bias =
to_bias(LayerNorm(pair)), key mask, softmax, sigmoid gate, to_out, residual) on H100. Module summary: [h100.md](../h100.md).
Contract (same code as B200, `integrations/attention_pair_bias_b200.py`): bf16 activations, B = 1, d_pair 128, no QK-norm,
(heads, d_single) in 8 x 48 / 12 x 32 / 16 x 24 / 24 x 16 at 384 and 16 x 32 at 512, inference L % 16 == 0, training L % 128 == 0.
fp32 master parameters are rounded to bf16 inside the packing pass (no cast launch); gradients are fp32, never rounded to bf16.

## Path

CUDA + cuBLAS only, no Triton. Pair kernels (`pair_bias`, `pair_bias_bwd`: TMA / bulk-copy rings, `mma.sync`, LayerNorm folded
into the projection) and row passes in `kernels/augmented_attention/cuda/apb/apb_rows.cu`; sm_90a wgmma attention cores
`attn_fwd.cu` (inference, and training with the gate in its epilogue: O fp32 + og bf16 + LSE), `attn_dq.cu` (dQ + dbias),
`attn_dkv.cu`.

- Inference: `ln_rows` -> cuBLAS q|k|v|g -> `pair_bias` -> gated `attn_fwd` -> cuBLAS to_out accumulated onto the residual seed.
- Training forward: `prep` (cast + pack, one launch) -> `ln_rows` -> cuBLAS -> `pair_bias` -> `attn_fwd` (gate fused) -> cuBLAS.
  Backward: cuBLAS dO, dWo -> `gate_bwd` -> `attn_dq`, `attn_dkv` -> `qkv_bwd` -> cuBLAS dxa, dW -> `ln_bwd` -> `pair_bias_bwd`
  -> `finalize` (fixed-order tree reduce of the block partials + every gradient into its own tensor).

## Accuracy

Reference: fp32 PyTorch module vs fp64 truth (random parameters +0.1 N, 20 % key mask, L 384 / 768, master parameters under
autocast; probe and numbers in the [record](../../../records/h100_attention_pair_bias_20261013.md)). Relative Frobenius error vs fp64; ours / bf16 PyTorch / Triton agree on every tensor:

| tensor | L384 ours | L384 bf16 PyTorch | L768 ours | L768 bf16 PyTorch |
|---|---|---|---|---|
| output | 1.03e-2 | 1.04e-2 | 1.06e-2 | 1.07e-2 |
| dsingle | 1.65e-2 | 1.67e-2 | 1.67e-2 | 1.70e-2 |
| dpair | 1.67e-2 | 1.67e-2 | 1.68e-2 | 1.68e-2 |
| dW to_bias | 1.61e-2 | 1.59e-2 | 1.66e-2 | 1.68e-2 |
| dW to_out | 1.15e-2 | 1.17e-2 | 1.19e-2 | 1.21e-2 |

The bf16 noise floor is the same for every implementation (cosine to the fp32 module 0.9998-0.9999). The criterion is
the guide's: every tensor within 1.3x of the bf16 PyTorch module (`tests/gpu/sm90/test_h100_apb_training_gpu.py`, 105 H100
tests with the inference ones, repeatability bitwise, CUDA-graph replay, `torch.compile`).

`settings.apb_h100_exact_dwb` (default **off**): recompute the to_bias.weight gradient in fp32 from bf16-rounded inputs
(CUTLASS 4.2 + `CUTLASS_PATH`). It lowers that one gradient's error 1.7e-2 -> 0.8e-2 and costs 2.4x the step (L768 1.28 ms vs
0.27 ms), which put training behind Triton at L384 / 768 before; every other gradient of the block stays at 1.7e-2.

## Measurements (2026-10-13, H100 SXM, CUDA graph, `bench.py ... precision=bf16-mixed`, ms)

| L | inference ours | Triton | PyTorch | cuEquivariance | × | training ours | Triton | PyTorch | cuEquivariance | × |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.0225 | 0.0431 | 0.0518 | 0.0549 | 1.91 | 0.0661 | 0.1116 | 0.1364 | 0.1251 | 1.69 |
| 256 | 0.0300 | 0.0580 | 0.0701 | 0.0740 | 1.94 | 0.0865 | 0.1629 | 0.2071 | 0.1785 | 1.88 |
| 384 | 0.0419 | 0.0886 | 0.1061 | 0.1084 | 2.11 | 0.1226 | 0.2546 | 0.3089 | 0.2643 | 2.08 |
| 512 | 0.0558 | 0.1279 | 0.1518 | 0.1542 | 2.29 | 0.1656 | 0.3648 | 0.4334 | 0.3688 | 2.20 |
| 640 | 0.0722 | 0.1716 | 0.2010 | 0.2033 | 2.38 | 0.2153 | 0.4924 | 0.5887 | 0.5190 | 2.29 |
| 768 | 0.0902 | 0.2238 | 0.2590 | 0.2608 | 2.48 | 0.2716 | 0.6431 | 0.7379 | 0.6528 | 2.37 |

× = ours vs the fastest other row (Triton). Before this work: training L384 0.498 ms, L768 1.28 ms (shadow on; 0.5x / 0.5x Triton).

## Per-kernel (ncu / profiler, L768 training, us)

pair_bias_bwd 115 (2.57 TB/s; same-size copy 2.95), pair_bias 59 (2.74 TB/s, read-only), attn_dq 15, attn_fwd 15, attn_dkv 14,
cuBLAS GEMMs ~38, row kernels + finalize ~25. Pair kernels are 87-93 % of the measured copy bandwidth; stage depth 2 / 3 / 4
did not change them. Step time-roofline SoL (pair bytes / 3.35 TB/s) ~55 % at L768, ~30 % at L384 (launch-latency bound).

## What changed and what did not work

- The natural-unit attention builds divided every score by a float constant (IEEE division, ~10 instructions): attn_fwd 59 -> 15 us,
  attn_dq 62 -> 15, attn_dkv 84 -> 14 at L768 (multiply by the reciprocal).
- Shadow off by default (above); `acc_reduce` 12 -> 3 us (16-warp tree, no zero fill); cast + pack and gate fused.
- Key splits (2 warpgroups per tile, merged in smem; build flag `APB_KS`): 1-2 us per core, 0.5 % of the step, default 1.
- Not done: overlapping `pair_bias_bwd` with the dK / dV chain on a second stream (~10 %), bf16 dbias (3 % of bytes).

## Open

- fp32 activations and QK-norm have no CUDA path: on H100 they run the PyTorch reference (no Triton fallback,
  [dispatch.md](../dispatch.md)). Carried over from the 2026-09-30 status snapshot
  ([docs/records/h100_status_20260930.md](../../../records/h100_status_20260930.md), item 9).

## Reproduce

```sh
# one GPU, engine environment's python (see ../h100.md for the srun line)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python benchmarks/runners/bench.py target=attention_pair_bias level=module mode=<inference|training> \
  min_seq_len=128 max_seq_len=768 seq_len_step=128 'implementations=[pytorch,triton,miniworld,cuequivariance]' \
  +apb_n_head=8 d_single=384 precision=bf16-mixed cudagraph=manual
python -m pytest tests/gpu/sm90/test_h100_apb_training_gpu.py tests/gpu/sm100/test_b200_apb_gpu.py -k apb_h100
```

Other layouts: `+apb_n_head=12|16|24` at `d_single=384`, `+apb_n_head=16 d_single=512`. The fp32 pair-weight oracle needs
`CUTLASS_PATH` (CUTLASS 4.2 headers) and `settings.configure(apb_h100_exact_dwb=True)`. Dated measurements and the
investigation: [records/h100_attention_pair_bias_20261013.md](../../../records/h100_attention_pair_bias_20261013.md).
