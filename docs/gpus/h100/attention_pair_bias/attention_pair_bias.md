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

- fp32 activations and QK-norm have no CUDA path: on H100 they run the PyTorch reference (no portable-kernel fallback,
  [dispatch.md](../dispatch.md)). Carried over from the 2026-09-30 status snapshot
  ([docs/records/h100_status_20260930.md](../../../records/h100_status_20260930.md), item 9).

## Reproduce

```sh
# one GPU, engine environment's python (see ../h100.md for the srun line)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python benchmarks/runners/bench.py target=attention_pair_bias level=module mode=<inference|training> \
  min_seq_len=128 max_seq_len=768 seq_len_step=128 'implementations=[pytorch,portable,miniworld,cuequivariance]' \
  +apb_n_head=8 d_single=384 precision=bf16-mixed cudagraph=manual
python -m pytest tests/gpu/sm90/test_h100_apb_training_gpu.py tests/gpu/sm100/test_b200_apb_gpu.py -k apb_h100
```

Other layouts: `+apb_n_head=12|16|24` at `d_single=384`, `+apb_n_head=16 d_single=512`. The fp32 pair-weight oracle needs
`CUTLASS_PATH` (CUTLASS 4.2 headers) and `settings.configure(apb_h100_exact_dwb=True)`. Dated measurements and the
investigation: [records/h100_attention_pair_bias_20261013.md](../../../records/h100_attention_pair_bias_20261013.md).

## d_pair 256 (2026-10-26)

The native path above is d_pair 128. At d_pair 256 the module now runs the pair-fused core (`kernels/augmented_attention/cuda/fused_pair.py`: `pair_mma.cu` forward on mma.sync, `pair_rows.cu` backward, the
wgmma attention cores): 8 x 48, L 384, bf16, ladder CUDA / portable 0.62 (inference, 0.091 ms) and 0.86 (training, 0.349 ms); before, the call ran the PyTorch statements (1.07 / 1.03). Record:
[attention_cuda_ladder_20261026.md](../../../records/attention_cuda_ladder_20261026.md).

## Gate bias and Chai-1's width: 16 x 24 heads, d_pair 256 (2026-10-27)

`AttentionPairBias(gate_bias=c)` (the constant on the gate logits, `sigmoid(g + c)`; Chai-1's Pairformer single attention uses 1.0) and Chai-1's shape
(`d_single` 384 as 16 x 24, `d_pair` 256) run the native path on H100 for **inference** (bf16, B = 1, `L % 128 == 0`, the key mask as before); before, both fell to the composed
PyTorch statements (`gate_bias != 0` was excluded from the native dispatch, and the native pair kernel was 128 wide for 12 / 16 / 24 heads). Training with a gate bias or at 16-head
d_pair 256 keeps the composed route.

- The gate bias is not a new kernel term: the packed q | k | v | g bias (`bvec`, added by the one cuBLAS `addmm`) gets `c` on its g rows, once per parameter version (cached with the packs), so the
  attention core's `sigmoid(g)` epilogue reads `g + c`. A bias of 0 packs the same bytes as before (d_pair 128 calls are unchanged).
- d_pair 256 at 16 heads: `pair_bias_fwd` (`apb_rows.cu`) instantiates its 256-wide TMA ring for 16 heads (3 stages, one block per SM, 213 KB of shared memory); `Wf = to_bias.weight * ln_pair.weight * log2 e`
  [16, 256] is made in torch per parameter version (the prep pass is 128 wide). Everything else (row LayerNorm, q | k | v | g GEMM, the wgmma core with heads padded to 32, to_out accumulated onto the residual seed) is the existing 16 x 24 path.
- Ladder (H100, CUDA graph, `apb_chai`, `apb_gb`, median of 3; ms): record [../../../records/apb_gate_bias_chai_20261027.md](../../../records/apb_gate_bias_chai_20261027.md).
- Tests: `tests/gpu/sm90/test_apb_gate_bias_h100_gpu.py` (44: output within 1.3x of the bf16 PyTorch module's error against fp32, native route asserted, mask / no mask, bf16 and fp32-master parameters, compiled and CUDA-graph replay bitwise).

Not done: gate bias / 16-head d_pair 256 training (the pair backward kernels are 128 wide for 16 heads); the B200 path keeps `gate_bias == 0` (not measured there).
