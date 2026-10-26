# Boltz-2 / RoseTTAFold3 / Chai-1 variants on H100 (sm90), inference and training

The modules of these three models whose math differs from the AF3 modules the engine already serves (FoldForge's
`docs/engine-kernel-gaps.md`, items 1-3b). **Inference** (bf16, no grad): every heavy product is cuBLAS or an existing engine kernel; the new
CUDA is small row / gather / attention kernels. **Training** (bf16 under autocast with fp32 master parameters, or fp32 on TF32; 2026-10-26): hand-CUDA row
kernels (forward and backward), cuBLAS GEMMs and the engine's sm_90a attention cores in one autograd `Function` per block; see
[Training](#training-2026-10-26). A call with another dtype, shape or card runs each module's torch statements (or the portable kernels with
`force_portable_kernels`). Measurements: [../../../records/model_variants_h100_20261022.md](../../../records/model_variants_h100_20261022.md),
[../../../records/dit_variants_cuda_ladder_20261026.md](../../../records/dit_variants_cuda_ladder_20261026.md).

The table is the inference path (the training path follows it).

| what | module / option | how | vs compiled torch (CUDA graph) |
|---|---|---|---|
| token DiT, all three | `modules.dit_variant.DiTVariantStack(n, **BOLTZ2 / ROSETTAFOLD3 / CHAI1)` | per step two conditioning GEMMs for the whole stack, the pair bias of every block in one LN + GEMM (cached per pair), per block CUDA rows (`kernels/dit_variants/cuda/rows.cu`: LN + sigmoid / identity AdaLN, kq LayerNorm, (up-)GLU, residual + gates, parallel residual) around cuBLAS and the AF3 token step's attention core | 2.8-4.1x (L 384 / 768) |
| atom DiT, all three | `AtomDiTVariantStack(n, **BOLTZ2_ATOM / ROSETTAFOLD3_ATOM / CHAI1_ATOM)` | the same rows at 128 wide, the local DiT's window attention and pair-bias kernels, Chai-1's same-conformer mask as -inf in the bias, the slid key window's edge windows in torch | 1.6x (8192 atoms) |
| triangle attention, RF3 gate / out biases | `TriangleAttention(gate_bias=True, out_bias=True)` | the sm90 kernels (heads of 32), the gate bias added to the front's gate, the out bias to the result | - |
| triangle attention, 64-channel heads (Chai-1 trunk, RF3 template) | any `TriangleAttention` with 64-channel heads | `kernels/triangle_attention/cuda/h100_dh.py`: generic LN, cuBLAS projections, a mma.sync FlashAttention-2 core (`triattn_dh_sm90.cu`, heads of 32 / 64, gate in the epilogue) | 3.2-6.5x (L 256-768) |
| triangle attention, Chai-1 outputs | `untransposed_output=True` (ending node), `dual_output=True` (confidence head) | the same path, the update laid back as the variant has it | as above |
| OPM, RF3 projection biases | `OuterProductMean(projection_bias=True)` | the wide H100 OPM composition with the biases added in its prologue (else the fused OPM plus the bias terms `(Rs A^T + Ls Bm^T) / cnt + c`) | 1.2-1.5x (1.6-2.1x eager) |
| OPM, Chai-1 groups | `OuterProductMean(groups=8)` | one projection GEMM, a group split, one batched GEMM over the MSA rows, the products laid out per pair row with the product LayerNorm (`kernels/outer_product_mean/cuda/gather.cu`), the out-projection | 1.1-1.2x |
| single attention gate bias (Chai-1) | `AttentionPairBias(gate_bias=1.0)` | the composed path (on H100 that is SDPA) with the constant on the gate logits | - |
| parallel Pairformer (Chai-1, RF3) | `PairformerBlock.from_components(..., parallel=True)` | every part on the block input; `pair + sum_i (y_i - pair)` from the parts' residual outputs in one CUDA pass (fp32 sums, one rounding) | - |

Second pass: [../../../records/model_variants_h100_speed_20261022.md](../../../records/model_variants_h100_speed_20261022.md). Not done: Chai-1's grouped OPM is a composition (1.1-1.2x); the token / atom DiT
variants keep the pair bias of a block per call unless the pair tensor is unchanged (FoldForge recomputes the pair conditioning every
diffusion step, so a fold-level split of the diffusion head would remove that work); FoldForge wiring of these modules is separate.

## Training (2026-10-26)

`DiTVariantStack` / `AtomDiTVariantStack` called with a gradient (any of `single`, `cond`, `pair` or a parameter requires grad) on H100 take
`kernels/dit_variants/cuda/train.py` (`settings.dit_variant_train_cuda`, `MINIWORLD_DIT_VARIANT_TRAIN_CUDA`; off = the portable / reference route).
Contract: bf16 (autocast, fp32 master parameters) or fp32 (TF32 matmuls) activations; **token stack** 768 wide x 16 heads of 48, `L` a multiple of 128, one
conditioning row set `[L, c]` shared by the samples, key mask `[L]`, pair `[L, L, 128 | 256]`, any number of samples; **atom stack** 128 wide x 4 heads of 32,
`N` a multiple of 64 (the 32 x 128 windows, slid or padded key window, Chai-1's same-conformer `uid`), conditioning `[N, 128]`, trunked pair `[N / 32, 32, 128, 16]`.

| part | what runs |
|---|---|
| conditioning, per block | the AdaLN scales / shifts and the residual gates' logits as ONE `[L, K D]` table made by the modules' own layers under autograd (the sigmoid form's weight-only LayerNorm, Chai-1's `+ 1`); the block's kernels return the table's gradient summed over the samples |
| pair bias | token: one CUDA LayerNorm of the pair rows (`pair_ln_fwd` / `pair_ln_bwd`) and one stacked GEMM `[nb H, C] x [C, L L]` for every block, `log2 e` folded into the weights (the LN bias of Chai-1 only shifts a row's logits: its gradient is exactly zero); atom: `pair_bias_mma` / `pair_bias_bwd_mma` per block |
| block forward | `ln_mod_fwd` (LN + the AdaLN modulation, one pass for the parallel blocks' two or three outputs) -> cuBLAS q / k / v (/ g) -> `qkv_prep` (scale; RF3's LayerNorm over the whole width) -> attention core -> gate -> out-projection -> `res_ln_fwd` / `res_out_fwd` (residual + gate + the transition's AdaLN) -> cuBLAS expand -> `glu_fwd` (SwiGLU, Boltz-2's up-gate) -> cuBLAS squeeze |
| block backward | `res_bwd`, `glu_bwd`, `res_adaln_bwd`, `gate_o_bwd` (do, dg and D = rowsum(dog . og) from the saved gated output alone), the attention backward, `qkv_bwd` (RF3's LayerNorm backward + column sums), `adaln_bwd`; weight gradients are cuBLAS (fp32 accumulation, fp32 results). Every reduction over the samples is a fixed-order partial row per (token, split of the samples): no atomics, bitwise repeatable |
| attention cores | token bf16: `attn_fwd` (gated epilogue) + `attn_dqb` / `attn_dkv`; token fp32: the TF32 `attn_tf32` core (operands rounded to nearest TF32 in the row kernels, v^T and the key-permuted layouts by a tiled transpose); atom: the local DiT's `lattn90` window kernels (`attn_fwd_mma` / `attn_bwd_mma`, bf16 or TF32) |
| slid key window (atoms) | the five edge windows (0, 1 and the last three real ones) whose keys differ from the clipped window the kernel reads: `edge.cu` (fp32, register-blocked) forward, backward compute (their dO / D rows zeroed for the main kernel) and apply, and the edge bias `LN(z[w]) gamma Wb` with its gradient |

Parity (`tests/gpu/sm90/test_dit_variant_train_h100_gpu.py`): the output and every gradient (inputs, conditioning, pair, each parameter) of the six stacks, bf16 and fp32, are within
1.5x the error of the same-dtype PyTorch reference against fp64 (measured 0.8-1.0x). Numbers and the ladder: [records/dit_variants_cuda_ladder_20261026.md](../../../records/dit_variants_cuda_ladder_20261026.md).
