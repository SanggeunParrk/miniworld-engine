# Boltz-2 / RoseTTAFold3 / Chai-1 variants on H100 (sm90), inference

The modules of these three models whose math differs from the AF3 modules the engine already serves (FoldForge's
`docs/engine-kernel-gaps.md`, items 1-3b). Inference only (bf16, no grad): a call with grad, another dtype or another card runs each
module's torch statements. Every heavy product is cuBLAS or an existing engine kernel; the new CUDA is small row / gather / attention
kernels. Measurements: [../../../records/model_variants_h100_20261022.md](../../../records/model_variants_h100_20261022.md).

| what | module / option | how | vs compiled torch (CUDA graph) |
|---|---|---|---|
| token DiT, all three | `modules.dit_variant.DiTVariantStack(n, **BOLTZ2 / ROSETTAFOLD3 / CHAI1)` | per step two conditioning GEMMs for the whole stack, the pair bias of every block in one LN + GEMM (cached per pair), per block CUDA rows (`kernels/dit_variants/cuda/rows.cu`: LN + sigmoid / identity AdaLN, kq LayerNorm, (up-)GLU, residual + gates, parallel residual) around cuBLAS and the AF3 token step's attention core | 1.3-2.9x (L 384 / 768) |
| atom DiT, all three | `AtomDiTVariantStack(n, **BOLTZ2_ATOM / ROSETTAFOLD3_ATOM / CHAI1_ATOM)` | the same rows at 128 wide, the local DiT's window attention and pair-bias kernels, Chai-1's same-conformer mask as -inf in the bias, the slid key window's edge windows in torch | 1.2-1.9x (8192 / 24576 atoms) |
| triangle attention, RF3 gate / out biases | `TriangleAttention(gate_bias=True, out_bias=True)` | the sm90 kernels (heads of 32), the gate bias added to the front's gate, the out bias to the result | - |
| triangle attention, 64-channel heads (Chai-1 trunk, RF3 template) | any `TriangleAttention` with 64-channel heads | `kernels/triangle_attention/cuda/h100_dh.py`: generic LN, cuBLAS projections, a mma.sync FlashAttention-2 core (`triattn_dh_sm90.cu`, heads of 32 / 64, gate in the epilogue) | 3.2-6.5x (L 256-768) |
| triangle attention, Chai-1 outputs | `untransposed_output=True` (ending node), `dual_output=True` (confidence head) | the same path, the update laid back as the variant has it | as above |
| OPM, RF3 projection biases | `OuterProductMean(projection_bias=True)` | the fused H100 OPM (weights only) plus the bias terms `(Rs A^T + Ls Bm^T) / cnt + c` (two small GEMMs, one CUDA pass) | 0.9-1.0x (1.1-1.4x eager) |
| OPM, Chai-1 groups | `OuterProductMean(groups=8)` | one projection GEMM, a group split, one batched GEMM over the MSA rows, the products laid out per pair row with the product LayerNorm (`kernels/outer_product_mean/cuda/gather.cu`), the out-projection | 1.1-1.2x |
| single attention gate bias (Chai-1) | `AttentionPairBias(gate_bias=1.0)` | the composed path (on H100 that is SDPA) with the constant on the gate logits | - |
| parallel Pairformer (Chai-1, RF3) | `PairformerBlock.from_components(..., parallel=True)` | every part on the block input; `pair + sum_i (y_i - pair)` from the parts' residual outputs in one CUDA pass (fp32 sums, one rounding) | - |

Not done: RF3 OPM is not faster than `torch.compile` of its statements (the fused OPM's GEMM is most of the time); the token / atom DiT
variants keep the pair bias of a block per call unless the pair tensor is unchanged (FoldForge recomputes the pair conditioning every
diffusion step, so a fold-level split of the diffusion head would remove that work); FoldForge wiring of these modules is separate.
