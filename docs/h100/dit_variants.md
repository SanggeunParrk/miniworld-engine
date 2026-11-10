# DiT variants on H100 (sm90): Boltz-2, RoseTTAFold3, Chai-1

Scope: the atom blocks `AtomDiTBoltz2`, `AtomDiTRoseTTAFold3`, `AtomDiTChai1` (`AtomDiT(model=...)` with `boltz2`, `rosettafold3`,
`chai1`): their `"cuda"` backend is the mixin `VariantAtomCuda` (`modules/dit_variant_cuda.py`) on `kernels/dit_variants`. The token
blocks `TokenDiTBoltz2`, `TokenDiTRoseTTAFold3`, `TokenDiTChai1` are NOT here any more: on H100 they run the one token DiT algorithm
(TokenDiTProtenix's `"cuda"` blocks with their model options, [token_dit.md](token_dit.md#model-options-one-algorithm)); their A100 paths
are `VariantTokenCuda`'s. The other backends are the module's own (`"pytorch"`, `"triton"`; `"anthropic"` for `AtomDiTBoltz2`).
Requirement rows: `atom_single`, N 1024-8192, 128 wide x 4 heads, cond 128, pair 16, `key_window` `pad` (default) or `slide`; dtypes
`float32` and `master`; modes `eval` and `train`. Measurement and judging: [../measure.md](../measure.md); conventions:
[../modules.md](../modules.md), [../kernels.md](../kernels.md).

What differs from the AF3 blocks:

| class | AdaLN | attention | transition | schedule |
|---|---|---|---|---|
| Boltz-2 | sigmoid form | gated query; one AdaLN for q and k / v | up-gate `silu(W_a h) (W_b h) (W_u h)` | sequential |
| RoseTTAFold3 | sigmoid form | bias-free query, affine LayerNorm of the flat q / k over all heads | SwiGLU | parallel `a + dA(a) + dT(a)` |
| Chai-1 | identity `(1 + W_s s) LN(a) + W_b s`, eps 0.1 | no gating query, no out projection, same-conformer key mask (`uid`) | SwiGLU | parallel |

## How a module call reaches the kernels

The kernels were written against a stack of variant blocks; `_Stack` is a read-only one-block view onto the module's own submodules
(nothing copied). It bridges three naming differences: `expand` is `cat(expand_a, expand_b)`, a bias-free query is a zero bias, and an
AdaLN's `to_bias` is the kernels' `to_shift`. The path is the first whose `serves` holds:

| call | atom |
|---|---|
| bf16, no gradient | `interface.atom_inference` |
| fp32, no gradient | `cuda/infer32.atom_inference` |
| a gradient (bf16 or fp32) | `cuda/train.atom_stack_train` |

The kernels take one conditioning row set shared by the samples; a `cond` that differs per sample runs one kernel call per sample.

## Kernels

| kernel | role | what it does |
|---|---|---|
| `dit_variants_ln_mod_rows_fwd_sm90` | forward_infer | `rows.cu ln_mod_rows`: `LN(x)` once, then up to three `xhat * f(scale) + shift` outputs (f = sigmoid or identity); copies the step input into the fp32 residual |
| `dit_variants_res_rows_fwd_sm90` | forward_infer | `res_rows`: `x += sigmoid(ga) ya [+ sigmoid(gt) yt]`, then the next AdaLN or the final output in bf16 |
| `dit_variants_qk_ln_rows_fwd_sm90` | forward_infer | RF3 q / k LayerNorm over all columns, in place, logit scale folded into q's affine |
| `dit_variants_gate_rows_fwd_sm90` | forward_infer | `o <- o * sigmoid(g)` in place |
| `dit_variants_glu_rows_fwd_sm90` | forward_infer | `silu(g) v [u]` (Boltz-2's up-gate) |
| `dit_variants_ln_mod_fwd_sm90`, `_qkv_prep_fwd_sm90`, `_res_ln_fwd_sm90`, `_res_out_fwd_sm90`, `_gate_fwd_sm90`, `_glu_fwd_sm90` | forward | `train_rows.cu`: the training forward rows (also the fp32 inference rows) |
| `dit_variants_res_bwd_sm90`, `_glu_bwd_sm90`, `_res_adaln_bwd_sm90`, `_gate_o_bwd_sm90`, `_qkv_bwd_sm90`, `_adaln_bwd_sm90` | backward | the training backward rows |
| `dit_variants_edge_attn_fwd_sm90` | forward | `edge.cu`: the slid key window's five edge windows (fp32), plus their bias and backward entry points |
| `augmented_attention_local_pair_bias_fwd_sm90`, `augmented_attention_local_attn_*_sm90`, `augmented_attention_local_*bias_bwd*_sm90` | forward / backward | atom window attention (`attn_fwd_mma` / `attn_bwd_mma`, bf16 and TF32) and the window pair bias (`pair_bias_mma`) of the local atom DiT |

## Atom stack

128 wide x 4 heads of 32, query windows of 32 atoms, 128 keys each. Conditioning tables per atom (`[N, nb 6 128]` modulation: q, k,
transition; `[N, nb 2 128]` gates). Per block the same rows at D = 128, cuBLAS, the local DiT's window attention (`attn_fwd_mma`;
N padded to a multiple of 64 for its 64-query tiles) and `pair_bias_mma` (per window, log2 units, -inf at keys outside `[0, N)` or
masked; Chai-1's other-conformer keys -inf). `key_window="slide"`: windows 0, 1 and the last three real ones are the only ones whose slid
window differs from the clipped window the kernel reads; their results are recomputed over their own keys -- in torch for bf16
inference, with `edge.cu` for fp32 inference and training (`edge_bias_fwd/bwd`, `edge_attn_fwd`, `edge_bwd_compute` zeroes their dO / D
rows before the main backward, `edge_bwd_apply` adds their dq / dk / dv after it).

## Numerics

bf16 GEMM and attention operands, fp32 accumulation; the residual stream between the two halves of a sequential block, LayerNorm
statistics and every reduction are fp32. fp32: TF32 GEMMs and cores (operands rounded to nearest TF32 in the row kernels). Reductions
over the samples are fixed order: one partial row per (token, split of the samples), the splits summed on the host; no atomics.

## Launch configs

No launch-config table: the CUDA rows are one block per row (a thread per 4 columns), the cores are fixed builds. The `"triton"`
backend's cores read `kernels/augmented_attention/configs/h100_sm90.json` (token 16 x 48, atom 4 x 32 rows).

## Limits (`_supports_cuda`)

| call | result |
|---|---|
| not CUDA / not sm_90 | refused |
| autocast on (every `master` requirement row) | refused: the kernels fix their own precision |
| single / cond / pair not all bf16 or all fp32; `single` not `[A, 1, L, d]` | refused |
| AdaLN epsilons differ within the block, or a sigmoid-form cond LayerNorm eps != 1e-5 | refused |
| atom: `key_window` other than `pad` / `slide`, not 128 x 4 heads / pair 16, pair not `[1, N / 32, 32, 128, 16]`, mask not bool `[1, N]` | refused |
| atom fp32 inference or any gradient: N not a multiple of 64 (bf16 inference: of 32) | refused |

## History

Old note (git HEAD): `docs/gpus/h100/model_variants/model_variants.md`. Its rows for TriangleAttention variants, OuterProductMean
biases / groups, `AttentionPairBias(gate_bias)` and the parallel Pairformer are not DiT modules and are not covered here. Historical
(from the 2026-10-22 note, not re-measured under the current measurer): token stacks 2.8-4.1x and atom stacks 1.6x against compiled
torch (CUDA graph, L 384 / 768 and 8192 atoms).
