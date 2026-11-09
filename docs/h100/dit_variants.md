# DiT variants on H100 (sm90): Boltz-2, RoseTTAFold3, Chai-1

Scope: the token blocks `TokenDiTBoltz2`, `TokenDiTRoseTTAFold3`, `TokenDiTChai1` and the atom blocks `AtomDiTBoltz2`,
`AtomDiTRoseTTAFold3`, `AtomDiTChai1` (`TokenDiT(model=...)` / `AtomDiT(model=...)` with `boltz2`, `rosettafold3`, `chai1`). Their
`"cuda"` backend is the mixin `VariantTokenCuda` / `VariantAtomCuda` (`modules/dit_variant_cuda.py`) on `kernels/dit_variants`; the
other backends are the module's own (`"pytorch"`, `"triton"`; `"anthropic"` for the three token classes and `AtomDiTBoltz2`).
Requirement rows: token -- `token_single`, L 128-768, 768 wide x 16 heads, n = 2, Boltz-2 d_cond 768 / pair 128, RF3 cond 384 / pair
128, Chai-1 cond 384 / pair 256 with `adaln_eps` 0.1 (default) or 1e-5; atom -- `atom_single`, N 1024-8192, 128 wide x 4 heads, cond 128,
pair 16, `key_window` `pad` (default) or `slide`; dtypes `float32` and `master`; modes `eval` and `train`. Measurement and judging:
[../measure.md](../measure.md); conventions: [../modules.md](../modules.md), [../kernels.md](../kernels.md).

What differs from the AF3 blocks ([token_dit.md](token_dit.md)):

| class | AdaLN | attention | transition | schedule |
|---|---|---|---|---|
| Boltz-2 | sigmoid form | gated query; atom: one AdaLN for q and k / v | up-gate `silu(W_a h) (W_b h) (W_u h)` | sequential |
| RoseTTAFold3 | sigmoid form | bias-free query, affine LayerNorm of the flat q / k over all heads | SwiGLU | parallel `a + dA(a) + dT(a)` |
| Chai-1 | identity `(1 + W_s s) LN(a) + W_b s`, eps 0.1 | no gating query; atom: no out projection, same-conformer key mask (`uid`) | SwiGLU | parallel |

## How a module call reaches the kernels

The kernels were written against a stack of variant blocks; `_Stack` is a read-only one-block view onto the module's own submodules
(nothing copied). It bridges three naming differences: `expand` is `cat(expand_a, expand_b)`, a bias-free query is a zero bias, and an
AdaLN's `to_bias` is the kernels' `to_shift`. The path is the first whose `serves` holds:

| call | token | atom |
|---|---|---|
| bf16, no gradient | `interface.inference` | `interface.atom_inference` |
| fp32, no gradient | `cuda/infer32.inference` | `cuda/infer32.atom_inference` |
| a gradient (bf16 or fp32) | `cuda/train.stack_train` | `cuda/train.atom_stack_train` |

The kernels take one conditioning row set shared by the samples; a `cond` that differs per sample runs one kernel call per sample.

## Kernels

| kernel | role | what it does |
|---|---|---|
| `dit_variants_ln_mod_rows_fwd_sm90` | forward_infer | `rows.cu ln_mod_rows`: `LN(x)` once, then up to three `xhat * f(scale) + shift` outputs (f = sigmoid or identity); copies the step input into the fp32 residual |
| `dit_variants_res_rows_fwd_sm90` | forward_infer | `res_rows`: `x += sigmoid(ga) ya [+ sigmoid(gt) yt]`, then the next AdaLN or the final output in bf16 |
| `dit_variants_qk_ln_rows_fwd_sm90` | forward_infer | RF3 q / k LayerNorm over all columns, in place, logit scale folded into q's affine |
| `dit_variants_gate_rows_fwd_sm90` | forward_infer | atom `o <- o * sigmoid(g)` in place |
| `dit_variants_glu_rows_fwd_sm90` | forward_infer | `silu(g) v [u]` (Boltz-2's up-gate) |
| `dit_variants_ln_mod_fwd_sm90`, `_qkv_prep_fwd_sm90`, `_res_ln_fwd_sm90`, `_res_out_fwd_sm90`, `_gate_fwd_sm90`, `_glu_fwd_sm90`, `_pair_ln_fwd_sm90` | forward | `train_rows.cu`: the training forward rows (also the fp32 inference rows) and the pair LayerNorm |
| `dit_variants_res_bwd_sm90`, `_glu_bwd_sm90`, `_res_adaln_bwd_sm90`, `_gate_o_bwd_sm90`, `_qkv_bwd_sm90`, `_adaln_bwd_sm90`, `_pair_ln_bwd_sm90` | backward | the training backward rows |
| `dit_variants_edge_attn_fwd_sm90` | forward | `edge.cu`: the slid key window's five edge windows (fp32), plus their bias and backward entry points |
| `augmented_attention_fwd_sm90`, `_fwd_tf32_sm90`, `_bwd_dqb_sm90`, `_bwd_dq_sm90`, `_bwd_dkv_sm90`, `_bwd_dqb_tf32_sm90`, `_bwd_dkv_tf32_sm90` | forward / backward | token attention cores (see [token_dit.md](token_dit.md)) |
| `augmented_attention_local_pair_bias_fwd_sm90`, `augmented_attention_local_attn_*_sm90`, `augmented_attention_local_*bias_bwd*_sm90` | forward / backward | atom window attention (`attn_fwd_mma` / `attn_bwd_mma`, bf16 and TF32) and the window pair bias (`pair_bias_mma`) of the local atom DiT |
| `cond_transition_token_layernorm_rows_sm90`, `cond_transition_token_transpose_sm90` | forward | shared LayerNorm rows (pair rows of the bf16 inference bias) and helpers attributed to these classes |

## Token stack

- **bf16 inference** (`interface.py`): per call two conditioning GEMMs for every block (`[L, nb 4 D]` AdaLN scale / shift with the
  sigmoid form's weight-only cond LayerNorm folded into the weights, Chai-1's `+ 1` into the bias; `[L, nb 2 D]` output gates); the pair
  bias of every block from one LayerNorm of the pair rows and one GEMM, `[nb 16, L, L]` bf16 in log2 units, masked keys -inf, cached per
  pair / mask (`_capture.lookup_inputs`). Per block: `ln_mod_rows` -> q\|k\|v(\|g) GEMM (logit scale folded into `W_q` / `b_q` or the
  kq-norm affine) -> `qk_ln_rows` (RF3) -> the AF3 gated core `gated_inference` (a block without a gating query gets gate logits of
  +1e4, `sigmoid = 1`) -> out GEMM -> parallel: expand GEMM, `glu_rows`, `res_rows` with both terms; sequential: `res_rows` + AdaLN,
  expand, `glu_rows`, squeeze, `res_rows`.
- **fp32 inference** (`infer32.py`): the training rows without saved tensors, TF32 GEMMs, weights folded once per parameter version,
  the bias key-permuted for the TF32 core, v^T from its own GEMM, `tf32_gated_inference`.
- **Training** (`train.py`): conditioning tables and pair biases are made by ordinary torch ops outside the block Function (autograd
  chains their gradients): one pair LayerNorm (`pair_ln_fwd` / `pair_ln_bwd` as an autograd Function) and one stacked GEMM
  `[nb H, C] x [C, L L]` with log2 e folded into the weights (Chai-1's LN offset is dropped: it only shifts a row's logits). Per block one
  `_TokenBlock` Function: `ln_mod_fwd` -> q\|k\|v\|g GEMM -> `qkv_prep` -> core (bf16 `attn_fwd` with the gate epilogue + `attn_dqb` /
  `attn_dkv`; fp32 the TF32 core with v^T and the key-permuted layouts by `transpose2d`) -> out GEMM -> `res_ln_fwd` / `res_out_fwd`
  -> expand GEMM -> `glu_fwd` -> squeeze GEMM. Backward: `res_bwd`, `glu_bwd`, `res_adaln_bwd`, `gate_o_bwd` (do, dg and
  `D = rowsum(dog og)` from the saved gated output), attention backward, `qkv_bwd`, `adaln_bwd`; weight gradients cuBLAS.

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
| token: `compute_dtype`, `extra_pair_bias`, not 768 x 16 heads / n = 2 / pair 128 or 256, L not a multiple of 128, a key mask that differs across samples | refused |
| atom: `key_window` other than `pad` / `slide`, not 128 x 4 heads / pair 16, pair not `[1, N / 32, 32, 128, 16]`, mask not bool `[1, N]` | refused |
| atom fp32 inference or any gradient: N not a multiple of 64 (bf16 inference: of 32) | refused |

## History

Old note (git HEAD): `docs/gpus/h100/model_variants/model_variants.md`. Its rows for TriangleAttention variants, OuterProductMean
biases / groups, `AttentionPairBias(gate_bias)` and the parallel Pairformer are not DiT modules and are not covered here. Historical
(from the 2026-10-22 note, not re-measured under the current measurer): token stacks 2.8-4.1x and atom stacks 1.6x against compiled
torch (CUDA graph, L 384 / 768 and 8192 atoms).
