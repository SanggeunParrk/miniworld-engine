# Token DiT on H100 (sm90)

Scope: `TokenDiTProtenix` (AF3 Alg. 23: AdaLN + `AugmentedAttentionPairBias` + conditioned SwiGLU transition; selected by
`TokenDiT(model=...)` for `af3`, `openfold3`, `openfold3_preview2`, `protenix`, `protenix_v2`, `opendde`, `intellifold2`, `esmfold2`) and
`TokenDiTMiniWorld` (the same block with the per-head q / k RMSNorm, `miniworld`). Backends: `"pytorch"` (the reference), `"triton"`
(the block's own statements with only the attention core on the Triton pair-bias kernel), `"cuda"` (the fused H100 blocks below) and,
for every class but `TokenDiTMiniWorld`, the baselines `"anthropic"` (each model's release attention core, eval only) and `"cueq"`
(the pair-bias producer on cuEquivariance's `pair_repr_to_attention_bias`, the rest the module's statements). `TokenDiTBoltz2`,
`TokenDiTRoseTTAFold3` and `TokenDiTChai1` run the SAME `"cuda"` blocks with their model options
([Model options](#model-options-one-algorithm) below; [dit_variants.md](dit_variants.md)). The bias-only block (a different algorithm:
no query / key) is in [bias_only_dit.md](bias_only_dit.md); the atom DiT blocks in [local_atom_dit.md](local_atom_dit.md). A100
(sm_80): `kernels/integrations/augattn_sm80.py` + `conditioned_transition_sm80.py` (variants: `VariantTokenCuda`). Requirement rows
(`requirements_module.csv`): `token_single`, L 128-768, d_single 768 (1024 engine-own), 16 heads (12 / 24 engine-own), d_cond 384
(ESMFold2: 768), d_pair 128 / 256 (Protenix-v2) / 512 (IntelliFold2), n = 2; dtypes `float32` and `master` (bf16 activations, fp32
parameters, bf16 autocast); modes `eval` and `train`. How accuracy and speed are judged: [../measure.md](../measure.md); module and
backend conventions: [../modules.md](../modules.md), [../kernels.md](../kernels.md).

## Model options (one algorithm)

Every token DiT class but the bias-only one is ONE implementation: the inference block (`kernels/integrations/token_dit.py` ->
`conditioned_transition.cuda.token_inference`) and the training block (`kernels/integrations/token_dit_train.py`), the same row
kernels, packs, pair-bias hoist and attention cores. A model's differences are options read off the block's own modules
(`token_dit.options`), and its parameters are mapped onto the canonical leaves (`token_dit.leaves`; a part the model lacks is a cached
constant leaf):

| option | models | inference | training |
|---|---|---|---|
| parallel `a + dA(a) + dT(a)` | RoseTTAFold3, Chai-1 | `adaln_in_rows` writes both AdaLNs of the one LN(x); `resgate_out_rows` adds both updates | `adaln_a` writes xt (`OPT_PAR`), `res_adaln_b` skips the AdaLN of x1, `adaln_a_bwd` takes dxa and dxt into one LN backward |
| identity AdaLN `(1 + W_s c) LN(a) + W_b c`, eps 0.1 / 1e-5 | Chai-1 | tables from the raw c (ln_cond leaf 1, scale bias leaf 1: the "+ 1"); rows without the sigmoid | `OPT_IDENT` rows; the cond gradient is dchat + dcg (no cond LN) |
| no gating query | Chai-1 | gate columns +1e4 (sigmoid = 1); the q\|k(\|v) GEMM writes the other columns only | gate rows zero, gate bias +1e4 (its gradient is 0) |
| up-gate transition `silu(W_a h) (W_b h) (W_u h)` | Boltz-2 | cuBLAS `[a \| b \| u]` + `swiglu_rows(up)` | `[W_a; W_b; W_u]`, `swiglu_bwd(up)` |
| flat q / k LayerNorm with affine | RoseTTAFold3 | `qk_ln_rows` (logit scale folded into q's affine) | `qknorm` / `qknorm_bwd` mode 2 (`dwq dwk dbq dbk` partials) |
| bias-free query | RoseTTAFold3 | zero to_query.bias leaf | the same |
| per-head q / k RMSNorm | MiniWorld | `qknorm_rows` | `qknorm` mode 1 |
| fp32 q / k / v^T rounded to TF32 (nearest) before the TF32 core | Boltz-2, RoseTTAFold3, Chai-1 | `rna_rows` (fused into `qk_ln_rows` for RF3) | -- |
| per-sample conditioning | all | `[A L, *]` tables | `[A L, *]` rows |
| fp32 (TF32) / bf16 / master, eval / train | all | | |

The default options are Protenix's (bit-identical to the pre-unification block). The options other than the defaults are H100 kernels;
on B200 the block serves the defaults only. The training block packs a model with options through its torch pack and finishes the weight
gradients in torch (the fused `train_pack90` / `train_final90` kernels map the canonical leaves of the default block).

## Kernels

Names from `requirements_kernel.csv`; the `cond_transition_*` rows belong to the `conditioned_transition` family, the attention cores to
`augmented_attention`. GEMMs not listed are cuBLAS.

| kernel | role | what it does |
|---|---|---|
| `cond_transition_token_pack_sm90` | forward_infer | packs the block's weights once per weight version: q pre-scaled by log2 e / sqrt(head dim), `Wf = to_bias.weight * ln_pair.weight * log2 e`, the six conditioning projections as two GEMM operands |
| `cond_transition_token_cond_rows_sm90` | forward_infer | conditioning LayerNorm and its GEMM-dtype copy, one warp-per-row pass |
| `cond_transition_token_adaln_in_rows_sm90` | forward_infer | copies the step input into the fp32 residual and writes `AdaLN(x)`, one pass |
| `cond_transition_token_qknorm_rows_sm90` | forward_infer | per-head q / k RMSNorm in place on the q / k columns of the q\|k\|v\|g GEMM output |
| `cond_transition_token_qk_ln_rows_sm90` | forward_infer | RoseTTAFold3's flat q / k LayerNorm with affine in place (optionally rounded to TF32) |
| `cond_transition_token_rna_rows_sm90` | forward_infer | fp32 rows rounded to TF32 to nearest in place (q \| k, v^T) before the TF32 core, which truncates |
| `cond_transition_token_resgate_adaln_rows_sm90` | forward_infer | `x += sigmoid(gate) y` (fp32) then the transition's AdaLN; optional RNA rounding to TF32 |
| `cond_transition_token_swiglu_rows_sm90` | forward | `silu(a) b` where the fused expand GEMM is not used |
| `cond_transition_token_resgate_out_rows_sm90` | forward_infer | the last residual, written in the output dtype |
| `cond_transition_token_layernorm_rows_sm90` | forward_infer | pair LayerNorm rows (no affine; optional key permutation) |
| `cond_transition_gemm_swiglu_sm90` | forward_infer | `gemm_swiglu_sm90.cu`: expand GEMM with SwiGLU in the epilogue, `[a \| b]` never stored |
| `augmented_attention_apb_pair_bias_fwd_sm90` | forward | bf16 pair bias: LN(pair) . Wf on mma.sync, head-major, masked keys -inf (shared with AttentionPairBias) |
| `bias_only_dit_f32_pair_bias_tc` | forward | fp32 pair bias at d_pair 128 (`pair_bias_tok`): LN + projection + key permutation + mask in one pass |
| `augmented_attention_pair_bias_fwd_fp32_sm90` | forward_infer | `pair_rows.cu` `bias32_k`: the same for the 256-wide fp32 pair at 16 heads |
| `augmented_attention_fwd_sm90` | forward | `attn_fwd.cu`: bf16 wgmma core (gated inference `attn_inf`; training `attn_fwd_og`) |
| `augmented_attention_fwd_tf32_sm90` | forward | `attn_tf32.cu`: the fp32 core on TF32 wgmma (gated inference and training forward) |
| `augmented_attention_bwd_dqb_sm90`, `augmented_attention_bwd_dq_sm90`, `augmented_attention_bwd_dkv_sm90` | backward | bf16 backward: dQ + dbias (`attn_dqb`), dQ alone (`attn_dq`), dK / dV (`attn_dkv`) |
| `augmented_attention_bwd_dqb_tf32_sm90`, `augmented_attention_bwd_dkv_tf32_sm90` | backward | the TF32 backward (`attn_tf32_dqb`, `attn_tf32_dkv`) |
| `cond_transition_token_train_*`, `cond_transition_token_transpose_sm90` | forward_train / backward | `token_dit_train_rows.cu`: the training row passes (cond prep, AdaLN, QK-norm, pair LN / bias, key mask, residual + gate + AdaLN, and their backwards; weight pack, final weight-gradient reduce, dbias reduce, tiled transpose) |
| `cond_transition_caller_grad_sum_sm90`, `cond_transition_caller_grad_add_sm90` | backward | the caller-gradient hooks the H100 training block installs on its inputs |
| `augmented_attention_fwd`, `_bwd_pre`, `_bwd_split`, `_bwd_reduce` | forward / backward | the Triton core (`triton/main.py`) of the `"triton"` backend |

## Inference block (`kernels/integrations/token_dit.py` -> `conditioned_transition.cuda.token_inference`)

No autograd, one opaque op per block call (`token_dit_h100_infer`), autocast off: the step's precision is `single`'s dtype.

1. **Weight pack**, cached per (pointer, version) of every weight and scoped to the CUDA-graph capture (`kernels/_capture`): a capture
   records the pack once; each replay re-packs the weights as they are then.
2. **Pair bias**, hoisted: `bias [H, L, L]` in log2 units, masked keys -1e30, built once per (pair, mask, Wf) and cached through
   `_capture.lookup_inputs` (the pair is held by a weak storage reference, so a freed pair evicts its entry). bf16: the APB pair kernel
   (d_pair 128) or CUDA LayerNorm rows + one cuBLAS GEMM (d_pair 256 / 512). fp32: written in the TF32 core's key order (below), by
   `pair_bias_tok` (128) or `pair_bias_rows32` (256, 16 heads), else LayerNorm rows + GEMM + a torch masked fill of the keys in the core's order. An `extra_pair_bias`
   [L, L] is added once here (times log2 e; key-permuted in fp32).
3. **Conditioning**: `cond_rows` (d_cond 384 or 768; the weight pack is built for the weights' width), then two GEMMs: `g1 [P, 4, D]` (AdaLN scale / shift of both halves) and `g2 [P, 2, D]` (the two
   output gates). P = L rows when the samples share the conditioning (sample axis 1 or stride 0), A L rows otherwise.
4. **Per block**: `adaln_in_rows` -> q\|k\|v\|g GEMM (bf16) or q\|k\|g GEMM + a second GEMM for v^T (fp32) -> `qknorm_rows`
   (TokenDiTMiniWorld; the logit scale is folded into the q-norm weight) -> gated core (writes `sigmoid(g) o` over q) -> out GEMM ->
   `resgate_adaln_rows` -> expand + SwiGLU -> squeeze GEMM -> `resgate_out_rows`. The residual stream is fp32.
5. **Expand + SwiGLU**: bf16 always `gemm_swiglu_sm90` (128 x 256 tiles whose B rows are 128 of Wa then the same 128 of Wb, wgmma
   m64n256, SwiGLU in the epilogue). fp32: the TF32 build of the same kernel at M >= 768 rows, with `xa` RNA-rounded to TF32 by its row
   kernel and `[Wa; Wb]` rounded once per pack (the TMA-fed operands are otherwise truncated); below 768 rows cuBLAS + `swiglu_rows`.
6. **B > 1 trunk_seeds** (bf16 only): one pair per trunk_seed, biases `[H, B, L, L]`, the core reads group `s % B` for sample `s`; the
   key mask must be one `[L]` row expanded over B (stride 0).

## Training block (`kernels/integrations/token_dit_train.py`)

One `autograd.Function` per block whose forward and backward are each one opaque op (torch.compile keeps them as single nodes); autocast
off inside, precision from the tensors: bf16 operands when `single` (or `compute_dtype`) is bf16, else fp32 operands on TF32 tensor cores.
L is padded to a multiple of 128 inside (zero rows, padded keys masked) and the result sliced; any A.

- Forward: cond LN + two GEMMs (every in-block LayerNorm of cond shares its statistics, so its weight folds into the projection) ->
  `adaln_a` -> q\|k\|v\|g GEMM -> QK-norm rows -> pair bias -> core (writes `og = sigmoid(g) o`; O is not kept) -> out GEMM ->
  `res_adaln_b` -> expand GEMM + `swiglu_rows` (`a | b` saved) -> squeeze GEMM -> `res_c`.
- Pair bias: bf16 uses the one-pass pair kernel (LN(pair) Wf^T, `extra` added in the same pass, LN(pair) not saved) and its one-pass
  backward; fp32 uses `pair_ln` + GEMM in log2 units, key-permuted for the TF32 core.
- bf16 attention backward: with A a multiple of 3, `attn_dqb` (the CTA's three warpgroups are three samples of one query tile; their dS
  tiles are summed in shared memory and added with one TMA bulk reduce) into `C` chunk buffers (C = largest divisor of A / 3 up to 4)
  summed in order, then `attn_dkv -DDBIAS=0`; otherwise `attn_dq` + `attn_dkv -DDB_PER_SAMPLE=1` and a fixed-order reduce. The
  backward takes `D = rowsum(dO O) = rowsum(d og * og)`.
- fp32 attention backward: `attn_tf32_dqb` (per-sample dbias buffer) + `attn_tf32_dkv`; q, k, dO are transposed by a tiled CUDA
  transpose for the products that contract over tokens; bias / LSE / D are passed in the permuted orders.
- Weight gradients: cuBLAS with fp32 accumulation, reduced by the `final` kernel in fixed order.

## Attention core design

- **bf16 (`attn_fwd.cu` and backward)**: q pre-scaled by log2 e / sqrt(head dim), bias seeded into the score accumulator times log2 e,
  so every exponential is one `ex2`; the running max moves only when a block exceeds it by 8 (log2 units, `LAZY`). 64-key blocks
  through a 3-stage TMA / mbarrier ring, two consumer warpgroups (128 query rows) per CTA, q held in registers and its tile parked in a
  bias slot. Built per head layout (`DHP` 16 / 32 / 48 / 64, `ROW_WIDTH`). Gated inference reads q / k / v / g as column views of the
  GEMM output through TMA and writes `sigmoid(g) o` over its own q tile. Stages read with ldmatrix are released after
  `fence.proxy.async` (the generic-proxy read is not ordered by the mbarrier release alone).
- **fp32 (`attn_tf32.cu`)**: TF32 wgmma reads both operands K-major only. Hence v is passed as v^T `[D, tokens]`; P feeds the A
  operand from registers by loading K rows permuted within each 8-row group (shared-memory row `2j + i` holds key `4i + j`, a 4-D TMA
  box), so S column `2t + i` is key `t + 4i` and the accumulator is the A fragment with no shuffle; bias and key mask are laid out in
  that order (`conditioned_transition.cuda.key_perm`). Head dim 48 is two 128-B-swizzled chunks (32 + 16). 64-key blocks, 2-stage ring.

## Numerics

- bf16: operands rounded to bf16 (q after its scale, the bias after log2 e), P and dS bf16 into the tensor core, every sum fp32,
  residual fp32, output in the input dtype.
- fp32: GEMMs on TF32 tensor cores (forced, whatever `allow_tf32`); the TF32 cores truncate their TMA-fed operands to TF32, which is
  the largest error term of the core; softmax, statistics and residual fp32.
- Masked keys: -1e30 / -inf in the bias. A fully masked key row is outside the `"anthropic"` range; the reference gives 0 there.
- Every attention kernel writes each output element from one writer or reduces in a fixed order: bitwise repeatable run to run.

## Launch configs

The CUDA paths take no launch-config table: tiles, stages and warpgroups are build defines fixed per head layout. The `"triton"`
backend's core reads `kernels/augmented_attention/configs/h100_sm90.json` (`augmented_attention_fwd_triton`, `_bwd_pre_triton`,
`_bwd_split_triton`, `_bwd_reduce_triton`; keys by dtype, `H`, `HEAD_DIM` and the L bucket; the forward has rows for 16 x 48 / 24 /
32 / 64, 12 x 32 / 64, 24 x 16 / 32, 8 x 48 and 4 x 32), rank 1 by default; other layouts take the kernel's default.

## Limits (`_supports_cuda`)

| call | result |
|---|---|
| not CUDA / not sm_90 (A100: the sm_80 route above) | refused |
| key mask that differs across samples (`[A, B, L]` not stride-0) | refused |
| block outside `token_dit.options` (mixed AdaLN forms / epsilons, sigmoid-form conditioning LayerNorm eps != 1e-5, flat q / k LayerNorms of different eps or without bias) or pair LayerNorm eps != 1e-5 | refused |
| training: outside bf16 / fp32, `[A, 1, L, d]`, `(heads, d)` in 16 x 768, 24 x 768, 12 x 768, 16 x 1024, cond 384 / 768, pair 128 / 256 / 512, n = 2, key mask `[1, L]` | refused |
| inference: the same, plus L a multiple of 128; B > 1 only bf16 with one pair per trunk_seed and a stride-0 mask; fp32 needs fp32 parameters and a bool `[1, L]` mask | refused |
| `"anthropic"`: grad, `compute_dtype`, QK-norm, head dim > 64, a fully masked row (plus each model's own: Boltz-2 head dim 48 only and one mask for the samples; RF3 / Chai-1 no key mask) | refused |
| `"cueq"`: `compute_dtype`, QK-norm, L at or below cuEquivariance's fallback threshold | refused |
| `"triton"`: head dim > 64, dtype other than bf16 / fp32 | refused |

## History

Old notes: `git show 65f37dc3c^:docs/gpus/h100/token_dit/token_dit.md`. Historical (from the 2026-10-15 note, not re-measured under the current
measurer; H100 80GB HBM3, CUDA graph, ms per block, 16 x 48): inference bf16 A = 5 shared cond L384 / L768 0.102 / 0.179 against
PyTorch compiled 0.282 / 0.666; training bf16 A = 48 3.32 / 7.66 against 5.63 / 16.97; training fp32 7.25 / 16.41 against 9.12 / 24.10.
Rejected there: an FA2-layout mma.sync TF32 core (re-read K / V per 64-row CTA), a one-pass warp-per-row pair-bias kernel for
training (latency-bound shuffles), per-warpgroup dbias reduces (3x the L2 reduce traffic), a SwiGLU GEMM that also stores `[a | b]`
for training (the epilogue serialises).
