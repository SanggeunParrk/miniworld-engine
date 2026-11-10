# Bias-only token DiT on H100 (sm90)

Scope: `TokenDiTMiniWorldBiasOnly` (`TokenDiT(model="miniworld_bias_only")`): AF3 Alg. 23 with the attention's query-key half removed --
`BiasOnlyAttention` computes `a + sigmoid(to_scale(s)) * to_out(sigmoid(g) * softmax(bias) v)` with `bias = to_bias(LN(pair))` -- then
the conditioned SwiGLU transition. Backends: `"pytorch"` (the reference) and `"cuda"` (below); there is no `"triton"` and no
`"anthropic"` backend (no release has this block). `pair_bias_all(blocks, pair)` hoists every block's bias from one LayerNorm and
one GEMM over the pair (PyTorch on H100; the fused hoist kernels are B200-only). Requirement rows: `token_single`, L 128-768, d_single 768,
16 heads x 48, n = 2, d_cond 384 (MiniWorld) and the engine-own d_cond 768 / d_pair 256 rows; dtypes `float32` and `master`; modes
`eval` (5 samples) and `train` (1 and 48 samples). Measurement and judging: [../measure.md](../measure.md); conventions:
[../modules.md](../modules.md), [../kernels.md](../kernels.md).

With no query and no key, `P = softmax(bias)` depends neither on the sample nor on the solver step, only on the pair. The H100 path is
therefore a different algorithm from the token DiT's: P is made once per pair, and the attention of every sample is a GEMM per head
(`P_h [L, L] x v_h [L, dh]`) with the gate in its epilogue -- no online softmax.

## Kernels

| kernel | role | what it does |
|---|---|---|
| `bias_only_dit_pv_gate_sm90` | forward | `pvx` (`pv_gate_inf.cu`, bf16 wgmma): `sigmoid(g) * (P v)` per head and sample group; without the gate it is `dV = P^T dO` |
| `bias_only_dit_pv_gate_tf32_sm90` | forward | `pvx32`: the same on TF32 wgmma for fp32 (v passed as v^T) |
| `bias_only_dit_dpb_sm90` | backward | `dpb2`: `dbias = P o (sum_a dO_a v_a^T - D)`, 128 x TN output tile per CTA (bf16, L >= 384) |
| `bias_only_dit_dpb_pipe_sm90` | backward | `dpb_pipe`: the 64 x 64-tile form of the same (L < 384) |
| `bias_only_dit_ln_rows_sm90`, `_adaln_in_rows_sm90`, `_resgate_adaln_rows_sm90`, `_resgate_out_rows_sm90`, `_swiglu_rows_sm90` | forward | `bias_only_dit_rows.cu`: conditioning LN, input AdaLN into the fp32 residual, residual + gate + AdaLN, last residual in the output dtype, SwiGLU |
| `bias_only_dit_softmax_rows_sm90`, `bias_only_dit_softmax_t_sm90` | forward | row softmax of the bias with the key mask folded in (inference); the same writing P and P^T (training) |
| `bias_only_dit_gate_bwd_rows_sm90` | backward | `do = da sigmoid(g)`, `dg`, `D = sum da a` per row and head from the gated output alone |
| `bias_only_dit_cond_tables_gemm_sm90` | forward_infer | `bo_gemm_sm90.cu cond_tables`: both conditioning-table GEMMs as one wgmma launch (up to 384 table rows) |
| `bias_only_dit_train_*_sm90` | forward_train / backward | `bias_only_dit_train_rows.cu`: weight pack, cond LN, AdaLN, pair bias (mma.sync), residual passes and their backwards, unfold, finalize |
| `bias_only_dit_f32_*` | forward / backward | `bias_only_dit_f32_rows.cu`: the fp32 rows, the fp32 pair bias (exact FMA or TF32 mma.sync) and its backward, the fp32 softmax |
| `cond_transition_gemm_swiglu_sm90` | forward_infer | the token DiT's expand GEMM with the SwiGLU epilogue (bf16 inference) |

## Inference (`kernels/integrations/bias_only_dit.py` -> `bias_only_dit/cuda/runner.py`, `FusedBiasOnlyDiT`)

H100 runs the 12-launch step (`_step`). The three-kernel step of the same runner (`bo_front_*` / `bo_tail_*`) is tcgen05 (sm_100a) and
is never taken on H100.

1. **Hoist** (once per pair and mask, cached on the runner while the pair / mask tensors' pointer and version are unchanged, capture
   scoped): bf16 -- LayerNorm rows (`layernorm128_rows`, or the 256-wide row template), one cuBLAS GEMM for every block, `softmax_rows`
   in place: P `[nb H, L, L]` bf16. fp32 -- `pair_bias_cuda` per block at d_pair 128 (LayerNorm and the H projections on the FMA pipe, exact fp32,
   LN(pair) never stored), `pair_bias_tc_cuda` at 256 (the same pass with the products on TF32 mma, split fp32: 3 terms; the FMA form was
   ~1 ms of a 1.85 ms L768 step),
   then fp32 `softmax_rows`. A block given `bias=` makes P per call from it instead.
2. **Conditioning tables**: `ln_rows` (d_cond 384 or 768) then the two table GEMMs (bf16 up to 384 table rows: one `cond_tables` wgmma launch; otherwise two
   cuBLAS GEMMs). T = L rows for a shared conditioning, A L rows otherwise.
3. **Per block**: `adaln_in_rows` (the first residual pass, `resgate_adaln_rows_in`, reads the input itself: no fp32 copy of it) ->
   v\|g GEMM (fp32: v^T and g as two GEMMs, since TF32 wgmma takes K-major operands) -> `pvx` / `pvx32` -> out GEMM ->
   `resgate_adaln_rows` -> expand: bf16 `gemm_swiglu_sm90` unless its 128 x 256 tiles fill at most half a wave or just over one
   (then cuBLAS + `swiglu_rows`); fp32 cuBLAS + the fp32 SwiGLU rows -> squeeze GEMM -> `resgate_out_rows` (or the next block's AdaLN).

## Training (`kernels/integrations/bias_only_dit_train.py`)

One autograd Function per block (`_Block` bf16, `_Block32` fp32, `_BlockH` with a hoisted bias), forward and backward each one opaque op.

- bf16 forward: `cond_ln` + two GEMMs -> `adaln_a` -> v\|g GEMM -> pair bias (`pair_bias`: LN(pair) on chip, fp32 logits on H100) ->
  `softmax_t` (P and P^T, bf16) -> `pvx` -> out GEMM -> `res_adaln_b` -> expand GEMM + `swiglu_rows` (`a | b` saved) -> squeeze GEMM ->
  `res_c`.
- bf16 backward (per-step launches on H100): `res_c_bwd`, dh GEMM + `swiglu_bwd`, `res_adaln_b_bwd`, dog GEMM, `gate_bwd_rows`,
  `pvx` ungated on P^T (dV), `dpb2` / `dpb_pipe` (dbias), dxa GEMM, `adaln_a_bwd`, the conditioning backward, `unfold`, `pair_bias_bwd`
  and `finalize` (fixed-order reduction of the per-block partials); weight gradients are cuBLAS, one GEMM each.
- fp32 forward: the same steps on TF32 GEMMs and the fp32 rows; pair bias `pair_bias_tc` (TF32 mma.sync); `pvx32` reads v^T and g from
  one flat buffer. fp32 backward: `gate_bwd_fold` writes dO head-major with the samples folded into the columns, so `dV^T = dO^T P` and
  `dP = dO v^T` are each ONE batched cuBLAS TF32 GEMM over the heads (no permute copies); `dbias = P o (dP - D)`.
- With `bias=` (from `pair_bias_all`): the block skips the pair kernels; its dbias leaves as the gradient of the `bias` input,
  and `ln_pair` / `to_bias` get their gradients through `pair_bias_all`.

## Kernel design

- **`pvx`**: CTA = (head, 64 x WGS query rows, a group of samples); the consumer warpgroups keep the accumulators of the whole group and
  a producer thread streams `[P rows | v of every sample]` 64-key k-blocks through a TMA ring, so P is read once per (head, rows, group).
  The samples of a group form the N dimension of one wgmma per k step. Gate rows arrive by TMA before the epilogue; the bf16 result
  leaves by TMA store. `(WGS, samples per CTA)` come from a host cost model so small L still fills the SMs (groups 1-48).
- **`pvx32`**: sample j of a group is rows `dh j .. dh j + dh - 1` of the B tile (v^T), so the group is one `m64n(dh x samples)k8`
  wgmma without padding; N = dh x samples <= 256 (at most 4 samples per group at width 64).
- **`dpb2`**: two consumer warpgroups own 64 rows each of a 128 x TN tile (TN 256 when L % 256 == 0, else 128), one `m64nTNk16` chain
  over the samples, both operands K-major; P is read and the result written in 64-column chunks through a 128-B-swizzled TMA staging
  tile; `dsum = sum_a D` precomputed.
- **Hoist** (`pair_bias_all`): `ln_pair` has no offset, so `ln_pair_b(pair) = LN0(pair) diag(g_b)` and every block's bias is
  `LN0(pair) W'_b^T` with `W'_b = W_b diag(g_b)`: one LN0, one projection for all blocks. Backward: `d LN0 = dbias_all^T W'_all`, the
  LN0 backward rows, `dW' = dbias_all LN0` (bf16: against the split `[bf16(LN0) | bf16(LN0 - bf16(LN0))]`, so dW' sees LN0 to ~2^-17).

## Numerics

bf16: bf16 GEMM operands, fp32 accumulation, residual, LayerNorm statistics and softmax; on H100 the training pair bias enters the
softmax as fp32 logits (not rounded to bf16 first). fp32: TF32 GEMMs and cores, fp32 everything else; the fp32 inference hoist is exact
fp32. Masked keys get probability 0 in the softmax kernels (the reference fills the largest negative finite logit, so its fully masked
row is uniform). Reductions over samples and rows are fixed order (per-block partial rows reduced by `finalize`).

## Launch configs

No CUDA kernel here reads a launch-config table: tiles and groups are build defines or host cost models (`pick_group`,
`_h100_group`, the `dpb2` / `dpb_pipe` length rule). `kernels/bias_only_attention/configs/h100_sm90.json` belongs to the Triton
`bias_only_attention` family, which no module in the current tree reaches.

## Limits (`_supports_cuda`)

| call | result |
|---|---|
| not CUDA / not sm_90 | refused |
| inference: dtype other than bf16 / fp32 (fp32 = fp32 weights, no autocast), not `[A, 1, L, 768]` with L % 128 == 0 and L <= 3072, `(heads, head width)` not 16 x 48 / 24 x 32 / 12 x 64 / 16 x 64, cond not 384 / 768, pair not 128 / 256, n != 2, eps != 1e-5, mask not `[1, L]` | refused |
| training: the same with L <= 768, cond 384 or 768, pair 128 / 256 | refused |
| B > 1 (a pair per sample, `single [1, B, L, d]`) | refused (PyTorch backend only) |

## History

Old notes (git HEAD): `docs/gpus/h100/bias_only_dit/bias_only_dit.md`, `docs/kernels/bias-only-attention.md` (the latter is about the
Triton bias-only TriangleAttention benchmark and does not apply to this block). Historical (from the 2026-10-15 note, not re-measured
under the current measurer; ms per block, 16 x 48, CUDA graph): inference bf16 A = 5 shared cond L384 / L768 0.089 / 0.146 against
PyTorch compiled 0.207 / 0.424; fp32 0.163 / 0.296 against 0.365 / 0.772. Rejected on H100: the B200 three-kernel step (cluster-split
front / tail; 1.4-1.6x slower, register accumulators cap the per-CTA tile), squeeze GEMM + last residual in one kernel (`sq_resout`,
built, off: 1.1-1.5x slower), dog GEMM + gate backward in one kernel (`dog_gatebwd`, built, not dispatched), bf16 dV / dbias as batched
cuBLAS GEMMs (permute copies cost what the GEMMs save), row-split weight-gradient GEMMs.
