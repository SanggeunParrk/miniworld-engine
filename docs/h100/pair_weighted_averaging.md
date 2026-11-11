# MSAPairWeightedAveraging on H100 (sm90)

Scope. `modules.MSAPairWeightedAveraging` (`modules/msa.py`):

    y = LN_m(msa); v = W_v y                              # [B, M, L, H, C]
    w = softmax_j(W_b LN_z(pair), masked keys at the dtype's min)   # [B, H, L, L]
    o = sigmoid(W_g y) * sum_j w[h, i, j] v[m, j, h]
    return msa + drop_msa(W_o o)                          # row-broadcast dropout (training, p_drop)

`ln_pair` has no bias (a per-head constant on the logits cancels in the softmax; a checkpoint's `ln_pair.bias` is dropped on load).
Backends: `pytorch` (reference), `cuda` (H100 kernels in `kernels/integrations/pwa_train.py` and `pwa_wide.py`; A100 (sm_80):
`integrations/pwa_sm80.py`, `pwa_wide_sm80.py`), `triton` (`kernels/pair_weighted_averaging/triton/main.py`), `anthropic` (external
baseline, inference only, not described here). Requirements rows: MSA stream, L 128 .. 768, depth 1024 (one row also 384 / 512 /
2048), 8 heads; (d_msa, d_pair, C) = (64, 128, 8), (64, 128, 32), (64, 256, 32), (128, 256, 8), (128, 256, 16), (128, 384, 8),
(256, 512, 32); dtype `master` only; eval and train (p_drop 0.15 / 0.25). Every path below applies the residual and the row dropout
itself.

Judging: [measure](../measure.md), [modules](../modules.md), [kernels](../kernels.md).

## Kernels

`requirements_kernel.csv` rows of family `pair_weighted_averaging`. MiniWorld-width sources: `kernels/integrations/csrc/pair3.cu`,
`ln_vg.cu`, `pwa_fwd3.cu` (also holds `pwa_fwd2_kernel`), `pwa_fwd2.cu`, `pwa_gateout.cu`, `pwa_glue3.cu`, `dgv_bwd.cu`; wide:
`pair_wide.cu`, `pwa_wide.cu`.

| name | role | what it does |
|---|---|---|
| `pair_weighted_averaging_mask_sm90` | forward | `pwa_mask_kernel` (pair3): the key mask in the layout the pair kernel reads |
| `pair_weighted_averaging_pair_softmax_sm90` | forward | `pair_fwd_kernel` (pair3, `pair_fwd3`): per pair row i, LN_z(128) -> proj_z (8 heads) -> key mask -> softmax over j -> w [H, N, N] bf16 |
| `pair_weighted_averaging_layernorm_value_sm90` | forward | `ln_vg_kernel`: y = bf16(LN(m)) natural [S][N][64], v head-major [H][N][S*C], all TMA, persistent |
| `pair_weighted_averaging_contract_gate_out_sm90` | forward | `pwa_fwd3`: o_h = w_h v_h, gate, out-projection, dropout, residual in one warp-specialized kernel (o saved for training) |
| `pair_weighted_averaging_gate_out_sm90` | forward | `pwa_gateout_kernel`: the gate / out pass over a cuBLAS contraction (N >= 512) |
| `pair_weighted_averaging_bwd_glue_sm90`, `..._bwd_glue_reduce_sm90` | backward | `pwa_glue3_kernel` (+ `pwa_glue_reduce`): from the saved o, do (head-major), dgp into the [dgp \| dv] buffer, dWo partials in-kernel |
| `pair_weighted_averaging_bwd_contract_dv_sm90` | backward | `pwa_plain2` (`pwa_fwd2.cu`): dv = w^T do into the [dgp \| dv] buffer (used where cuBLAS does not run it) |
| `pair_weighted_averaging_bwd_dgv_sm90`, `..._bwd_dgv_reduce_sm90` | backward | `dgv_bwd_kernel` (+ `pwa_dgv_reduce`): dy = dgv [Wg; Wv] in registers, dWgv, LN backward + residual -> dm, dgamma, dbeta |
| `pair_weighted_averaging_bwd_pair_sm90`, `..._bwd_pair_reduce1_sm90`, `..._bwd_pair_reduce2_sm90` | backward | `pair_bwd_kernel` (pair3) + `pwa_pair_reduce1/2`: softmax backward -> proj_z backward -> LN_z backward -> dz, dWb, dgamma_z |
| `pair_weighted_averaging_transpose_sm90`, `..._wgv_pack_sm90` | backward | `pwa_transpose_kernel` (Wo^T, w^T), `pwa_wgv_kernel` ([Wg; Wv]^T) |
| `pair_weighted_averaging_bwd_pair` | forward_train / backward | Triton `_pair_fwd_kernel` / `_pair_bwd_kernel` inside `pwa_train.py` |
| `pair_weighted_averaging_pair_softmax_wide_sm90`, `..._bwd_pair_wide_sm90` | forward / backward | `pair_wide.cu`: the pair kernels with d_pair a compile-time width (128 / 256 / 384) |
| `pair_weighted_averaging_layernorm_value_wide_sm90` | forward | `ln_vg_w` (`pwa_wide.cu`): y = LN(m) + statistics, v head-major |
| `pair_weighted_averaging_gate_out_wide_sm90` | forward | `gate_out_w`: gate, o * sigmoid(gate) from registers into the out projection, dropout, residual |
| `pair_weighted_averaging_bwd_gate_wide_sm90` | backward | `gate_bwd_w`: do' = dout Wo, gate recomputed: go, do (head-major), dgp |
| `pair_weighted_averaging_bwd_dm_wide_sm90` | backward | `dm_bwd_w`: dy = [dgp \| dv][Wg; Wv], LN backward + residual -> dm, dgamma / dbeta partials; dv copied into the natural-layout buffer |
| `pair_weighted_averaging_gate_big_sm90`, `pair_weighted_averaging_out_big_sm90` | forward | BIG build (d_msa 256, 8 x 32): `gate_w` (gate + gated product, sigmoid gate saved in training) and `out_w` (out projection, dropout, residual), `gate_out_w` split in two |
| `pair_weighted_averaging_bwd_gate_big_sm90`, `pair_weighted_averaging_bwd_layernorm_big_sm90` | backward | BIG build: `gate_bwd_big_w` (from the saved gate) and `ln_bwd_w` (LN backward of an fp32 dy + residual; dy = [dgp \| dv][Wg; Wv] on cuBLAS) |
| `pair_weighted_averaging_layernorm_gemm_softmax`, `_layernorm_gemm`, `_gate_gemm`, `_bwd_gate_gemm`, `_bwd_layernorm_gemm_dw`, `_bwd_layernorm_gemm_dx_dlnw`, `_bwd_layernorm_gemm_softmax` | forward / backward | the Triton backend |

The contractions (o = w v where split, dv = w^T do on the native path, dw = do v^T) and the wide path's weight gradients are cuBLAS.

## Design: the `cuda` backend

`_supports_cuda` -> `pwa_train.refusal`; `_forward_cuda` runs `pair_weighted_averaging` (grad, or training mode with p_drop > 0) or
`pair_weighted_averaging_inference`. Widths other than d_msa 64, 8 x 32 go to `pwa_wide`; d_msa 64, 8 x 32 at d_pair 256 / 384 / 512 (`pair_only_wide`,
`PAIR_WIDE`) stays on `pwa_train` with `pwa_wide`'s pair kernels (N <= 1024).

### MiniWorld width (d_msa 64, d_pair 128, 8 heads x 32): `pwa_train.py`

Range: batch 1, N a multiple of 128, S even, N <= 2048 (inference) / 1024 (training).

Forward (3 launches, plus the mask):

1. `pair_fwd3`: a warp takes 16 j at a time; the [16 j][128 d] z tile is cp.async-staged (swizzled), read as mma A fragments with
   ldmatrix, normalised in registers (row statistics by quad shuffles); the projection is eight `m16n8k16` mma.sync with Wb^T as B;
   mask and softmax over a shared-memory row of N logits (hence N <= 2048).
2. `ln_vg`: one 128-thread block per (token n, 64 consecutive s), persistent with a 3-stage TMA load pipeline and the weight staged
   once; every global access is a full line.
3. `pwa_fwd3` (N < 512, and every plain-build call), 384 threads: one producer thread streams per (head, j-chunk) a 24 KB stage {W chunk for each consumer, the shared
   v chunk} through a 3-deep TMA ring; two consumer warpgroups (64 rows each) run six `m64n64k16` wgmma per head, the gate GEMM against
   their y tiles, sigmoid x o packed into register-A fragments, and the out-projection as an RS wgmma accumulated over heads; the
   residual is TMA-loaded into the output staging and added before the TMA store. N >= 512 on the native builds (below): cuBLAS
   `bmm(w, v)` (head-major o) and
   `pwa_gateout`, the same epilogue reading o tiles; o is rounded to bf16 before the gate there, so the forward output is not
   bit-identical to the fused kernel's (the gradients are).

Backward (`_PwaMath.backward`): `pwa_glue3` (do head-major, dgp, dWo partials; the gated product never reaches memory) -> dv = w^T do
(cuBLAS on the native path, read tile by tile by `dgv_bwd`; else `pwa_plain2`) -> dw = do v^T (cuBLAS) -> `dgv_bwd` (one pass over the
[M][512] dgv buffer: dy in registers, dWgv, LN backward + residual) -> `pair_bwd3`.

Native vs plain builds. The six training shapes S in {1024, 2048, 4096} x N in {384, 768} (`_h100_training_default`) and every H100
inference call run natively: `pair3.cu`, `dgv_bwd.cu` and `pwa_glue3.cu` are built with `-DMW_PWA_CUDA_AUX=1` (per-CTA partials with
fixed-order reduce kernels), the key mask is packed by `pwa_mask_kernel`, the transposes / weight packs are kernels, dv is cuBLAS, and
the training forward's dropout keep-mask comes from the TriAttn extension's `drop_keep`. Other training shapes use the plain builds
(float atomics for dWgv, the LN and pair weight sums), a dense bf16 [N, N] mask, `pwa_fwd3` at every N and the Triton `pair_bwd`. The Triton `pair_fwd` is reached only when `pair_fwd3` cannot run (N % 16 or N > 1024 in
training, N > 2048 in inference), which the refusal already excludes on H100.

### Other widths (`pwa_wide.py`)

d_msa 64 / 128, d_pair 128 / 256 / 384, 8 heads x C 8 / 16, and the BIG build (d_msa 256, d_pair 512, 8 x 32), any depth, N <= 1024 (padded to a multiple of 16 with masked tokens),
inference and training. Kernels are built per (d_msa, C) and per d_pair; pair-kernel warps / CTAs per SM: `PAIR_CFG` (128: 8 x 2,
256: 4 x 3, 384: 6 x 1, 512: 4 x 1). BIG: `gate_out_w` is split into `gate_w` + `out_w` (one 128 KiB weight resident each),
the backward uses `gate_bwd_big_w` with the saved gate and `ln_bwd_w` after a cuBLAS dy.

- Forward: `pair_fwd3` (wide) -> `ln_vg_w` -> cuBLAS `o = w v` -> `gate_out_w`. Inference keeps no y: `gate_out_w` normalises m again.
- Backward: `gate_bwd_w` -> cuBLAS dWo, `dv = w^T do`, `dw = do v^T` -> `dm_bwd_w` -> cuBLAS `[dWg; dWv] = dgv^T y` -> `pair_bwd3`
  (wide). The MSA-side kernels are one warpgroup per CTA, persistent over (token n, 64 MSA rows) tiles, epilogues in the wgmma
  accumulator layout so the gated product is the next GEMM's A operand from registers. The pair-side block reductions run warp by warp
  in a fixed order (no float atomics).

## Triton backend

Forward: `layernorm_gemm_softmax` (per row i: LN_z -> Wb -> mask -> two-pass softmax), `layernorm_gemm` (LN_m -> Wv head-major), cuBLAS
`o`, `gate_gemm` (gate, out-projection summed over heads, dropout, residual). Backward: `bwd_gate_gemm` (g recomputed, do, dgp, dWo_h /
dWg_h in registers), cuBLAS dv / dw, `bwd_layernorm_gemm_dw`, `bwd_layernorm_gemm_dx_dlnw`, `bwd_layernorm_gemm_softmax`. The dropout
keep-mask `[B, L, D]` is drawn by the module and applied as `x * keep / (1 - p)`. Head widths below 16 are zero-padded tiles.

## Numerics

- LayerNorm statistics fp32. The pair logits stay fp32 into the softmax (`pair3`; `-DMW_PWA_ROUND_LOGITS=1` is an opt-in bf16 rounding
  build, not used). The wide pair kernels keep the logits as bf16 in shared memory.
- `dgv_bwd` keeps dy in fp32 registers (no bf16 rounding before the LN backward).
- Native builds and the wide path reduce weight / affine gradients in a fixed order (bit-identical replays); the plain builds
  (MiniWorld width, training shapes outside the six) use float atomics.
- Masked keys use the bf16 minimum (`torch.finfo(bfloat16).min`), as the module's `masked_fill`.
- bf16 only; there is no fp32 PWA kernel.

## Launch configs

`kernels/pair_weighted_averaging/configs/h100_sm90.json` holds the seven Triton ops (`pair_weighted_averaging_*_triton`); a key
without an entry runs the decorator default (warned once). The CUDA kernels have fixed schedules and the shape rules above (`_split_contraction`, `PAIR_CFG`).

## Limits

- `cuda`: CUDA, sm_90, bf16 msa and pair, 4-D shapes, bool `[B, L]` mask, batch 1. MiniWorld width: N % 128, S even, N <= 1024
  (training) / 2048 (inference); with d_pair 256 / 384 / 512 (wide pair kernels) N <= 1024. Wide: d_msa in {64, 128}, d_pair in
  {128, 256, 384}, 8 heads, C in {8, 16}, or (256, 512, 8 x 32) (BIG); 0 < N <= 1024. Every requirements row has a CUDA path.
- `triton`: bf16, d_msa a power of two >= 16, 1 <= C <= 128, n_head * C % 16 == 0, d_pair >= 16.
- A `dead` instance returns msa in inference.

## History

The previous, longer note was deleted; see git history (speed-of-light and comparison tables for the wide widths, the open items list).
