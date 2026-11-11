# OuterProductMean on H100 (sm90)

Scope. `modules.OuterProductMean` (`modules/msa.py`):

    m = LN(msa); a = mask * (W_l m [+ b_l]); b = mask * (W_r m [+ b_r])
    o[i, j] = sum_s a[s, i] (x) b[s, j]                       # d_hidden^2 per pair
    pair = W_out (o / n) + b_out        normalize_before_proj (AF3 SI order)
    pair = (W_out o + b_out) / n        normalize_before_proj=False (ESMFold2: the bias is divided too)
    n[i, j] = sum_s mask[s, i] mask[s, j] (fp32), clamp(n, 1), or n + count_offset (AF3 release)

Variants: `projection_bias` (RoseTTAFold3), `groups` (Chai-1: 8 groups of 8-wide halves, the 512 products LayerNorm'd by `product_norm`,
no count division), `mask_interchain`, an optional cross-tensor `residual` (returns `residual + OPM`).

Backends: `pytorch` (reference), `cuda` (H100 kernels below), `triton` (portable Triton path), `anthropic` (external baseline, not
described here). Requirements rows: MSA stream, L 128 ..
768, depth 1024 (one row also 384 / 512 / 2048); (d_msa, d_pair, d_hidden) = (64, 128, 32) with and without projection bias and in both
normalisation orders, (64, 256, 64) with groups 8, (128, 256, 32), (128, 384, 32), (256, 512, 32); dtype `master` only (bf16
activations, fp32 parameters); eval and train.

Judging: [measure](../measure.md), [modules](../modules.md), [kernels](../kernels.md).

## Kernels

`requirements_kernel.csv` rows of family `outer_product_mean`. The H100 sources are `kernels/integrations/csrc/opm_epilogue.cu` (built
once per (d_msa, d_pair) and variant flags), `kernels/outer_product_mean/cuda/chai.cu` and `cuda/gather.cu`.

| name | role | what it does |
|---|---|---|
| `outer_product_mean_pack_weights_sm90` | forward | `opm_pack_native`: LN affine to fp32, the projection / out weights to bf16 (and Wo^T, the gamma-folded backward weight in training); inference caches the packed forms per parameter version |
| `outer_product_mean_prologue_sm90` | forward | `opm_prologue_wgmma_native`: one 64-MSA-row tile per token, LN + both projections + mask on wgmma, written in the grouped layouts A2 `[(i,c), s]`, BT `[(j,e), s]` |
| `outer_product_mean_prologue_tma_sm90` | forward | `opm_prologue_tma_hopper` (d_msa 64): the persistent TMA version of the prologue (one warp loads, eight warps LayerNorm 128 MSA rows in place, one warpgroup projects on wgmma, TMA stores) |
| `outer_product_mean_prologue_wide_sm90` | forward | `opm_prologue_tma_wide<CM>`: the same persistent TMA prologue for d_msa 64 / 128 / 256, with the optional projection biases (RF3) in its epilogue |
| `outer_product_mean_mask_bits_sm90` | forward | `opm_mask_bits_native`: the [N, S/32] bit mask |
| `outer_product_mean_count_sm90` | forward_train | `opm_count_tiled`: n[i, j] = popcount of the AND of two bit rows, exact, 16 x 16 pair tiles (inference takes the count inside the epilogue instead) |
| `outer_product_mean_epilogue_sm90` | forward | `opm_epilogue_kernel`: reads the cuBLAS product in its grouped layout, [i,j,c,e] -> [(i,j),(c,e)] permute, / n in the fp32 accumulator, projection to d_pair on wgmma, bias, residual; d_pair cut into 128-wide CTA slices |
| `outer_product_mean_fused_infer_sm90` | forward_infer | `opm_fused_kernel` (`opm_fused_infer`): product and projection in one kernel; the grouped product never reaches HBM |
| `outer_product_mean_bwd_dgrad_sm90` | backward | `opm_dgrad_kernel`: dO = (dz / n) Wo straight into the grouped layout, dbias on the side |
| `outer_product_mean_bwd_dwo_sm90` | backward | `opm_dwo_kernel`: dWo off the grouped O, split-K partials |
| `outer_product_mean_bwd_reduce_sm90` | backward | `opm_reduce_partials_chunked`: fixed-order sum of per-CTA partials (dWo, the dgrad's dbias, the prologue backward's weight / affine sums) |
| `outer_product_mean_bwd_prologue_sm90` | backward | `opm_prologue_bwd_kernel`: mask, both projection backwards and the LN backward in one pass -> dm and per-CTA partials of dWa, dWb, dgamma, dbeta |
| `outer_product_mean_bwd_finish_sm90` | backward | `opm_finish_native` (d_msa 64 native path): turns the reduced partials into dWa, dWb (gamma / beta applied) and dgamma, dbeta |
| `outer_product_mean_gather_sm90` | forward_infer | `gather_kernel` (`gather.cu`) of the composed inference path |
| `outer_product_mean_chai_proj_sm90`, `..._chai_epilogue_sm90` | forward | Chai-1: LN + projections + mask into group-major a, b; per-pair product LayerNorm + projection + bias + residual |
| `outer_product_mean_chai_bwd_sm90`, `..._chai_projbwd_sm90` | backward | Chai-1: product-LN backward scattered into the grouped layout; projection + MSA LN backward |
| `outer_product_mean_layernorm_gemm`, `_epilogue`, `_bwd_epilogue_dx`, `_bwd_epilogue_dw`, `_bwd_layernorm_gemm`, `_epilogue_ln`, `_bwd_epilogue_ln_dx`, `_bwd_epilogue_ln_dw` | forward / backward | the Triton backend (`triton/main.py`) |

The outer product itself (O = A2 BT^T), dA / dB and the composed paths' projections are cuBLAS.

## Design: the `cuda` backend

`OuterProductMean._cuda_route` picks one H100 path per call (bf16 activations, bool [B, M, L] mask, CUDA, sm_90):

| form | grad | path |
|---|---|---|
| groups 1, no projection bias | any | `opm_train.outer_product_mean` when `opm_train.refusal` is None; else, without grad, `composed.inference` |
| projection bias (RF3) | grad | `opm_train` (biases in the prologue) |
| projection bias | no grad | `composed.biased_wide` (S % 256, N % 64, d_msa 64 / 128, d_pair 128 / 256 / 384), else `composed.inference` |
| groups > 1 (Chai-1) | any | `chai.forward` (8 x 8, d_msa 64, N % 64, batch 1, d_pair % 128; training without interchain) |

A100 (capability (8, 0)): `integrations/opm_sm80.py` (`outer_product_mean/cuda/*_sm80.py`); Chai-1 on `chai.py` with `cuda/chai_sm80.py`.

### `opm_train` (groups 1)

Range (`refusal`): d_hidden 32, (d_msa, d_pair) in `SHAPES` = {(64, 128), (128, 256), (128, 384), (256, 512)}, bf16, batch 1, no interchain mask. A length that
is not a multiple of 64 or a depth that is not a multiple of 256 is padded with masked rows and the output cut back; the residual is fused
in the epilogue only for an unpadded length. `count_offset` is taken as the divisor `max(count, dmin) + doff` (`divisor_params`).

Forward boundaries: prologue (LN + projections + mask -> A2, BT, and in training the LN statistics and bit mask) -> cuBLAS
`O = A2 BT^T` in the grouped layout -> epilogue (permute, / n, Wo, bias, residual). The prologue is chosen by `_h100_path` (the
Triton `fused_prologue` kept in `opm_train.py` is not reached: every former Triton-prologue cell takes the wide CUDA prologue):

- d_msa 64 / d_pair 128, AF3 order, no bias: training at S 1024 / 2048 / 4096 uses the persistent TMA prologue, other S the non-TMA
  wgmma prologue; inference uses the TMA prologue at seven measured (S, N) cells and otherwise the wide prologue (the former Triton
  prologue cells; `_CUDA_PROLOGUE_FOR_LEGACY = True`) or the non-TMA one.
- Every other width, the after-projection order or projection biases: `_forward_wide` (`opm_prologue_tma_wide<CM>` + cuBLAS + the
  c_z-sliced epilogue).
- Inference at d_msa 64 / d_pair 128, N % 16, S % 64 (`_fused_enabled`): `opm_fused_infer` replaces the cuBLAS product and the epilogue.
  Persistent, 384 threads: a producer warpgroup (TMA for A / B and the weight) and two consumer warpgroups. A work item is 16 i x 8 j =
  128 pairs and four passes over c; each pass is a `m64n256k16` GEMM over S, rounded to bf16 into a shared-memory Opair tile (k permuted
  so the fragment stores are conflict-free), then `z^T += W_q Opair^T` (`m64n128k16`, the pass's 64 KiB weight slice, k-permuted once
  per parameter version by `_fused_weight`) with the 128 x 128 fp32 result kept in registers across passes. The epilogue divides by the
  popcount of the two tokens' mask rows (or after the bias), stages a [pair][o] tile and writes rows (+ residual).
- Training keeps O for dWo (`save_o = True`).
- Two exact shapes use graph-owned cuBLAS workspaces (`_owned_blas_for`: training S1024/L768, inference S4096/L128). The gathered-TMA
  epilogue variants serve inference at (1024, 384) and (2048, 384) (written for the d_msa 64 / d_pair 128 width only).

Backward: `opm_dgrad` (dz / n times Wo into the grouped layout, dbias) -> two cuBLAS GEMMs for dA, dB -> `opm_dwo` off the kept O ->
`opm_prologue_bwd` (mask, projection and LN backward; the wide path folds gamma into the backward weight on the host, the d_msa 64
native path applies it in `opm_finish_native`). Every partial goes through `opm_reduce_partials_chunked` (no atomics). RF3's
projection-bias gradients are two fp32-accumulated GEMVs of the mask with dA / dB.

Design notes kept from the kernel sources: the epilogue moved from wmma to wgmma with 128-B-swizzled operands because the wmma
versions were L1 / MIO bound (shared-memory fragment loads outnumbered the math); the epilogue pipeline is three stages (one barrier per
chunk); `opm_dgrad` uses 64-column passes so three CTAs fit per SM, and larger CTAs that would cut the L2 re-read of Wo^T measured slower
(latency bound, not bandwidth bound).

### Composed inference (`composed.py`)

For the groups-1 forms `opm_train` does not take (d_hidden 32), without grad: `m = LN(msa)` (the module's own `ln_msa`), the
projections as cuBLAS GEMMs (bias in the epilogue, mask after it), the outer product as one batched GEMM `[b, (i k), (j l)]`, then
`gather` writes the `[B N N, K K]` rows, the out-projection, and the fp32 count division before or after it.

### Chai-1 (`chai.py`)

`chai_proj` (LN + both projections + mask -> group-major a, b `[8, S N, 8]`) -> one cuBLAS batched GEMM per group -> `chai_epilogue`
(one persistent CTA per 64 consecutive pairs x 128 outputs: a producer warp TMA-loads the pair tile straight into the K-major wgmma layout
through a 4-D box, a warpgroup LayerNorms the 512 products in place and multiplies by the folded weight `W' = W_out gamma`,
`b' = b_out + W_out beta` with `m64n128k16`, + bias + residual; training also stores the normalised tile y and the statistics).
Backward: `dy = dz W'`, `dW' = dz^T y`, dbias (cuBLAS), `chai_bwd` (product-LN backward into the grouped layout), two batched GEMMs for
dA / dB, `chai_projbwd` (projection and MSA LN backward), and dgamma / dbeta of the product norm from `dW'`.

## Triton backend

`triton/main.py::triton_outer_product_mean`: `layernorm_gemm` (LN -> projections -> mask, a / b [S L, CH]) -> cuBLAS grouped product ->
epilogue (permute folded into the loads, / n, projection, bias, residual); backward `bwd_epilogue_dx` (dz / n, dO grouped), cuBLAS dA /
dB, `bwd_epilogue_dw` (split-K over i), `bwd_layernorm_gemm` (mask, projection backward, LN backward). Projection biases ride the same
kernels; groups use the `*_epilogue_ln_*` kernels with one batched GEMM.

## Numerics

- LN statistics and the pair count in fp32; the count is exact (popcount of bit masks), so depths beyond 256 are not rounded through
  bf16.
- The epilogue divides in the fp32 accumulator (projection is linear: (O / n) W == (O W) / n), dropping one bf16 rounding of the
  normalised product. The fused inference kernel rounds each pass's product to bf16 before the projection (Opair), as the unfused
  path's bf16 cuBLAS output is.
- Weight gradients come back in the parameters' dtype (fp32 masters get fp32); partials are reduced in a fixed order.
- `opm_train` is bf16-only; there is no fp32 OuterProductMean kernel.

## Launch configs

`kernels/outer_product_mean/configs/h100_sm90.json` holds the eight Triton ops (`outer_product_mean_*_triton`); a key without an
entry runs the decorator default (warned once). The CUDA kernels have fixed schedules; the per-shape choices are the
`_h100_path` / `_fused_enabled` / `_epilogue_*_enabled` / `_owned_blas_for` rules above.

## Limits

- `cuda` (`_common_refusal` + `_cuda_route`): CUDA, sm_90, bf16 msa, `msa [B, M, L, d_msa]`, bool mask `[B, M, L]`, residual in msa's
  dtype. Training needs `opm_train` (batch 1, the four `SHAPES` widths, no interchain mask) or, for groups, `chai` (training without interchain).
- `triton`: bf16 msa; no `count_offset`; no `mask_interchain`; d_msa and d_hidden powers of two >= 16; d_pair % 32; grouped products a
  power of two in 16 .. 1024 with groups a power of two; a fused residual must be bf16 `[B, L, L, d_pair]`.
- A `dead` instance returns the residual in inference.

Historical (from the deleted OPM note, record `opm_fused_ladder_20261026`, not re-measured under the current measurer): at L256 S1024 the
fused inference kernel took 212 us against 190 us (cuBLAS product) + 59 us (epilogue).

## History

The previous, longer note was deleted; see git history (per-cell timings, speed-of-light tables, comparisons with anthropic and PyTorch,
the B200 comparison).
