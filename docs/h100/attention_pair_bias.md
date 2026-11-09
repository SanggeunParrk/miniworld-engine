# AttentionPairBias on H100 (sm90)

Scope: `AttentionPairBias` (the Pairformer single track, AF3 Alg. 24 without conditioning: `LN(single)` -> q / k / v / gate projections,
`bias = to_bias(LN(pair))`, key mask, softmax, `sigmoid(g + gate_bias)` gate, `to_out`, residual). Backends: `"pytorch"` (the reference,
SDPA), `"triton"` (the module's statements with the attention core on the Triton pair-bias kernel), `"cuda"` (below), `"anthropic"` (the
release's pair-bias provider, inference only) and `"cueq"` (`cuequivariance_torch.attention_pair_bias`). Requirement rows:
`token_single`, L 128-768, d_single 384 with 16 heads (16 x 24; engine-own 12 x 32, 24 x 16) or 512 with 16 heads (16 x 32), d_pair
128 / 256 / 384 / 512 (OpenDDE also 8 heads at d_pair 384), Chai-1's `gate_bias=1.0`; dtypes `float32` and `master`; modes `eval` and
`train`. Measurement and judging: [../measure.md](../measure.md); conventions: [../modules.md](../modules.md),
[../kernels.md](../kernels.md).

The `"cuda"` backend is `kernels/integrations/attention_pair_bias_b200.py` (its H100 branches; the file also holds the B200 path). It
serves bf16 activations only, so of the requirement dtypes it takes `master` (bf16 activations, fp32 master parameters) and refuses
`float32`.

## Kernels

| kernel | role | what it does |
|---|---|---|
| `augmented_attention_apb_prep_sm90` | forward | `prep`: casts (an fp32 master is rounded here, no separate cast launch) and packs `W q\|k\|v\|g` with its bias, `Wo`, `Wf = to_bias.weight * ln_pair.weight * scale`, padding 16 x 24 heads to 32 wide |
| `augmented_attention_apb_ln_rows_sm90` | forward | `ln_rows`: `LN(single)` (bf16), its (mean, rstd), and the residual seed that `to_out` accumulates onto |
| `augmented_attention_apb_pair_bias_fwd_sm90` | forward | `pair_bias_fwd`: LayerNorm of the pair folded into the projection, mma.sync, TMA ring over 128-row pair tiles; head-major bias, masked keys -inf |
| `augmented_attention_fwd_sm90` | forward | `attn_fwd.cu` built per head width: inference `attn_inf` (`sigmoid(g) o` over q), training `attn_fwd_o_og` (O fp32, og bf16, LSE) |
| `augmented_attention_bwd_dq_sm90` | backward | `attn_dq.cu apb_dqb`: dQ and the full dbias (one sample, natural units) |
| `augmented_attention_bwd_dkv_sm90` | backward | `attn_dkv.cu -DDBIAS=0`: dK, dV |
| `augmented_attention_apb_gate_bwd_sm90` | backward | `gate_bwd`: dO, `D = rowsum(dO O)` and dg from the saved O and q\|k\|v\|g |
| `augmented_attention_apb_qkv_bwd_sm90` | backward | `qkv_bwd`: dQ / dK / dV into the packed q\|k\|v\|g gradient, column sums into the accumulator |
| `augmented_attention_apb_ln_bwd_sm90` | backward | `ln_bwd`: LayerNorm backward plus the residual gradient |
| `augmented_attention_apb_pair_bias_bwd_sm90` | backward | `pair_bias_bwd`: dpair and the dWf partials, recomputing LN(pair) |
| `augmented_attention_apb_acc_reduce_sm90`, `augmented_attention_apb_finalize_sm90` | backward | fixed-order tree reduce of the per-block partial rows; every parameter gradient into its own fp32 tensor |
| `augmented_attention_fwd`, `_bwd_pre`, `_bwd_split`, `_bwd_reduce` | forward / backward | the Triton core of the `"triton"` backend |

`requirements_kernel.csv` also attributes the shared `layernorm*` and `cond_transition_caller_grad_*` kernels to
this module; they are documented with their own families.

## Inference (one opaque op, `apb_b200_inference`)

`ln_rows` -> one cuBLAS `addmm` for q\|k\|v\|g (q pre-scaled by log2 e / sqrt(head dim): logits in exp2 units) -> `pair_bias` (Wf in
log2 units, masked keys -inf) -> gated `attn_inf` (one sample, `NWG=1`, `ROW_WIDTH` = the padded width) -> `to_out` accumulated in
place onto the residual seed `ln_rows` wrote (cuBLAS beta = 1; no copy of x into the output). The packs are cached per parameter
(pointer, version) and capture-scoped. `gate_bias` is not a kernel term: the constant is added to the gate rows of the packed bias
vector once per pack, so the core's `sigmoid(g)` reads `g + c`. At d_pair 256 (16 x 24 heads only) `pair_bias_fwd` is instantiated for
the 256-wide ring and Wf is made from the full 256 columns in torch per parameter version.

## Training (autograd Function, opaque forward / backward)

Forward: `prep` (natural units) -> `ln_rows` -> q\|k\|v\|g GEMM -> `pair_bias` -> `attn_fwd_o_og` (gate in the epilogue: O fp32 for the
backward's D, og bf16, LSE) -> `to_out` onto the residual seed. Backward: dO GEMM, dWo GEMM -> `gate_bwd` -> `apb_dqb` + `attn_dkv` ->
`qkv_bwd` -> dxa / dW GEMMs -> `ln_bwd` -> `pair_bias_bwd` -> `finalize`. Every producer owns one partial row per block; `finalize`
reduces them in block order (no atomics, no zero fill), so the step is bitwise repeatable. Parameter gradients are fp32 and cast once
to each parameter's dtype.

Pair-weight gradient repair: at L 384 and 768 (the qualified lengths) and the five supported layouts, `to_bias.weight`'s gradient is
recomputed by `_pair_weight_shadow` (native CUDA + cuBLAS, the `apb_rows` extension built with `-DAPB_H100_FP32_SHADOW=1`, which needs
CUTLASS 4.2 headers through `CUTLASS_PATH`) to reproduce the fp32 reference's rounding; it materialises extra fp32 workspaces. Other
lengths keep the fused dWb from `pair_bias_bwd`.

## Design notes

- 16 x 24 heads run 32 wide: the packed q\|k\|v\|g weights carry 8 zero rows after each head's 24 (Wo 8 zero columns), so q, k, v, g,
  O and their gradients are 512 wide with exact zeros in the pads; the cores see heads of 32 (`DHP=32`) with the softmax scale of 24
  (`RSQDV`); `finalize` gathers the real rows / columns. Geometry: `apb.GEOMETRY` (8 x 48, 12 x 32, 16 x 24 -> 32, 24 x 16, 16 x 32 at 512).
- The natural-unit builds (`APB_NATURAL=1`) multiply scores by the reciprocal head scale instead of dividing.
- `APB_KS` (key splits per 64 query rows, merged in shared memory) is a build option; the default is 1 for every kernel.
- Rejected (from the old note): overlapping `pair_bias_bwd` with the dK / dV chain on a second stream and a bf16 dbias were not done;
  pipeline depth 2 / 3 / 4 did not change the pair kernels.

## Numerics

bf16 GEMM operands and attention operands (P, dS bf16 into the tensor core), fp32 accumulation, LayerNorm vectors fp32, every gradient
produced in fp32. Masked keys are -inf in the H100 core (the B200 path keeps a finite -1e30), matching the fp32 SDPA reference's zero
update for an all-false key mask.

## Launch configs

The CUDA path has no launch-config table (fixed builds per geometry). The `"triton"` backend's core reads
`kernels/augmented_attention/configs/h100_sm90.json`; it has rows for `H=16, HEAD_DIM=24` (and 16 x 48, 4 x 32), rank 1 by default.

## Limits (`_supports_cuda` = `refusal`)

| call | result |
|---|---|
| not CUDA / not sm_90 | refused |
| `use_qk_norm` | refused (no norm between the packed GEMM and the core) |
| activations not bf16 (incl. every `float32` requirement row) | refused |
| `single` not one contiguous `[1, L, d]` row (B > 1) | refused |
| (heads, d_single) not in 8 x 384, 12 x 384, 16 x 384, 24 x 384, 16 x 512, or `n_head * d_hidden != d_single` | refused |
| d_pair 256 except 16 heads x 384; d_pair 256 with a gradient; d_pair 384 / 512 | refused |
| `gate_bias` with a gradient | refused |
| L not a multiple of 128; pair not contiguous `[1, L, L, d_pair]`; mask not bool `[1, L]` | refused |
| `extra_pair_bias` | refused (the bias is made inside the pass) |
| `"anthropic"`: a gradient; `"cueq"`: QK-norm, `extra_pair_bias`, different single / pair LayerNorm eps | refused |
| `"triton"`: head dim > 64 | refused |

## History

Old note (git HEAD): `docs/gpus/h100/attention_pair_bias/attention_pair_bias.md`. Historical (from the 2026-10-13 note, not re-measured
under the current measurer; 8 x 48, d_pair 128, bf16-mixed, CUDA graph, ms): inference L384 / L768 0.042 / 0.090 against Triton
0.089 / 0.224 and PyTorch 0.106 / 0.259; training 0.123 / 0.272 against Triton 0.255 / 0.643. That note's d_pair 256 pair-fused core
(`fused_pair.py`) no longer exists.
