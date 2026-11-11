# Triangle attention on H100 (sm90)

Scope: the triangular gated self-attention classes of `modules/triangle.py` — `TriangleAttention`, `TriangleAttentionAF3`,
`TriangleAttentionRF3`, `TriangleAttentionChai1`, `TriangleAttentionChai1Confidence` (backends `pytorch`, `cuda`, `triton`, `cueq`,
`anthropic`) and `BidirectionalTriangleAttention` (`pytorch`, `cuda`). The requirement rows (`requirements/requirements_module.csv`)
ask for L 128-768 (multiples of 128), starting and ending node, `p_drop=0.25`, dtype `master` (bf16 activations, fp32 master
parameters), modes `eval` and `train`, at (d_pair, heads[, d_hidden]) = (64, 2), (64, 4), (64, 4, 128), (128, 4), (256, 8),
(384, 12) for `TriangleAttention`; (64, 4), (128, 4), (256, 8), (512, 8) for AF3; (64, 4, 256), (128, 4) for RF3; (64, 4, 128),
(256, 4) for Chai-1; (256, 4) for the Chai-1 confidence head; d_pair 128 for the bidirectional class. Backend selection, the
`_supports_<backend>` contract and the measurer are described in [../modules.md](../modules.md), [../kernels.md](../kernels.md) and
[../measure.md](../measure.md).

## Backends

| backend | what runs | range (`_supports_*`) |
|---|---|---|
| `pytorch` | the module's statements (`_attention` / `_attention_fp32_softmax`) | reference, every class |
| `cuda` | the whole update (LN, projections, core, gate, out-projection, residual, dropout) on hand-written sm_90a kernels: `integrations/triattn_h100_sm90.py` ("sm90" route) or, inference only, `triangle_attention/cuda/h100_dh.py` ("dh" route) | see [Limits](#limits--unsupported-calls) |
| `triton` | the attention core only (`triangle_attention/triton/main.py`, forward + backward); LN, projections, gate, out-projection, residual and dropout are the module's statements | CUDA tensor, bf16 pair, self-attention, square pair, bool `[B, L]` mask |
| `cueq` | the core on `cuequivariance_torch.triangle_attention` | fp32 masters run the projections under autocast (`master_autocast`); bf16 head width <= 128 and % 8; fp32 head width <= 32 and % 4, inference only; L above the library's PyTorch-fallback threshold |
| `anthropic` | `TriangleAttention` / `...AF3`: first the fused block `opt_core.attn.pair_fused.tri_attn_block` (LN + projections, core `tier:fast`, gate + W_o); else, and for the other classes, the core on `opt_core.kernels.triattn` (tier word `fast`) | inference only; bf16 / fp32; fp32 masters under `master_autocast`; the provider's own cell selection |

Coverage of the requirement rows by the `cuda` backend (head width = d_hidden / heads):

| class | d_pair, heads, head width | eval | train |
|---|---|---|---|
| `TriangleAttention` | 128, 4, 32 | fused d128 path | fused d128 path |
| `TriangleAttention` | 256, 8, 32 | wide fused front (d256 x 8 heads) | width-generic path |
| `TriangleAttention` | 64 (2 x 32, 4 x 16, 4 x 32), 384 (12 x 32) | width-generic path | width-generic path |
| `TriangleAttentionAF3` | 64 (4 x 16), 128, 256 (8 x 32) | sm90 (ending node: bias transposed after the front, `flip_bias`) | starting: as `TriangleAttention`; ending: width-generic path (bias flip, dbias flipped back) |
| `TriangleAttentionAF3` | 512, 8, 64 | dh | width-generic path, dh training core |
| `TriangleAttentionRF3` | 128, 4, 32 | fused d128 path (gate / out biases in the tail) | width-generic path (gate bias in the projection GEMM epilogue, out bias in the out-projection epilogue) |
| `TriangleAttentionRF3` | 64, 4, 64 | dh | width-generic path, dh training core |
| `TriangleAttentionChai1`, `...Confidence` | all rows | dh (untransposed / dual output) | width-generic path (untransposed / dual output in the glue; 64-channel heads on the dh training core) |
| `BidirectionalTriangleAttention` | 128 (bias-only) | bidirectional path (`integrations/bidir_triattn_h100.py`) | bidirectional path |

A100 (sm_80): `_cuda_route_sm80` -> `kernels/integrations/triattn_sm80.py` (`triangle_attention/cuda/sm80*.py`, `cuda/sm80/`); the
bidirectional path builds its kernels for sm_80 in `cuda/sm80_bidir.py`.

## Kernels

Names from `requirements/requirements_kernel.csv` (family `triangle_attention`); sources under
`kernels/triangle_attention/cuda/h100_sources/` unless noted.

| kernel | role | what it does |
|---|---|---|
| `triangle_attention_pack_sm90` | forward | `tri_pack`: [Wq; Wk; Wv; Wg] -> bf16 [512, 128], Wb zero-padded to 16 rows, Wo bf16, one launch (fp32 or bf16 parameters) |
| `triangle_attention_front_sm90` | forward | `tri_front` (d128): LN in registers + q / k / v / g + head-major `bf16(bias / scale)`, masked keys -inf |
| `triangle_attention_fwd_sm90` | forward | `triattn_fwd` (`triattn_sm90.cu`): bf16 attention core, head dim 32, base-2 LSE when the backward needs it, optional gate epilogue |
| `triangle_attention_fwd_fp16_sm90` | forward_infer | `triattn_fwd_h16` (`triattn_h16_sm90.cu`): the inference core on fp16 q / k / v / bias, bf16 out |
| `triangle_attention_tail_sm90` | forward | `tri_tail` (d128): `x + drop o ((sigmoid(g) o o) Wo^T [+ bo])` in one pass |
| `triangle_attention_front_wide_sm90`, `triangle_attention_pack_wide_sm90` | forward_infer | `tri_front_wide` / `tri_pack_wide`: the d128 front design at d_pair 256, 8 heads (groups of four CTAs, one projection each) |
| `triangle_attention_gate_bwd_sm90` | backward | `tri_gate_bwd` (d128): `dyd = dy o drop`, `du = dyd Wo`, do, dg, Delta per head, dWo partials, dpair seed = dy |
| `triangle_attention_bwd_kv_sm90`, `triangle_attention_bwd_q_sm90` | backward | `triattn_bwd_kv` / `triattn_bwd_q` (`triattn_bwd_sm90.cu`): key-owned dK / dV and query-owned dQ / dbias, P recomputed from the LSE (both paths) |
| `triangle_attention_head_bwd_sm90` | backward | `tri_head_bwd` (d128): input gradient of the five projections, LN backward from recomputed statistics, bf16 TMA reduce-add into dpair |
| `triangle_attention_wgrad_sm90`, `triangle_attention_wgrad_reduce_sm90`, `triangle_attention_param_finish_sm90` | backward | `tri_wgrad` (d128): weight / LN-parameter gradients from per-CTA partial slots, summed in a fixed order, then the parameter algebra `dW_t = gamma M_t + beta cs_t` |
| `triangle_attention_generic_pack_sm90` | forward | glue `pack`: Wcat = [Wq; Wk; Wv; Wg; Wb (heads rounded up to 8 rows)] bf16 and Wo; 16-channel heads zero-padded to 32 |
| `triangle_attention_generic_layernorm_fwd_sm90` | forward | glue `ln_fwd`: y = LN(x) (bf16, or fp16 for fp16 inference GEMMs) + fp32 mean / rstd; also the dh path's LN |
| `triangle_attention_front_fused_sm90` | forward_infer | `front_fused3_sm90.cu` `front`: LN + [Wq; Wk; Wv; Wg; Wb] in one kernel for short inference (`_FRONT_L`: d_pair 64 up to L768, wider up to L128) |
| `triangle_attention_generic_front_bias_sm90` | forward | glue `front_bias`: the bias GEMM's [M, Hp] output -> head-major scaled / masked sbias |
| `triangle_attention_generic_residual_sm90` | forward_train | glue `residual`: `x + drop o t` (training with dropout; without dropout the residual is in the cublasLt epilogue) |
| `triangle_attention_generic_gate_bwd_in_sm90`, `..._gate_bwd_out_sm90` | backward | `ddrop = dy o drop`; do, dg into `dall`, Delta per head |
| `triangle_attention_generic_assemble_bwd_sm90` | backward | glue `assemble`: fp32 dbias -> the bf16 bias columns of `dall` |
| `triangle_attention_generic_layernorm_bwd_sm90` | backward | glue `ln_bwd`: dx = dy + LN backward of `dall Wcat`, per-CTA gamma / beta partials |
| `triangle_attention_dh_fwd_sm90` | forward_infer | `triattn_dh_sm90.cu`: mma.sync core for head widths 32 / 64 with the gate fused (`u = sigmoid(g [+ gb]) o softmax(...) v`) |
| `triangle_attention_dh_fwd_train_sm90` | forward_train | the same kernel with `TRAIN`: also writes O (bf16) and the base-2 LSE for the backward (64-channel heads in training) |
| `triangle_attention_dh_bwd_kv_sm90`, `triangle_attention_dh_bwd_q_sm90` | backward | `triattn_dh_sm90.cu`: mma.sync FlashAttention-2 backward, P recomputed from the LSE; key side dK / dV, query side dQ and dbias into per-slab fp32 partials (deterministic) |
| `triangle_attention_fwd`, `..._bwd_pre`, `..._bwd_dkdv`, `..._bwd_dq` | forward / backward | Triton core (`triton/main.py`): flash-style forward with fp32 LSE, FA2-split backward (delta, key-side dK / dV / per-row dbias, query-side dQ) |
| `triangle_attention_generic_gate_u_sm90` | forward | glue `gate_u` (`triattn_generic_sm90.cu`): `u = sigmoid(g) o O`; used by the bidirectional path |
| `triangle_attention_bidir_gate_bwd_sm90` | backward | `triattn_bidir_sm90.cu` `gate_bwd` (`cuda/h100_bidir.py`): do, dg of the bidirectional gate on channel planes |
| `triangle_attention_drop_scale_sm90`, `triangle_attention_graph_rng_write_sm90` | forward_train | `pack.cu` (`h100_triattn.extension("pack")`): Philox keep-mask; used by `integrations/pwa_train.py` (MSA row dropout), not by the triangle modules |

## Fused path: d_pair = d_hidden = 128, 4 heads x 32

`cuda/h100_sm90.py`; raw PTX wgmma / TMA / mbarrier in `triattn_mod_sm90.cu` and `triattn_sm90.cu`, no CUTLASS or CuTe.
Inference: front -> core (fp16 core, `INFERENCE_FP16 = True`) -> tail. Training: front -> bf16 core with LSE -> tail with the
dropout scale; backward gate_bwd -> bwd_kv -> bwd_q -> head_bwd -> wgrad.

- **Front.** Tile = 128 rows (one (b, i), 128 consecutive j); warpgroup 0 is the TMA producer (weights once, x tiles through a
  two-stage ring), warpgroups 1 / 2 take rows 0-63 / 64-127. x is ldmatrix-loaded straight into the wgmma A-fragment layout and
  normalised in registers (a row's 128 channels on a quad: statistics are two quad sums); q / k / v / g are four m64n128 RS chains
  over the resident [Wq; Wk; Wv; Wg] (128 KiB), the bias one m64n16 chain over Wb. y never touches memory; outputs go from the
  accumulators to global memory with 16-byte stores per lane. HBM-bound (writes four times what it reads).
- **Core forward** (`triattn_sm90.cu`). q / k / v in the projection layout `[B, N, S, H, 32]`; sbias `[B, H, S, S]` arrives
  pre-divided by the scale and shared by the N pair rows. A CTA task = (b, h, 128-query tile) x 2 pair rows on a persistent grid
  that walks the query tile fastest, then the head (CTAs running together share K / V rows in L2; `TA_ORDER 2`: from S >=
  `TA_ORDER1_MINS` = 2048 the row group comes before the head); the task order is a dynamic
  global counter (`TA_DYN=1`). 384 threads: one TMA producer warpgroup, two consumer warpgroups (64 queries each). The bias
  fragment initialises the score accumulator (no per-logit bias add), `S += Q K^T` is an RS wgmma with Q in registers for the task,
  and `[O | l] += P [V | 1]` is one m64n40 RS wgmma per 16 keys (row sums on the tensor core from the same bf16 P). The softmax
  offset is the row's running maximum, raised lazily (only when a tile exceeds it by more than `TA_TAU = 8` log2 units; o and l are
  then rescaled), so `p <= 2^8` for any logit spread. From L >= `TA_SPEC_MINS` = 512 (`TA_SPEC 1`) the offset is speculative: no
  maximum after a row's first finite tile, the tile's row-sum increment (from the tensor core) decides a raise (`TA_TS`), and a tile
  past `TA_TOVF` goes on a fix list that a second, exact launch recomputes. Descriptors are built once per CTA and moved by one add
  (`TA_DOFF`, `desc_off`). Sub-steps run in FlashAttention-3 order (QK^T of sub-step y + 1 issued before
  the exponentials of y), nothing in flight at a loop back edge.
- **Core forward, fp16 inference** (`triattn_h16_sm90.cu`). Same schedule; the score GEMM accumulates in fp16 pairs, the bias
  fragment is the accumulator's initial value, `x = s scale log2e - m` (HFMA2) and `p = 2^x` (`ex2.f16x2`) run on pairs, p is
  directly the A fragment of the PV GEMM (fp32 accumulation). The row maximum is taken only when needed (`TA_SPEC 2`: a row
  without a finite logit, or a lane past offset + `TA_TAU`, one warp vote), so fp16 cannot overflow.
- **Tail.** Tile = 128 rows; g, o and x are ldmatrix-loaded in the A-fragment layout, which is also the accumulator layout:
  `u = sigmoid(g) o o` is the A operand of the m64n128 chain over Wo, `drop o acc + x` is added in place, one rounding. RF3's
  gate / out biases are added in fp32 inside the tail before that rounding.
- **Backward.** `tri_gate_bwd` forms do, dg, Delta and dWo (SS wgmma, register accumulator over the CTA's tiles) and writes the
  residual seed dpair = dy. `triattn_bwd_kv`: task = (b, h, 128-key tile) x 2 rows, 64-query steps; `S^T` starts from `bias^T`
  (ldmatrix.trans of the same tile), P^T recomputed from the LSE, `dV += P^T dO`, `dK += dS^T Q`, FA3 order. `triattn_bwd_q`: dQ
  and dbias; the step's dbias partial goes out by fp32 TMA reduce-add into slab p of `dbias [P][B H S][S]`, every element written by
  one thread in a fixed order and the slabs summed in a fixed order (deterministic). Launches of g samples, `g = max(1, 96 //
  (H L / 128))` (`BWD_Q_WAVE = 96`). `tri_head_bwd`: the four projections' input gradient on the tensor core, the bias share in
  fp32 on CUDA cores, LN backward from recomputed statistics, bf16 TMA reduce-add into dpair. `tri_wgrad`: `M_t = xhat^T d_t`,
  `cs_t`, `B_h = xhat^T db` on the tensor core (db as a bf16 hi + lo pair, near-fp32); H100 has no tensor memory, so a CTA owns a
  pair of the four 128 x 128 fp32 accumulators (x is read twice).
- Saved for the backward: q, k, v, g, sbias, o, lse.

## Wide fused front: d_pair = d_hidden = 256, 8 heads x 32 (inference)

`h100_sm90.inference_wide`. [Wq; Wk; Wv; Wg] (512 KiB) does not fit one CTA: CTAs come in groups of four, CTA `4n + p` holding
projection p (128 KiB resident) and walking the same 64-row tiles (x read from HBM once, from L2 three times); the g CTA also
writes the head-major bias. Packed weight rows are permuted (`tri_pack_wide`) so each lane stores 16 contiguous bytes from
registers. Then the fp16 core with the gate in its epilogue and `x + u Wo^T` on cublasLt with the residual in the epilogue.
Training at this width runs the width-generic path.

## Width-generic path: d_pair <= 512 (multiple of 8), heads of 16 / 32 channels, <= 16 heads

`cuda/h100_sm90_generic.py` + `triattn_generic_sm90.cu`. The core is the fused path's (head dim 32 at any head count; 16-channel
heads are zero-padded to 32 in the packed weights, the softmax scale stays `1/sqrt(16)`). Around it: hand-written LN / bias /
residual kernels and cuBLAS GEMMs (bf16 in, fp32 accumulate).

- forward: `ln_fwd` -> `[q | k | v | g | bias] = y Wcat^T` -> `front_bias` -> core with the gate in its epilogue (`u = sigmoid(g)
  o O`; O itself only when the backward needs it) -> `x + u Wo^T` (cublasLt, residual and RF3 out bias in the epilogue) or, with
  dropout, `t = u Wo^T` then `residual`. Short inference uses one GEMM for q | k | v | g | bias (L <= 256) or the one-kernel front
  (`front_fused3`, `_FRONT_L`); fp16 inference beyond L 256 uses fp16 GEMMs.
- backward: `gate_bwd_in` -> `du = ddrop Wo`, `dWo = ddrop^T u` (fp32 result) -> `gate_bwd_out` -> `triattn_bwd_kv` /
  `triattn_bwd_q` write dk / dv / dq into their columns of `dall [M, 4 HD + Hp]` -> `assemble` -> `dyln = dall Wcat` (bf16),
  `dW = dall^T y` (fp32 result) -> `ln_bwd` (dx = dy + LN backward; gamma / beta partials summed by torch).
- Saved for the backward: y, mean, rstd, q | k | v | g, sbias, o, lse, u.
- The ending node's rows are the pair's columns: the kernels read / write the transposed layout (`trans`); the inference
  out-projection writes transposed rows as a strided cublasLt batch over i (one GEMM per sample).
- Training variants (`h100_sm90_generic.train_forward` / `train_backward`, flags from `integrations/triattn_h100_sm90._variant`):
  AF3 / RF3 / Chai-1 ending-node bias (`flip`: sbias transposed after `front_bias`, the core's dbias transposed back before
  `assemble`); RF3 gate bias (a bias vector `[0 | 0 | 0 | gb]` in the q | k | v | g GEMM epilogue, `dgb` = column sum of dg) and out
  bias (cublasLt bias epilogue, `dbo` = column sum of the update's gradient); Chai-1's untransposed output (`u Wo^T` in the pair's own
  row order) and the confidence head's dual output (the untransposed projection, then the other one's rows transposed onto it with
  cublasLt's strided batch; backward: the update's gradient and its transposed copy, `du = dA Wo + dB Wt`); with dropout the residual
  is `addcmul(x, update, drop)` on the broadcast scale. Heads of 64 channels run the dh core (`h100_dh.train_core` / `backward`:
  bias in natural units, Delta from `gate_bwd_out` per 32-channel group summed in pairs, dS as a bf16 hi + lo pair in the dQ / dK
  MMAs, dbias in `h100_dh.slabs` fp32 slabs summed in order). The variants keep `dyln` in fp32 (the standard module: bf16).

## dh path: heads of 32 / 64 channels, Chai-1 output variants (inference; its core also trains 64-channel heads)

`cuda/h100_dh.py` + `triattn_dh_sm90.cu`. LN (`ln_fwd`; the ending node reads the pair transposed) -> `q | k | v | g = y W^T` and
the pair bias on cuBLAS (from the transposed pair or, `transposed_bias=False`, from the pair as it is; masked keys -inf) ->
mma.sync core (m16n8k16 bf16, fp32 accumulation, FlashAttention-2 schedule: CTA = 64 queries of one (b, n, h), 4 warps x 16
queries, 64-key tiles through a two-stage cp.async ring, P rounded to bf16 in registers) with the gate fused -> `u Wo^T [+ bo]` on
cuBLAS, laid back as the class requires (standard: ending node transposed back; Chai-1: untransposed; confidence head: `W_o u +
(W_t u)^T` / `(W_o u)^T + W_t u`). L is padded to a multiple of 64 with masked keys, so any L is taken.

## Bidirectional path: `BidirectionalTriangleAttention`, bias-only

`kernels/integrations/bidir_triattn_h100.py` (forward, inference and training, and backward). Bias-only weights do not depend on
the row, so the softmax is formed once per (sample, head) and each direction is a plain GEMM over channel planes `[R, L L]`: the
width-generic glue `ln_fwd` -> `[Wv; Wg; Wb] y^T` (one GEMM) -> fp32 softmax (starting: bias plane as it is; ending: transposed),
bf16 P -> per-head GEMMs -> glue `gate_u` -> `pair + U^T Wo^T` (residual in the GEMM's C). Backward: `gate_bwd`
(`cuda/h100_bidir.py`, `triattn_bidir_sm90.cu`) -> softmax gradients -> `dW`, `dy_ln` (fp32) -> glue `ln_bwd`. Refusals
(`refusal`): bf16 pair, `use_self_attention=False`, d_pair % 8 == 0 and <= 512, L % 8 == 0, head width % 8 == 0, fp32 LN affine.

## Batching and chunking

Every sm90 kernel takes B samples in one launch (rows (b, i, j), tasks (sample, head, tile); mask `[B, L]` and dropout scale
`[B, L, C]` per sample). `MAX_BATCH = 48` per launch for both paths and `B Lp^2 < 2^31` (int32 TMA row coordinate). A call with
more samples, or whose working set exceeds `CHUNK_MEMORY_FRACTION = 0.85` of the device's total memory (inference: also
`INFERENCE_WORKSET_FRACTION = 0.25`), runs in balanced chunks (`plan`), a pure function of shapes, mode and total memory (stable
under torch.compile and CUDA-graph capture). Training chunks are one autograd node (`_Chunked`): chunks keep their saved tensors
when they fit, else the backward reruns each chunk's forward; parameter gradients are summed in fp32 in chunk order and rounded once.
One inference sample above the cap runs in blocks of pair rows (`_generic_rows`). L that is not a multiple of 128 is padded to the
next one (pad keys masked, pad rows cut), so L = 144 costs an L = 256 call. The launches are `opaque` custom ops
(`triattn_h100_infer`, `triattn_h100_fwd`, `triattn_h100_bwd`): one op forward and one backward per autograd node.

## Triton core

`triton/main.py`, used by the `triton` backend for every class (the module computes everything else). q / k / v are read as
strided views of the projection layout `[B, L, L, H*D]` (no copies); the bias is made contiguous. Forward: grid
`(L / BLOCK_M1, B H, L rows)`, online softmax in base 2 with fp32 m / l / accumulator, P cast to the V dtype for the PV dot, fully
masked rows guarded to output 0; stores the base-2 LSE (fp32, `[B, H, L, L]`). Backward: `bwd_pre` (delta = sum(o dO)), `bwd_dkdv`
(key-owned dK, dV and a per-row dbias), `bwd_dq` (query-owned dQ); dq / dk / dv are written in bf16 straight into the projection
layout. The per-row dbias buffer is `[B, H*L, L, L]` in the bias dtype, reduced over the row axis by torch afterwards.

## Numerics

- Fused / generic, training: LN statistics fp32, projections bf16 out of fp32 accumulators, sbias = `bf16(bias / scale)`, core
  operands bf16 with fp32 softmax and accumulation, the gated out-projection and residual in one rounding. Parameters may be fp32
  (master) or bf16; they are packed to bf16 and the gradients come back in the parameters' dtype (fp32 outputs whenever any
  parameter is fp32).
- Inference (`INFERENCE_FP16 = True`): q / k / v / sbias in fp16 (3 more mantissa bits than bf16), the score sum rounded to fp16
  once per k16 step; range `|q.k + bias/scale| < 65504`.
- dh core: bf16 operands, fp32 accumulation, P rounded to bf16; bias in natural units.
- The generic path's cublasLt plan picks its algorithm by timing the heuristic's top candidates once per shape (outside a capture);
  `dyln` is bf16 (the reference's own precision there).
- Determinism: parameter gradients come from fixed-order partial sums (no float atomics); dbias is accumulated in slabs in a fixed
  order.

## Launch configs

`kernels/triangle_attention/configs/h100_sm90.json` has entries only for the Triton core: `triangle_attention_fwd_triton` (keys
`bfloat16+float32|H,HEAD_DIM,L`: (H, HEAD_DIM) = (2, 32), (4, 16), (4, 32), (4, 64), (8, 32), (8, 64), (12, 32)), `..._bwd_pre_triton`,
`..._bwd_dkdv_triton`, `..._bwd_dq_triton` (`HEAD_DIM` 16/32/64); every key at L 128, 256, 384, 512, 640, 768. A key without an entry
runs the kernel's default config (warned once). The CUDA kernels take no table: their schedules are compile-time constants (`TA_TAU`, `TA_SPEC`, `TA_ORDER`, `TA_DYN`, tile 128, 2 rows per task).

## Limits / unsupported calls

`cuda` (`_cuda_route`): CUDA tensor on capability (9, 0) (capability (8, 0) goes to `_cuda_route_sm80`, the A100 path); bf16 pair (no fp32 kernel); self-attention, no q / k RMSNorm; square
pair, bool `[B, L]` mask. sm90 route: d_pair = to_query in-features, multiple of 8, <= 512; head width 16 / 32; <= 16 heads; L % 16
== 0 (training: head width 16 / 32 / 64). Training (grad or live dropout) runs on the sm90 route for every class and variant (fused for
the standard module at d128 x 4 heads, else width-generic). dh route (inference only): head width 32 / 64, d_pair % 8 == 0 and <=
512. Everything else returns a reason (`BackendUnsupported`). `triton`: bf16 pair only; the bias-only form
(`use_self_attention=False`) is refused by every library / engine backend of the single-direction classes. Heads of 96 / 128 channels have no engine kernel.

## Rejected approaches (forward / backward core)

From the 2026-10 H100 notes: fp32 bias via LDS.128 (shared-memory bandwidth), bias fragments held across rows (spills), 3-4 pair
rows per task (register pressure), exponentials split onto the FMA pipe, head-slowest task order (K / V re-read from DRAM 2.5x),
bf16 packing on the integer pipes, a single-pass backward with dQ in shared
memory (dbias partials too large). A wgmma left in flight across a loop back edge makes ptxas serialize every wgmma (C7514);
stages read with ldmatrix and released to TMA need `fence.proxy.async` before the release.

## Historical numbers

Historical (from the 2026-10-15 note and its bench script, since removed; H100 80GB, CUDA graph, B = 1, starting node, no mask, fp32 masters; not
re-measured under the current measurer). ms of this path, and in parentheses the faster of PyTorch-compiled / cuEquivariance
divided by it:

| mode, d_pair | L128 | L384 | L768 |
|---|---|---|---|
| inference 64 | 0.025 (2.34x) | 0.152 (2.38x) | 0.735 (2.73x) |
| inference 128 | 0.033 (2.41x) | 0.265 (2.70x) | 1.345 (2.94x) |
| inference 256 | 0.065 (2.00x) | 0.610 (2.30x) | 3.096 (2.53x) |
| inference 512 | 0.147 (1.69x) | 1.423 (1.97x) | 7.442 (2.44x) |
| training 64 | 0.122 (1.70x) | 0.656 (2.58x) | 2.909 (3.40x) |
| training 128 | 0.133 (2.29x) | 0.942 (3.43x) | 4.955 (3.94x) |
| training 256 | 0.296 (1.75x) | 2.231 (2.88x) | 11.78 (3.44x) |
| training 512 | 0.569 (1.81x) | 4.882 (2.60x) | 27.54 (3.23x) |

Same note, inference against Anthropic's cell: 1.28-1.57x at d_pair 128 and 1.07-1.25x at d_pair 256 (L128-768); its cell served
no other width.

## History

The previous, longer notes were deleted; see git history.
