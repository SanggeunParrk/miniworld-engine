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
| `triangle_attention_fwd_fp16_sm90` | forward_infer | `triattn_fwd_h16` (`triattn_h16_sm90.cu`): the inference core on fp16 q / k / v / bias, bf16 out; takes the key mask (live key tiles counted in the kernel) and, for AF3's ending node from the one-kernel front, reads the bias tiles transposed |
| `triangle_attention_tail_sm90` | forward | `tri_tail` (d128): `x + drop o ((sigmoid(g) o o) Wo^T [+ bo])` in one pass |
| `triangle_attention_front_wide_sm90`, `triangle_attention_pack_wide_sm90` | forward_infer | `tri_front_wide` / `tri_pack_wide`: the d128 front design at d_pair 256, 8 heads (groups of four CTAs, one projection each) |
| `triangle_attention_gate_bwd_sm90` | backward | `tri_gate_bwd` (d128): `dyd = dy o drop`, `du = dyd Wo`, do, dg, Delta per head, dWo partials, dpair seed = dy |
| `triangle_attention_bwd_kv_sm90`, `triangle_attention_bwd_q_sm90` | backward | `triattn_bwd_kv` / `triattn_bwd_q` (`triattn_bwd_sm90.cu`): key-owned dK / dV and query-owned dQ / dbias, P recomputed from the LSE (both paths) |
| `triangle_attention_head_bwd_sm90` | backward | `tri_head_bwd` (d128): input gradient of the five projections, LN backward from recomputed statistics, bf16 TMA reduce-add into dpair |
| `triangle_attention_wgrad_sm90`, `triangle_attention_wgrad_reduce_sm90`, `triangle_attention_param_finish_sm90` | backward | `tri_wgrad` (d128): weight / LN-parameter gradients from per-CTA partial slots, summed in a fixed order, then the parameter algebra `dW_t = gamma M_t + beta cs_t` |
| `triangle_attention_generic_pack_sm90` | forward | glue `pack`: Wcat = [Wq; Wk; Wv; Wg; Wb (heads rounded up to 8 rows)] bf16 and Wo; 16-channel heads zero-padded to 32 |
| `triangle_attention_generic_layernorm_fwd_sm90` | forward | glue `ln_fwd`: y = LN(x) (bf16, or fp16 for fp16 inference GEMMs) + fp32 mean / rstd; also the dh path's LN |
| `triangle_attention_front_fused_sm90` | forward_infer | `front_fused3_sm90.cu` `front`: LN + [Wq; Wk; Wv; Wg; Wb] in one kernel for short inference (`_FRONT_L`: d_pair 64 up to L768, wider up to L128; `_FRONT_L_COMPACT`: AF3 d_pair 64 with 16-channel heads kept compact up to L2048) |
| `triangle_attention_generic_front_bias_sm90` | forward | glue `front_bias`: the bias GEMM's [M, Hp] output -> head-major scaled / masked sbias |
| `triangle_attention_generic_residual_sm90` | forward_train | glue `residual`: `x + drop o t` (training with dropout; without dropout the residual is in the cublasLt epilogue) |
| `triangle_attention_generic_gate_bwd_in_sm90`, `..._gate_bwd_out_sm90` | backward | `ddrop = dy o drop`; do, dg into `dall`, Delta per head |
| `triangle_attention_generic_assemble_bwd_sm90` | backward | glue `assemble`: fp32 dbias -> the bf16 bias columns of `dall` |
| `triangle_attention_generic_layernorm_bwd_sm90` | backward | glue `ln_bwd`: dx = dy + LN backward of `dall Wcat`, per-CTA gamma / beta partials |
| `triangle_attention_dh_fwd_sm90` | forward_infer | `triattn_dh_sm90.cu`: mma.sync core for head widths 32 / 64 with the gate fused (`u = sigmoid(g [+ gb]) o softmax(...) v`) |
| `triangle_attention_dh_fwd_train_sm90` | forward_train | the same kernel with `TRAIN`: also writes O (bf16) and the base-2 LSE for the backward (64-channel heads in training) |
| `triangle_attention_dh_bwd_kv_sm90`, `triangle_attention_dh_bwd_q_sm90` | backward | `triattn_dh_sm90.cu`: mma.sync FlashAttention-2 backward, P recomputed from the LSE; key side dK / dV, query side dQ and dbias into per-slab fp32 partials (deterministic) |
| `triangle_attention_d64_fwd_sm90` | forward_infer | `triattn_d64_sm90.cu` (`h100_dh.ext_d64`): the inference core of 64-channel heads at L % 128 == 0 on wgmma, fp16 operands and fp16 scores (bias as the score accumulator's initial value, f16x2 exponentials, [O \| l] += P [V \| 1]), the gate fused |
| `triangle_attention_fwd`, `..._bwd_pre`, `..._bwd_dkdv`, `..._bwd_dq` | forward / backward | Triton core (`triton/main.py`): flash-style forward with fp32 LSE, FA2-split backward (delta, key-side dK / dV / per-row dbias, query-side dQ) |
| `triangle_attention_generic_gate_u_sm90` | forward | glue `gate_u` (`triattn_generic_sm90.cu`): `u = sigmoid(g) o O`; used by the bidirectional path |
| `triangle_attention_bidir_gate_bwd_sm90` | backward | `triattn_bidir_sm90.cu` `gate_bwd` (`cuda/h100_bidir.py`): do, dg of the bidirectional gate on channel planes |
| `triangle_attention_bidir_logits_sm90` | forward | `triattn_bidir_sm90.cu` `logits` (`cuda/h100_bidir.py`): the bias planes -> fp32 softmax logits in one pass (widen, ending half transposed, masked keys -inf); `torch.softmax` follows |
| `triangle_attention_bidir_tail_sm90` | forward | `triattn_bidir_sm90.cu` `tail`: `out = x + (sigmoid(G) o O)^T Wo^T` in one mma.sync kernel (u formed in shared memory, product rounded to bf16 then x added: the bits of gate_u + torch.addmm); d_pair 128; in training (u saved) the kernel also writes u |
| `triangle_attention_bidir_gate_bwd_fused_sm90` | backward | `triattn_bidir_sm90.cu` `gate_bwd_fused`: du = Wo^T dy^T (mma.sync, k16 steps in order, bits of torch.mm) and the gate backward in one kernel, no du planes; d_pair 128, K in (128, 256) |
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
  Inference (`tri_tail_sm90<true>`): the gate is projected here from x (`g = LN(x) Wg^T`, fp32). Ending node without a dropout scale
  (`tr = 2`): the tiles are rows of the pair as stored -- x and the output go through plain 2-D maps and only o is read through the
  4-D transposed map (one strided stream instead of two, the store contiguous; the same values bit for bit; same-job A/B 2026-11-13,
  ending tail L1024 / 1536 / 2048: 0.314 / 0.748 / 1.307 -> 0.281 / 0.618 / 1.108 ms, equal to the starting node).
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
The LayerNorm is folded into the pack (`tri_pack_wide(..., lnw, lnb)`): `y W^T = r (x Wf^T - mu S1) + S0` with `Wf = W diag(gamma)`
(bf16), `S1 = Wf 1`, `S0 = W beta` (fp32, summed in fp64; `fold [2, 4 C + 8]`), so the GEMM operand is x as loaded (no normalised
bf16 copy of the tile, one rounding fewer), the chains are issued right after the ldmatrix loads and the row statistics (two-pass
mean / variance in fp32, as before) are computed while they run; the epilogue is two FMAs per output. The front is bound by the
tensor core: four 256 x 256 projections of L^2 rows, GEMM-only ablation (no loads / stores / statistics) 1.07 ms of 1.25 ms at L1024;
one m64n256 chain or step-interleaved halves were not faster (1.09 / 1.20 ms), the CTA-group lockstep `FW_LAG` 3 still wins over
free-running (1.26 vs 1.32 ms).
Training at this width runs the width-generic path.

## Width-generic path: d_pair <= 512 (multiple of 8), heads of 16 / 32 channels, <= 16 heads

`cuda/h100_sm90_generic.py` + `triattn_generic_sm90.cu`. The core is the fused path's (head dim 32 at any head count; 16-channel
heads are zero-padded to 32 in the packed weights, the softmax scale stays `1/sqrt(16)`). Around it: hand-written LN / bias /
residual kernels and cuBLAS GEMMs (bf16 in, fp32 accumulate).

- forward: `ln_fwd` -> `[q | k | v | g | bias] = y Wcat^T` -> `front_bias` -> core with the gate in its epilogue (`u = sigmoid(g)
  o O`; O itself only when the backward needs it) -> `x + u Wo^T` (cublasLt, residual and RF3 out bias in the epilogue) or, with
  dropout, `t = u Wo^T` then `residual`. Short inference uses one GEMM for q | k | v | g | bias (L <= 256) or the one-kernel front
  (`front_fused3`, `_FRONT_L`); fp16 inference beyond L 256 uses fp16 GEMMs.
- inference at d_pair 64 x 4 heads x 16 (2026-11-14; 3 launches: front, fp16 core, cublasLt out-projection): the fp16 core takes the key
  mask itself (`h100_sm90.key_args`: its producer warpgroup counts each batch element's live 128-key tiles once per CTA, after requesting
  the first task's Q and first key tile -- the seven `key_tiles` glue launches were ~9 us of a 46 us L128 call); 16-channel heads issue one
  k16 score step (the padded half adds exact zeros) and, since 2.5.10, run every other exponential pair on the FMA polynomial (`TA_POLY16` = 8 of 16
  pairs; before: all on the MUFU). Not bit-identical (P changes within its fp16 rounding): over 42 served cases (TriangleAttention 64:4:16 and AF3 d_pair
  64, both nodes, L128-2048, B1 / 2, masks none / tail / random) the update's error vs a float64 PyTorch reference is 1.002x the previous core's on
  average (max 1.61e-2 vs 1.57e-2 of the update's max; module output error max 3.37e-3 both); 32-channel heads unchanged (66 / 66 cases bit-identical).
  Module time, CUDA-graph replay, previous core vs this one interleaved in one process (L512 / 768 / 1024 / 1536 / 2048, B1): 0.97 / 0.95 / 0.93 /
  0.91 / 0.89x (TriangleAttention both nodes and AF3's starting node; AF3 ending node 0.95 / 0.93 / 0.92 / 0.91 / 0.91x);
  the front runs on two weight stages at d_pair 64 and stores no LayerNorm statistics; AF3's ending node stores its bias as the front's
  rows have it (whole masked rows) and the core reads those tiles with `ldmatrix.trans` (`bias_transposed`, the `BTR` template) instead
  of 16-bit stores at stride L (1.3 us at L128, 10 us at L512). Measurer 4 (graph, cold L2) ms before -> after, anthropic: L128 0.045-0.047
  -> 0.033-0.035 (0.035); L256 0.111-0.118 -> 0.093-0.098 (0.101-0.102); L512 0.449-0.465 -> 0.416-0.422 (0.458-0.464).
- fp16 inference with the unfused front at d_pair 128 < C <= 512 (AF3 / Protenix d_pair 384 x 12 heads x 32; `FRONT_LNB`, default on; 2026-11-15): the
  pair bias comes out of the LayerNorm kernel (`ln_bias_fwd`: LN rows in shared memory, the bias GEMM on `mma.sync` m16n8k16 -- the bits of the cuBLAS
  fp16 bias GEMM -- stored head-major, scaled and masked), no `y` / `[L^2, Hp]` round trip and no `front_bias` kernel. Starting node only (`not trans`);
  the row-blocked path (L >= 2048) calls `ln_bias_blocks` per block (no `y` written, the q | k | v | g GEMM re-reads the pair). Module output
  bit-identical (`tests/gpu/test_triattn_h100_front_lnb.py`). Measured on the module (warm): +0.5..1.7 % at L256-1536, within noise at L2048.
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

### 64-channel heads: the fp16 wgmma core (2.5.13 candidate, `cand/ta-d64`, 2026-11-17)

At L % 128 == 0 the 64-channel heads (AF3 d_pair 512 x 8, RF3 d_pair 64 x 4 x 64, Chai-1 / confidence d_pair 256 x 4) take
`triattn_d64_sm90.cu` (`ext_d64`), rebuilt on the design of the 32-channel fp16 core (`triattn_h16_sm90.cu`); it replaced the
bf16-operand / fp32-softmax wgmma core of 2.5.12 (same file, same kernel name `triangle_attention_d64_fwd_sm90`).

- Front (`h100_dh._front_fp16`, outside any autocast): the glue's LayerNorm in fp16 (`ln_fwd(h16)`; for d_pair in (128, 512]
  `ln_bias_fwd`, which also writes the head-major `fp16(bias / scale)`; the AF3-orientation ending node's bias from the pair as it is by
  `ln_bias_blocks`), `q | k | v | g = y W^T` as one cuBLAS fp16 GEMM (weights rounded to fp16 from the fp32 masters); d_pair <= 128:
  `ln_fwd` + fp16 bias GEMM + `front_bias`. Tail unchanged.
- Core: CTA task = (b, h, 128 queries) x 2 pair rows; warpgroup 0 = TMA producer, warpgroups 1 / 2 = 64 queries each; 128-key tiles;
  the bias fragment (ldmatrix of the fp16 tile, shared by the two rows) IS the initial value of the fp16 score accumulator
  (`m64n128k16.f16`, 4 k16 steps); `x = s scale log2e - m` (HFMA2), `p = 2^x` on f16x2 pairs (MUFU.EX2.F16x2, every other pair on the
  FMA-pipe polynomial), the lazy row offset with a warp vote (TAU 8); P is the A fragment of `[O | l] += P [V | 1]` (`m64n72k16`,
  fp32), the ones block is ONE 2 KiB block reached through each V descriptor's leading-byte offset (`desc_mv`), and the Q tiles travel
  through the K | V ring (the stage also carries the task id and its live key tiles: no separate Q buffer, room for 4 stages of 32 KiB);
  the epilogue's gate rows are prefetched into L1 before the last sub-step; dynamic task counter; head groups
  whose bias fits 20 MB of L2 run together.
- Profile of the 2.5.12 core (ncu, 512:8:64 L1024, locked clocks): tensor pipe 33 % active, issue 38 %; stall samples: the bias
  bf16 -> fp32 unpack (IMAD.U32) 23 %, MUFU.EX2 19 %, FMNMX 11 %, mbarrier spin 9 %; FA4 (no bias) 51 % tensor. The fp16 design removes
  the unpack, 3/4 of the MUFU work and the per-logit max.
- Variants measured (each one's speed relative to the default, -D knobs of the source): 64-key tiles 0.87-0.96x; FA3 pingpong of the two
  consumer warpgroups 0.91-1.01x; 4 / 0 of 16 pairs on the polynomial 0.93-1.01x / 0.86-0.95x; no gate prefetch 0.92-0.98x (sustained); all within the same accuracy.

Core alone (one process, CUDA-graph replay, interleaved, H100 SXM 700 W cap; B = 1, N = L pair rows; us sustained / cool; the 2.5.12
core built from 6bb7d21fb's source; FA4 4.0.0b32 / cuDNN 9.2 SDPA in fp16, without the pair bias unless noted):

| cfg | L | 2.5.12 core | fp16 core | speed-up | FA4 (no bias) | cuDNN (no bias) | cuDNN + bias |
|---|---|---|---|---|---|---|---|
| 512:8:64 | 512 | 1315 / 1213 | 964 / 927 | 1.36 / 1.31 | 896 / 797 | 695 / 656 | 1414 / 1235 |
| 512:8:64 | 768 | 3800 / 3452 | 2653 / 2532 | 1.43 / 1.36 | 2334 / 2084 | 2281 / 2076 | 4321 / 3886 |
| 512:8:64 | 1024 | 8216 / 7540 | 6442 / 6019 | 1.28 / 1.25 | 5736 / 5065 | 5215 / 4750 | 9944 / 8883 |
| 512:8:64 | 1536 | 24867 / 24056 | 18529 / 16342 | 1.34 / 1.47 | 15433 / 14394 | 17989 / 15552 | 36432 / 33682 |
| 256:4:64 | 512 | 670 / 618 | 515 / 484 | 1.30 / 1.28 | 449 / 403 | 348 / 334 | 695 / 626 |
| 256:4:64 | 768 | 1845 / 1692 | 1331 / 1244 | 1.39 / 1.36 | 1164 / 1046 | 1143 / 1040 | 2194 / 1949 |
| 256:4:64 | 1024 | 4081 / 3671 | 2838 / 2659 | 1.44 / 1.38 | 2920 / 2544 | 2655 / 2380 | 4941 / 4401 |
| 256:4:64 | 1536 | 12352 / 11124 | 8690 / 8240 | 1.42 / 1.35 | 8006 / 7122 | 8852 / 7766 | 17310 / 14797 |
| 256:4:64 | 2048 | 27945 / 25851 | 18688 / 18031 | 1.50 / 1.43 | 18837 / 17570 | 21857 / 18192 | 39788 / 36093 |

With the pair bias the fp16 core is 1.35-2.13x faster than cuDNN SDPA with the bias as `attn_mask` (sustained); it runs at 0.83-1.03x
of the bias-free FA4's speed (0.87-0.93x at L <= 768, where a task has only 4-6 key tiles and its prologue / epilogue are not overlapped).
Error vs fp64 of the
core (random q / k / v, bias std 2, gate 1): rms 6.3e-4 / 7.2e-4 against 1.7e-3 / 2.0e-3 of the 2.5.12 core, max 1.1-1.9e-2 against
3.0-3.8e-2 -- about 0.37x (fp16 operands carry 3 more bits; the bf16 P was the 2.5.12 core's largest rounding).

Whole module (`h100_dh.forward`, CUDA graph, interleaved, mask with the last L / 8 keys padded, us sustained; speed-up sustained / cool):

| class (d_pair, heads x 64) | L 512 | 768 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|
| AF3 512, 8, starting | 2983 -> 2683 (1.11 / 1.10) | 7343 -> 6533 (1.12 / 1.12) | 14491 -> 12337 (1.17 / 1.16) | 38127 -> 31380 (1.22 / 1.19) | 75141 -> 64088 (1.17 / 1.27) |
| AF3 512, 8, ending | 3311 -> 3010 (1.10 / 1.10) | 8160 -> 7183 (1.14 / 1.13) | 15444 -> 13268 (1.16 / 1.17) | 41048 -> 35214 (1.17 / 1.21) | 81248 -> 69240 (1.17 / 1.26) |
| RF3 64 (d_hidden 256), 4, starting | 1047 -> 948 (1.11 / 1.11) | 2674 -> 2270 (1.18 / 1.17) | 5246 -> 4185 (1.25 / 1.24) | 14481 -> 12093 (1.20 / 1.17) | 31276 -> 24251 (1.29 / 1.28) |
| RF3 64 (d_hidden 256), 4, ending | 1100 -> 991 (1.11 / 1.12) | 2734 -> 2364 (1.16 / 1.16) | 5413 -> 4373 (1.24 / 1.23) | 15042 -> 12359 (1.22 / 1.16) | 30814 -> 24447 (1.26 / 1.28) |
| Chai-1 256, 4, starting | 1319 -> 1164 (1.13 / 1.13) | 3276 -> 2750 (1.19 / 1.18) | 6423 -> 5046 (1.27 / 1.23) | 17425 -> 14230 (1.22 / 1.22) | 35391 -> 28223 (1.25 / 1.32) |
| Chai-1 256, 4, ending | 1440 -> 1257 (1.15 / 1.14) | 3567 -> 2986 (1.19 / 1.18) | 6821 -> 5426 (1.26 / 1.23) | 18247 -> 14891 (1.23 / 1.21) | 36741 -> 29128 (1.26 / 1.27) |
| Chai-1 confidence 256, 4, starting | 1621 -> 1493 (1.09 / 1.10) | 3963 -> 3452 (1.15 / 1.14) | 7550 -> 6293 (1.20 / 1.19) | 19697 -> 16130 (1.22 / 1.18) | 39934 -> 32886 (1.21 / 1.25) |
| Chai-1 confidence 256, 4, ending | 1788 -> 1616 (1.11 / 1.11) | 4340 -> 3812 (1.14 / 1.14) | 8273 -> 6892 (1.20 / 1.18) | 21332 -> 18027 (1.18 / 1.18) | 42356 -> 35801 (1.18 / 1.23) |

Accuracy (`measure/accuracy.py`, the eval cases of requirement rows 80 / 81 / 84 / 85, pytorch + cuda): all 48 measurable cases
(L 128-768; L >= 1024 the fp64 gold does not fit one H100 for either core) GREEN with both cores; `max|y - y64| / max|y64|` of the
fp16 core / the 2.5.12 core 0.84-1.32 (mean 0.98: the module output's bf16 rounding dominates that metric), worst err / tol 0.63
(Chai-1 confidence L640 starting, 5.04e-3 against tol 8e-3).

## Bidirectional path: `BidirectionalTriangleAttention`, bias-only

`kernels/integrations/bidir_triattn_h100.py` (forward, inference and training, and backward). Bias-only weights do not depend on
the row, so the softmax is formed once per (sample, head) and each direction is a plain GEMM over channel planes `[R, L L]`: the
width-generic glue `ln_fwd` -> `[Wv; Wg; Wb] y^T` (one GEMM) -> fp32 softmax (starting: bias plane as it is; ending: transposed),
bf16 P -> per-head GEMMs -> glue `gate_u` -> `pair + U^T Wo^T` (residual in the GEMM's C). Backward: `gate_bwd`
(`cuda/h100_bidir.py`, `triattn_bidir_sm90.cu`) -> softmax gradients -> `dW`, `dy_ln` (fp32) -> glue `ln_bwd`. Refusals
(`refusal`): bf16 pair, `use_self_attention=False`, d_pair % 8 == 0 and <= 512, L % 8 == 0, head width % 8 == 0, fp32 LN affine.

Notes (2026-11-16, `fp8/bidir-ta`; the TriAttn q.k core work of the single-direction paths (Y16 front, TA_SW / N24 / K16, FR, FP8 PV) has no counterpart here: this path has no flash core, its contractions are cuBLAS GEMMs over channel planes).

- Users: no model of the engine's registry or of FoldForge / MiniWorld builds the class (registry: engine-own, d_pair 128, 4 heads, L 128-768, eval and train; the single-direction bias-only `TriangleAttention` rows share the path). Protenix / Boltz / OpenFold3 / AF3 / RF3 / ESMFold2 / OpenDDE run `TriangleAttention*` with q.k attention, not this route.
- Default route, inference (outputs bit-identical to 2.5.9, `tests/gpu/test_bidir_triattn_h100.py`): `logits` (one pass in place of float / cat / transpose copy / masked_fill), the planes GEMM `[Wv; Wg; Wb] y^T` through the fastest bit-equal cublasLt algorithm (calibrated once per shape against `torch.mm`, outside a graph capture; none equal or not calibrated: `torch.mm`), the fused tail (`tail`: no u planes, no gate_u pass, no residual copy), `dyln` written in place in the backward. Switches: `BIDIR_FUSED_TAIL=0` (tail off). The weights are cast to bf16 and concatenated in one launch (`cast_weights`). Graph replay (inference, one sample) L = 128 / 256 / 384 / 512 / 640 / 768: 64.3 / 181.6 / 387.7 / 655.2 / 1025.7 / 1507.6 us (2.5.9) -> 54.6 / 142.7 / 294.2 / 485.4 / 767.0 / 1140.0 us. L = 768 per stage (device us): LayerNorm 109, planes GEMM 294, starting contraction 166, ending contractions (4 launches) 181, fused tail 337, small kernels (logits, softmax, casts) ~45. Training (forward + backward): L = 768 4624 -> 4351 us (planes GEMM, dyln in place).
- Tried, no gain: a fused LayerNorm + planes GEMM kernel (mma.sync, bit-identical output): 488-555 us at L = 768 against 109 + 294 us for ln_fwd + the cublasLt GEMM (shared-memory bound: every warp reads the whole weight chunk; kept in `~/miniworld-engine-scratch/fp8/bidir-ta/experiments/`, not in the tree).
- FP8 PV (e4m3 P and V) for the starting contraction: no-go. 4 x `torch._scaled_mm` at L = 768 131 us against 171 us bf16, plus the V quantization pass (>= 74 us at the byte floor, 190 us as two torch ops), and a relative error of 3.8e-2 (single-direction route: hi ~2 %); the ending contraction (`P V`, V contracted along its row axis) would need a transposed V plane as well.

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
- dh mma.sync core (L % 128 != 0, and training): bf16 operands, fp32 accumulation, P rounded to bf16; bias in natural units.
- dh wgmma core (64-channel heads, L % 128 == 0, inference): y, q / k / v / g and bias / scale in fp16 (fp32 GEMM accumulators rounded once),
  the score sum rounded to fp16 once per k16 step, P in fp16, O and the row sums in fp32; range `|q.k + bias/scale| < 65504` (the bias
  is clamped at +-6e4 by the glue).
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

## Register / concurrency experiments (2026-11)

fp16 inference core `triattn_h16_sm90.cu`, measured from 2.5.12 (core only, B = 1, N = L rows, CUDA graph, interleaved, sustained +
cool, against a separately built identical cubin as the noise control: 0.96-1.03 sustained, 0.99-1.04 cool). The experiment macros
live on branch `exp/ta-regs` only; the release keeps `TA_RC` (default 1 from 2.5.13).

**Where the consumer registers went (2.5.12, 32-channel kernel, 232 allotted, peak 178 live, 0 spills).** 48 of the 178 were an
artefact of the loop shape: the other row's probabilities (32: its PV had retired, but the loop's `j + 1 < nk` test gave ptxas a path to
a later read) and the gate prefetch `gpre` (16: defined under `j == nk - 1`, so loop-carried through every sub-step).

**Kept: `TA_RC=1`** (2.5.13 default) -- the last key tile peeled out of the consumer loop, the gate prefetch after the loop. Same
operations in the same order: bit-identical. ptxas spends the freed registers hoisting the next sub-step's bias ldmatrix over the
exponentials (peak 188, 0 spill). Core ratio 2.5.12 / `TA_RC` over 128:4:32, 64:4:16, 256:8:32 x L 512 / 1024 / 2048: sustained
1.00-1.11 (64:4:16 1.07-1.11), cool 1.03-1.16. ncu (128:4:32, base clocks): L1024 2.18 -> 2.11 ms, L2048 15.70 -> 15.31 ms; issue busy
+0.5-0.9 pt, eligible warps / scheduler 0.76 -> 0.80, instructions -1.8 %.

**Tried, no gain** (ratio vs 2.5.12, sustained):

* lower `setmaxnreg` with `TA_RC` (160 / 176 / 192): 0.92-0.98 / 1.03-1.09 / 0.98-1.08 (all below 232);
* 3 consumer warpgroups (192-query tiles, 160 registers, 108-168 B spill): 0.89-1.04, also with Q from shared memory;
* QK^T as an SS wgmma (Q read from shared memory, double-buffered, -16 registers): 0.97-1.05;
* 2 CTAs per SM (producer + one consumer warpgroup each, 64-query tiles, 216 registers, K / V streamed twice): 0.92-1.07, a win only
  at 64:4:16 L <= 1024;
* 64-key sub-steps (peak 155 registers): 0.88-0.99, and not bit-identical (the offset may rise every 64 keys);
* `TA_FR=1` / `TA_SW=0` on top of `TA_RC`: below `TA_RC` alone.

The core is not starved for warps: more warpgroups lose to the register cap or the doubled K / V stream, and 64-key steps double the
wgmma issue / fence work. The one win is the loop shape.

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

Notes (2026-11-16, `bidir-ta2`, training and the tail; all bit-identical to 2.5.9: outputs and every gradient hashed against the 2.5.9 tree on 114 cases, `tests/gpu/test_bidir_triattn_h100.py`, and 420 kernel-level random trials of 6.3e9 elements with Gaussian / heavy-tailed / outlier data without a mismatching bit).

- The fused tail also serves training (it writes the saved u): gate_u + residual copy + the badd out-projection become one kernel (L = 768 train 4146 -> 3953 us). The tail itself runs 16-channel chunks in a 4-deep cp.async ring (each thread forms u for the units it loaded, one barrier per chunk): L = 768 338 -> 323 us.
- `gate_bwd_fused`: du = Wo^T dy^T (mma.sync, k16 steps in order = the cuBLAS bits) fused into the gate backward, no du planes: 173 + 523 us -> 586 us. Ablation of the fused kernel at L = 768: GEMM alone 154 us, memory traffic alone 527 us, both 545 us (the GEMM hides under the traffic; the 1.36 GB of the four planes at 3.35 TB/s is 406 us). 128-token tiles with 256-byte row segments (594 us) and an early prefetch of the g / o units (565 us) were slower than 64-token tiles with 128-byte segments through a per-warp smem staging (545-586 us).
- `softmax_bwd_planes`: d logits = p (dp - s) written into the (transposed) bias-gradient planes in one pass (s = the torch sum, so the reduction order is torch's): about -60 us at L = 768.
- Training L = 768 (fwd + bwd, CUDA-graph replay, same harness): 4428 -> 3775 us (-14.7 %); inference 1508 -> 1122 us (-25.6 %, of which this branch's 2.5.9-relative gains before the tail ring are the 2.5.10 candidate's own).
- Tried, no gain: the ending contraction as one `cublasGemmBatchedEx` (pointer arrays, 128 batches): bit-equal, but 177 us with prebuilt pointers against 190 us for the 4 strided bmm launches (event timing), 209-212 us with the pointer arrays built per call: no pay-off once the build is counted, and the bit-equality of other cuBLAS algorithm choices (other L) would have to be proven per shape. Tail with the epilogue in registers and a ring that runs across tiles: 352 us against 323 us (slower). Fusing dyln = dc^T W with `ln_bwd`: not bit-identical possible (the gamma / beta partials of `ln_bwd` are accumulated in a grid-stride row order of its own).

## Opt-in FP8 PV route (inference, head width 32 / 16; `h100_sm90.FP8_PV`, default off)

`triattn_h16_sm90.cu` built with `-DTA_F8PV=1` (`h100_sm90._core_f8pv`); the default build is unchanged bit for bit (checked against the previous
source on five shapes, with / without gate and key mask). `FP8_PV = True` makes every fp16-core call site (`core_h16()`: the fused d128 path,
the wide d256 path, the width-generic path) run the variant; the LSE and anything else it does not take run the fp16 core.

- Score GEMM unchanged: q, k, the bias and S stay fp16 (FP8 Q / K made the output error 1.4-5 % on the real dumps and, with an fp32 score
  accumulator instead of the fp16 pairs, would give up the f16x2 softmax: measured slower at 4 heads, <= 9 % faster at 8).
- P and V on e4m3: `[O | l] += P [V | 1]` is four `wgmma m64n40k32 e4m3` per 128 keys (eight k16 before). V is stored transposed `[B, N, H, dh, S]`
  (K-major B operand), the keys of each group of 32 permuted (`vt_perm`) so the accumulator layout of S is already the k32 A fragment of P (no
  shuffles); 8 rows of e4m3 ones behind the 32 V^T rows give the row sums from the quantized P. Heads of 16 channels: the TMA box's rows 16-31 are
  zero fill (3-D map).
- P as e4m3 bits straight from the logits, no MUFU, no `cvt` (a `cvt.e4m3x2.f16x2` per pair costs as much as the exponential: the F2F rate): with
  `x` = the logit in log2 units minus (offset - 8), `byte = round(8 max(x, -7) + 56)` by one magic add (1024 + y in fp16) and one PRMT per four
  probabilities. That is the piecewise-linear 2^x (up to 6 % high mid-octave) in e4m3 bits, the largest probability 2^8..2^8.5, so the lazy offset
  uses `TA_TAU 0.5` here (8 in the fp16 core).
- V scale per (batch, head): amax / 448, by `vt_quant` (`vt_amax_k` + `vt_quant_k`, one extra read of v), or a fixed amax (`FP8_PV_AMAX`, no
  pass; larger values saturate). The epilogue multiplies by scale / l.
- Where it stands: V^T is made by a standalone pass (the fronts still write fp16 v), so the module gain at L <= 1024 is about nothing
  (d128 L1024: 2.42 -> 2.37 / 2.33 ms dynamic / static scale); the core alone is 14-34 % faster. Writing V^T from the fronts is the remaining step.
- Accuracy (2026-11-14, real dumps of Protenix v1 / Boltz-2 / AF3 / OpenDDE, 6 layers of 5I28 and 8c3l): output error against fp32 1.1-2.7 % per call
  (fp16 core 0.18 %, bf16 rounding 0.17 %); V on e4m3 is the larger part (1.0-2.3 %), the bit-mapped P 0.65-1.4 % (exact e4m3 P 0.4-0.75 %), Q / K on e4m3 would be 1.4-5 %.
  End to end (FoldForge, 200 steps, trunk seeds 0-3, same protocol as the bf16 baseline): Protenix v1, 18 FoldBench picks: the change at the same trunk seed is
  inside the bf16 trunk-seed spread on 17 (mean RMSD 0.06-0.9 A against 0.13-6.1 A); 8taa (DNA) has one seed 7.0 A from its bf16 twin (bf16 seed pairs up to 2.6 A).
  Boltz-2, 11 picks: 10 inside, 9c90 outside (3 of 4 seeds flip to ipTM ~0.41, RMSD 12.6 A; bf16 seeds agree to 0.2 A); V rounded through e4m3 on the fp16 core
  (`FP8_PV_MODE = "v_only"`) flips it too.  (2026-11-14, later: 9c90 is bistable, see the hi / lo section below -- the flip does not isolate V.)  Not for default use.

### FP8 PV, second round (2026-11-14): where the module time goes, V as hi / lo planes, V^T written by the front

Per-kernel time of the real inference module (d_pair 128, 4 x 32, torch profiler, ms per sample, starting node, no mask; compiled cubins of the default route):

| L | front | core | tail | module | front GB/s | tail GB/s |
|---|---|---|---|---|---|---|
| 256 | 0.028 | 0.048 | 0.020 | 0.100 | | |
| 512 | 0.103 | 0.249 | 0.070 | 0.428 | | |
| 1024 | 0.403 | 1.711 | 0.275 | 2.407 | 2.66 TB/s | 2.92 TB/s |
| 2048 | 1.598 | 13.98 | 1.061 | 17.06 | 2.69 TB/s | 3.04 TB/s |

Bound type: the front (reads the pair once, writes q, k, v fp16 + the bias: 1032 B per pair element) and the tail (o, x in, out) run at 88-100 % of the 3.0 TB/s a
torch copy reaches on this card (`bw.py`): HBM-bound, only fewer bytes help. The core is 70-82 % of the module from L 512 up and is not bandwidth-bound (0.5 TB/s):
the 2026-11-14 ablation of its pieces (`TA_ABL`, since removed) at L 1024 / 2048, 4 x 32: no QK^T wgmma -1.4 % / -7.9 %, no exponentials -23 % / -1 %, no PV wgmma
(its exponentials become dead code) -39 % / -43 % (8 x 32 at L 2048: -17 % / -23 % / -46 %): MUFU + conversion, the PV wgmma and the softmax bookkeeping, not the score GEMM.
Bytes: e4m3 q / k / v writes would cut the front by up to a third (6 % of the module at L 1024) but Q / K on e4m3 is 1.4-5 % output error (no-go); V as hi + lo
e4m3 planes is the same bytes as fp16; the tail's out-projection GEMM is not what bounds it, an e4m3 `o` would be the 2.65 % mantissa floor again.

Opt-in additions (default route bit-identical to origin/main ac7b5f5d7: the whole inference module, six shapes / orientations / masks, compared as bytes):
- `FP8_PV_HL = True` (`-DTA_F8HL=1`): V on two e4m3 planes (hi, and the rounding residual x 16 rounded again; `vt` is `[B, N, H, 2, dh, S]`), PV on `m64n72k32`
  ([O_hi | O_lo | l], the epilogue adds O_lo / 16).  V error ~2^-8 instead of 2^-4.  Real dumps (Protenix v1 / Boltz-2, 5I28 + 8c3l, all layers): output error against fp32
  0.5-1.5 % (hi only 1.7-2.8 %, fp16 core 0.18 %); what is left is the e4m3 probabilities.
- `FP8_PV_FUSED = True` + `FP8_VSCALE` (fp32 [4], powers of two) or `FP8_VSCALE_AUTO = True` (calibrated per weight pack on its first call: the power of two above 4 x the V
  absmax): the d_pair-128 front (`tri_front_vt`) writes V^T planes itself with the static scale, the standalone `vt_quant` pass is gone; values past 448 x scale saturate and
  set a device flag (`take_overflow()`; `rescale_all()` then redo).  With `FP8_PV_FUSED` only this route feeds the FP8 core (the wide / generic call sites stay fp16:
  the standalone pass costs more than the core gains there).  `FP8_PV_HL = False` writes the hi plane alone.
- Module time (ms per sample, median of 9 interleaved rounds, same-cubin control = default measured twice; the control differs by <= 1.6 %):

| d128 4 x 32 | L256 | L512 | L1024 | L2048 |
|---|---|---|---|---|
| default | 0.0974 | 0.4391 | 2.604 | 17.80 |
| default, control | 0.0974 | 0.4394 | 2.604 | 17.51 |
| hi + lo fused (`FP8_PV_HL`, fused) | 0.0987 (+1 %) | 0.4305 (-2 %) | 2.434 (-6.5 %) | 16.22 (-9 %) |
| hi fused | 0.0902 (-7 %) | 0.3979 (-9 %) | 2.236 (-14 %) | 14.56 (-18 %) |
| hi + lo, standalone V^T pass | 0.1153 (+18 %) | 0.503 (+15 %) | 2.713 (+4 %) | 16.99 (-4 %) |
| hi, standalone V^T pass | 0.1071 (+10 %) | 0.4654 (+6 %) | 2.528 (-3 %) | 15.11 (-15 %) |

  (d256 8 x 32, standalone pass only: hi +14 % / +9 % / -1 % / -10 % at L256 / 512 / 1024 / 2048, hi + lo +26 % / +19 % / +7 % / 0 %: no gain below L2048.)
  (d64 4 x 16, standalone pass, generic front: hi +5 % / -4 % / -15 % / -19 %, hi + lo +12 % / +3 % / -7.5 % / -12 % at L256 / 512 / 1024 / 2048; control <= 0.3 %.)
- Accuracy at the end level (FoldForge, 200 steps, trunk seeds 0-3, bf16 baselines from the same protocol): hi + lo and hi, standalone and fused (calibrated scales, no overflow
  in any run): Protenix v1, 9-18 FoldBench picks, Boltz-2 10 picks: same-seed RMSD inside the bf16 trunk-seed spread everywhere except Boltz-2 9c90 (see below).
  9c90 is bistable under numerical noise: relative noise of 0.1 / 0.2 / 0.5 / 1 / 2 % on the fp16 core's output (independent per element, four to six draws each) flips it in
  1/6, 3/6, 2/4, 1/4, 4/4 draws (to ipTM ~0.42, RMSD 12.6 A, bf16 ipTM 0.71-0.79); the bf16 seeds only agree to 0.2 A because the bf16 path is deterministic.  hi + lo (output
  error 0.5-1.5 %) flips 3 of 4 trunk seeds standalone and 2 of 4 fused, hi (fused) 4 of 4 deviate; no e4m3 P path is expected to hold it.  Protenix v1 on 9c90 does not flip.

### FP8 PV, third round (2026-11-15, branch `fp8/triattn`): all configs, the standalone pass made cheap, 12 heads, whitelist per call site

Measured against the 2.5.9 default (which keeps its N24 / K16 / software-pipelined 16-channel variants; F8PV still forces them off). Tables and the end-level protocol: `docs/fp8.md`.
- `vt_quant` served 8 heads at most and fell to `quantize_v_torch` (the 12-head d_pair 384 module was 1.4-2.4x slower than the default at L512..2048); it now serves 16 heads (`vt_amax_k` has one running max per thread: the block size is a multiple of C / 8, so a thread's head is fixed).
  `_F8Core` falls back to the fp16 core past 16 heads.
- Static per-site scale (`FP8_VSCALE_AUTO`, `quantize_v_static`): the first eager call of a site (weight pack, set by the integration around `_infer`) takes the dynamic scales and keeps the power of two above 4 x their absmax per head; later calls run
  one pass (`vt_quant(v, 0, vinv, ovf)`: no amax pass, no amax allocation, saturation sets the `_OVF` flag read by `take_overflow()`). Hi-plane module time, dynamic -> static: d128 L512 / 1024 / 2048 -5 % / -3.6 % / -1.1 %.
  `rescale_all` now scales in place, so a captured graph (which reads `1 / scale` inside the capture) follows a rescale.
- S gate for the unfused sites (`FP8_PV_MIN_S`, set by `enable_fp8_pv_for` for models whitelisted for "unfused"): 32-channel heads 1024, 16-channel heads 512 (hi), twice that for hi + lo. Below it the fp16 core runs (bit-identical).
- Real activations, core only (tiled to L1024 / L2048): the FP8 core is 24-28 % faster than the 2.5.9 core on Protenix v1 / Boltz-2 / AF3 / OpenDDE dumps, hi + lo 17-22 %; the dynamic V^T pass takes 0.23 / 0.88 ms (4 x 32) and 0.81 / 3.2 ms (12 x 32) of that.
- Not done, measured as the remaining ceiling: a fused V^T front for d_pair 256 / 384 / 64 (the fused d128 route vs the same route unfused: 9 % of the module at L1024, 3 % at L2048); (the N24 variant is now on the F8 core, see the fourth round below; K16 and SW do not help there).
- Default route: bit-identical to main ec11768fc on ten shapes (d128 start / end / flip / masked, d256 x 8, d384 x 12, d64 x 4 x 16, d64 x 4 / 2 x 32, L256..1536), compared as bytes.

### FP8 PV, fourth round (2026-11-15, branch `fp8/vt`): the V^T pass removed where the front is ours

The standalone `vt_quant` pass (one read of fp16 V, one write of e4m3 planes) ate 40-55 % of the FP8 core's gain at module level (it runs 52-95 % of the byte floor: launch bound below L512, near the floor for d_pair 64 from L1024); the extra read of V is the avoidable part. Now:
- `front_fused3_sm90.cu` `-DMW_F3_VTW=1|2` (bf16 front, d_pair 64, and the fp16-exact `MW_F3_Y16` front): the V block of the projection leaves as e4m3 hi (/ lo) V^T planes, quantized from the fp16-rounded accumulator with the static per-head scale (same cvt as `vt_quant_k`).
  The accumulator holds rows (16 wi + g8, + 8) x columns (8 i + 2 t4, +1); the key position of row `r` is `16 wi + 4 (g8 / 2) + 2 hf + (g8 & 1)`, so the four keys of one 32-bit word are the rows of a lane pair (g8 even / odd) at both halves:
  one `shfl.xor 4` and two `prmt` give each lane the whole word of one channel (lane parity picks channel 2 t4 + parity), one 32-bit store into a [channel][64 keys] stage (words XOR-swizzled by 4 x ((channel / 2) mod 4): conflict free),
  then 16-byte global stores of 64 contiguous keys per (head, plane, channel). The first version (byte stores per element) made the front 25 us slower than the pass it replaced; the word version costs about nothing (front with planes 0.125 vs plain 0.141 ms at 64:4:32 L512).
- `triattn_mod_sm90.cu` `tri_front_wide_sm90<CW, VTW>`: the group-2 CTA of the d_pair-256 wide front writes the hi plane the same way (4 channels a lane, 8 KiB stage past the kernel's smem per warpgroup; the default instance is unchanged).
- Gated per config (`FP8_PV_UNFUSED_FRONT_MIN_S`), with the first eager call of a site taking its static scale through the old pass; scale-derived tensors cached (`_VINV` / `_VSB`, updated in place by `rescale_all`).
- Tests: `tests/gpu/test_triattn_f8pv_h100.py::test_front_made_v_planes_equal_the_standalone_pass` (module output as bytes, hi / hi + lo, padded mask).
Module time (before -> after, hi, percent vs the 2.5.9 default): 64:4:32 L256 / 512 / 1024 / 2048 -0.3 / +0.5 / -4.6 / -13.9 -> -9.7 / -10.4 / -9.7 / -17.1; 64:2:32 +0.1 / 0.0 / -2.8 / -12.4 -> -4.7 / -6.4 / -5.6 / -15.9; 64:4:16 +0.2 / -7.9 / -17.0 / -17.6 -> -4.2 / -12.6 / -19.4 / -19.6;
256:8:32 (gate 768) -0.7 / +2.7 / -3.2 / -12.9 -> -0.3 / fp16 core / -7.3 / -13.6; the cuBLAS-front site (384:12:32) and the fused d128 site are unchanged. Full tables, the pass's share of the byte floor and the removal ablation: `docs/fp8.md`.

### FP8 PV core, fourth round (2026-11-15, branch `fp8/core`): ablation of the F8 core, the 16-channel PV as n24

- The F8 core has no MUFU: P is two HFMA2 + HMNMX2 (+ one HFMA2) per pair and a PRMT per four probabilities, so there is no exp share and `TA_POLY` does not apply. Removal ablation (`TA_ABL`, time only; the F8 bits are the PV wgmma 2, QK^T 1, the whole conversion 4,
  its parts 8 / 4096 / 8192 / 2048, the whole X phase 128, the offset machinery 16 / 32768 / 131072, bias ldmatrix 32, K / V^T loads 65536 / 256, epilogue stores 16384): PV 21-30 % of the core time, QK^T 3-20 % (real activations 3-8 %),
  conversion 3-12 %, whole X phase 7-28 %; loads and epilogue stores are not limiting (+-6 %). The offset machinery's 10-30 % ablation is an upper bound (it also zeroes the data and with it the power draw): halving the exact-path firing rate
  with the data kept (`TA_F8K=7 TA_F8TAU10=15`, not bit-identical) gains 0-9 % on synthetic logits and nothing on real activations. Full table: `docs/fp8.md`.
- Ported / tried on the F8 core: `TA_F8N24` (16-channel heads: PV as m64n24k32 over V^T rows 0-15 + an 8-row ones block, the V^T box 16 rows instead of 32 zero-filled): bit-identical, core -3.2 .. -4.9 % on 4 x 16, module 64:4:16 unfused hi -1.7 .. -2.8 %; default on.
  Level or worse (kept as macros, default off, all bit-identical except K / TAU): `TA_F8SW` (software pipelined sub-step; -3.4 .. +7.8 %), `TA_F8SWN`, `TA_F8QM=1 / 2` (fp16-domain quad max: hv -4 %, real +2 .. +10 %).
- Module against the 2.5.9 default (hi, static scale): 128:4:32 fused -6.2 / -9.7 / -13.9 / -17.0 %, unfused +5.3 / +1.3 / -6.6 / -12.8 %, 256:8:32 +8.5 / +2.5 / -5.4 / -11.4 %, 384:12:32 +0.5 / +2.4 / -4.9 / -11.3 %, 64:4:16 +3.8 / -8.6 / -18.4 / -21.4 % (L256 / 512 / 1024 / 2048).
- Default route bit-identical to main ec11768fc (ten module shapes, bytes); FP8 module output bit-identical to `fp8/triattn` (ten shapes). The F8 tests were not re-run on this branch (the output is unchanged; run through the code graph before merging).

### FP8 QK (EXPERIMENTAL, 2026-11-17, branch `fp8x/triattn`; `h100_sm90.FP8_QK`, default off)

On top of the FP8 PV route, d_pair 128 fused route only: the front writes q / k as e4m3 (per-head product-1 power-of-two scales) and the F8 core runs `S += Q K^T` as
one `m64n128k32.f16.e4m3.e4m3` wgmma (`-DTA_F8QK=1`). Module 0.93-0.95x of the FP8 PV route from L512 (0.80-0.82x of 2.5.12), attention-output error on real
activations 3.1-4.0 % against 2.1-2.4 % (max-norm up to 24 % against 6 %), judge form 0.85-0.95 % (> 8e-3). The stage analysis of every other e4m3 placement
(tail o, front GEMM operands, cuBLAS projections, out-projection, bidirectional) and why they were not built: `docs/fp8.md`, "TriAttn: where else FP8 can go".

## Training backward, query side: row-split kernel (3.0.1 candidate, branch `bf16/triattn-bwd`, 2026-11-18)

The query-side kernel `triattn_bwd_q` is bound by the fp32 dbias TMA reduce-adds (removal ablation, 4 heads, L768: 2.43 ms -> 1.43 ms; every
second flush removed 1.59 ms, three of four 1.49 ms; L512 0.716 -> 0.465 ms; L384 0.314 -> 0.234 ms; L128-256 not bound).  `TB_Q8` (default 1,
used from S = 512; 0 = the old kernel, 2 = always) restructures it: a task is a 64-query tile x 8 pair rows, warpgroup 1 / 2 own rows 0-3 / 4-7
of all 64 queries; per key step warpgroup 2 stages its 64 x 64 fp32 dbias partial in shared memory, warpgroup 1 adds its own in place and the
storer issues ONE reduce-add pair: S / 8 partials per element instead of S / 4.  ncu (4 heads, L768, default slab count): L2 reduce sectors
84.9 M -> 42.5 M (halved), DRAM 5.07 -> 2.94 GB, kernel time 2.39 -> 2.42 ms: the gain comes with the slab count.  Slabs: the count (<= 8)
whose tiles x slabs tasks fill whole SM waves best (48 tiles: 5 slabs = 1.82 waves, against 4 (1.45) and 6 (2.18)): kernel L768 2.43 -> 2.01 ms
(-17 %), 8 heads 5.06 -> 4.03, 12 heads 7.70 -> 6.1, L512 0.72 -> 0.61 (4 slabs); at L384 and below it is not faster, so S < 512 keeps the old
kernel.  Determinism: fixed order (warpgroup 2's tile, then warpgroup 1's added; slabs summed in order); dq bitwise unchanged, dbias differs in
fp32 summation order (rms 1e-7 relative).
Module training step (fwd + bwd, CUDA graph, interleaved with a same-cubin control, 3.0.0 kernel = `TB_Q8=0`), sustained, ms:
128:4:768 6.116 -> 5.768 (0.943), 128:4:512 2.139 -> 2.030 (0.949), 256:8:768 13.72 -> 12.78 (0.931), 384:12:768 21.38 -> 20.07 (0.939),
64:4:768 6.391 -> 5.970 (0.934), 64:2:768 3.489 -> 3.341 (0.958); control ratios 0.999-1.022.  All nine gradients vs the old kernel: rms <= 4e-5
relative, both runs bitwise equal.  Test: `tests/gpu/test_triattn_h100_bwd_q_rowsplit.py`.  Not repeated from the lists above: more slabs, ping-pong.

## Training, 64-channel heads: wgmma bf16 forward + two-row query-side backward (3.0.1 candidate, branch `bf16s/triattn-train`, 2026-11-18)

Profile (H100, L1024, B1, one training step, kernel time): d_pair 64 x 4 heads (16 or 32 channels) 12.3 ms, d_pair 128 fused 11.7 ms, d_pair 64 x 2 heads 6.7 ms. The
core (`triattn_fwd` 2.05 + `bwd_kv` 2.73 + `bwd_q8` 4.29 ms) is 9.1 ms of the 12.3 and does not depend on d_pair: the generic path's glue (3.2 ms) is
bound by the bytes of the head-width tensors (q | k | v | g, dall: 4 x 128 columns), the same bytes the fused d128 kernels move (2.7 ms). d_pair 64 x 4 x
16-channel heads are padded to 32 channels, so they cost the same as 32-channel heads. Extending the fused front / tail / head_bwd / wgrad to d_pair 64 would
save at most ~0.5 ms (4 %); not done. The open lever for 16-channel heads is a compact (unpadded) layout through the three core kernels (~10 %, est.).
RF3 / Chai-1 (64-channel heads, L1024): 33.5 / 36.8 ms, of which mma.sync `dh_bwd_kv` 9.6, `dh_bwd_q` 8.9, `dh_fwd` 6.3 and the sum of the bf16 dbias planes 3.3 ms.

Changes (outputs change within rounding; `backend_versions["cuda:h100_sm90"]`: AF3 13, RF3 14, Chai1 / Chai1Confidence 9):
- `triattn_d64t_sm90.cu` (`h100_dh.ext_d64t`, kernel row `triangle_attention_d64_fwd_train_sm90`): the 2.5.12 bf16 wgmma core of 64-channel heads (git 68d6be5cf)
  with the training outputs O (bf16) and the base-2 LSE; `h100_dh.train_core` uses it for S % 128 == 0 (training always pads to 128; `TRAIN_WGMMA`).
- `triattn_dh_bwd_q` (kernel version 4): `BWD_Q_ROWS` = 2 pair rows a CTA (two warpgroups, shared bias tile); the rows' fp32 dS fragments are added in
  shared memory (fixed order) before one bf16 plane is stored: half the planes (and half the plane sum's read), each plane element the rounding of a sum.
  The fp32 slab variant was removed (one implementation). `plan` counts the planes by length.
- `TB_POLY` (default 0 = unchanged code): experiment flag, recomputed P of the wgmma backward kernels on the FMA polynomial.
- Tests: `tests/gpu/test_triattn_h100_d64_train.py`.
Status of the measurements: see the report of the branch (GPU queue was saturated; numbers are added when the job `all1` ran).
