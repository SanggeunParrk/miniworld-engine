# AttentionPairBias on B200 (sm100)

Kernel-level status of AttentionPairBias (the Pairformer single track: LayerNorm(single) -> q (with bias) / k / v / gate
projections, pair bias = to_bias(LayerNorm(pair)), key mask, softmax, sigmoid gate, to_out, residual; d_pair 128, no
QK-norm) on B200; the module-level summary is in [b200.md](../b200.md). Columns are (Length, d_pair) from the shape registry.
Five head layouts are served: 8 x 48 (the registry row), 12 x 32, 16 x 24 (AF3's Pairformer single attention) and 24 x 16 at
d_single 384, and 16 x 32 at d_single 512.

Summary (2026-10-01). Inference and training in bf16, B = 1, every layout; inference at L % 16 == 0, training at L % 128 ==
0 (both tested and measured at L128-768). Against the fastest other implementation (`bench.py`, PyTorch compiled / Triton /
cuEquivariance / Anthropic): inference 1.17-1.66x in every layout (Anthropic's shipped module row is the runner-up),
training with CUDA graphs 1.38-1.90x (cuEquivariance or Triton; Anthropic has no backward). Training without CUDA graphs
is host-bound: 0.58-1.30x the fastest other row, behind at short lengths (see "Limits and next"). Accuracy matches the
bf16 PyTorch module (output and every gradient within 1.3x its error against the fp32 module, every layout).
Time-roofline SoL (8 x 48, measured 2026-09-30): inference 6-51 % (L128 -> L768), training 26 % (L384) / 49 % (L768);
the pair kernels run at 69-95 %, the attention cores are latency-bound at B = 1.

On B200 the module runs **hand-written CUDA and cuBLAS only** (no Triton, no quack), from `AttentionPairBias.forward`
through `integrations/attention_pair_bias_b200.py` when the call matches its contract: implementation MINIWORLD with the
engine backend not forced to Triton, bf16 single and pair, B = 1, (heads, d_single) one of `SHAPES` = (8, 384), (12, 384),
(16, 384), (24, 384), (16, 512), d_pair 128, no QK-norm, key mask [1, L] or none, capability 10.0. Everything else keeps
the module path.

- **Inference** (no autograd; one opaque op): `ln_rows` (LayerNorm of the single rows, bf16, and a copy of the input as the
  output's residual seed) -> one cuBLAS GEMM for q | k | v | g (the weights packed once per parameter version, q and its
  bias pre-scaled by log2(e) / sqrt(head dim) so the logits come out in exp2 units) -> `pair_bias` (LayerNorm(pair) . Wf,
  Wf = to_bias.weight x ln_pair.weight x log2(e), masked keys -1e30) -> the gated attention core `attn_inf -DQPAIR=1`
  (sigmoid(g) o written over the q columns) -> cuBLAS to_out accumulated in place onto the residual seed (beta = 1; no
  separate residual pass, and no copy of the input into the output first as an out-of-place addmm would make).
- **Training** (one autograd Function; forward and backward each one opaque op, kept as nodes by torch.compile). Forward:
  `prep` (weight pack, Wf) -> `ln_rows` (+ mean / rstd, the residual seed) -> cuBLAS q | k | v | g -> `pair_bias` (natural
  units) -> `attn_fwd2 -DQPAIR=1` (O fp32, LSE) -> `gate_rows` -> cuBLAS to_out in place onto the seed. Backward: cuBLAS dO
  and dWo -> `gate_bwd` (dO for the core, D = rowsum(dO o) per head, dg) -> `bias_transpose`, `attn_dkv`, `attn_dqb
  -DDQPART=1` -> `qkv_bwd` -> cuBLAS dxa and dW q | k | v | g -> `ln_bwd` (+ the residual gradient) -> `pair_bias_bwd` ->
  `finalize` (every parameter gradient into its own tensor, one launch).
- **Layouts.** The attention cores take the layout as build flags (`-DNHEAD=<heads> -DDHP=<width> -DRSQDV=<1 / sqrt(head
  dim)>`, `sm100.APB_GEOMETRY`), the pair kernels as a template on the head count, the row passes on the row width (384 or
  512). 16 x 24 runs 32 wide per head: `prep` packs q | k | v | g with 8 zero rows after each head's 24 (and Wo with 8 zero
  columns), so q, k, v, g, O and their gradients are 512 wide with exact zeros in the pads; the cores see heads of 32 and
  the softmax scale of 24, and `finalize` gathers the real rows / columns of the weight gradients back. The other layouts
  have no padding.
- **Deterministic dQ.** `attn_dqb` stores each 128-key chunk's dQ partial into its own slice (`-DDQPART=1`, [L / 128, L,
  W] fp32) instead of reduce-adding into one buffer, and `qkv_bwd` adds the slices in chunk order on its way to the bf16
  dq columns. The atomic order had made runs differ by ~1e-4 relative after the bf16 rounding, which the bench's repeat
  check rejects (16 x 24, L640). The parameter-gradient column sums (dWf, ln weights, dbq) still add one block-reduced
  partial per block atomically into fp32; their order varies run to run within that check.
- ln_pair's bias adds Wb . b to every logit of a head: the softmax cancels it, so the forward drops it, and its gradient
  (sum_j dbias[h, i, j] = 0 for every query) is exactly zero -- returned as 0 (the fp32 module returns rounding noise
  there, ~1e-5 of ln_pair.weight's gradient).
- Masked keys get -1e30 (finite) where the module uses finfo.min: the same softmax whenever a sample has a valid key.
  A fully masked sample (no valid key; not expected in use) stays finite in inference but gives NaN in training: the
  training cores compute exp2(t log2 e - m log2 e) with one FMA, and at |t| ~ 1e30 the rounding of m log2 e (~1e23)
  overflows the exponent.

성능 확인: △ (fastest measured, the step below 70 % SoL). cache build ✓: nothing on these paths autotunes -- the CUDA
kernels have fixed launch shapes, the sm_100a attention cores are cubins built on first use into
`MINIWORLD_ENGINE_JIT_ROOT` (keyed by source and flags, one per layout), the row kernels an extension built on first use.

## Kernels

`kernels/augmented_attention/cuda/apb/apb_rows.cu` (pair kernels and row passes; built for sm_100 only) and
`kernels/augmented_attention/cuda/sm100/` (the attention cores, shared with the token DiT through build flags).

### K1 · `pair_bias` (pair -> bias: LayerNorm + projection, one read of the pair)

Persistent, two blocks per SM of 8 warps; a block walks tiles of 128 consecutive keys of one query row (32 KB of pair,
contiguous), each brought by one `cp.async.bulk` into a 2-stage ring. A warp takes 16 rows: `mma.sync m16n8k16` on the
raw bf16 words (the K order permuted to the load order, Wf's B fragments on the same map), the LayerNorm folded into the
projection -- Wf . LN(x) = rstd (Wf . x - mean sum_c Wf) -- and the row statistics on the tensor cores too: the row sums
from a column of ones, sum x^2 from the Gram block X X^T (the A fragment reused as the B operand; its diagonal). Results are
staged in shared memory and leave per head in 8-byte stores, with -1e30 on masked keys. No per-element float work.
Templated on the head count: N = 8 heads per mma, so 8 / 16 / 24 heads take 1 / 2 / 3 mma groups, and 12 heads run as 16
(the 4 pad heads have zero Wf and are not stored). 8 heads at L768: 27-32 us, 80-93 % of the HBM floor.

### K2 · `pair_bias_bwd` (dpair, dWf)

Persistent, 8 warps per block over the forward's tiles. TMA brings the pair tile as two 128-B-swizzled boxes of 64 columns
(the swizzle keeps the row reads and the `ldmatrix.trans` conflict-free) and dbias[0..H-1][i][j0..] (fp32, H x 512 B).
With a = rstd dbias:
dWf[h, c] = sum_r a[r, h] x[r, c] - sum_r a[r, h] mean_r (mma: M = columns through `ldmatrix.trans`, N = heads, K = rows;
the second term one scalar per head); dx^ = Wf^T dbias (K = the heads: `m16n8k16` per two groups of 8, `m16n8k8` for an odd
one; the result lands on the lane's own columns); dpair = rstd dx^ + k x + c with the row scalars from S1 = sum_h dbias s_h
and S2 = sum_h dbias y_h. dWf leaves block-reduced into the step's fp32 accumulator (one atomic per entry per block).
- 8 heads: two blocks per SM (128 registers), S2 from y = Wf . x recomputed on the raw words. L768: 59 us, 86 % of the HBM
  floor.
- 16 and 24 heads (12 run as 16): each warp holds 64 / 96 dWf accumulators, so one block per SM (up to 255 registers)
  with a 3-stage ring, and S2 from a first dx^ pass over the tile (S2 = sum_c dx^ x) instead of holding Wf's
  projection fragments.

### K3 · attention cores (sm_100a, tcgen05; the token DiT kernels)

Built per layout with `-DQPAIR=1 -DNHEAD=<heads> -DDHP=<width> -DRSQDV=<1 / sqrt(head dim)>` (`sm100._apb_defs`):

| layout | NHEAD | DHP (width in memory and in the MMAs) | row width W |
|---|---|---|---|
| 8 x 48 | 8 | 48 | 384 |
| 12 x 32 | 12 | 32 | 384 |
| 16 x 24 | 16 | 32 (24 + 8 zeros) | 512 |
| 24 x 16 | 24 | 16 | 384 |
| 16 x 32 (d 512) | 16 | 32 | 512 |

- `attn_inf` (inference), `attn_fwd2` (training forward): with one sample a work item pairs two 128-query tiles of one
  head (the two softmax warpgroups), which then share each K / V block (`QPAIR`; the token DiT pairs samples). Tile tails
  past L load as zeros and are not stored.
- `attn_dkv`, `attn_dqb -DDQPART=1` (training backward, A = 1) and `bias_transpose` (`glue.cu`). dQ leaves as one fp32 slice
  per 128-key chunk (see "Deterministic dQ"); dK / dV are [L, W].
- The width sets the MMA K steps (DH / 16), the q / k / v / dO boxes and the fp32 output staging (48: a 32-column box in the
  128-B swizzle + a 16-column box in the 64-B swizzle; 32: the 32-column box; 16: the 16-column box in the 64-B swizzle).
- The token DiT builds keep QPAIR = 0 and NHEAD = 16 (its tests pass unchanged).

### K4 · row passes and glue (`apb_rows.cu`)

A warp per row, lane l on CPL = W / 32 contiguous columns (12 at W = 384, 16 at 512), 4 rows per block: `ln_rows`
(LayerNorm, bf16 out, mean / rstd saved, and the input copied out as the residual seed), `gate_rows` (og = sigmoid(g) o),
`gate_bwd` (dO for the core, D per head, dg), `qkv_bwd` (dq = the dQ chunk slices added in order | dk | dv, fp32 -> bf16
into the dqkvg buffer, dbq), `ln_bwd` (dx = dy + LayerNorm backward, dw, db). Head sums are lane shuffles where a head
spans whole lanes (8 x 48: 4 lanes; 16 x 24 and 16 x 32: 2); where it does not (24 x 16 and 12 x 32 at CPL 12),
`gate_bwd_hl` gives each head one lane (lanes past the head count idle). Column sums (dw, db, dbq) are block-reduced and
added into one zeroed fp32 accumulator. `prep` packs W q | k | v | g (padded rows for 16 x 24), its bias, Wo (padded
columns) and Wf in one launch; `finalize` writes every parameter gradient into its own tensor (custom-op outputs may not
alias) in one launch.

## Measurements (2026-10-01)

`bench.py target=attention_pair_bias level=module sweep_axis=seq_len min_seq_len=128 max_seq_len=768 seq_len_step=128
precision=bf16-mixed +apb_n_head=<heads> d_single=<d>`, compiled, one GPU through `gpuq` with nothing else on it, no key
masking (the config's `mask_prob` 0), ms. Inference with a CUDA graph; training forward + backward with CUDA graphs off /
on. × = ours against the fastest other row. Every implementation runs the same layout (`apb_n_head` and `d_single` set the
module for all of them). The rows of one layout come from one run: 8 x 48 and 16 x 24 from 12:32-12:47 (after the
deterministic dQ and this page's code), 24 x 16 / 12 x 32 / 16 x 32 from 10:09-10:50 the same day (the same kernels;
the later edits were the package move and the dh-64 TMEM clear, which these widths do not reach). Script:
`scratch/apb/bench_layout.sh` on the B200 host (`APB_LAYOUTS="8:384 16:384"`).

Inference:

| layout | (Length, d_pair) | PyTorch compiled | Triton | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|---|---|
| 8 x 48 | (128, 128) | 0.041 | 0.039 | 0.045 | 0.029 | **0.023** | 1.27 |
| 8 x 48 | (256, 128) | 0.057 | 0.047 | 0.059 | 0.035 | **0.029** | 1.22 |
| 8 x 48 | (384, 128) | 0.074 | 0.063 | 0.078 | 0.045 | **0.035** | 1.29 |
| 8 x 48 | (512, 128) | 0.104 | 0.086 | 0.106 | 0.057 | **0.043** | 1.33 |
| 8 x 48 | (640, 128) | 0.141 | 0.113 | 0.141 | 0.072 | **0.053** | 1.35 |
| 8 x 48 | (768, 128) | 0.176 | 0.139 | 0.176 | 0.090 | **0.063** | 1.42 |
| 12 x 32 | (128, 128) | 0.043 | 0.039 | 0.047 | 0.029 | **0.022** | 1.28 |
| 12 x 32 | (256, 128) | 0.057 | 0.047 | 0.061 | 0.037 | **0.029** | 1.29 |
| 12 x 32 | (384, 128) | 0.076 | 0.061 | 0.080 | 0.047 | **0.035** | 1.36 |
| 12 x 32 | (512, 128) | 0.108 | 0.088 | 0.108 | 0.061 | **0.043** | 1.43 |
| 12 x 32 | (640, 128) | 0.143 | 0.113 | 0.143 | 0.082 | **0.053** | 1.54 |
| 12 x 32 | (768, 128) | 0.181 | 0.141 | 0.178 | 0.102 | **0.063** | 1.62 |
| 16 x 24 | (128, 128) | 0.041 | 0.039 | 0.045 | 0.029 | **0.025** | 1.17 |
| 16 x 24 | (256, 128) | 0.057 | 0.049 | 0.059 | 0.037 | **0.029** | 1.29 |
| 16 x 24 | (384, 128) | 0.076 | 0.063 | 0.078 | 0.047 | **0.035** | 1.35 |
| 16 x 24 | (512, 128) | 0.106 | 0.090 | 0.109 | 0.061 | **0.043** | 1.43 |
| 16 x 24 | (640, 128) | 0.145 | 0.119 | 0.149 | 0.082 | **0.053** | 1.54 |
| 16 x 24 | (768, 128) | 0.184 | 0.149 | 0.188 | 0.104 | **0.064** | 1.64 |
| 24 x 16 | (128, 128) | 0.041 | 0.039 | 0.045 | 0.029 | **0.025** | 1.17 |
| 24 x 16 | (256, 128) | 0.057 | 0.049 | 0.061 | 0.039 | **0.031** | 1.27 |
| 24 x 16 | (384, 128) | 0.078 | 0.063 | 0.080 | 0.049 | **0.037** | 1.33 |
| 24 x 16 | (512, 128) | 0.110 | 0.088 | 0.113 | 0.063 | **0.045** | 1.41 |
| 24 x 16 | (640, 128) | 0.147 | 0.115 | 0.149 | 0.086 | **0.055** | 1.56 |
| 24 x 16 | (768, 128) | 0.188 | 0.143 | 0.188 | 0.108 | **0.065** | 1.66 |
| 16 x 32 (d 512) | (128, 128) | 0.043 | 0.041 | 0.045 | 0.029 | **0.024** | 1.17 |
| 16 x 32 (d 512) | (256, 128) | 0.057 | 0.049 | 0.061 | 0.037 | **0.031** | 1.20 |
| 16 x 32 (d 512) | (384, 128) | 0.078 | 0.065 | 0.080 | 0.049 | **0.037** | 1.33 |
| 16 x 32 (d 512) | (512, 128) | 0.104 | 0.088 | 0.108 | 0.061 | **0.045** | 1.37 |
| 16 x 32 (d 512) | (640, 128) | 0.147 | 0.121 | 0.151 | 0.082 | **0.055** | 1.48 |
| 16 x 32 (d 512) | (768, 128) | 0.184 | 0.148 | 0.187 | 0.104 | **0.065** | 1.60 |

Anthropic = the release's own module row for this cell (`opt_core.kernels.apb`, row `composed` of `mod_pf_c384cz128`;
`bench.py` `_anthropic_apb_composition`): `ln_proj.pair_bias` (LayerNorm(z) + Linear in one Triton kernel, head-major
planes) and the `apb_attn` core (Triton; key mask and sigmoid gate fused), with torch LayerNorm / one q | k | v | g GEMM /
to_out + residual around them, weights packed once, at the same head layout. It runs `compile=false` (its upstream calls
are compiler-disabled, so a compiled row would execute no compiled graph), with the CUDA graph. The release's table has no
B200 measurement of this cell (its selection inherits the H100 one, where cuEquivariance led); its Pairformer core cells
are 16 heads x 24. With `+apb_anthropic_core=fpf_apb` (`fpf_apb.apb_views`, the token DiT comparison's core; 8 x 48,
2026-10-01): 0.029 / 0.037 / 0.047 / 0.059 / 0.076 / 0.094 ms, the same or slower. Kernels at L768 (8 x 48, key mask 20 %,
profiler, before the residual fold): Anthropic 90.5 us (`_ln_proj_kernel` 56.1, `_apb_fwd` 17.8, GEMMs and torch glue 16.6)
against ours 56.1 us (`pair_bias` 27.3, `attn_inf` 17.3, GEMMs 7.0, `ln_rows` 2.4): the lead is the pair-bias producer, the
attention cores are even (both latency-bound at B = 1). Accuracy of the attention branch against the fp32 module, L128-768:
Anthropic 1.28-1.39e-2, ours 1.32-1.44e-2, the bf16 PyTorch module about the same. Anthropic has no backward, so it has no
training row.

Training (CUDA graph off / on):

| layout | (Length, d_pair) | PyTorch compiled | Triton | cuEquivariance | ours | × (graph on) |
|---|---|---|---|---|---|---|
| 8 x 48 | (128, 128) | 0.403 / 0.119 | 1.086 / 0.106 | 0.505 / 0.106 | 0.657 / **0.070** | 1.53 |
| 8 x 48 | (256, 128) | 0.555 / 0.195 | 1.114 / 0.149 | 0.572 / 0.166 | 0.665 / **0.084** | 1.78 |
| 8 x 48 | (384, 128) | 0.545 / 0.254 | 1.093 / 0.199 | 0.900 / 0.164 | 0.650 / **0.101** | 1.62 |
| 8 x 48 | (512, 128) | 0.573 / 0.336 | 1.093 / 0.274 | 0.717 / 0.223 | 0.492 / **0.125** | 1.79 |
| 8 x 48 | (640, 128) | 0.517 / 0.438 | 0.922 / 0.362 | 0.935 / 0.287 | 0.658 / **0.155** | 1.85 |
| 8 x 48 | (768, 128) | 0.617 / 0.536 | 0.933 / 0.448 | 0.732 / 0.354 | 0.497 / **0.186** | 1.90 |
| 12 x 32 | (128, 128) | 0.559 / 0.119 | 0.955 / 0.108 | 0.446 / 0.108 | 0.684 / **0.072** | 1.51 |
| 12 x 32 | (256, 128) | 0.614 / 0.194 | 0.955 / 0.147 | 0.556 / 0.169 | 0.730 / **0.088** | 1.68 |
| 12 x 32 | (384, 128) | 0.622 / 0.251 | 1.003 / 0.201 | 0.802 / 0.164 | 0.561 / **0.108** | 1.51 |
| 12 x 32 | (512, 128) | 0.516 / 0.336 | 0.969 / 0.274 | 0.770 / 0.213 | 0.556 / **0.138** | 1.54 |
| 12 x 32 | (640, 128) | 0.661 / 0.437 | 1.176 / 0.356 | 0.974 / 0.272 | 0.697 / **0.174** | 1.57 |
| 12 x 32 | (768, 128) | 0.656 / 0.539 | 1.146 / 0.457 | 0.986 / 0.334 | 0.686 / **0.213** | 1.57 |
| 16 x 24 | (128, 128) | 0.394 / 0.119 | 0.881 / 0.110 | 0.411 / 0.108 | 0.679 / **0.076** | 1.43 |
| 16 x 24 | (256, 128) | 0.439 / 0.194 | 1.127 / 0.154 | 0.582 / 0.168 | 0.693 / **0.092** | 1.67 |
| 16 x 24 | (384, 128) | 0.564 / 0.250 | 1.135 / 0.207 | 0.937 / 0.194 | 0.535 / **0.113** | 1.73 |
| 16 x 24 | (512, 128) | 0.601 / 0.338 | 0.914 / 0.291 | 0.728 / 0.244 | 0.696 / **0.143** | 1.70 |
| 16 x 24 | (640, 128) | 0.535 / 0.461 | 1.171 / 0.389 | 0.810 / 0.328 | 0.695 / **0.180** | 1.82 |
| 16 x 24 | (768, 128) | 0.646 / 0.559 | 1.220 / 0.485 | 0.957 / 0.405 | 0.736 / **0.229** | 1.77 |
| 24 x 16 | (128, 128) | 0.610 / 0.119 | 1.083 / 0.104 | 0.615 / 0.111 | 0.782 / **0.076** | 1.38 |
| 24 x 16 | (256, 128) | 0.659 / 0.197 | 1.164 / 0.149 | 0.733 / 0.176 | 0.858 / **0.092** | 1.62 |
| 24 x 16 | (384, 128) | 0.649 / 0.256 | 1.390 / 0.201 | 0.552 / 0.221 | 0.869 / **0.123** | 1.63 |
| 24 x 16 | (512, 128) | 0.616 / 0.350 | 1.200 / 0.278 | 0.577 / 0.287 | 0.734 / **0.156** | 1.79 |
| 24 x 16 | (640, 128) | 0.609 / 0.459 | 1.105 / 0.364 | 0.601 / 0.385 | 0.690 / **0.203** | 1.80 |
| 24 x 16 | (768, 128) | 0.652 / 0.569 | 0.937 / 0.467 | 0.603 / 0.479 | 0.526 / **0.256** | 1.82 |
| 16 x 32 (d 512) | (128, 128) | 0.612 / 0.121 | 1.258 / 0.110 | 0.629 / 0.110 | 0.806 / **0.078** | 1.42 |
| 16 x 32 (d 512) | (256, 128) | 0.680 / 0.194 | 1.346 / 0.151 | 0.671 / 0.170 | 0.867 / **0.094** | 1.61 |
| 16 x 32 (d 512) | (384, 128) | 0.675 / 0.256 | 1.270 / 0.207 | 0.708 / 0.180 | 0.518 / **0.117** | 1.54 |
| 16 x 32 (d 512) | (512, 128) | 0.685 / 0.338 | 1.273 / 0.278 | 1.083 / 0.229 | 0.814 / **0.147** | 1.56 |
| 16 x 32 (d 512) | (640, 128) | 0.718 / 0.459 | 1.294 / 0.381 | 0.816 / 0.312 | 0.826 / **0.184** | 1.70 |
| 16 x 32 (d 512) | (768, 128) | 0.701 / 0.557 | 1.357 / 0.469 | 1.145 / 0.389 | 0.616 / **0.231** | 1.68 |

The layouts cost ours the same within 0.003 ms in inference; in training (graph on) 24 x 16 is the slowest at L768 (0.256
against 0.186 ms for 8 x 48), the other layouts in between. Without CUDA graphs every row is host-bound (the GPU work is
0.07-0.26 ms): ours ranges 0.58-1.30x the fastest other row (compiled PyTorch or cuEquivariance), behind at short lengths
-- see "Limits and next". Before this path (the engine's module path, Triton attention, 8 x 48, 2026-09-30): inference
0.039 / 0.063 / 0.139 ms and training (graph on) 0.106 / 0.199 / 0.450 ms at L128 / L384 / L768.

Accuracy against the fp32 PyTorch module (`tests/integrations/test_b200_apb_gpu.py`, every layout, L128 / 208 / 384 / 768
inference and L128 / 384 training, key mask on / off): output and every input and parameter gradient within 1.3x the bf16
PyTorch module's error (8 x 48 probe at L128 / 384 / 768, 20 % key mask: output 1.1-1.2e-2 against 1.1-1.2e-2, gradients
0.99-1.13x, worst ~2e-2 relative); the compiled module matches eager.

Tests: `tests/numerics/test_apb_b200_gpu.py` (the pair kernels at 8 / 12 / 16 / 24 heads against fp64 at L128-768; per
layout the inference core at L128 / 200 / 256 / 384 / 640 / 768 and the training forward + backward at L128 / 256 / 384 /
768 against fp64), `tests/integrations/test_b200_apb_gpu.py` (module against fp32 PyTorch per layout: inference at L128 /
208 / 384 / 768, training at L128 / 384, key mask on / off; torch.compile against eager).

### Hardware limit (SoL) per kernel (8 x 48, 2026-09-30)

Measured before the residual fold (an extra residual pass then) and the deterministic dQ (`qkv_bwd` now also adds the dQ
chunk slices); the other layouts are not profiled per kernel.


The token DiT doc's method: floor = max(minimum HBM bytes / BW, FLOPs / tensor rate, exp2 count / MUFU rate) per kernel;
SoL = floor / measured (kernel duration, median over back-to-back eager steps; the measured us in parentheses). Ceilings:
HBM 6.31 TB/s (a 512 MB fp32 copy in the same process), tensor 2.23 PF/s bf16, MUFU 31 ex2 / ns / SM. B = 1, 20 % key
mask. "step" = the kernels' SoL weighted by their time (PyTorch fills counted at 0). Script: `sol_apb.py` (B200 scratch
`scratch/apb/sol`).

Inference:

| kernel | bound | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|---|
| attn_inf | HBM | 2 % (6.8) | 4 % (9.0) | 5 % (11.1) | 7 % (13.1) | 9 % (15.2) | 11 % (17.3) |
| pair_bias | HBM | 21 % (3.3) | 54 % (5.2) | 79 % (8.1) | 95 % (11.9) | 94 % (18.8) | 93 % (27.3) |
| ln_rows | HBM | 2 % (2.0) | 3 % (2.1) | 4 % (2.1) | 5 % (2.3) | 7 % (2.4) | 8 % (2.4) |
| cuBLAS GEMMs (2) | HBM | 6 % (6.2) | 7 % (6.5) | 9 % (6.4) | 10 % (6.9) | 11 % (7.3) | 13 % (7.0) |
| **step (time-weighted)** | | 6 % | 15 % | 26 % | 36 % | 44 % | 51 % |
| step kernel time, us | | 19.8 | 24.4 | 29.3 | 35.8 | 45.8 | 56.1 |
| sum of floors, us | | 1.2 | 3.7 | 7.6 | 13.1 | 20.0 | 28.5 |

Training (forward + backward):

| kernel | bound | L384 | L768 |
|---|---|---|---|
| attn_fwd2 | HBM | 6 % (10.0) | 12 % (16.2) |
| attn_dkv | HBM | 8 % (10.5) | 16 % (15.3) |
| attn_dqb | HBM | 13 % (10.5) | 29 % (17.4) |
| bias_transpose | HBM | 27 % (2.8) | 48 % (6.3) |
| pair_bias | HBM | 79 % (8.1) | 80 % (31.7) |
| pair_bias_bwd | HBM | 69 % (18.5) | 86 % (59.0) |
| ln_rows | HBM | 4 % (2.3) | 7 % (2.7) |
| gate_rows | HBM | 7 % (2.6) | 12 % (3.0) |
| gate_bwd | HBM | 11 % (2.7) | 19 % (2.9) |
| qkv_bwd | HBM | 16 % (2.6) | 26 % (3.3) |
| ln_bwd | HBM | 8 % (2.4) | 11 % (3.4) |
| prep | HBM | 16 % (2.3) | 16 % (2.4) |
| finalize | HBM | 27 % (2.6) | 24 % (2.9) |
| cuBLAS GEMMs (6) | HBM | 6 % (21.4) | 8 % (23.5) |
| **step (time-weighted)** | | 26 % | 49 % |
| step kernel time, us | | 102.2 | 193.4 |
| sum of floors, us | | 26.2 | 94.0 |

What bounds the step:
- The pair kernels, which move nearly all the bytes (the pair is L^2 x 128 bf16: 151 MB at L768), run at 69-95 %: a copy
  of the pair through the same TMA ring with no compute takes 60.5 us at L768 against the backward's 64.7 (kernel alone),
  and a PyTorch copy of the pair 49.5 us.
- The attention cores are latency-bound at B = 1: 8 heads x L / 128 query tiles give 24-48 work items for 148 SMs (paired
  tiles in the forward), and each walks the whole key axis in sequence. Their floor is 1-4 us at L768.
- The row kernels, GEMMs and glue are launch- and latency-bound (2-7 us each on a few MB).

## What was tried and not kept

| attempt | result |
|---|---|
| row kernels with a block of 96 threads over 8 rows (two block-wide sums per row) | 5-10 us each: latency-bound at L / 8 blocks; a warp per row (12 columns per lane) brought them to 2-3.5 us |
| per-block partial column sums added up by `torch.sum` | the five reductions took 20-40 us per step; block-reduced atomics into one accumulator instead (L / 4 blocks per address: no serialisation, unlike the token DiT's 768-wide sums over A L rows) |
| one copy kernel per parameter gradient | ~25 us of copies and memcpys per step; `finalize` writes them all in one launch |
| pair kernels with plain 16-B loads | forward 37-40 us, backward 74-94 us at L768 (3.8-4.1 TB/s); the bulk-copy / TMA rings and the folded LayerNorm took them to 27-32 / 59 us |
| forward ring variants (1 block of 4 or 8 warps per SM, 3-6 stages) | 40-44 us at one block per SM: the per-tile chain (wait, compute, barrier, stores) is serial in a block; two blocks per SM overlap it |
| row statistics with per-element fp32 (sum and sum of squares) | replaced by the ones / Gram mma; ~1 us gained in the forward, none in the backward |
| dpair written back in place and stored by TMA | no gain over 16-B stores from registers (69.6 vs 67.9 us incl. the accumulator fill) |
| 3 stages in the backward ring | no gain (69.3 vs 67.8 us) |
| ln_pair.bias gradient from the kernel's per-head dbias sums | rounding noise around the exact zero (up to 2x the bf16 module's noise); returned as 0 |
| 16 / 24 heads in the pair backward: dWf accumulated in a second phase over the tile instead of per-warp accumulators | slower; reverted to per-warp accumulators at one block per SM |
| 16 x 24 heads as 24 wide (no padding) | not built: 24 is not a multiple of the q k^T MMA's K step (16), and a 48-B row is no swizzle span (32 / 64 / 128 B); 32-wide heads with exact zero pads cost 33 % more q / k / v / g bytes and keep every kernel on the existing tiles |

## Limits and next

- The attention cores (2-29 % SoL) are the largest lever left: splitting the key axis into chunks (a second, small combine
  pass for the partial O / max / sum) would give ~3x more work items at B = 1 and cut the forward / inference core from
  11-17 us toward its few-us floor.
- Without CUDA graphs the step is host-bound (~0.5 ms of host time per training step against 0.1-0.2 ms of GPU work). The
  opaque ops' Python bodies (~64 / ~125 us) and the cuda-python launches of the attention cubins are the part this path
  controls; one C++ call per op (cubin launches and TMA descriptors included) is the lever.
- Served: bf16, B = 1, no QK-norm, d_pair 128, (heads, d_single) in (8, 384), (12, 384), (16, 384), (24, 384), (16, 512);
  inference L % 16 == 0 (the pair kernels' row groups), training L % 128 == 0 (the backward attention tiles). fp32, QK-norm
  and other shapes run the module path.
- Per-kernel SoL is measured for 8 x 48 only (2026-09-30, before the residual fold and the deterministic dQ).
