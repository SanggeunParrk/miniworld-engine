# AttentionPairBias on B200 (sm100)

Kernel-level status of AttentionPairBias (the Pairformer single track: LayerNorm(single) -> q (with bias) / k / v / gate
projections, pair bias = to_bias(LayerNorm(pair)), key mask, softmax, sigmoid gate, to_out, residual; d_pair 128, no
QK-norm) on B200; the module-level summary is in [b200.md](../b200.md). Columns are (Length, d_pair) from the shape registry.
Five head layouts are served: 8 x 48 (the registry row), 12 x 32, 16 x 24 (AF3's Pairformer single attention) and 24 x 16 at
d_single 384, and 16 x 32 at d_single 512.

Summary (2026-10-20). Inference and training in bf16, any number of samples B (2026-10-20; up to 48 per launch, bitwise the
one-sample results per sample), every layout; inference at L % 16 == 0, training at L % 128 == 0 (tested L128-1536, measured
L384 / 768 / 1536). Against main (59d910cf, B = 1, same GPU, same harness, back to back): inference 1.00-1.11x, training with
CUDA graphs 1.02-1.16x (the key-split forward cores, the query-split dK / dV core), training without CUDA graphs 1.04-2.74x
(the C++ launch sequences; host-bound cells, noisy). Against the fastest other implementation (PyTorch compiled /
cuEquivariance, 8 x 48 and 16 x 24): inference 2.06-2.78x, training with CUDA graphs 1.86-2.32x. B = 5 (the confidence head's
samples, 16 x 24): inference 2.18-2.67x, training 1.98-2.10x the fastest other row, 2.2-3.2x main (main ran B > 1 on the PyTorch
module path). Accuracy: the bf16 PyTorch module's (output and every gradient within 1.3x its error against the fp32 module,
every layout, every split count); with `apb_b200_ksplit = apb_b200_qsplit = 1` the output, dsingle, dpair and the weight-GEMM
gradients are main's bits. Every output and gradient is bitwise run to run (the parameter-gradient sums no longer use atomics).
Time-roofline SoL of the attention cores (8 x 48, L768): inference core 11 % -> 23 % (15 % with the combine pass), training
forward 12 % -> 24 %; the pair kernels (69-95 %, unchanged) dominate the step at L >= 768.

On B200 the module runs **hand-written CUDA and cuBLAS only** (no Triton, no quack), from `AttentionPairBias.forward`
through `integrations/attention_pair_bias_b200.py` when the call matches its contract: implementation MINIWORLD with the
engine backend not forced to Triton, bf16 single and pair, any number of samples B (since 2026-10-20; each sample its own pair
and key mask, up to `max_batch()` = 48 per launch, more in consecutive chunks of 48), (heads, d_single) one of `SHAPES` =
(8, 384), (12, 384), (16, 384), (24, 384), (16, 512), d_pair 128, no QK-norm, key mask [B, L] or none, capability 10.0.
Everything else keeps the module path. MiniWorld calls the module in the confidence head's MiniPairformer (16 x 24, d_single
384, its N diffusion samples as B); no MiniWorld config builds it with QK-norm (the trunk Pairformer sets `use_qk_norm: false`,
the confidence head leaves the False default), so QK-norm is not served.

Each op runs as **one C++ call** (since 2026-10-20): the sm_100 build of `apb_rows.cu` (`-DAPB_SM100_HOST`) registers the torch
ops `apb_b200::train_fwd`, `apb_b200::train_bwd` and `apb_b200::infer`, which issue the op's whole launch sequence -- the row and
pair kernels, the cuBLAS GEMMs through the same ATen calls as the Python sequence (so the same bits), the sm_100a attention
cubins through the driver with their TMA descriptors encoded (and cached by address) in C++. The cubins are built by the Python
side (nvcc 13.1) and fetched once per key through a resolver callback; `apb.host_ops()` loads the build, the ops' fake
implementations and the resolver (a constant to torch.compile). `settings.apb_b200_host = False` runs the Python sequence (same
kernels, same bits; the A/B switch). fp32 master weights go straight into `prep`, which rounds them to bf16 in its pass.

- **Inference** (no autograd; one opaque op): `ln_rows` (LayerNorm of the single rows, bf16, and a copy of the input as the
  output's residual seed) -> one cuBLAS GEMM for q | k | v | g (the weights packed once per parameter version -- inside a CUDA-graph capture per capture, unless `static_weights()` is declared --, q and its
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
  check rejects (16 x 24, L640). Since 2026-10-20 the parameter-gradient column sums (dWf, ln weights, dbq) are deterministic
  too: `ln_bwd` / `qkv_bwd` (grid capped at two blocks per SM, a block walking several rows) and `pair_bias_bwd` write one
  block partial row each and `finalize` reduces them in a fixed order (`acc_reduce_k`, as on H100): every output and gradient
  is bitwise run to run (five steps under a NaN-poisoned allocator, `test_b200_apb_host_gpu.py`). That held only most of the
  time until 2026-10-29: inside `attn_dkv` / `attn_dqb` the CTA's two MMA issuers accumulated dK / dV and a step's dQ onto one
  TMEM accumulator in the order their warpgroups got there, so B = 5 against five single calls (12 x 32, L768) differed about one
  run in five. The backward cores are now built `-DDET=1` (`_apb_defs`, both the Python sequence and the C++ host path's cubin
  resolver, and the token DiT per-sample-bias block that calls `apb_backward`): the issuer of a block waits for the other
  issuer's accumulating MMAs of the previous block to complete. Cost within the measurement noise (training step, CUDA graph,
  8 x 48: L256 / 384 / 768 112.4 / 143.8 / 272.1 us vs 112.1 / 143.6 / 272.1 without; token DiT over 48 proteins L384 / 768 6.92 /
  23.8 ms both); `test_backward_cores_are_bitwise_repeatable`.
- **Key split (forward / inference cores).** At B = 1 the cores had 16-48 work items for 148 SMs. Built `-DKSPLIT=1` they split
  the key blocks into NS equal chunks; a work item is (sample, head, query-tile pair, chunk) and leaves its chunk's
  normalised partial O (fp32) and partial LSE; `ksplit_combine` merges the chunks in order (weights 2^(LSE_c - max)), gates
  and writes og over q (inference) or O, LSE and og (training, the `gate_rows` pass folded in). NS depends on (L, heads)
  only (`sm100.apb_ksplit`: the smallest NS minimising waves / NS, each item keeping two key blocks; e.g. 8 heads L768 NS 6,
  16 heads L768 NS 3, L1536 NS 3), never on B, so a batch stays bitwise B single calls. The split builds form the exponent as
  (t - m) log2 e, so a chunk whose keys are all masked (a padded tail) stays finite and gets weight 0.
  `settings.apb_b200_ksplit` forces NS (1: the unsplit cores, the previous bits).
- **Query split (backward dK / dV core).** `attn_dkv -DQSPLIT=1`: a work item walks 1 / NQ of the query blocks and leaves its
  dK / dV as a slice of [NQ, S L, W]; `qkv_bwd` adds the slices in order. NQ by the same rule over the divisors of L / 128
  (`sm100.apb_qsplit`; 8 heads L768 NQ 3), `settings.apb_b200_qsplit` forces it.
- **Batches.** The forward cores' items carry the sample (bias planes [S H, L, L]), the backward cores take per-sample bias
  planes (`-DBPS=1`; the token DiT's builds share one bias across samples), the pair kernels run over the S L query rows
  (dbias map [L, L, S H]), D is [S, H, L]. A sample's output, dsingle and dpair are bitwise those of a one-sample call; the
  parameter gradients are the samples' sum (fp32 order).
- `ln_pair` has no bias since 2026-10-08 (the module's LayerNorm is `bias=False`; an old checkpoint's `ln_pair.bias` is dropped
  on load, exactly): its bias added Wb . b to every logit of a head, which the softmax cancels, and its gradient
  (sum_j dbias[h, i, j] = 0 for every query) is exactly zero. The kernels dropped it from the forward before as well; the
  cuEquivariance and Anthropic rows of the harness pass a zero offset, the same function.
- Masked keys get -1e30 (finite) where the module uses finfo.min: the same softmax whenever a sample has a valid key.
  A fully masked sample (no valid key; not expected in use) stays finite in inference but gives NaN in training: the
  training cores compute exp2(t log2 e - m log2 e) with one FMA, and at |t| ~ 1e30 the rounding of m log2 e (~1e23)
  overflows the exponent.

성능 확인: △ (fastest measured, the step below 70 % SoL). cache build ✓: nothing on these paths autotunes -- the CUDA
kernels have fixed launch shapes, the sm_100a attention cores are cubins built on first use into
`settings.engine_jit_root` (keyed by source and flags, one per layout), the row kernels an extension built on first use.

## Chai-1's width and the gate bias in training (2026-10-29)

Chai-1's Pairformer single attention (16 x 24 heads, d_single 384, **d_pair 256**, `gate_bias = 1.0`) now trains on the native path
too (inference since 2026-10-27), in both the C++ host path and the Python sequence; H100 keeps refusing training for it.

- **Gate bias.** The module's `gate_bias` is a constant (a float, not a parameter): `prep` writes it on the real g columns of the
  packed q | k | v | g bias, so qkvg's g is g + c and every gate pass (`gate_rows` / the key-split combine, `gate_bwd`) sees
  sigmoid(g + c) with no change. It has no gradient; dWg is dg^T xa as before (dg = dog o s (1 - s) at s = sigmoid(g + c)).
- **d_pair 256.** `prep` / `finalize` take the pair width off `to_bias.weight` (Wf, Wb and dWb [16, 256], dln_pair.weight [256]; the
  accumulator's dWf block is [16, 256]); the forward uses `pair_bias_fwd`'s 256-wide ring (as inference). `pair_bias_bwd` is
  templated on the width: its tile is four 64-column SW128 boxes (64 KB of pair + 8 KB of dbias, three stages, 230 KB of shared
  memory), the dx^ / dpair pass walks eight 32-column chunks. Holding the dWf tiles of all 256 columns per warp (128 accumulators)
  spilled 644 B, so at 256 the dWf runs **column-wise**: after the row pass (statistics into shared memory, dpair out) warp w takes
  columns 32 w .. 32 w + 31 over every 16-row group of the tile (16 accumulators, B = rstd dbias rebuilt from shared memory, the
  mean term over all the tile's rows), so a warp's columns are the block's and the end reduce sums one slot. 255 registers, no spills;
  the 128-wide builds are unchanged (same code path, same bits).

Measured (`bench.py target=attention_pair_bias level=module mode=training d_pair=256 +apb_n_head=16 +apb_gate_bias=1.0`, fp32 master
under autocast bf16, compiled, CUDA graph on / off, B = 1, GPU 7 of node 2 with only the utilisation filler beside it; before = d879e180,
where this call ran the module's PyTorch statements; ms):

| L | before (graph on / off) | PyTorch compiled (graph on) | now (graph on / off) | x before (graph on) |
|---|---|---|---|---|
| 256 | 0.223 / 0.764 | 0.216 | **0.113** / 0.318 | 1.98 |
| 384 | 0.317 / 0.915 | 0.299 | **0.152** / 0.574 | 2.09 |
| 768 | 0.778 / 0.902 | 0.717 | **0.371** / 0.475 | 2.10 |

Accuracy (`tests/gpu/sm100/test_b200_apb_gate_bias_gpu.py`): output and every input / parameter gradient within 1.3x the bf16 PyTorch
module's error against the fp32 module at (16 x 24, d_pair 256, gate 1.0 / 0.0), (16 x 24, 128, 1.0), (8 x 48, 128, 1.0), (12 x 32,
128, 0.5), L128 / 384, key mask on / off, bf16 parameters and an fp32 master, host and Python paths; the host path bitwise the Python
sequence, the step bitwise run to run, B = 3 bitwise its one-sample calls (output, dsingle, dpair), a captured training step bitwise
eager. `tests/gpu/sm100/test_apb_b200_gpu.py::test_pair_bias_d256_forward_and_backward`: the 256-wide pair kernels against fp64 (S 1 /
2, L128-768).

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
- `-DKSPLIT=1` (`attn_inf`, `attn_fwd2`): the key split above; `attn_inf` then stages fp32 partials (two 24 KB buffers, 225 KB
  of shared memory) instead of loading g. `-DQSPLIT=1` (`attn_dkv`): the query split; `-DBPS=1` (`attn_dkv`, `attn_dqb`):
  per-sample bias planes, `attn_dqb` items then (sample, head, query tile, key chunk).
- The token DiT builds keep QPAIR = 0 and NHEAD = 16 (its tests pass unchanged).

### K4 · row passes and glue (`apb_rows.cu`)

A warp per row, lane l on CPL = W / 32 contiguous columns (12 at W = 384, 16 at 512), 4 rows per block: `ln_rows`
(LayerNorm, bf16 out, mean / rstd saved, and the input copied out as the residual seed), `gate_rows` (og = sigmoid(g) o),
`gate_bwd` (dO for the core, D per head, dg), `qkv_bwd` (dq = the dQ chunk slices added in order | dk | dv, fp32 -> bf16
into the dqkvg buffer, dbq), `ln_bwd` (dx = dy + LayerNorm backward, dw, db). Head sums are lane shuffles where a head
spans whole lanes (8 x 48: 4 lanes; 16 x 24 and 16 x 32: 2); where it does not (24 x 16 and 12 x 32 at CPL 12),
`gate_bwd_hl` gives each head one lane (lanes past the head count idle). `ksplit_combine` (the key split's merge) is a thread
per four columns, templated on NS so the NS partial loads issue together (a runtime count left them one latency each: 6.4 ->
4.0 us at 8 x 48 L768 NS 6). Column sums (dw, db, dbq) are block-reduced and
added into one zeroed fp32 accumulator. `prep` packs W q | k | v | g (padded rows for 16 x 24), its bias, Wo (padded
columns) and Wf in one launch; `finalize` writes every parameter gradient into its own tensor (custom-op outputs may not
alias) in one launch.

## Measurements (2026-10-20)

`bench.py target=attention_pair_bias level=module sweep_axis=seq_len min_seq_len=L max_seq_len=L precision=bf16-mixed
+apb_n_head=<heads> d_single=<d> [+apb_batch=5]` (fp32 master parameters under autocast bf16, compiled, CUDA graph for
inference, training with CUDA graphs off / on; no key masking, `mask_prob` 0), ms; GPU 7 of the B200 box (driver 580.95.05,
torch 2.13.0+cu129, sm_100a cores built by nvcc 13.1) with only the ~2.3 GB utilisation dummy beside it, checked before every
run. main = the engine at 59d910cf (its own worktree and harness), measured in the same session (cells reproduced to 4 digits
on repeats); PyTorch compiled / cuEquivariance measured at 8 x 48 and 16 x 24 only. × = main / ours and (fastest of PyTorch
compiled, cuEquivariance) / ours; training cells are graph off / on, the ratios graph on. The graph-off cells are host-bound and
vary run to run (main 0.58-0.86 ms, ours 0.31-0.47 ms at L384 / 768 across runs). The B = 5 main cells ran the PyTorch module
path (main serves B = 1 only).

#### B = 1, inference

| layout | L | PyTorch compiled | cuEquivariance | main | ours | × main | × fastest other |
|---|---|---|---|---|---|---|---|
| 8 x 48 | 384 | 0.072 | 0.074 | 0.035 | **0.033** | 1.06 | 2.19 |
| 8 x 48 | 768 | 0.145 | 0.151 | 0.063 | **0.057** | 1.11 | 2.54 |
| 8 x 48 | 1536 | 0.392 | 0.399 | 0.150 | **0.141** | 1.07 | 2.78 |
| 12 x 32 | 384 |  |  | 0.035 | **0.035** | 1.00 |  |
| 12 x 32 | 768 |  |  | 0.063 | **0.059** | 1.07 |  |
| 12 x 32 | 1536 |  |  | 0.154 | **0.147** | 1.04 |  |
| 16 x 24 | 384 | 0.072 | 0.076 | 0.037 | **0.035** | 1.06 | 2.06 |
| 16 x 24 | 768 | 0.149 | 0.154 | 0.063 | **0.061** | 1.04 | 2.44 |
| 16 x 24 | 1536 | 0.408 | 0.418 | 0.158 | **0.158** | 1.00 | 2.59 |
| 24 x 16 | 384 |  |  | 0.037 | **0.035** | 1.06 |  |
| 24 x 16 | 768 |  |  | 0.065 | **0.063** | 1.03 |  |
| 24 x 16 | 1536 |  |  | 0.166 | **0.166** | 1.00 |  |
| 16 x 32 (d 512) | 384 |  |  | 0.037 | **0.035** | 1.06 |  |
| 16 x 32 (d 512) | 768 |  |  | 0.065 | **0.061** | 1.07 |  |
| 16 x 32 (d 512) | 1536 |  |  | 0.160 | **0.160** | 1.00 |  |

#### B = 1, training

| layout | L | PyTorch compiled | cuEquivariance | main | ours | × main (graph on) | × fastest other (graph on) |
|---|---|---|---|---|---|---|---|
| 8 x 48 | 384 | 0.480 / 0.254 | 0.477 / 0.204 | 0.787 / 0.113 | 0.448 / **0.100** | 1.12 | 2.04 |
| 8 x 48 | 768 | 0.625 / 0.514 | 0.658 / 0.407 | 0.577 / 0.201 | 0.325 / **0.178** | 1.13 | 2.29 |
| 8 x 48 | 1536 | 1.465 / 1.351 | 1.270 / 1.135 | 0.604 / 0.510 | 0.531 / **0.489** | 1.04 | 2.32 |
| 12 x 32 | 384 |  |  | 0.587 / 0.121 | 0.316 / **0.104** | 1.16 |  |
| 12 x 32 | 768 |  |  | 0.580 / 0.227 | 0.324 / **0.201** | 1.13 |  |
| 12 x 32 | 1536 |  |  | 0.823 / 0.622 | 0.625 / **0.580** | 1.07 |  |
| 16 x 24 | 384 | 0.485 / 0.254 | 0.634 / 0.209 | 0.584 / 0.125 | 0.452 / **0.110** | 1.13 | 1.89 |
| 16 x 24 | 768 | 0.657 / 0.541 | 0.658 / 0.430 | 0.594 / 0.237 | 0.470 / **0.223** | 1.06 | 1.93 |
| 16 x 24 | 1536 | 1.567 / 1.445 | 1.375 / 1.236 | 0.747 / 0.681 | 0.713 / **0.666** | 1.02 | 1.86 |
| 24 x 16 | 384 |  |  | 0.794 / 0.135 | 0.319 / **0.125** | 1.08 |  |
| 24 x 16 | 768 |  |  | 0.624 / 0.272 | 0.476 / **0.258** | 1.06 |  |
| 24 x 16 | 1536 |  |  | 0.884 / 0.821 | 0.850 / **0.806** | 1.02 |  |
| 16 x 32 (d 512) | 384 |  |  | 0.810 / 0.127 | 0.317 / **0.115** | 1.11 |  |
| 16 x 32 (d 512) | 768 |  |  | 0.863 / 0.243 | 0.315 / **0.227** | 1.07 |  |
| 16 x 32 (d 512) | 1536 |  |  | 0.750 / 0.688 | 0.717 / **0.673** | 1.02 |  |

#### B = 5, inference

| layout | L | PyTorch compiled | cuEquivariance | main | ours | × main | × fastest other |
|---|---|---|---|---|---|---|---|
| 16 x 24 | 384 | 0.174 | 0.182 | 0.194 | **0.080** | 2.44 | 2.18 |
| 16 x 24 | 768 | 0.492 | 0.506 | 0.578 | **0.203** | 2.85 | 2.43 |
| 16 x 24 | 1536 | 1.792 | 1.816 | 2.135 | **0.672** | 3.18 | 2.67 |

#### B = 5, training

| layout | L | PyTorch compiled | cuEquivariance | main | ours | × main (graph on) | × fastest other (graph on) |
|---|---|---|---|---|---|---|---|
| 16 x 24 | 384 | 0.709 / 0.600 | 0.696 / 0.558 | 0.744 / 0.622 | 0.475 / **0.283** | 2.20 | 1.98 |
| 16 x 24 | 768 | 1.971 / 1.836 | 1.889 / 1.737 | 2.109 / 1.965 | 0.889 / **0.848** | 2.32 | 2.05 |
| 16 x 24 | 1536 | 6.755 / 6.577 | 6.385 / 6.180 | 7.288 / 7.095 | 2.989 / **2.944** | 2.41 | 2.10 |

Host time per training step without CUDA graphs (`host_parts.py` / `host_probe.py` style probe, fp32 master parameters under
autocast bf16, the module compiled with `triton.cudagraphs` off, B = 1; us of host time per step, which is the step time here;
same-session values, +-30 % between runs on this shared host):

| layout, L | main | Python sequence (`apb_b200_host = False`) | C++ host path |
|---|---|---|---|
| 16 x 24, 384 | 847-1070 | 632-847 | 487-498 |
| 16 x 24, 768 | 613-876 | 618-629 | 326-336 |
| 8 x 48, 768 | 617-619 | 609-726 | 338-493 |

Inside the C++ ops (instrumented build, 16 x 24, L128): train_fwd 37 us, train_bwd 68 us of host work, of which the six cuBLAS
GEMMs through ATen take 6-8 us each; the rest of the step's host time is torch's (autograd engine, autocast, the module call).

Attention cores, kernel durations (CUPTI, eager steps, B = 1, bf16 parameters; us; split counts as chosen by `apb_ksplit` /
`apb_qsplit`):

| kernel | layout | L384 | L768 | L1536 |
|---|---|---|---|---|
| `attn_inf` unsplit -> split + `ksplit_combine` | 8 x 48 | 11.7 -> 6.5 + 3.2 (NS 3) | 18.4 -> 8.4 + 4.0 (NS 6) | 30.8 -> 15.5 + 4.2 (NS 3) |
| | 16 x 24 | 11.3 -> 7.1 + 3.2 (NS 3) | 18.1 -> 10.5 + 3.8 (NS 3) | 31.3 (unsplit) |
| `attn_fwd2` unsplit -> split (combine as above) | 8 x 48 | 10.4 -> 7.3 | 16.5 -> 8.2 | 28.4 -> 15.2 |
| | 16 x 24 | 10.1 -> 6.9 | 16.2 -> 10.0 | 28.8 (unsplit) |
| `attn_dkv` unsplit -> query split | 8 x 48 | 10.1 -> 7.3 (NQ 3) | 14.9 -> 9.1 (NQ 3) | 26.2 -> 21.0 (NQ 3) |
| | 16 x 24 | 9.1 -> 6.7 (NQ 3) | 13.7 -> 12.2 (NQ 3) | 47.0 -> 36.3 (NQ 3) |

The split cores move the forward / inference core from 2-29 % to 15-24 % of its time floor at L768 (floors from the
2026-09-30 table below: 1.9 us for both cores at 8 x 48 L768); what is left is per-item latency (prologue, pipeline fill,
partial store) and the combine launch. The split also makes the pair kernel that follows ~1 us slower (L2 state), which is why
the split is kept only where its estimate halves the unsplit core.

Accuracy and determinism (`tests/gpu/sm100/test_b200_apb_ksplit_gpu.py`, `test_b200_apb_host_gpu.py`, the existing
`test_b200_apb_gpu.py` / `test_apb_b200_gpu.py` / `test_b200_fp32_master_gpu.py`): every key-split and query-split count
against the fp32 module within 1.1x the unsplit error (padded key tails, whole masked chunks finite); the C++ path bitwise the
Python sequence (every layout, bf16 and fp32-master parameters, inference and training); five steps under a NaN-poisoned
allocator bitwise equal; CUDA-graph capture bitwise eager; B = 2 / 5 at L256 / 768 (samples of different lengths) and a chunked
B = 5 (max_batch 2) bitwise the one-sample calls for the output, dsingle and dpair, parameter gradients within 1e-5 of the
samples' sum; torch.compile against eager. 225 tests, plus the token DiT core tests (`test_augmented_attention_bf16_sm100_gpu.py`,
`test_b200_token_dit*_gpu.py`, `test_tdit_heads_sm100_gpu.py`; 309 in the combined run).

## Measurements (2026-10-01; CUDA-graph rows re-measured 2026-10-10; before the 2026-10-20 changes)

`bench.py target=attention_pair_bias level=module sweep_axis=seq_len min_seq_len=128 max_seq_len=768 seq_len_step=128
precision=bf16-mixed +apb_n_head=<heads> d_single=<d>`, compiled, one GPU with no other process on it, no key
masking (the config's `mask_prob` 0), ms. Inference with a CUDA graph; training forward + backward with CUDA graphs off /
on. × = ours against the fastest other row. Every implementation runs the same layout (`apb_n_head` and `d_single` set the
module for all of them). The rows of one layout come from one run: 8 x 48 and 16 x 24 from 12:32-12:47 (after the
deterministic dQ and this page's code), 24 x 16 / 12 x 32 / 16 x 32 from 10:09-10:50 the same day (the same kernels;
the later edits were the package move and the dh-64 TMEM clear, which these widths do not reach).

Re-measured 2026-10-10 (engine v2.2.0 plus unreleased changes, torch 2.13.0+cu129, triton 3.7.1, driver 580.95.05, sm_100a kernels built by nvcc 13.1, one GPU with no **2026-10-11:** every CUDA-graph cell of these tables was re-measured with fp32 master parameters (`precision=bf16-mixed`, the harness default since that date: fp32 parameters over bf16 activations under `torch.autocast`; one run per cell, `parameter_dtype` = float32 in the CSV, the engine environment's python called directly); the cells above are those values. The no-CUDA-graph training tables are host-bound and keep their earlier values (bf16 parameters).
other process on it; the Anthropic cells with `compile=false`): the inference table and the graph-on figures of the training table. A cell that
reproduced within 8 % keeps its value, a cell that moved was repeated and the median of the runs is shown. The PyTorch compiled and cuEquivariance
columns are 20-25 % faster than on 2026-10-01 at L512-768 since LayerNorm / RMSNorm on B200 became PyTorch ops that `torch.compile` fuses
(2026-10-02); the graph-off training figures are host-bound and keep their 2026-10-01 values.

Inference:

| layout | (Length, d_pair) | PyTorch compiled | Triton | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|---|---|
| 8 x 48 | (128, 128) | 0.047 | 0.049 | 0.051 | 0.039 | **0.024** | 1.59 |
| 8 x 48 | (256, 128) | 0.057 | 0.057 | 0.061 | 0.047 | **0.030** | 1.54 |
| 8 x 48 | (384, 128) | 0.070 | 0.074 | 0.074 | 0.055 | **0.035** | 1.59 |
| 8 x 48 | (512, 128) | 0.092 | 0.096 | 0.096 | 0.068 | **0.043** | 1.57 |
| 8 x 48 | (640, 128) | 0.121 | 0.123 | 0.125 | 0.082 | **0.053** | 1.54 |
| 8 x 48 | (768, 128) | 0.147 | 0.149 | 0.152 | 0.100 | **0.063** | 1.58 |
| 12 x 32 | (128, 128) | 0.049 | 0.051 | 0.053 | 0.039 | **0.024** | 1.61 |
| 12 x 32 | (256, 128) | 0.061 | 0.059 | 0.064 | 0.047 | **0.031** | 1.53 |
| 12 x 32 | (384, 128) | 0.074 | 0.074 | 0.078 | 0.057 | **0.035** | 1.64 |
| 12 x 32 | (512, 128) | 0.096 | 0.100 | 0.100 | 0.072 | **0.043** | 1.67 |
| 12 x 32 | (640, 128) | 0.123 | 0.125 | 0.129 | 0.092 | **0.053** | 1.73 |
| 12 x 32 | (768, 128) | 0.152 | 0.154 | 0.158 | 0.113 | **0.064** | 1.77 |
| 16 x 24 | (128, 128) | 0.047 | 0.051 | 0.051 | 0.039 | **0.024** | 1.59 |
| 16 x 24 | (256, 128) | 0.059 | 0.061 | 0.063 | 0.049 | **0.030** | 1.60 |
| 16 x 24 | (384, 128) | 0.071 | 0.076 | 0.076 | 0.059 | **0.035** | 1.71 |
| 16 x 24 | (512, 128) | 0.094 | 0.102 | 0.098 | 0.071 | **0.045** | 1.60 |
| 16 x 24 | (640, 128) | 0.121 | 0.129 | 0.127 | 0.092 | **0.053** | 1.73 |
| 16 x 24 | (768, 128) | 0.149 | 0.162 | 0.155 | 0.115 | **0.065** | 1.75 |
| 24 x 16 | (128, 128) | 0.047 | 0.049 | 0.051 | 0.039 | **0.024** | 1.59 |
| 24 x 16 | (256, 128) | 0.059 | 0.059 | 0.063 | 0.049 | **0.031** | 1.60 |
| 24 x 16 | (384, 128) | 0.074 | 0.076 | 0.078 | 0.057 | **0.037** | 1.56 |
| 24 x 16 | (512, 128) | 0.096 | 0.098 | 0.100 | 0.074 | **0.045** | 1.64 |
| 24 x 16 | (640, 128) | 0.123 | 0.125 | 0.129 | 0.096 | **0.055** | 1.74 |
| 24 x 16 | (768, 128) | 0.151 | 0.155 | 0.158 | 0.119 | **0.067** | 1.76 |
| 16 x 32 (d 512) | (128, 128) | 0.049 | 0.051 | 0.053 | 0.043 | **0.024** | 1.76 |
| 16 x 32 (d 512) | (256, 128) | 0.059 | 0.061 | 0.064 | 0.049 | **0.031** | 1.60 |
| 16 x 32 (d 512) | (384, 128) | 0.074 | 0.078 | 0.078 | 0.059 | **0.037** | 1.62 |
| 16 x 32 (d 512) | (512, 128) | 0.094 | 0.100 | 0.100 | 0.076 | **0.045** | 1.68 |
| 16 x 32 (d 512) | (640, 128) | 0.125 | 0.131 | 0.129 | 0.094 | **0.055** | 1.70 |
| 16 x 32 (d 512) | (768, 128) | 0.151 | 0.162 | 0.158 | 0.117 | **0.066** | 1.78 |

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
| 8 x 48 | (128, 128) | 0.403 / 0.135 | 1.086 / 0.125 | 0.505 / 0.121 | 0.657 / **0.080** | 1.51 |
| 8 x 48 | (256, 128) | 0.555 / 0.199 | 1.114 / 0.166 | 0.572 / 0.166 | 0.665 / **0.094** | 1.76 |
| 8 x 48 | (384, 128) | 0.545 / 0.254 | 1.093 / 0.219 | 0.900 / 0.205 | 0.650 / **0.113** | 1.82 |
| 8 x 48 | (512, 128) | 0.573 / 0.336 | 1.093 / 0.296 | 0.717 / 0.262 | 0.492 / **0.135** | 1.94 |
| 8 x 48 | (640, 128) | 0.517 / 0.442 | 0.922 / 0.381 | 0.935 / 0.356 | 0.658 / **0.164** | 2.17 |
| 8 x 48 | (768, 128) | 0.617 / 0.515 | 0.933 / 0.469 | 0.732 / 0.412 | 0.497 / **0.197** | 2.09 |
| 12 x 32 | (128, 128) | 0.559 / 0.133 | 0.955 / 0.123 | 0.446 / 0.123 | 0.684 / **0.084** | 1.47 |
| 12 x 32 | (256, 128) | 0.614 / 0.194 | 0.955 / 0.166 | 0.556 / 0.168 | 0.730 / **0.098** | 1.69 |
| 12 x 32 | (384, 128) | 0.622 / 0.256 | 1.003 / 0.219 | 0.802 / 0.209 | 0.561 / **0.119** | 1.76 |
| 12 x 32 | (512, 128) | 0.516 / 0.334 | 0.969 / 0.291 | 0.770 / 0.268 | 0.556 / **0.149** | 1.79 |
| 12 x 32 | (640, 128) | 0.661 / 0.430 | 1.176 / 0.375 | 0.974 / 0.348 | 0.697 / **0.184** | 1.89 |
| 12 x 32 | (768, 128) | 0.656 / 0.517 | 1.146 / 0.473 | 0.986 / 0.414 | 0.686 / **0.225** | 1.84 |
| 16 x 24 | (128, 128) | 0.394 / 0.133 | 0.881 / 0.129 | 0.411 / 0.123 | 0.679 / **0.086** | 1.43 |
| 16 x 24 | (256, 128) | 0.439 / 0.195 | 1.127 / 0.172 | 0.582 / 0.168 | 0.693 / **0.102** | 1.64 |
| 16 x 24 | (384, 128) | 0.564 / 0.254 | 1.135 / 0.229 | 0.937 / 0.209 | 0.535 / **0.125** | 1.67 |
| 16 x 24 | (512, 128) | 0.601 / 0.334 | 0.914 / 0.305 | 0.728 / 0.269 | 0.696 / **0.155** | 1.73 |
| 16 x 24 | (640, 128) | 0.535 / 0.457 | 1.171 / 0.407 | 0.810 / 0.371 | 0.695 / **0.192** | 1.93 |
| 16 x 24 | (768, 128) | 0.646 / 0.545 | 1.220 / 0.506 | 0.957 / 0.430 | 0.736 / **0.240** | 1.79 |
| 24 x 16 | (128, 128) | 0.610 / 0.133 | 1.083 / 0.123 | 0.615 / 0.125 | 0.782 / **0.086** | 1.43 |
| 24 x 16 | (256, 128) | 0.659 / 0.198 | 1.164 / 0.166 | 0.733 / 0.174 | 0.858 / **0.104** | 1.59 |
| 24 x 16 | (384, 128) | 0.649 / 0.258 | 1.390 / 0.221 | 0.552 / 0.219 | 0.869 / **0.135** | 1.62 |
| 24 x 16 | (512, 128) | 0.616 / 0.347 | 1.200 / 0.295 | 0.577 / 0.279 | 0.734 / **0.166** | 1.68 |
| 24 x 16 | (640, 128) | 0.609 / 0.456 | 1.105 / 0.381 | 0.601 / 0.373 | 0.690 / **0.215** | 1.73 |
| 24 x 16 | (768, 128) | 0.652 / 0.550 | 0.937 / 0.485 | 0.603 / 0.456 | 0.526 / **0.268** | 1.70 |
| 16 x 32 (d 512) | (128, 128) | 0.612 / 0.137 | 1.258 / 0.127 | 0.629 / 0.125 | 0.806 / **0.090** | 1.39 |
| 16 x 32 (d 512) | (256, 128) | 0.680 / 0.199 | 1.346 / 0.170 | 0.671 / 0.170 | 0.867 / **0.106** | 1.60 |
| 16 x 32 (d 512) | (384, 128) | 0.675 / 0.258 | 1.270 / 0.226 | 0.708 / 0.213 | 0.518 / **0.129** | 1.65 |
| 16 x 32 (d 512) | (512, 128) | 0.685 / 0.338 | 1.273 / 0.301 | 1.083 / 0.274 | 0.814 / **0.158** | 1.74 |
| 16 x 32 (d 512) | (640, 128) | 0.718 / 0.459 | 1.294 / 0.401 | 0.816 / 0.379 | 0.826 / **0.197** | 1.93 |
| 16 x 32 (d 512) | (768, 128) | 0.701 / 0.544 | 1.357 / 0.489 | 1.145 / 0.434 | 0.616 / **0.244** | 1.78 |

The layouts cost ours the same within 0.003 ms in inference; in training (graph on) 24 x 16 is the slowest at L768 (0.256
against 0.186 ms for 8 x 48), the other layouts in between. Without CUDA graphs every row is host-bound (the GPU work is
0.07-0.26 ms): ours ranges 0.58-1.30x the fastest other row (compiled PyTorch or cuEquivariance), behind at short lengths
-- see "Limits and next". Before this path (the engine's module path, Triton attention, 8 x 48, 2026-09-30): inference
0.039 / 0.063 / 0.139 ms and training (graph on) 0.106 / 0.199 / 0.450 ms at L128 / L384 / L768.

Accuracy against the fp32 PyTorch module (`tests/gpu/sm100/test_b200_apb_gpu.py`, every layout, L128 / 208 / 384 / 768
inference and L128 / 384 training, key mask on / off): output and every input and parameter gradient within 1.3x the bf16
PyTorch module's error (8 x 48 probe at L128 / 384 / 768, 20 % key mask: output 1.1-1.2e-2 against 1.1-1.2e-2, gradients
0.99-1.13x, worst ~2e-2 relative); the compiled module matches eager.

Tests: `tests/gpu/sm100/test_apb_b200_gpu.py` (the pair kernels at 8 / 12 / 16 / 24 heads against fp64 at L128-768; per
layout the inference core at L128 / 200 / 256 / 384 / 640 / 768 and the training forward + backward at L128 / 256 / 384 /
768 against fp64), `tests/gpu/sm100/test_b200_apb_gpu.py` (module against fp32 PyTorch per layout: inference at L128 /
208 / 384 / 768, training at L128 / 384, key mask on / off; torch.compile against eager).

### Hardware limit (SoL) per kernel (8 x 48, 2026-09-30)

Measured before the residual fold (an extra residual pass then) and the deterministic dQ (`qkv_bwd` now also adds the dQ
chunk slices); the other layouts are not profiled per kernel.


The token DiT doc's method: floor = max(minimum HBM bytes / BW, FLOPs / tensor rate, exp2 count / MUFU rate) per kernel;
SoL = floor / measured (kernel duration, median over back-to-back eager steps; the measured us in parentheses). Ceilings:
HBM 6.31 TB/s (a 512 MB fp32 copy in the same process), tensor 2.23 PF/s bf16, MUFU 31 ex2 / ns / SM. B = 1, 20 % key
mask. "step" = the kernels' SoL weighted by their time (PyTorch fills counted at 0).

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
| ln_pair.bias gradient from the kernel's per-head dbias sums | rounding noise around the exact zero (up to 2x the bf16 module's noise); returned as 0 (the parameter no longer exists) |
| 16 / 24 heads in the pair backward: dWf accumulated in a second phase over the tile instead of per-warp accumulators | slower; reverted to per-warp accumulators at one block per SM |
| key-split combine as a warp per row with a runtime chunk count (2026-10-20) | 4-10 us: the guarded partial loads issued one latency at a time; a thread per four columns, templated on NS: 3-5 us |
| key split wherever waves / chunks improves (2026-10-20) | at two waves of a third (16 heads L1536, 24 heads L1024) the partial stores, the combine and a ~1 us slower pair kernel after it ate the gain (16 x 24 L1536 0.160 vs 0.158 ms unsplit); the split is kept only where its estimate halves the unsplit core |
| batched indexing in the pair kernels for every B (2026-10-20) | the one-sample forward 1.6-6 % slower at L1536 (24 heads: 123.5 vs 115.7 us); B = 1 compiles the single-sample indexing |
| 16 x 24 heads as 24 wide (no padding) | not built: 24 is not a multiple of the q k^T MMA's K step (16), and a 48-B row is no swizzle span (32 / 64 / 128 B); 32-wide heads with exact zero pads cost 33 % more q / k / v / g bytes and keep every kernel on the existing tiles |

## Limits and next

- Inference at L >= 768 is the pair kernel's (HBM-bound at 80-95 %: 27 of 50 us at 8 x 48 L768, 104 of 148 us at 16 x 24
  L1536); the attention core with its combine is 12-16 us of it.
- The key split's combine is its own launch (3-5 us). The last chunk of an item could merge the others' partials in the core's
  epilogue (a counter per item; the merge order fixed, so still deterministic), removing the launch and the extra pass.
- Backward: `attn_dqb` writes dbias in fp32 ([S H, L, L], 151 MB at 16 x 24 L1536) that `pair_bias_bwd` reads again
  (dqb 98 us, pair backward 268 us there); `bias_transpose` (11 / 44 us at 16 x 24 L768 / 1536) exists only because `attn_dkv`
  reads bias^T -- loading the bias tile through TMA and reading it transposed in shared memory would remove that pass.
- Host: without CUDA graphs the step is still host-bound below L ~1024 (C++ ops ~105 us of host work, half of it the six
  ATen GEMM calls; the rest torch's autograd / autocast / module call). Calling cuBLAS directly with cached algorithms would
  save ~30 us but must reproduce ATen's algorithm choice to keep the bits.
- Not served: QK-norm (no MiniWorld config uses it with this module), fp32 activations, other shapes (module path). H100 keeps
  B = 1.
- Per-kernel SoL beyond the attention cores is the 2026-09-30 table (8 x 48).
