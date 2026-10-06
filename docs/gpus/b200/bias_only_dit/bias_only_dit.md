# Bias-only token DiT on B200 (sm100)

Kernel-level status of the bias-only token DiT block on B200; the module-level summary is in [b200.md](../b200.md). The block
(`modules/bias_only_dit`, `BiasOnlyDiTBlock`) is the token DiT block with the attention's query-key half removed: AdaLN +
**bias-only attention** (the logits are the pair bias, `softmax(to_bias(LN(pair))) v`, gated by `sigmoid(g)` and by the
conditioning scale) + conditioned SwiGLU transition. Widths as the `dit` target: d_single 768, d_cond 384, d_pair 128,
transition n = 2. Columns below are (Length, Dimension, dtype) with the token width 768 (`token_single`); the pair-bias kernels
take (Length, d_pair). No other implementation of this op exists, so every comparison is against the PyTorch module.

Why a module of its own rather than a flag on `DiTBlock`: with no query and no key the attention weights `P = softmax(bias)`
depend on neither the augmented sample nor the single representation -- only on the pair, which carries no noise level. One `P`
per block serves every sample and every solver step. The fast path is therefore a different algorithm: `P` is hoisted, and the
per-sample attention is a GEMM per head (`P_h [L, L] x v_h [L, d_head]`) with the gate in its epilogue -- no online softmax, no
running max, no rescale.

## Scope and dispatch

Both paths run **hand-written CUDA and cuBLAS only** (no Triton, no quack), bf16 -- or fp32 on TF32 tensor cores (an fp32 block:
fp32 inputs and weights, no CUDA autocast; see [fp32 path](#fp32-path-tf32)) -- B == 1, L a multiple of 128 up to 768, a key
mask [1, L] or none, LayerNorm eps 1e-5, implementation MINIWORLD or TRITON. Everything else -- other cards, other widths, fp32
under autocast or with bf16 weights -- runs the module's PyTorch composition. Each integration's `serves()` is the whole gate.

- **Inference** (no autograd): `BiasOnlyDiTBlock.forward` -> `integrations/bias_only_dit.py` -> `kernels/bias_only_dit/cuda/runner.py`
  (`FusedBiasOnlyDiT`). One conditioning per sample or one shared by the samples (sample axis 1 or stride 0). The weight pack and
  `P` are cached on the tensors' (pointer, version), CUDA-graph replays included.
- **Training** (autograd on): `integrations/bias_only_dit_train.py`. One conditioning per sample. One autograd Function per block
  whose forward and backward are each one opaque op (`torch.compile` keeps them as nodes); CUDA-graph safe (launches take their
  row counts as arguments, no host syncs).

**Head layouts.** Four (heads, head width) layouts are served; the layout is read off the weights (`to_bias.weight` rows = heads,
`to_value.weight` rows = attention channels), is a compile-time constant of the sm_100a cubins (`-DNHEAD`, `-DDHEAD`; none for
16 x 48) and a template or argument of the row kernels.

| layout | attention channels | module | bench |
|---|---|---|---|
| 16 heads x 48 (default, the `dit` target's) | 768 | `BiasOnlyDiTBlock()` | (default) |
| 24 heads x 32 | 768 | `BiasOnlyDiTBlock(n_head=24)` | `+n_head=24` |
| 12 heads x 64 | 768 | `BiasOnlyDiTBlock(n_head=12)` | `+n_head=12` |
| 16 heads x 64 | 1024 (v / g / out projections 768 <-> 1024) | `BiasOnlyDiTBlock(d_head=64)` | `+d_head=64` |

**Switches** (default on): `MINIWORLD_BIAS_ONLY_DIT` (inference), `MINIWORLD_BIAS_ONLY_DIT_TRAIN` (training),
`MINIWORLD_BIAS_ONLY_DIT_CORE` (the sm_100a core). Experiment knobs: `MINIWORLD_BIAS_ONLY_DIT_SG` (samples per core work item),
`MINIWORLD_BIAS_ONLY_DIT_PV_VB` (keys per v tile), `MINIWORLD_BIAS_ONLY_DIT_RESLN` / `_COND` (inference GEMM-epilogue experiments,
off: slower, see below), `MINIWORLD_BO_ROWS_BLOCKS_PER_SM` (cap on the training row kernels' resident blocks).

**cache build ✓** everywhere: nothing on these paths autotunes. The row kernels have fixed launch shapes (grids sized from the
occupancy API), the sm_100a kernels are cubins built on first use into `MINIWORLD_ENGINE_JIT_ROOT` (keyed by source and
definitions), the choices that depend on the shape (core sample group, v tile, `dpb` key tile) are closed-form cost models, and
the GEMMs are cuBLAS.

**성능 확인** (2026-10-01, judged per kernel and shape from the per-kernel probes `sol_bo.py` (training, A = 48) and `sol_inf.py`
(inference, S = 5): kernel time against its floor = max(minimum HBM bytes / 6.31 TB/s, FLOPs / 2.23 PF/s, the energy of those
bytes and FLOPs at the 1000 W cap)): ✓ = at least 90 % of the floor in all four head layouts; △ = the fastest implementation
measured (there is no other), below 90 %. Training columns are the measured training lengths (384, 768).

## Inference

Per call the runner

1. (once per pair and mask) makes `P` for every block: a CUDA LayerNorm of the pair rows, one cuBLAS GEMM for the per-head bias
   (the LayerNorm weight folded into the projection), and the CUDA row softmax written in place;
2. makes the conditioning tables: a CUDA LayerNorm of the conditioning rows and two cuBLAS GEMMs (AdaLN scale / shift of both
   halves, the two output gates), over L rows when the samples share one conditioning, over S L rows otherwise;
3. runs each block: input AdaLN rows -> cuBLAS v|g GEMM -> **`pv_gate_inf`** -> cuBLAS out GEMM -> residual + gate + AdaLN rows
   -> expand GEMM + SwiGLU -> cuBLAS squeeze GEMM -> residual + gate rows in the output dtype. The residual stream is fp32.

### I1 · attention core `pv_gate_inf` (sigmoid(g) · (P v), per head and sample)

`kernels/bias_only_dit/cuda/pv_gate_inf.cu`. Work item = (head, 128-query tile, group of SG samples); SG from a cost model (fewest
bytes into the busiest SM; SG = 2 at L384 and 5 at L768 for S = 5, 16 for A = 48). The item's `P` tile [128 x L] is copied once
into **tensor memory** (TMA in 64-key SW128 chunks, `tcgen05.cp` into the A-operand layout, L / 2 columns) and stays there while
the group's samples stream their v tiles [VB keys x d_head] through the shared-memory ring (VB = 128 for few samples, 192 / 256
for A = 48: larger TMA boxes); per sample `M 128 x N d_head` products, A from TMEM, into one of two d_head-column accumulators at
the top of TMEM. The loop is sample-outer, so the epilogue of sample k (g in by TMA, `sigmoid(g) * o` in packed f32x2 with the
reciprocal on the FMA pipe, the gated tile out by TMA) runs under the products of sample k + 1. A head row (d_head x 2 bytes) sits
in one 128-byte swizzle row: 96 B for 16 x 48, 64 B for 24 x 32, the whole row for d_head 64. The training backward runs the same
kernel without the gate on `P^T` (`dV = P^T dO`).

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I2 · row kernels `ln_rows`, `adaln_in_rows`, `resgate_adaln_rows`, `resgate_out_rows`, `swiglu_rows`, `softmax_rows`

`kernels/bias_only_dit/cuda/bias_only_dit_rows.cu`: one warp per row, every load of the row issued before any math, the
statistics as warp reductions over registers (no block barrier); 5-6 TB/s at 1920 rows. `softmax_rows` takes the key mask
(masked keys at the largest negative finite logit: a fully masked row is uniform, as in the PyTorch module). The pair LayerNorm
is the token DiT's CUDA `layernorm128_rows`.

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I3 · expand GEMM + SwiGLU (from M = S L >= 3840 the token DiT's `gemm_swiglu2_sm100`; below it cuBLAS + `swiglu_rows`)

| (Length, Dimension, dtype) | (128, 768, bf16) | (256, 768, bf16) | (384, 768, bf16) | (512, 768, bf16) | (640, 768, bf16) | (768, 768, bf16) |
|---|---|---|---|---|---|---|
| implementation | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | cuBLAS + CUDA | CUDA |
| 성능 확인 | △ | △ | △ | △ | △ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Training

bf16 activations, fp32 accumulation; weight gradients leave cuBLAS in the parameter's dtype (the engine's LayerNorm weights stay
fp32 in a bf16 block: their gradients are written in fp32). What the forward keeps: the conditioning tables and statistics, `xa`,
`v|g`, `P` and `P^T`, the gated attention output, `y`, `xt`, `a|b`, `h`, `z` and the row statistics -- not the attention's
pre-gate output (the backward gets it from the gated one) and not the fp32 residual `x1 = x + sigmoid(g1) y` (rebuilt with one
FMA wherever it is needed: 4 B per element less traffic and memory).

- Forward: pair bias (T1) -> softmax writing `P` and `P^T` (T2) -> conditioning LN (T4) -> cuBLAS tables -> AdaLN (T4) -> cuBLAS
  v|g -> `pv_gate_inf` (I1) -> cuBLAS out -> residual + AdaLN (T4) -> expand GEMM + SwiGLU keeping a | b (T5) -> cuBLAS squeeze ->
  residual out (T4).
- Backward: residual / SwiGLU / AdaLN backwards (T4) around the cuBLAS data and weight gradients (one GEMM per group of
  parameters that share an input); the attention: `do = da sigmoid(g)`, `dg = da a (1 - sigmoid(g))`, `D = sum da a` per row
  and head (`gate_bwd_rows`, T4; `a = sigmoid(g) o`, so `o` is never stored); `dV = P^T do` (I1 without the gate); `dbias =
  P o (sum_a do v^T - D)` (T3); `dbias` -> d pair and `dWf` (T1); last, `finalize` (T4) writes the small weight gradients.

### T1 · pair bias on tensor cores: `pair_bias` / `pair_bias_bwd` (bias = LN(pair) Wf^T and its backward)

`kernels/bias_only_dit/cuda/bias_only_dit_train_rows.cu`, templates on the head count. The R = L^2 pair rows of 128 go through
`mma.sync` (bf16, fp32 accumulate), so LN(pair) never reaches HBM. Forward: 16-row tiles per warp, LN in registers, the
[16 rows x 128] x [128 x heads] product (heads in n-tiles of 8; 12 heads padded to 16 with zero weights), the bias staged per warp
and stored head-major; writes only the bias and the row statistics. Backward: 128 rows per block item, the next item's pair rows
and dbias streamed in by `cp.async` (double buffer, chunks XOR-swizzled by row): `d LN = dbias^T Wf` on mma (K = heads: one k16
step, plus a k8 step for heads 16-23), the LN backward reduced algebraically (pass 1 sums `d LN` and `d LN * pair`, pass 2 one FMA
pair per element), d pair out; LN(pair) is written back into the buffer it was read from, and each warp adds its 16 columns of
`dWf^T += LN(pair)^T dbias^T` over the item's rows on mma (M = columns, N = heads, K = rows; fragments by `ldmatrix`). dWf leaves
as one partial per block. The products' K / N slots are assigned so that every lane's pair loads and d pair stores are 64
contiguous bytes per row. Issue-bound (element-wise fp32 math around small products): 51-65 % of its floor.

| (Length, Dimension, dtype) | (384, 128, bf16) | (768, 128, bf16) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | △ | △ |
| cache build | ✓ | ✓ |

### T2 · softmax writing `P` and `P^T`: `softmax_t`

`bias_only_dit_rows.cu`. A block takes 32 query rows of one head (four per warp, the row softmax of I2), keeps them in shared
memory (row pitch L + 2: the column gathers hit distinct banks) and writes the [L keys x 32 queries] tile of `P^T` in 64-byte row
pieces, so the backward's `dV = P^T dO` needs no transpose kernel. 24-73 % of its floor (few rows per block at L384).

| (Length, Dimension, dtype) | (384, 768, bf16) | (768, 768, bf16) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | △ | △ |
| cache build | ✓ | ✓ |

### T3 · bias gradient `dpb_sm100` (dbias = P o (sum_a do v^T - D))

`kernels/bias_only_dit/cuda/dpb_sm100.cu`. Work item = (head group, 128-query tile, NJ-key tile); its K loop walks the samples
(per sample one do tile [128 x d_head] and the key tile [NJ x d_head] in two halves, `M 128 x N NJ / 2 x K d_head` tcgen05 products
into TMEM); the epilogue streams `P` in [128 x 32] pieces and applies `P o (dP - D)`. With 32-wide heads an item takes two heads
(their 64 columns fill the 128-byte row: 16 KB TMA boxes instead of 8, one accumulator per head). An item is bound by its SM's TMA
intake, so NJ comes from a cost model -- rounds of items over the SMs x (128 + NJ) rows per sample: 16 x 48 L384 NJ 128, L768
NJ 256; 24 x 32 L768 NJ 192. 69-98 % of its floor (TMA intake per SM).

| (Length, Dimension, dtype) | (384, 768, bf16) | (768, 768, bf16) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | △ | △ |
| cache build | ✓ | ✓ |

### T4 · training row kernels (conditioning LN, AdaLN, residual + AdaLN, residual out, their backwards, gate backward, finalize)

`bias_only_dit_train_rows.cu` and `gate_bwd_rows` in `bias_only_dit_rows.cu`. The 768-wide rows take **two warps per row**
(64 threads, 12 columns each; one warp per row spilled in the backward) and move 5.5-6.5 TB/s; LayerNorm statistics merge per-
thread (mean, M2) pairs in one exchange between the two warps; the forward LayerNorm kernels keep their loads packed until used
(four blocks per SM). Persistent blocks (as many as are resident) accumulate the bias gradients' column sums in registers or
shared memory and leave one partial row per block. `unfold` writes the AdaLN projections' gradients (from the folded GEMM) in
the parameters' dtype; `swiglu_bwd` is a 16-byte elementwise pass using the forward's `h`. `finalize`, the step's last kernel,
sums every partial (bias sums, cond-LN weights, dWf -> `ln_pair` and `to_bias`) into the small gradients in the parameters'
dtypes (no torch glue, no copies). `gate_bwd_rows` covers 768 or 1024 attention channels. Together 91-95 % of their floor.

| (Length, Dimension, dtype) | (384, 768, bf16) | (768, 768, bf16) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ |
| cache build | ✓ | ✓ |

### T5 · expand GEMM + SwiGLU keeping a | b (`gemm_swiglu2_sm100` with SAVE_AB)

The token DiT's kernel built with SAVE_AB (`GemmSwigluAB`): `h` and `a | b` from one kernel, so the forward does not read
`a | b` back for `h`. A = 48: L384 82 us against cuBLAS 59 + SwiGLU 29; L768 152 against 170. 95-102 % of its floor.

| (Length, Dimension, dtype) | (384, 768, bf16) | (768, 768, bf16) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✓ | ✓ |
| cache build | ✓ | ✓ |

**Host side.** The sm_100a launches bind their TMA descriptors once per buffer set and reuse a prebuilt argument block (the
shared driver rebuilt ctypes arguments per call, 25-35 us each). The descriptors do not keep their tensors alive: holding them
pinned a step's activations per call (7 cudaMallocs per step, 13 GB reserved after a few steps). A step issues in ~0.9 ms of host
time at L384 against ~1.0 ms of GPU time, so without CUDA graphs the GPU stays the bound.

## fp32 path (TF32)

Status (2026-10-06): written, **not yet built or measured on B200** -- the kernels below compile on first use; the tests are
`tests/integrations/test_b200_bias_only_dit_tf32_gpu.py`. Every head layout of the bf16 path (16 x 48, 24 x 32, 12 x 64,
16 x 64), the same lengths (L a multiple of 128 up to 768), masks, shared / per-sample conditioning, eager, `torch.compile`
(the same opaque ops) and CUDA graphs.

**Gate.** `serves()` takes fp32 when single, cond and pair are fp32, the block's weights are fp32 (`to_value.weight`) and CUDA
autocast is off (under autocast the module path keeps its casts, as before), and `tf32_ready` (a `device_constant`) has built the
fp32 kernels: the fp32 row extension, the bf16 row extensions whose dtype-generic passes the fp32 path shares, the attention core
(and for training the ungated core and `dpb32_sm100`). A failed build warns once and keeps the module path for the process.
`MINIWORLD_BIAS_ONLY_DIT_TF32=0` turns the fp32 path off.

**Recipe.** Every activation, saved tensor, table and gradient fp32; the residual stream fp32. General GEMMs are cuBLAS on TF32
tensor cores (`tf32_gemms`: forced whatever the caller's `allow_tf32`, restored after -- the token DiT fp32 path's rule). The
products of the attention are tcgen05 `kind::tf32` (fp32 operands rounded to tf32 by the MMA, fp32 accumulation). The training
pair bias and its backward are TF32 `mma.sync` (operands rounded to nearest, as cuBLAS rounds the module's `to_bias` GEMM and its
gradients; `MINIWORLD_BIAS_ONLY_DIT_PAIR_EXACT=1`: split fp32, three products, ~exact); the inference hoist keeps the exact fp32
FMA kernel (once per pair). Weight gradients fp32 (cuBLAS), the small ones through the bf16
extension's dtype-generic `unfold` / `finalize`.

| step | inference | training forward | training backward |
|---|---|---|---|
| pair bias `LN(pair) Wf^T` | `pair_bias` (f32 rows) per block | `pair_bias` + row stats | `pair_bias_bwd`: d pair, dWf partials |
| softmax | `softmax_rows` (f32) | `softmax_t` (P and P^T, f32) | -- |
| attention | **`pv_gate_tf32`** (gated) | **`pv_gate_tf32`** (gated) | `gate_bwd` rows, **`pv_gate_tf32`** ungated on P^T (dV), **`dpb32_sm100`** |
| conditioning | `ln_rows` (fp32 in / out) + 2 cuBLAS TF32 GEMMs | `cond_ln` + 2 GEMMs | `cond_bwd`, `unfold`, `finalize` |
| rows | `adaln_in_rows`, `resgate_*_rows` (dtype-generic), `swiglu` (f32) | `adaln_a`, `res_adaln_b`, `swiglu`, `res_c` | `res_c_bwd`, `swiglu_bwd`, `res_adaln_b_bwd`, `adaln_a_bwd` |
| v\|g, out, expand, squeeze | cuBLAS TF32 | cuBLAS TF32 | cuBLAS TF32 (data and weight gradients) |

### F1 · `pv_gate_tf32.cu` (a = sigmoid(g) (P v), or P v)

P no longer fits tensor memory in fp32 (a 128-query tile is L columns, 768 > 512), so the loop is turned around: **key-outer,
sample-inner**. A 32-key chunk of P [128 i][32 j] (128-B rows, SW128, 16 KB) is the K-major A operand from shared memory and
feeds the products of the item's SG samples, each into its own accumulator. v is the B operand with the head's channels as N:
MN-major, which `kind::tf32` reads only from the 128-B swizzle with 32-B atoms (`SWIZZLE_128B_ATOM_32B`, UMMA layout type 1:
the operand form `attn_fwd_tf32` / `attn_dkv_tf32` use) -- a sample's chunk is DH / 32 (rounded up) boxes [32 keys][32 channels],
4 KB each, LBO 4 KB, 1 KB per K step of 8 keys. Accumulators double-buffered by item parity (2 SG DH <= 512 TMEM columns), so an
item's epilogue (fp32 gate, staged g / a tiles by TMA: channels 0-31 SW128 + 32..DH-1 SW64 / SW128) runs under the next item's
products.

| budget | 16 x 48 | 24 x 32 | 12 x 64 / 16 x 64 |
|---|---|---|---|
| sample groups SG built | 1, 2, 4, 5 | 1, 2, 4, 8 | 1, 2, 4 |
| stage (P chunk + SG v chunks), at the largest SG | 16 + 5 x 8 KB = 56 KB, 2 stages (SG 4: 48 KB, 3) | 16 + 8 x 4 KB = 48 KB, 3 stages | 16 + 4 x 8 KB = 48 KB, 2 stages |
| g / a staging (3 tiles) | 3 x 24 KB | 3 x 16 KB | 3 x 32 KB |
| TMEM (2 SG DH columns) | 480 -> 512 | 512 | 512 |

Warps: 0 TMA producer, 1 MMA (whole warp waits, `elect_one()` issues), 2 TMEM allocator, 4-7 epilogue (a query row per thread);
`__launch_bounds__(256, 2)` caps the registers at 128 (shared memory keeps one CTA per SM). SG per call: the fewest bytes into the
busiest SM (`pick_group_tf32`: rounds x (P once + SG v tiles + their g / a)); `MINIWORLD_BIAS_ONLY_DIT_SG` forces one.

### F2 · `dpb32_sm100.cu` (dbias = P o (sum_a dO v^T - D))

`dpb_tf32.cu` was removed on 2026-10-06: identical reruns gave different dbias at L 640 / 768 for every head layout (a CTA with more
than one work item; L <= 384 had one per CTA and was bit-identical), and 16 x 64 at L 640 missed the fp64 bound (8.4e-3 > 3e-3).
Its replacement is a port of the bf16 `dpb_sm100.cu` -- correct in bf16 -- with its structure kept: work item (head group, 128-query
tile, NJ-key tile), K loop over the samples, warps 0 TMA / 1 MMA / 2 TMEM / 3 P producer / 4-7 epilogue, ONE accumulator released
after the item, the two-slot P ring, three per-warp dbias staging tiles. What fp32 forces: a head row is DH / 32 (rounded up) boxes
of 32 channels in 128-byte swizzle rows (48-wide heads 32 + 16, the 16-channel box in the first 64 bytes of its rows, as the bf16
kernel's 96-byte rows), 4 / 2 `kind::tf32` K steps each; P pieces of 16 KB and dbias staging tiles of 4 KB (SW128); and the key tile
NJ = 128 so that two 64 KB stages fit (bf16: NJ up to 384), loaded as one box and multiplied as ONE N = 128 product per K step (bf16
splits its larger tile into two halves; with fp32's NJ = 128 the halves only re-read the do tile: 38 % of the MMA floor, L768 159 us,
as first ported). Tests: reruns bit-identical at L 128..768 for every layout, within 3e-3 of fp64, no spills.

### F3 · fp32 rows (`bias_only_dit_f32_rows.cu`)

The training rows keep the bf16 kernels' layout (768-wide rows on two warps, persistent blocks, per-block column-sum partials that
the shared `finalize` sums); every kernel at most 128 registers. `pair_bias`: a block takes 128 pair rows into shared memory (pitch
129), a thread one row: two-pass statistics, then the H dot products with Wf^T read as float4 broadcasts -- exact fp32, no
LN(pair) in memory. `pair_bias_bwd`: 32 rows per step, a thread one channel (Wf's column and its dWf column in registers,
dbias^T as broadcasts), then a warp per row for the LayerNorm backward. `softmax_t` keeps 16 rows in shared memory (pitch L + 4;
two rows per warp, both rows' loads first: 32 rows a block, one per warp at a time, was latency-bound, 58 us at L768) and writes
P^T with 16-byte stores.

Training pair bias on TF32 tensor cores (`pair_bias_tc`, `pair_bias_bwd_tc`; the FMA kernels above were issue-bound, L768 185 + 299 us
against memory floors of ~60 + ~105): the bf16 kernels' scheme on `mma.sync m16n8k8 .tf32`. Forward: a warp takes 16 pair rows
(8 float4 per lane and row, 64 contiguous bytes per row across a lane quad), the LayerNorm statistics over the quad, LN(pair) Wf^T
with the K slots mapped onto the loaded columns, the bias stored head-major from the accumulators. Backward: 128 rows per block item
through a cp.async double buffer (rows at pitch 136 floats, dbias at 132: conflict-free fragment reads), d LN = dbias^T Wf (K = the
heads), the LayerNorm backward over the quad (d LN kept in registers between its two passes), d pair out, LN(pair) written back in
place, then dWf^T += LN(pair)^T dbias^T per warp's 16 channels; one dWf partial per block.

**Weight pack** (`pack`): the fp32 step repacks its weights inside every captured step (the cache is scoped to the capture); the
eight torch cats / products (31 us cold) are one launch of up to 16 copy segments, the LayerNorm weights folded in as a column scale
(a grid row per segment, the table in parameter space; a per-element search with a 64-bit modulo ran 42 us).
`dxt` and `dxa` (the d shift halves of dG) leave cuBLAS straight in their dG columns, so the two LayerNorm backward rows read them
there and do not copy them (2 x [M, 768] fp32 writes less).

### F4 · `gemm_glu_tf32.cu` (the transition's GEMMs with the SwiGLU in their epilogue, training)

The fp32 step's expand GEMM and its SwiGLU were cuBLAS + a row pass that read `a | b` back (L768: 264 + 104 us), and the backward's
`dh = dZ Wsq` was cuBLAS + `swiglu_bwd` reading `dh` back (144 + 168 us). Both are one kernel now, the fp32 twin of
`gemm_swiglu2_sm100.cu`'s MMA side: 2-CTA clusters, M = 256, N = 256 `kind::tf32` products with B split by N across the pair (forward:
`Wa_j` / `Wb_j`; backward: `Wsq^T` rows, `Wsq^T` packed once per step), 32-fp32 K-blocks (16 KB per 128-row operand tile, 4 MMAs of
K = 8), accumulators double-buffered in TMEM. Epilogue: two warpgroups, each with an IO thread owning a ring of 16-column x 128-row
staging tiles (64-B swizzle, 8 KB per tensor). Forward: `h = silu(a) b`, `h`, `a`, `b` stored by TMA (the fp32 rows' sigmoid, same
formula). Backward: the IO thread loads `a`, `b` of tile u + NB as tile u's stores drain; the warps write `da = dh b s (1 + a (1 - s))`,
`db = dh a s` in place, the IO thread stores them; `dh` never reaches memory. Measured (L768, per launch): forward 329 us against
cuBLAS 254 + the SwiGLU pass 99 (its GEMM runs ~530 TF/s against cuBLAS's ~680; the whole step was not better), so the forward is
opt-in (`MINIWORLD_BIAS_ONLY_DIT_TF32_GLU=1`); backward 253 against 138 + 165, on by default (`MINIWORLD_BIAS_ONLY_DIT_TF32_GLU_BWD=0`:
cuBLAS + the rows; a failed build warns once and does the same).

**Attention core, batched N** (`pv_gate_tf32.cu -DBATCHN`, default; `MINIWORLD_BIAS_ONLY_DIT_PV_BATCH=0`: one MMA per sample): the
group's v chunks sit NA 4-KB atoms apart, so all SG samples are ONE MN-major B operand of N = SG x NA x 32 (LBO = 4 KB): one MMA per
K step reads the P chunk once instead of SG times (the per-sample loop issued SG small N = DH products, ~320 TF/s at L768). 48-wide
heads compute 16 unused columns per sample (the next head's channels), so SG <= 4 there (two accumulator sets in 512 columns).

**`res_adaln_b`** (forward, 4.7 TB/s where `adaln_a` runs 6.1): built for three blocks per SM (<= 80 registers) with its loads in two
waves (residual inputs, then the AdaLN tables before the row statistics): L768 141 -> 107 us, L384 73 -> 57
(`MINIWORLD_BIAS_ONLY_DIT_F32_RESB_MINB=2`: the two-block build; read per call). `res_adaln_b_bwd` loads `dout` in a second wave
before the row sums' barrier (all six streams at once spilled 24 B at 128 registers; a three-block build was slower, +99 us per L768
step). Weight gradients: `MINIWORLD_BIAS_ONLY_DIT_WGRAD_SPLIT=S` runs each of the six as a bmm over S row chunks plus a sum of the
partials (cuBLAS takes these long-K, small-output TF32 GEMMs at 360-500 TF/s, the data gradients at 600-670); off by default until
measured (`bo32_train_breakdown.py --wgrad-probe`).
`bench_scripts/bo32_train_breakdown.py --ab VAR=VAL ...` times the whole step against each switch in alternation in one process
(single whole-step runs differ by ~50 us at L768 on this power-capped card). The row extension reports every kernel's registers and local memory (`func_attrs`); the tests hold the new ones to
<= 128 registers without spills, as the cubins' (`Kernel.regs`, `.lmem`).

### fp32 training: speed of light per launch (16 x 48, A = 48)

Minimum time = max(FLOPs / 720 TF/s, unavoidable HBM bytes / 7 TB/s): the measured ceilings of this power-capped card (the best cuBLAS
TF32 GEMM of the step; a streaming copy). Bytes are every activation read or written once at 4 bytes (weights, row statistics and
partials neglected), not the launch arguments' sizes (L2 hits make those over-count). `bench_scripts/bo32_train_breakdown.py` prints
it for EVERY node of the step's CUDA graph (each kernel node timed alone from its own launch parameters; a launch's floor on its first
node, so split-K / batch reductions and copies show as pure gap), with the step's floor / whole step and the time between the node
sum and the whole step (`sol_model`, `node_times`). Measured: the round-5 breakdown (L384) and round 6 (L768), hot, us, before
the weight-gradient split (`_wgrad32`, S = 4: the six weight gradients 985 -> 809 us at L768) and `dpb32_sm100`; sorted by the
L768 gap.

| launch | L384 meas | L384 SoL | L384 gap | L768 meas | L768 SoL | L768 gap | SoL % (L768) | bound |
|---|---|---|---|---|---|---|---|---|
| dh + SwiGLU bwd (`gemm_glu_tf32` BWD) | 130.1 | 72.8 | 57.3 | 255.6 | 145.6 | 110.0 | 57% | HBM |
| `pv_gate` | 46.2 | 25.6 | 20.6 | 137.2 | 60.4 | 76.8 | 44% | MMA |
| mm dWn = dG^T chat | 106.7 | 60.4 | 46.3 | 190.6 | 120.8 | 69.8 | 63% | MMA |
| `pv_dv` | 43.3 | 17.5 | 25.8 | 126.5 | 60.4 | 66.1 | 48% | MMA |
| `pair_bias_bwd` | 47.3 | 22.9 | 24.4 | 154.5 | 91.7 | 62.8 | 59% | HBM |
| mm dWgg = dGg^T c | 62.5 | 30.2 | 32.3 | 122.6 | 60.4 | 62.2 | 49% | MMA |
| mm dWsq = dz^T h | 92.2 | 60.4 | 31.8 | 176.8 | 120.8 | 56.0 | 68% | MMA |
| mm dWvg = dvg^T xa | 107.3 | 60.4 | 46.9 | 173.5 | 120.8 | 52.7 | 70% | MMA |
| `dpb` (the removed `dpb_tf32`) | 34.8 | 18.9 | 15.9 | 112.5 | 60.4 | 52.1 | 54% | MMA |
| mm dchat = dG Wn | 73.7 | 60.4 | 13.3 | 172.4 | 120.8 | 51.6 | 70% | MMA |
| mm dWab = dab^T xt | 145.4 | 120.8 | 24.6 | 290.1 | 241.6 | 48.5 | 83% | MMA |
| mm dWo = dy^T og | 53.0 | 30.2 | 22.8 | 105.1 | 60.4 | 44.7 | 57% | MMA |
| `res_adaln_b_bwd` | 97.0 | 80.9 | 16.1 | 200.5 | 161.8 | 38.7 | 81% | HBM |
| mm ab = xt Wab^T | 119.1 | 120.8 | -1.7 | 276.8 | 241.6 | 35.2 | 87% | MMA |
| mm G = chat Wn^T | 72.4 | 60.4 | 12.0 | 156.0 | 120.8 | 35.2 | 77% | MMA |
| `pair_bias` | 19.2 | 12.1 | 7.1 | 73.4 | 48.5 | 24.9 | 66% | HBM |
| mm Gg = c Wg^T | 44.3 | 30.2 | 14.1 | 80.8 | 60.4 | 20.4 | 75% | MMA |
| `res_adaln_b` | 55.9 | 48.5 | 7.4 | 116.0 | 97.1 | 18.9 | 84% | HBM |
| mm dxt = dab Wab | 127.7 | 120.8 | 6.9 | 260.2 | 241.6 | 18.6 | 93% | MMA |
| mm dcg = dGg Wg | 40.6 | 30.2 | 10.4 | 77.3 | 60.4 | 16.9 | 78% | MMA |
| `gate_bwd` | 48.8 | 40.4 | 8.4 | 96.5 | 80.9 | 15.6 | 84% | HBM |
| `res_c_bwd` | 47.2 | 40.4 | 6.8 | 95.7 | 80.9 | 14.8 | 85% | HBM |
| mm y = og Wo^T | 35.1 | 30.2 | 4.9 | 73.0 | 60.4 | 12.6 | 83% | MMA |
| `swiglu` | 55.8 | 48.5 | 7.3 | 108.0 | 97.1 | 10.9 | 90% | HBM |
| `adaln_a_bwd` | 55.2 | 48.5 | 6.7 | 106.6 | 97.1 | 9.5 | 91% | HBM |
| `softmax_t` | 7.6 | 4.0 | 3.6 | 25.1 | 16.2 | 8.9 | 64% | HBM |
| `adaln_a` | 40.0 | 32.4 | 7.6 | 73.6 | 64.7 | 8.9 | 88% | HBM |
| mm dog = dy Wo | 37.9 | 30.2 | 7.7 | 68.1 | 60.4 | 7.7 | 89% | MMA |
| `res_c` | 56.4 | 48.5 | 7.9 | 103.8 | 97.1 | 6.7 | 94% | HBM |
| mm z = h Wsq^T | 62.8 | 60.4 | 2.4 | 127.0 | 120.8 | 6.2 | 95% | MMA |
| weight pack, `finalize`, `unfold` | 14.6 | 0 | 14.6 | 15.1 | 0 | 15.1 | -- | -- |
| `cond_bwd` | 17.2 | 16.2 | 1.0 | 37.6 | 32.4 | 5.2 | 86% | HBM |
| mm dxa = dvg Wvg | 61.6 | 60.4 | 1.2 | 120.8 | 120.8 | 0.0 | 100% | MMA |
| mm vg = xa Wvg^T | 62.4 | 60.4 | 2.0 | 120.6 | 120.8 | -0.2 | 100% | MMA |
| `cond_ln` | 6.2 | 8.1 | -1.9 | 15.9 | 16.2 | -0.3 | 102% | HBM |
| **sum** | 2128 | 1613 | 514 | 4446 | 3362 | 1084 | 76% | |

What the gaps say, and what was done about them (2026-10-06): the weight-gradient GEMMs (six rows, ~330 us of gap at L768) are
cuBLAS's single long-K launches -- split into batches of row chunks (`_wgrad32`), whole step -168 us (L768) / -82 (L384). The
dh + SwiGLU-backward kernel (110 us) waited for every tile's store to finish reading before refilling its ring -- one store now
stays in flight. Left, without a profile: the attention core (`pv_gate`, `pv_dv`: 44-48 %, MMA-bound by the model, ~320 TF/s), the
pair-bias backward (59 %), dchat / G / Gg (70-77 %, cuBLAS with N = 384-4608 over K = 384-3072).

`res_adaln_b_bwd` takes three warps per 768-row (8 columns per thread): the bf16 twin's two-warp layout holds its bf16 operands
packed until used; in fp32 the same layout needs 72 registers of row data per thread and spilled (24-64 bytes at 128).

**Not ported:** `gemm_resln_sm100` and `cond_tables_sm100` (opt-in, measured slower than the default composition in bf16; the
cond-table kernel's resident [128 x 384] A tile is 192 KB in fp32 and does not fit beside its B stages). The fp32 step runs the
default composition (`MINIWORLD_BIAS_ONLY_DIT_RESLN` / `_COND` are ignored for fp32).

**Accuracy contract** (the tests): output and every gradient against an fp64 reference within 1.5x the error of the PyTorch fp32
module with TF32 GEMMs (`allow_tf32 = True`) + 1e-3. A TF32 operand keeps 10 mantissa bits (unit roundoff 2^-11 ~ 4.9e-4):
every product of the block carries O(1e-3) relative error whoever computes it, so IEEE fp32 is not the bar; the floor is one
truncation step, 2^-10 (a `kind::tf32` MMA drops the operands' low 13 bits where cuBLAS may round to nearest; the token DiT's
fp32 tests take 1.5x + 3e-3). Kernel tests: the TF32 products (core, dV, dbias) within 3e-3 of fp64 einsum; the exact-fp32 kernels (softmax, pair bias and
its backward, gate backward) within 1e-5.

## Measurements (2026-09-30 / 2026-10-01)

B200 (148 SMs, 1000 W power cap), the v2.2.0 pixi env (torch 2.13.0+cu129), one GPU through `gpuq` with nothing else on it.
`benchmarks/runners/bench.py target=bias_only_dit level=module` (one block, bf16, compiled; inference S = 5 with a CUDA graph,
training A = 48 with CUDA graph off and on, mask_prob 0), ms per block, median. Runs differ by up to 5 % on this card (power
cap). × = ours vs PyTorch compiled on the same layout; there is no cuEquivariance or Anthropic implementation of this op.

Accuracy, relative Frobenius error against the fp32 IEEE PyTorch block on the same weights and inputs: inference 3.3-3.5e-3
against PyTorch compiled bf16's 4.1e-3 (the bench's columns); training (`tests/integrations/test_b200_bias_only_dit_train_gpu.py`)
the output 3.6e-3 against 4.3e-3 and every input and parameter gradient at 0.96-0.99x PyTorch bf16's error, for all four layouts.

### 16 heads x 48 · Inference (S = 5)

The attention weights are made once per pair (as a sampler does over its steps; the timed calls reuse one pair), so the rows do
not include `P`; PyTorch recomputes its softmax every call. 2026-09-30.

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16), per-sample cond | 0.078 | — | — | **0.051** | 1.52 |
| (256, 768, bf16), per-sample cond | 0.098 | — | — | **0.063** | 1.55 |
| (384, 768, bf16), per-sample cond | 0.119 | — | — | **0.072** | 1.66 |
| (512, 768, bf16), per-sample cond | 0.154 | — | — | **0.086** | 1.79 |
| (640, 768, bf16), per-sample cond | 0.192 | — | — | **0.096** | 2.00 |
| (768, 768, bf16), per-sample cond | 0.227 | — | — | **0.102** | 2.22 |
| (128, 768, bf16), shared cond | 0.078 | — | — | **0.049** | 1.58 |
| (256, 768, bf16), shared cond | 0.100 | — | — | **0.057** | 1.75 |
| (384, 768, bf16), shared cond | 0.120 | — | — | **0.065** | 1.83 |
| (512, 768, bf16), shared cond | 0.154 | — | — | **0.076** | 2.03 |
| (640, 768, bf16), shared cond | 0.194 | — | — | **0.086** | 2.26 |
| (768, 768, bf16), shared cond | 0.229 | — | — | **0.090** | 2.55 |

### 16 heads x 48 · Training (A = 48)

2026-10-01.

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (384, 768, bf16), CUDA graph | 1.247 | — | — | **0.985** | 1.27 |
| (768, 768, bf16), CUDA graph | 2.586 | — | — | **2.057** | 1.26 |
| (384, 768, bf16), no graph | 1.425 | — | — | **1.118** | 1.27 |
| (768, 768, bf16), no graph | 2.623 | — | — | **2.183** | 1.20 |

### 24 heads x 32 · Inference (S = 5) and Training (A = 48)

2026-10-01.

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16), inference, per-sample / shared cond | 0.078 / 0.078 | — | — | **0.053 / 0.051** | 1.46 / 1.52 |
| (384, 768, bf16), inference, per-sample / shared cond | 0.127 / 0.127 | — | — | **0.072 / 0.065** | 1.77 / 1.94 |
| (768, 768, bf16), inference, per-sample / shared cond | 0.238 / 0.240 | — | — | **0.104 / 0.092** | 2.28 / 2.60 |
| (384, 768, bf16), training, CUDA graph | 1.305 | — | — | **1.086** | 1.20 |
| (768, 768, bf16), training, CUDA graph | 2.528 | — | — | **2.198** | 1.15 |
| (384, 768, bf16), training, no graph | 1.451 | — | — | **1.138** | 1.27 |
| (768, 768, bf16), training, no graph | 2.770 | — | — | **2.299** | 1.20 |

### 12 heads x 64 · Inference (S = 5) and Training (A = 48)

2026-10-01.

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16), inference, per-sample / shared cond | 0.080 / 0.080 | — | — | **0.053 / 0.051** | 1.50 / 1.56 |
| (384, 768, bf16), inference, per-sample / shared cond | 0.119 / 0.121 | — | — | **0.072 / 0.066** | 1.66 / 1.84 |
| (768, 768, bf16), inference, per-sample / shared cond | 0.230 / 0.231 | — | — | **0.100 / 0.088** | 2.30 / 2.63 |
| (384, 768, bf16), training, CUDA graph | 1.308 | — | — | **1.075** | 1.22 |
| (768, 768, bf16), training, CUDA graph | 2.628 | — | — | **2.080** | 1.26 |
| (384, 768, bf16), training, no graph | 1.439 | — | — | **1.121** | 1.28 |
| (768, 768, bf16), training, no graph | 2.756 | — | — | **2.242** | 1.23 |

### 16 heads x 64 · Inference (S = 5) and Training (A = 48)

1024 attention channels: the PyTorch block also does a third more attention work. 2026-10-01.

| (Length, Dimension, dtype) | PyTorch compiled | cuEquivariance | Anthropic | ours | × |
|---|---|---|---|---|---|
| (128, 768, bf16), inference, per-sample / shared cond | 0.080 / 0.080 | — | — | **0.053 / 0.053** | 1.50 / 1.50 |
| (384, 768, bf16), inference, per-sample / shared cond | 0.123 / 0.123 | — | — | **0.074 / 0.067** | 1.67 / 1.82 |
| (768, 768, bf16), inference, per-sample / shared cond | 0.232 / 0.233 | — | — | **0.108 / 0.098** | 2.14 / 2.38 |
| (384, 768, bf16), training, CUDA graph | 1.357 | — | — | **1.115** | 1.22 |
| (768, 768, bf16), training, CUDA graph | 2.617 | — | — | **2.175** | 1.20 |
| (384, 768, bf16), training, no graph | 1.478 | — | — | **1.138** | 1.30 |
| (768, 768, bf16), training, no graph | 2.822 | — | — | **2.174** | 1.30 |

### Training step: where the time goes and speed of light

Kernel durations (probe `sol_bo.py`: torch.profiler over eager steps, gradients reset each step), us; the floor row sums every
kernel's floor (the larger of its minimum HBM bytes / 6.31 TB/s, its FLOPs / 2.23 PF/s and their energy at the cap):

| | 16 x 48 L384 / L768 | 24 x 32 | 12 x 64 | 16 x 64 |
|---|---|---|---|---|
| cuBLAS GEMMs (forward 5, data gradients 6, weight gradients 6, split-K reduces) | 503 / 906 | 503 / 905 | 505 / 905 | 545 / 976 |
| expand GEMM + SwiGLU (T5) | 79 / 147 | 79 / 147 | 79 / 147 | 79 / 147 |
| row kernels (T4 without `finalize`, `gate_bwd_rows`) | 284 / 554 | 284 / 553 | 286 / 555 | 293 / 572 |
| attention: core twice (I1), `dpb` (T3), `softmax_t` (T2) | 68 / 183 | 80 / 200 | 65 / 156 | 76 / 196 |
| pair bias forward + backward (T1) | 37 / 122 | 40 / 135 | 37 / 121 | 37 / 124 |
| `finalize` | 8 / 8 | 10 / 10 | 7 / 7 | 8 / 8 |
| kernel sum | 979 / 1920 | 996 / 1950 | 978 / 1891 | 1037 / 2021 |
| floor (energy) | 851 / 1789 | 854 / 1802 | 849 / 1783 | 915 / 1932 |

SoL = the sum of the floors / the measured step. This card runs at a **1000 W cap**: sustained work sits at ~985 W and the clocks
drop to fit, so cuBLAS sustains 1.21-1.23 PF/s on the step's own GEMM shapes, not the 2.23 PF/s of the MMA alone, and the GEMMs
are half the step. Three ceilings, L384 / L768:

| ceiling | 16 x 48 | 24 x 32 | 12 x 64 | 16 x 64 |
|---|---|---|---|---|
| nominal roofline: 6.31 TB/s copy, 2.23 PF/s MMA (burst kernel durations, `sol_bo.py`) | 69 / 73 % | 68 / 72 % | 69 / 74 % | 70 / 74 % |
| + energy: (read 115 / write 72 pJ/B, 0.47 pJ/FLOP) / 750 W dynamic (same durations) | 87 / 93 % | 86 / 92 % | 87 / 94 % | 88 / 96 % |
| power-capped, sustained: a CUDA graph of steps replayed 3 s, ceilings measured the same way (6.77 TB/s, cuBLAS 1.21-1.23 PF/s; `t_sustain.py`) | 94.9 / 98.5 % | 94.0 / 97.0 % | 96.4 / 98.9 % | 95.7 / 99.8 % |

On the nominal roofline 90 % is out of reach on this card: with every non-GEMM kernel at its floor the step would be at ~78 %
(L384) / ~83 % (L768), the rest is cuBLAS at the power cap. In the sustained regime the step is within 1-6 % of the ceiling. What keeps L384 below 90 %
of the energy floor: the cuBLAS GEMMs (87 %), the pair-bias kernels (T1), `softmax_t` and `dpb` (△ above).

### Inference step: where the time goes (16 x 48, per-sample conditioning, probe)

| kernel | L384 | L768 |
|---|---|---|
| cuBLAS v\|g, out and squeeze GEMMs (and expand at L384) | 27.7 us (4) | 24.6 us (3) |
| expand GEMM + SwiGLU `gemm_swiglu2_sm100` (I3) | — (cuBLAS above + row pass) | 19.3 us |
| cuBLAS conditioning tables (two GEMMs over S L rows) | 12.5 us | 18.8 us |
| `pv_gate_inf` (I1) | 8.0 us | 16.8 us |
| row kernels (I2) | 21.3 us (5) | 24.1 us (4) |
| sum | 69.6 us | 103.7 us |

At these sizes (M = S L = 640-3840 rows) every step is short, so the block is a chain of ~12 kernels that each leave part of the
GPU idle; the GEMMs run at 0.5-1.2 PFLOP/s and the row kernels near the memory rate.

## What was tried and not kept

Training:

| attempt | result |
|---|---|
| 768-wide training rows one warp per row (24 columns per lane) | ~200 registers in the backward: spills at 2 blocks/SM, latency-bound at 1 (res_adaln_b_bwd 131 us; two warps per row 57) |
| L2 bulk prefetch (`cp.async.bulk.prefetch.L2`) of the rows a warp takes next | slower at every distance (res_adaln_b_bwd 67 -> 71-84 us) |
| bias column sums by torch over the partial rows / serially by 24 blocks | 45 / 32 us; the partial-sum kernel (now `finalize`): a few us |
| pair LN + cuBLAS bias GEMM forward; pair backward on CUDA cores + cuBLAS dWf GEMM | 16 + 8 / 38 + 16 us at L384 (51 + 31 / 129 + 38 at L768); T1: 13 / 23 (39 / 82) |
| T1 backward: loads double-buffered by cp.async; fragments by `ldmatrix` | 84.9 -> 83.7 -> 81.8 us at L768 (it is issue-bound); the LN-backward algebra 74 |
| transpose of `P` as its own kernel | 5.4 / 14.4 us (L384 / L768) on top of the softmax; `softmax_t` writing both: 9.2 -> 7.8, 21.9 -> 16.3 |
| keeping x1 in fp32 | 4 B per element more traffic; rebuilt with one FMA instead (after moving res_adaln_b_bwd's column sums to shared memory: spills otherwise) |
| one exchange for the LayerNorm statistics alone (Chan merge) | no change for adaln_a (25.7 -> 25.1 us); packed loads and four blocks per SM: 23 |
| `pv_gate_inf` sample groups other than the cost model's pick at A = 48 | the pick (SG 16) was the fastest at L128-768, both 16 x 48 and 24 x 32 (24 x 32: within 4 %) |
| 24 x 32: one head per `dpb` item; heads padded to 32 in T1's backward | `dpb` 79 us at L768 (head pairs 57); T1 backward +20 % (k8 step and swapped dWf: +12 %) |
| torch glue for the small gradients (sum, mul, casts, zero fills) and autograd copying fp32 LayerNorm-weight gradients from a bf16 buffer | 21 us per step; `finalize` writes every small gradient in its parameter's dtype |

Inference:

| attempt | result |
|---|---|
| core: one work item per (head, query tile, sample), P tile in shared memory | 8.0 / 17.6 us (L384 / L768): the P tile is read once per sample |
| core: SG samples per item, P in shared memory, samples inner | 6.4 / 13.3 us: the epilogue only starts once the item is loaded and was ~40 % of the kernel |
| core: P in tensor memory in 32-key SW64 chunks, 64-key v tiles | 7.4 / 17.5 us: the ring runs at a few hundred clocks per TMA instruction whatever its size; 64-key / 128-key chunks (kept) 6.1 / 13.1 |
| core epilogue: scalar kit sigmoid (two MUFU ops) / scalar Newton reciprocal | MUFU-bound / issue-bound (one warp per SMSP); packed f32x2 with the reciprocal on the FMA pipe kept |
| core epilogue: shared loads, math and stores interleaved per 16-byte chunk | the volatile asm serialises six load latencies (~1500 clk per tile); all loads first |
| block-per-row row kernels (the token DiT's) | 5-6.5 us per pass at 1920 rows; one warp per row 4-5 us (L768 resgate 12.2 -> 8.9 us) |
| out / squeeze GEMM with residual + gate + AdaLN in its epilogue (`gemm_resln_sm100.cu`, 8-CTA clusters); conditioning LN + tables + sigmoid in one GEMM (`cond_tables_sm100.cu`) | correct, kept off: L384 69.7 -> 73.8 us, L768 102 -> 122 -- the epilogues sit on 128-256 threads per SM and are latency- / issue-bound where the row kernels run on every thread at the memory rate |
| the samples split over two / three CUDA streams | L384 70 -> 80 / 83 us: the core and cuBLAS's kernels each fill an SM's shared memory, the chains do not overlap |
| the output-gate table GEMM on a side stream | no change |

## Limits and next

- fp32 (TF32) inference and training: written (see [fp32 path](#fp32-path-tf32)); not yet built, tested or measured on B200.
- Training: half the step is cuBLAS at the power cap; the row kernels and the SwiGLU GEMM are at their floor (✓). Below it (△):
  the pair-bias kernels (T1, issue-bound), `softmax_t` (T2), `dpb` (T3, TMA intake), `finalize` (~8 us of reduction latency)
  and, with 32-wide heads, the core's N = 32 products (24 x 32 L768 at 80 % of its floor; two samples per N = 64 product would
  close it).
- Inference (S = 5, 640-3840 rows): every kernel is short and below 90 % of its floor (△): pv 11-57 %, the rows 27-100 % rising
  with L, the GEMMs 28-83 %.
- Inference: the remaining time is the chain of short kernels. What is left to try: programmatic dependent launch across the
  chain, which needs GEMMs of our own (cuBLAS launches cannot take part).
