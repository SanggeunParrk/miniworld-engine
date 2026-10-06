# Local atom DiT (AF3 block-local attention) on B200 (sm100)

Kernel-level status of the AF3 atom transformer block with **block-local attention** (engine `modules/local_dit.LocalDiTBlock`: AF3
Alg. 23 with the 32 x 128 trunking of Alg. 24): AdaLN -> q | k | v | gate -> attention in which atom `i` of query window
`w = i // 32` sees only the 128 atoms `[32 w - 48, 32 w + 80)` (clipped to `[0, N)`), with the per-window pair bias `LN(z) Wb^T` taken from
the *trunked* atom pair `[nwin, 32, 128, 16]` (shared by the A samples) -> gated out-projection + residual -> AdaLN + SwiGLU transition
(hidden 256) + residual; d_single = d_cond = 128, d_pair = 16, 4 heads x 32. This is the attention of the AF3 atom transformer
(n_queries 32, n_keys 128; see Not covered for the one difference of its blocks); the dense
block of [atom DiT](../atom_dit/atom_dit.md) attends over all N atoms instead and is a different op. The module-level summary is in
[b200.md](../b200.md).

**Where the code is.** `modules/local_dit/module.py` (the block, the PyTorch reference path, `to_windows` that makes the trunked pair
from a dense one), `integrations/local_dit.py` (the dispatch: two opaque ops `local_dit_block_fwd` / `local_dit_block_bwd` under an
autograd Function, served eagerly, under `torch.compile` and in CUDA graphs), and the attention and pair-bias kernels in
`kernels/augmented_attention/cuda/sm100_atom_local/`. The row kernels (conditioning, input projections, post-attention, transition and
their backward) are [atom DiT](../atom_dit/atom_dit.md)'s `sm100_atom`: they do not depend on what the attention attends to. The block
has the parameters of `DiTBlock` at atom widths, so both blocks share a checkpoint layout. `serves()` accepts: B200, implementation
MINIWORLD / TRITON, bf16 single / cond / pair, the atom widths, B = 1, any N, a `[B, N]` bool key mask or none, no QK-norm; anything else
keeps the PyTorch path, and `MINIWORLD_LOCAL_DIT_SM100=0` turns it off. N is padded to a multiple of 128 for the row kernels (zero rows,
the padded atoms masked as keys, the pair padded with zero windows); the first N rows come back.

- The attention and pair-bias kernels are hand CUDA, the attention on tcgen05 / TMEM / TMA; the row kernels are tcgen05 / TMA; the
  weight-gradient GEMMs are cuBLAS. No Triton, no quack, no N x N tensor anywhere.
- Attention geometry (all three kernels): a CTA owns 128 rows of one head and a range of samples; TMEM lane = row. 128 queries (four
  windows) see 224 keys, and 128 keys see 224 queries when the key chunk starts at `128 j - 112` (every TMEM lane quadrant's 32 keys then
  see the same four windows). Per sample: `S = Q K^T` and `dP = dO V^T` [128 x 224] are tcgen05 MMAs into TMEM; 16 warps (lane
  quadrant x band quarter) read their 32 x 32 band cells, the bias comes from shared memory (log2 domain, key mask folded in as -inf),
  and the softmax / gradient values are written back as packed bf16 over S -- the TMEM A operand of the next product (`umma_ts`).
  Outputs leave through shared memory with per-warp TMA stores (32 rows x 16 columns). TMA loads run three samples ahead.
- `lattn_fwd.cu`: query-centric; S in two TMEM buffers so the next sample's QK^T runs during this softmax; the quadrant's four warps
  exchange row maxima and sums through shared memory; O = P V; a 17th warp issues the TMA loads and the MMAs.
- `lattn_dq.cu`: query-centric; dS = P (dP - D) over S, dQ = dS K; a 17th warp issues the TMA loads and the MMAs.
- `lattn_dkv.cu`: key-centric; P^T and dS^T over S^T, dV = P^T dO and dK = dS^T Q; dbias is summed over the samples in registers (every
  dbias element belongs to one key, hence one thread; two sample halves add exactly, `0 + a + b`). TMEM is full (S^T 224 + dP^T 224 +
  dK / dV 64 columns), so dQ cannot be fused here (it needs dS with queries as rows plus 64 more columns).
- `lbias.cu`: the pair bias and its backward (dz, block partials of the gamma / Wb gradients reduced by a one-block kernel). The
  wrappers cache the TMA descriptors by address and geometry (encoding one costs ~5-10 us of host time).
- Dtypes: activations, weights, gradients bf16; pair bias, dbias, LSE and D fp32; every MMA accumulates in fp32.
- Keys outside `[0, N)` and keys with `mask == 0` get -inf; a query whose window has no valid key outputs 0 (and has zero gradient).

## Measurements

One B200 (shared box), bf16, `torch.set_float32_matmul_precision("medium")`, ms. "PyTorch" is the module's reference path
(`implementation=PYTORCH`, the unfused eager ops replayed as a graph, window gather by `unfold`), not a `torch.compile`d block; no
cuEquivariance row exists for this op. "previous" is the first version of these kernels (`mma.sync` attention, ce7fd93b).

Block, CUDA-graph replay:

| block | PyTorch | previous | engine | x PyTorch |
|---|---|---|---|---|
| training A48 N4096 (forward + backward) | 14.184 | 1.904 | 1.549 | 9.2 |
| training A48 N8192 | 27.029 | 3.510 | 2.838 | 9.5 |
| inference A48 N4096 | 5.693 | 0.530 | 0.427 | 13.3 |
| inference A5 N4096 | 1.198 | 0.120 | 0.105 | 11.4 |
| inference A5 N2048 | 0.649 | 0.082 | 0.075 | 8.7 |
| inference A1 N4096 | 0.814 | 0.072 | 0.064 | 12.7 |

The attention kernels alone (events over 20 eager launches, no graph; the small shapes are bounded by host launch time):

| (A, N) | bias fwd | attention fwd | attention bwd (dq + dkv) | bias bwd |
|---|---|---|---|---|
| (48, 4096) | 0.014 | 0.076 | 0.193 | 0.027 |
| (48, 8192) | 0.013 | 0.147 | 0.378 | 0.046 |
| (5, 4096) | 0.014 | 0.017 | 0.038 | 0.027 |
| (1, 2048) | 0.013 | 0.017 | 0.037 | 0.027 |

Speed of light (the atom DiT method: a CUDA graph of back-to-back calls replayed for >= 2 s, wall-clock time and NVML energy; floor =
max(bytes / HBM, FLOPs / 2.38 PF, exps / 4.65 T/s, dynamic energy / (P_max - P_idle)) with compulsory bytes; SoL = floor / time). The
large shapes run at the 1000 W cap, so further speed has to come from less energy, not more overlap.

| (A, N) | attention fwd | attention bwd | bias fwd | bias bwd |
|---|---|---|---|---|
| (48, 4096) | 72.7 us, 54 % | 190.6 us, 61 % | 6.5 us, 58 % | 18.3 us, 34 % |
| (48, 8192) | 144.2 us, 55 % | 379.8 us, 61 % | 10.5 us, 71 % | 32.2 us, 38 % |
| (5, 4096) | 12.3 us, 43 % | 31.0 us, 49 % | 6.5 us, 57 % | 18.3 us, 34 % |
| (1, 4096) | 6.5 us, 32 % | 16.4 us, 35 % | 6.5 us, 57 % | 18.3 us, 34 % |

Against Anthropic's window kernel (`opt_core/kernels/apb/fpf_apb/atom_triton.py` `atom_apb`: QK^T + pair bias + softmax + PV of the
same 32 x 128 windows, forward only, no LSE output), attention forward, graph replay, us (it has no backward, so training has no
Anthropic row):

| (A, N) | engine | Anthropic bf16 | Anthropic fp32 (its default) | x Anthropic bf16 |
|---|---|---|---|---|
| (48, 4096) | 72.3 | 187.7 | 482.7 | 2.60 |
| (48, 8192) | 143.8 | 370.2 | 959.2 | 2.57 |
| (5, 4096) | 12.4 | 20.7 | 48.6 | 1.67 |
| (5, 2048) | 9.5 | 12.1 | 25.3 | 1.28 |
| (1, 4096) | 6.5 | 6.1 | 12.1 | 0.94 |
| (1, 2048) | 6.3 | 4.0 | 6.7 | 0.63 |

At A = 1 the per-CTA fixed cost (TMEM allocation, the bias load, descriptor fetch, ~5 us) dominates; Anthropic's kernel is faster there.

Accuracy: `tests/integrations/test_b200_local_dit_gpu.py` checks the output and every gradient against an fp64 PyTorch block (no worse
than the bf16 module path), with and without a key mask, at N not a multiple of 32 or 128, eagerly, under `torch.compile` and in a CUDA
graph; the window definition is checked against a per-atom loop on CPU (`tests/numerics/test_local_dit_reference.py`). dQ, dK, dV and
dbias are bit-identical to the previous `mma.sync` kernels; the forward's row sum is now taken in fp32 (it was the sum of the bf16 P),
which moves O by 1.2e-3 relative (bf16 rounding level; relative error against fp64 1.77e-3 as before).

## Per-item hoist (inference)

The conditioning tables (AdaLN modulation; in cross mode the conditioning LayerNorm and the K/V modulation GEMM) depend only on the
conditioning tensor and the weights, and the windowed pair bias only on the pair tensor and its two weights. Across the steps of a
sampler they do not change, so an inference call (no gradients saved) computes them once per (tensor, in-place version, weights) and
later calls read the cache; Anthropic's `FastAtomStack` does the same through its `dit_hoist`. A cache entry holds a weak reference to its
tensor, so a freed-and-reused address never serves stale tables. With `kernels._capture.static_inputs()` (or
`MINIWORLD_STATIC_CONDITIONING=1`) a CUDA-graph capture serves from the eager entries and a replay launches none of these kernels: one
inference call goes from 11 to 6 kernels. Off by default; training never uses it. Tests: `tests/integrations/test_b200_local_dit_gpu.py`
(`-k hoist or static_inputs`).

## fp32 path (TF32 tensor cores)

fp32 single / cond / pair with fp32 parameters (no autocast) are served on B200 by a separate set of kernels: every product on tcgen05
`kind::tf32` (fp32 operands read at tf32 precision, fp32 accumulation), everything else fp32 -- the residual stream, LayerNorms / AdaLNs,
gates, the softmax and P, every saved activation and every gradient. The weight gradients are cuBLAS GEMMs with TF32 forced on (and the
caller's `allow_tf32` restored), the only non-hand-written products. No Triton, none of the bf16 kernels. Dispatch: `integrations/local_dit.py`
(`_fp32_call`, `_LocalBlock32` / `_infer32`, opaque ops `local_dit_block_fwd_tf32` / `local_dit_block_bwd_tf32`); the kernels load
separately from the bf16 ones (a build / load failure warns once and keeps the module path for fp32 calls only);
`MINIWORLD_LOCAL_DIT_TF32=0` keeps fp32 calls on the module path. A bf16 model fed fp32 activations, or a call under autocast, keeps the
module path. The per-item hoist applies (fp32 tables under their own keys).

Kernels (`kernels/augmented_attention/cuda/sm100_atom_local/`, `.../sm100_atom/`; budgets in each file's header):

| kernel (build entry) | design | smem / TMEM / threads |
|---|---|---|
| `lattn_fwd_tf32.cu` `local_attn_fwd_tf32` (`KERNELS_TF32["fwd32"]`) | the bf16 geometry (128 queries x 224 keys, TMEM lane = query); q / k K-major SW128, v MN-major in the 32-B-atom swizzle; P fp32 written in place over S (each warp its own 32 band cells + one zeroed out-of-band block); bias in registers; 2 stages | 168.3 KB / 480 cols / 17 warps |
| `lattn_dq_tf32.cu` `local_attn_dq_tf32` (`"dq32"`) | S, dP into TMEM, dS fp32 in place over S, dQ = dS K with K read a second time MN-major; q, dO, k, v, LSE, D in 2 stages, the MN-major k in one slot (S(it+1) precedes dQ(it+1)) | 222.3 KB / 480 cols / 17 warps |
| `lattn_dkv_tf32.cu` `local_attn_dkv_tf32` (`"dkv32"`) | key-centric (keys `128 j - 112`); P^T over S^T and dS^T over dP^T in place; q and dO read K-major (one slot: TMEM holds one S^T anyway) and MN-major (two slots, each doubling as its sample's dK / dV staging); dbias in registers, summed over the samples | 204.3 KB / 512 cols / 17 warps |
| `lbias_tf32.cu` `local_bias_{fwd,bwd,fin}_f32` (`"pb_f32"`, `"pb_b32"`, `"pb_fin32"`) | the pair bias and its backward on an fp32 pair (CUDA cores, one row per thread) | static |
| `gemm_tf32.cu` `atom_gemm_tf32` (`sm100_atom.KERNELS_TF32["gemm"]`, `gemm32`) | every projection and activation-gradient product: persistent 128 x NT (128 / 256) tiles, 3-stage TMA ring of 32-wide K slices, double-buffered TMEM accumulators; epilogues: bias + sigmoid on chosen 128-column blocks (conditioning tables), gated residual (`a1 = s + so (gated Wo^T)`, `out = a1 + st (h Ws^T)`, + the raw product for the backward), SwiGLU over an interleaved `[Wa_0; Wb_0; Wa_1; Wb_1]` pack; per-warp 4 KB TMA-store stagings; 4 warps round each A slice to tf32 in shared memory before the MMA | 208.3 KB / 512 cols / 16 warps, <= 128 registers |
| `rows_tf32.cu` `f32_*` (`"ln"`, `"adaln"`, `"gate"`, `"tail_b"`, `"swiglu_b"`, `"adaln_b"`, `"gate_b"`, `"ln_b"`) | the row-wise stages: conditioning LayerNorms, AdaLN and its backward (with the output gate fused), gates, SwiGLU backward, D = rowsum(dO O) per head, the conditioning LayerNorm backward with block-reduced v4 `red.add` | warp per row |

Operand rounding. A `kind::tf32` MMA reads an fp32 operand by dropping its low 13 mantissa bits; that truncation is a one-sided error
that accumulates coherently. Without rounding, the block output came out 1.6e-3 relative against fp64, 5.6x the cuBLAS TF32 module path
(2.8e-4), independent of shape. So every MMA operand is rounded to nearest (`cvt.rna.tf32.f32`) before the MMA reads it: the weight packs
on the host (`sm100_atom.round_tf32`), the GEMMs' A slices in shared memory, q / k / v in the projection's epilogue, dO in `f32_gate_bwd`,
and P / dS before they are written to TMEM. The stored activations stay exact fp32 (except q / k / v and dO, which only feed MMAs), as do
the row sums and dbias.

What the fp32 operands change in the attention kernels: a 32-wide fp32 head row is 128 B (one 128-B swizzle atom, four K = 8 MMA steps);
a tf32 MMA takes an MN-major operand only in the 128-B swizzle with 32-B atoms, so k (dQ), q and dO (dK / dV) are loaded a second time in
that layout; P / dS are as wide as S in fp32, so they go over S in place instead of packed; the bias (66 KB as a shared copy) lives in
registers (in `lattn_dkv_tf32` it is read per sample from L1 / L2 instead: 32 more registers spilled there). The fp32 tensors are about twice the bytes, so the stages drop from 3 to 2 (fwd, dq) or split by lifetime (dkv).

Layouts of the path: the conditioning table `[M, 768] = [s1 | bi1 | s2 | bi2 | so | st]` (three GEMMs from LN_g1(c), LN_g2(c), c; scales and
gates stored as sigmoids), the projection `[M, 512] = q | k | v | g` (cross mode: `q | g` from x1 and `k | v` from the K / V AdaLN, two
GEMMs), the transition's `[a | b]` interleaved per 128-column half. Forward: LN x2 + 3 GEMMs (hoisted in inference), AdaLN, 1 GEMM, pair
bias (hoisted), attention, gate, 3 GEMMs with 1 AdaLN between: an inference call with the hoist is 8 launches (cross: 10). Backward: 8 row
kernels, 7 GEMMs (cross: 10), dq + dkv, pair-bias backward, the cuBLAS weight gradients.

Tests: `tests/integrations/test_b200_local_dit_tf32_gpu.py` (output and every gradient against an fp64 block, no worse than the fp32
module path with TF32 on -- `ef < 1.5 em + 1e-3` -- and loosely bounded by the IEEE module path; cross / non-cross, masks, N not a multiple
of 32 / 128; eager, `torch.compile`, CUDA graph for inference and for a whole training step; the TF32 kernels ran and no Triton / bf16
kernel; no kernel spills (`lmem == 0`, <= 128 registers); the fp32 hoist). Measurements: not yet taken.

## Not covered

B > 1, QK-norm, fp16, other GPUs; fp32 activations with bf16 parameters or under autocast (module path). The cross-attention mode of the AF3 / Protenix atom blocks (`cross_attention=True`: keys and
values projected from a second AdaLN `attention.ada_ln_kv` of the already normalised atoms) is served, but its extra row work is plain
CUDA row kernels plus cuBLAS GEMMs (`lcross.cu`), not yet fused into the tcgen05 row kernels, so it is slower than the shared-AdaLN block.
