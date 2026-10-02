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

## Not covered

B > 1, QK-norm, fp32 / fp16, other GPUs. The cross-attention mode of the AF3 / Protenix atom blocks (`cross_attention=True`: keys and
values projected from a second AdaLN `attention.ada_ln_kv` of the already normalised atoms) is served, but its extra row work is plain
CUDA row kernels plus cuBLAS GEMMs (`lcross.cu`), not yet fused into the tcgen05 row kernels, so it is slower than the shared-AdaLN block.
