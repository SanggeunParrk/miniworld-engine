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

- The attention is hand CUDA with `mma.sync` (a 32-row query window is too small for tcgen05's 128-row tiles); the row kernels are
  tcgen05 / TMA; the weight-gradient GEMMs are cuBLAS. No Triton, no quack, no N x N tensor anywhere.
- `lattn_fwd.cu`: CTA = (window, head, sample chunk), 4 warps = 2 samples x 2 query tiles of 16 rows; the 128 keys of a window fit one
  softmax (no online rescaling); the window's bias stays in shared memory for the CTA's samples, q / K / V of the next sample pair are
  prefetched with `cp.async`. `lattn_dq.cu`: the same structure, dQ is window-local. `lattn_dkv.cu`: key-centric -- 16 keys are seen by exactly
  four query windows (128 consecutive queries), so a warp owns 16 keys, computes dK / dV for them and sums dS over the samples into dbias
  (every dbias element belongs to exactly one key: no atomics); a CTA of four such warps shares the 192 queries their windows span.
  `lbias.cu`: the pair bias and its backward (dz, and block partials of the gamma / Wb gradients).
- Dtypes: activations, weights, gradients bf16; pair bias, dbias, LSE and D fp32; every MMA accumulates in fp32.
- Keys outside `[0, N)` and keys with `mask == 0` get -inf; a query whose window has no valid key outputs 0 (and has zero gradient).

## Measurements

One B200 (gpu2 of the box, shared), bf16, `torch.set_float32_matmul_precision("medium")`, CUDA-graph replay, ms. "PyTorch" is the
module's reference path (`implementation=PYTORCH`, the unfused eager ops replayed as a graph, window gather by `unfold`), not a
`torch.compile`d block; no cuEquivariance or Anthropic row exists for this op, and no speed-of-light bound was measured, so no 성능 확인
mark is claimed.

| block | PyTorch | engine | x |
|---|---|---|---|
| training A48 N4096 (forward + backward) | 14.170 | 1.904 | 7.4 |
| training A48 N8192 | 27.008 | 3.510 | 7.7 |
| inference A48 N4096 | 5.686 | 0.530 | 10.7 |
| inference A5 N4096 | 1.199 | 0.120 | 10.0 |
| inference A5 N2048 | 0.649 | 0.082 | 7.9 |
| inference A1 N4096 | 0.814 | 0.072 | 11.3 |

The attention kernels alone (events over 20 launches, no graph):

| (A, N) | bias fwd | attention fwd | attention bwd (dq + dkv) | bias bwd |
|---|---|---|---|---|
| (48, 4096) | 0.029 | 0.182 | 0.418 | 0.066 |
| (48, 8192) | 0.028 | 0.352 | 0.820 | 0.090 |
| (5, 4096) | 0.028 | 0.028 | 0.064 | 0.066 |
| (1, 2048) | 0.029 | 0.015 | 0.031 | 0.060 |

Accuracy against an fp64 PyTorch block (`tests/integrations/test_b200_local_dit_gpu.py`; relative error of the output / the worst
gradient, the module path's bf16 error beside it): training A3 N300 output 4.42e-3 vs 4.22e-3, worst gradient 1.29e-2 vs 1.22e-2;
A3 N384 with a key mask 4.44e-3 vs 4.24e-3, 1.58e-2 vs 1.52e-2; A2 N1024 4.41e-3 vs 4.21e-3, 1.45e-2 vs 1.41e-2; inference 4.4e-3 vs
3.9e-3 at the same shapes. The window definition itself is checked against a per-atom loop on CPU
(`tests/numerics/test_local_dit_reference.py`), and the kernels against the PyTorch path under `torch.compile`, in a CUDA graph, with
and without a key mask, at N not a multiple of 32 or 128.

## Not covered

B > 1, QK-norm, fp32 / fp16, other GPUs. The cross-attention mode of the AF3 / Protenix atom blocks (`cross_attention=True`: keys and
values projected from a second AdaLN `attention.ada_ln_kv` of the already normalised atoms) is served, but its extra row work is plain
CUDA row kernels plus cuBLAS GEMMs (`lcross.cu`), not yet fused into the tcgen05 row kernels, so it is slower than the shared-AdaLN block.
