# Smooth lDDT on B200 (sm100)

Kernel-level status of `kernels/smooth_lddt`; the module-level summary is in [b200.md](../b200.md). The op is **not in the completion tables
yet**: it has no row in the shape registry and no bench target, so there is no **성능 확인** to give (the maintainer's call). **cache build ✓**:
nothing on this path autotunes (fixed launch shapes, a cubin built on first use into `MINIWORLD_ENGINE_JIT_ROOT`).

The op is the auxiliary structure loss of AlphaFold 3's diffusion module (SI Algorithm 27), the term MiniWorld adds inside `alpha_diffusion`:

    lddt_ab = sum_{i != j, both resolved, d_gt(i, j) < cut_i} mean_t sigmoid(t - |d_pred(a, i, j) - d_gt(i, j)|) / n_pairs(b),   t in (0.5, 1, 2, 4) A
    L = mean_{a, b} (1 - lddt_ab),   cut_i = 30 A for a nucleotide atom, 15 A otherwise

over A = 48 augments (rigid motions of the same structure, noised and denoised) of B batch elements, N up to 4096-8192 atoms. The PyTorch
loss builds `cdist`, the four sigmoids and the masks as [N, N] tensors per augment (48 x 4096^2 pairs; 3 GB intermediates, a checkpointed
forward re-run in the backward): about 0.3 s per micro-step in MiniWorld's phase-2 graph trainer, more than the whole diffusion step.

## Scope and dispatch

fp32 compute on B200 (sm_100a). `x_pred` [A, B, N, 3] fp32 or bf16 (the gradient goes back in its dtype), `gt` [B, N, 3], `is_nuc` / `mask`
[B, N] bool, any N (padded to a multiple of 128 inside), any A and B. `smooth_lddt.refusal(...)` returns `None` when the op serves a call
and otherwise the reason; it reads metadata only. The three ops are registered as `smooth_lddt_prep` / `smooth_lddt_fwd` /
`smooth_lddt_bwd` and covered by `tests/compile/test_compile_wrap_coverage.py`; the cubin is built on first use by the newest nvcc that
knows sm_100a and launched through the driver on torch's current stream, so the op is CUDA-graph capturable (tested).

## Kernels

`kernels/smooth_lddt/cuda/sm100/smooth_lddt.cu`, plain CUDA-core code, no tensor cores (the op is bound by the exponentials and the
shared-memory reads). The ground-truth pair set is kept as two N x N **bit masks** built once per call (N^2 / 8 bytes each, 2.5 MB at
N = 4471, L2 resident), so a pair outside the radius costs nothing in the forward and backward kernels:

    FM[b, i, j] = both atoms resolved, i != j, d_gt(i, j) < cut_i            (the pairs of the loss, directed; popcount = n_pairs)
    UM[b, i, j] = FM[b, i, j] | FM[b, j, i]                                  (symmetric: every pair that carries gradient to i or to j)

### P1 · `slddt_prep`
Grid (B, NB, NB), one CTA per 128 x 128 tile of the ground truth: writes the 4 words per row of FM and UM, the per-row pair counts
(integer atomics: the sum is exact and order independent) and the tile flags (bit 0: FM has a bit, bit 1: UM has a bit).

### F1 · forward `slddt_fwd` (tile) / `slddt_fwd_full` (whole structure)
One thread per atom row; it visits the set bits of its FM words only (`__popc(lowbit - 1)`), computes `x = |d_pred - d_gt|` from the
coordinates, one `exp(x)` shared by the four thresholds (`sigmoid(t - x) = 1 / (1 + exp(x) exp(-t))`), and sums.
- tile kernel, grid (A * B, N / 128): 128-atom column tiles staged in shared memory, tiles with no FM bit skipped by the CTA.
- whole-structure kernel, grid (A * B, ceil(N / 512)), 512 threads: the coordinates of one augment (x and g as float4, N * 32 B) staged once;
  each thread walks its whole row across all tiles, so the imbalance of the neighbour counts inside a warp costs once per row instead of
  once per tile. Used up to N = 3520 atoms (two CTAs per SM); above that one CTA per SM does not hide the latency and the tile kernel
  is as fast or faster (N = 4471: 1.69 vs 1.75 ms).

### B1 · backward `slddt_bwd` / `slddt_bwd_full`
The same two shapes over UM. The score is symmetric in the pair's distance, but the radius test is not (row-wise `cut_i`), so every
atom collects the gradient of **both** of its roles (first and second index of a pair, weights `[d_gt < cut_i] + [d_gt < cut_j]`)
from the symmetric mask. That removes every float atomic: the gradient is deterministic.

Compiled with `-ftz=true -prec-sqrt=false -prec-div=false`: the default IEEE `sqrt` and the denormal-safe reciprocal of `__fdividef`
cost about 20 instructions per pair (a slow-path call and four `FSETP/FSEL/FMUL` rescalings); with the approximate forms the
neighbour loop is 7 MUFU and about 40 other instructions.

## Measurements (2026-10-12)

B200 (sm_100a), torch 2.13.0+cu129; forward + backward, A = 48, B = 1, **real predicted structures** (FoldBench, all atoms resolved):

| N atoms | neighbours / atom | tile fill | first version (ms) | this version (ms) |
|---|---|---|---|---|
| 1492 | 369 | 0.89 | 0.86 | 0.45 |
| 2999 | 464 | 0.74 | 2.24 | 1.08 |
| 4471 | 557 | 0.72 | 3.25 | 1.69 |
| 6620 | 511 | 0.43 | 5.45 | 2.97 |
| 9027 | 662 | 0.49 | 7.26 | 3.50 |

(neighbours: ordered pairs inside the radius per atom; tile fill: share of 128 x 128 tiles with at least one pair.) At N = 4471 only
12.5 % of the pair slots are inside the radius, and the first version's per-augment distance test on the ground truth cost as much as
the useful work.

`ncu --set full` of the intermediate version (bit masks, IEEE sqrt / division; SM clock locked at 1.15 GHz), N = 4471: forward 1.79 ms,
backward 2.33 ms, issue slots busy 56-59 %, 17.2 active threads per warp (the imbalance of the per-tile neighbour counts), 40 registers,
occupancy 46 %. The approximate-math flags took the wall time from 2.51 to 1.69 ms; the whole-structure kernel takes the lane
efficiency up where it fits in shared memory.

Two things were tried and rejected. Sharing the ground-truth work between augments (`-DGA=2/4/8`: one thread owns a row for GA augments)
cuts instructions but also the number of warps, and the latency of the exponentials and shared-memory reads stops being hidden:
3.7 / 5.3 / 6.5 / 11.9 ms for GA = 1 / 2 / 4 / 8 at N = 4096 (removed). A whole-structure kernel at one CTA per SM (N > 3520) is no
faster than the tile kernel. An estimate from the instruction mix puts the floor of the pair loop near 0.4-0.5 ms at N = 4471 (about
35 % of that is reached); not measured by a profile.

## Tests

`tests/integrations/test_smooth_lddt_gpu.py` compares the value and the coordinate gradient with the dense fp32 reference (exact
`cdist`: the matmul form of `torch.cdist` cancels about 1e-4 A at structure scale) for A up to 48, ragged N, masks and nucleotide radii;
runs both the whole-structure and the tile kernels (N = 3000 / 3600 on either side of the switch, and the tile kernels forced);
checks the zero-resolved case, the bf16 input, bitwise determinism of the backward and a CUDA-graph replay.
