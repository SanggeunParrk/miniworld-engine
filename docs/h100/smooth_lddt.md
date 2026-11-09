# Smooth lDDT loss on H100

## Scope

`SmoothLddtLoss` (`modules/smooth_lddt.py`): the AlphaFold 3 Algorithm 27 diffusion loss, no parameters,

    loss = mean_{a, b} (1 - lddt_ab),   lddt_ab = sum_{pairs} keep_ij * mean_t sigmoid(t - |d_pred_ij - d_gt_ij|) / sum keep_ij

with `t in (0.5, 1, 2, 4)` A and `keep_ij` = both atoms resolved, `i != j`, `d_gt(i, j) < 30 A` for a nucleotide atom `i`, `15 A`
otherwise. `forward(x_pred [A, B, N, 3], gt [B, N, 3], is_nuc [B, N], mask [B, N])`; differentiable in `x_pred` only.

Backends: `pytorch` (dense reference, `functional.smooth_lddt_loss`), `cuda` (H100 sm_90a and B200 sm_100a cubins from one source),
`triton` (any CUDA card). Both kernel backends pad `N` to a multiple of 128 (padded atoms unresolved), cast `gt` to fp32 and the flags
to int8 (`SmoothLddtLoss._padded`).

Requirements row: `SmoothLddtLoss`, atom_single, 1024-8192 atoms, `float32 | master`, `eval | train`, 1 / 5 samples; `x_pred` in the
dtype, `gt` fp32; train = gradient with respect to `x_pred`.

Judging: [../measure.md](../measure.md); backends: [../modules.md](../modules.md); kernels: [../kernels.md](../kernels.md).

## Kernels

CUDA: `kernels/smooth_lddt/cuda/sm100/smooth_lddt.cu` (the directory name is historical; `_arch` builds `-arch=sm_90a` on H100),
plain CUDA-core code, launched through the CUDA driver on torch's current stream. Opaque ops `smooth_lddt_prep` / `_fwd` / `_bwd`
(`requirements_kernel.csv`: `smooth_lddt_prep_sm90`, `smooth_lddt_fwd_sm90` forward; `smooth_lddt_bwd_sm90` backward).

| kernel | grid | what it does |
|---|---|---|
| `slddt_prep` | (B, NB, NB), 128 threads | from `gt` only: bit masks `FM[b, i, :]` (pairs of the loss, directed) and `UM = FM | FM^T` (pairs carrying gradient to either atom), `N x N` bits each; per 128 x 128 tile a flag byte (bit 0: FM has a bit, bit 1: UM has one); per-row pair counts (integer atomics) |
| `slddt_fwd` | (A B, NB), 128 threads | tile kernel: one thread per atom row, walks the tiles whose flag is set and only the set bits of its 4 words per tile (tile coordinates staged in shared memory); partial sum per (augment * batch, row block) |
| `slddt_bwd` | (A B, NB), 128 threads | analytic gradient: every atom collects the gradient of both roles it plays in a pair (weight `[g < cut_i] + [g < cut_j]`), so no float atomics |
| `slddt_fwd_full_<T>s<S>` | (A B, ceil(N / (T / S))) | whole-structure forward: one augment's coordinates (`float4` x, y, z, cut^2) in shared memory (`N * 32` bytes), each row walks its whole neighbour list across all tiles; S threads per row deal out its word groups round-robin |
| `slddt_bwd_full_<T>` | (A B, ceil(N / T)) | whole-structure backward, one thread per row |

The value is `1 - (sum of the [A * B, NB] partials) / pair count`, averaged in torch; `SmoothLddtFunction` saves `x` (fp32), `gt`,
`nuc`, the tile flags and `UM`.

Triton (`kernels/smooth_lddt/triton/main.py`): `smooth_lddt_prep_triton` (32 x 32 tile flags and per-tile pair counts from `gt`),
`smooth_lddt_fwd_triton`, `smooth_lddt_bwd_triton` (BLOCK_I rows of one (augment, batch), loop over the BLOCK_J-column tiles whose flags
are set, recompute the ground-truth distance per visited pair). They are launched by the `"triton"` backend but are not listed in
`requirements_kernel.csv`.

## Design

- No `[N, N]` float tensor and no per-augment pair tensor: the augments are rigid motions of one structure, so the pair set, its
  per-tile skip flags and the pair count are built once per call from `gt`.
- A pair outside the radius costs nothing in the CUDA kernels (set-bit walk); the Triton kernels compute dense 32-row tiles of pairs, so
  their cost grows with the number of non-empty tiles rather than pairs.
- Pair score: with `w = exp(-x)`, `x = |d_pred - d_gt| >= 0` and `c_t = exp(-t)`, `sigmoid(t - x) = w / (w + c_t)`. The four-threshold
  score and its derivative are rational functions of `w` with one denominator `Q(w) = prod_t (w + c_t)` (positive coefficients, no
  cancellation, no overflow): `score = w Q'(w) / (4 Q(w))`, `-dscore/dx = w N(w) / (4 Q(w)^2)`. One exponential and one reciprocal per
  pair; the CUDA and Triton kernels share the form.
- Whole-structure vs tile kernels (`cuda/sm100/__init__.py`): the whole-structure kernels serve while `N * 32 <= 110 KB` (N <= 3520,
  two CTAs per SM), else the tile kernels. On H100 the forward variant is the largest of `512s2` / `256s2` whose grid gives at least
  1.5 CTAs per SM (else `256s2`); the backward CTA is the largest of 512 / 256 / 128 threads whose grid gives at least 2 CTAs per SM
  (else 128). B200 keeps `512s1` / 512.
- The bound is the MUFU work of the pair loop (2 sqrt + exp + rcp per pair) and lane imbalance from per-row neighbour-count
  differences, not memory.

## Numerics

- Every backend computes in fp32; `x_pred` bf16 is cast to fp32 before the CUDA kernels and the gradient is returned in `x_pred`'s
  dtype. The loss is returned in fp32 (fp64 for an fp64 call of the reference).
- The CUDA cubin is built with `-ftz=true -prec-sqrt=false -prec-div=false` and uses `__fdividef`.
- All reductions are fixed-order (partials per row block, summed by torch); value and gradient are bitwise repeatable.
- The pair score has a kink at `d_pred = d_gt`; an fp32 distance on the other side of it flips one pair's term, which bounds the
  gradient agreement with an fp64 definition.

## Launch configs

CUDA: fixed (128-thread tile kernels; whole-structure variant and CTA size chosen as above). Triton: `@configured`
`smooth_lddt_{prep,fwd,bwd}_triton`; there is no `kernels/smooth_lddt/configs/h100_sm90.json`, so they launch their defaults
(`BLOCK_P = 64`; `BLOCK_I = BLOCK_J = 64`; 4 warps, 2 stages).

## Limits (`_supports_kernels`, `_supports_cuda`)

- CUDA tensors; `x_pred` fp32 or bf16 `[A, B, N, 3]`; `gt [B, N, 3]`; `is_nuc` / `mask` bool `[B, N]`; non-empty.
- `"cuda"`: compute capability (9, 0) or (10, 0).

Historical (from the 2026-10-15 note, not re-measured under the current measurer): `ncu`, N = 8192, 7959 resolved atoms, tile kernels:
`slddt_prep` 88 us, `slddt_fwd` 470 us, `slddt_bwd` 609 us, 20 of 32 lanes active in the pair loops.

## Rejected approaches (brief)

- Raising the whole-structure threshold to one CTA per SM (224 KB) was slower than the tile kernels at N = 4096 on H100; the 110 KB
  switch point stays.
- Warp-cooperative flattening of the neighbour lists to recover idle lanes (estimated <= ~1.4x) was not built.

## History

Old page: `git show HEAD:docs/gpus/h100/smooth_lddt/smooth_lddt.md`.
