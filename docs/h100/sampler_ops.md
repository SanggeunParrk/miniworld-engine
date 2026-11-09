# Sampler glue (`sampler_ops`) on H100

## Scope

The rigid-motion glue around an EDM / AlphaFold 3 denoiser step and the atom encoder's pooling, as three parameter-free module
classes in `modules/sampler_ops.py` (eval only, no autograd; `_forward_pytorch` = the dense references in `modules/functional.py`):

| class | `forward` | what it computes |
|---|---|---|
| `AugmentNoise` | `(coords [B, N, 3], mask [N], rv [B, 2, 3], tr [B, 3], noise [B, N, 3], sigma, sigma_next)` -> `(noisy, sc [2])` | centre on the mask-weighted mean, rotate by the Gram-Schmidt frame of `rv`, translate, mask, re-noise with the churn magnitude; `sc = (sigma_hat, magnitude)` stays on the device |
| `AlignEuler` | `(noisy, denoised, mask, sc, sigma_next)` -> `[B, N, 3]` | weighted Kabsch alignment of `noisy` onto `denoised` (Horn's quaternion method), then `aligned + step_scale (sigma_next - sigma_hat) (aligned - denoised) / sigma_hat` |
| `PoolAtoms` | `(x [S, Q, C], idx [T, A], gather_mask, atom_mask)` -> `[S, T, C]` | `sum_a m relu(g x[s, idx]) / max(sum_a m, 1e-10)`, bf16 out, fp32 sums |

Backends: `pytorch`, `cuda` (H100 sm_90a), `triton` (any CUDA card). The random draws (`rv`, `tr`, `noise`) are made by the caller;
the churn arithmetic runs on device scalars, so nothing reads the host and every call can be captured.

Requirements rows: `AugmentNoise` (AF3 churn; and EDM churn `window=1, gamma=0.4, sigma_min=0.05, sigma_max=50, variance_floor=1e-6,
clamp=True`), `AlignEuler` -- float32, eval, 1024-8192 atoms, 1 / 5 / 48 samples; `PoolAtoms` -- master (bf16 `x`), eval, 128 channels,
24 atom slots per token.

Judging: [../measure.md](../measure.md); backends: [../modules.md](../modules.md); kernels: [../kernels.md](../kernels.md).

## Kernels

CUDA (`kernels/sampler_ops/cuda/sampler_ops.cu`, one shape-independent extension `sampler_ops_sm90a`; every size is a launch
parameter). Each op is one opaque op (`sampler_ops_augment_noise_sm90`, `sampler_ops_align_euler_sm90`, `sampler_ops_pool_atoms_sm90`
in `requirements_kernel.csv`, role `forward_infer`):

| kernel | op | what it does |
|---|---|---|
| `moments_k<K>` | augment_noise | per-chunk partials of the mask and the mask-weighted coordinates |
| `augment_noise_k<K>` | augment_noise | sums the sample's partials (chunk order), computes the frame and churn scalars, rotates + translates + masks + adds noise; writes `sc` |
| `moments2_k<K>` | align_euler | per-chunk partials of both point sets |
| `cov_k<K>` | align_euler | centred, weighted covariance partials |
| `align_euler_k<K>` | align_euler | sums the partials, Horn rotation (dominant eigenvector of the 4 x 4 matrix by 12 squarings, no SVD, warp 0), Euler update |
| `pool_atoms_wide_k` | pool_atoms | `C % 8 == 0` (the launched case): a thread owns 8 consecutive channels (one 16-byte load per atom row), SUB atoms requested at once, loop over the real atom count |
| `pool_atoms_fast_k` | pool_atoms | `C % 4 == 0`: one warp-batch of up to 32 atoms per round trip (a lane loads one atom's masks / index, shuffled to the others) |
| `pool_atoms_k` | pool_atoms | generic fallback for index / mask dtype combinations the templated kernels do not instantiate |

Triton (`kernels/sampler_ops/triton/main.py`): `_stats` (chunk partials), `_augment`, `_cov`, `_align` (with an in-kernel Horn
rotation), `_pool`. They are launched by the `"triton"` backend but are not listed in `requirements_kernel.csv`.

## Design

- Partial-sum passes: a block covers a chunk of 1024 atoms (`K = 4` per thread, 256 threads); `chunk_for` halves the chunk (down to
  256) while the launch would have fewer than 264 blocks (two per SM). Each block writes a 16-float partial record; no atomics.
- The consuming kernel stages its sample's chunk records in shared memory (one round trip of independent loads) and one thread per value
  adds them in chunk order, so the sum is the serial single-thread order.
- The kernels of one op are chained with programmatic dependent launch: the next kernel's input loads and scalar prologue (frame,
  churn scalars) run before `griddepcontrol.wait`; only the partials are read after it.
- `pool_atoms`: one pass over the atom rows; the channel sums run in atom order, the same order in the wide and fast kernels.

## Numerics

- fp32 throughout for `augment_noise` / `align_euler`, operation for operation the torch path's arithmetic: no fast-math, no
  contraction where the reference rounds between operations.
- Results depend on the chunk size only through the association of the fp32 sums (different chunk sizes are not bitwise equal); a
  given shape is run-to-run deterministic.
- `pool_atoms`: fp32 sums, bf16 output; the wide and fast kernels give the same bits.

## Launch configs

CUDA: fixed (256 threads, chunk by `chunk_for`, pool grid by channel width). Triton: `@configured` as
`sampler_ops_{stats,augment,cov,align,pool}_triton`; there is no `kernels/sampler_ops/configs/h100_sm90.json`, so they launch their
defaults (`BLOCK = 512`, 4 warps; pool `BLOCK_C = 256`, 2 warps).

## Limits

- `AugmentNoise` / `AlignEuler` (`_supports_kernels` + `_cuda_h100`): fp32 CUDA tensors of the documented shapes, scalar `sigma` /
  `sigma_next`, `sc` with at least two entries, non-empty; `"cuda"` needs sm_90.
- `PoolAtoms`: bf16 `x [S, Q, C]` on CUDA, `idx` int32 / int64 `[T, A]`, masks of `idx`'s shape in bool / fp32 / bf16 / int32 / int64,
  non-empty; `"cuda"` additionally `C % 4 == 0`, `C <= 4096`, sm_90.

Historical (from the 2026-10-25 note, not re-measured under the current measurer): CUDA graph, A = 48, 32768 atoms: augment_noise
27.0 us, align_euler 64.6 us, pool_atoms 96.4 us.

## History

Old page: `git show HEAD:docs/gpus/h100/sampler_ops/sampler_ops.md`.
