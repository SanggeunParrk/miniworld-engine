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

`kernels/smooth_lddt/cuda/sm100/smooth_lddt.cu`, plain CUDA-core code: one thread per atom row, a 128-atom column tile staged in shared
memory, no tensor cores (the op is bound by the exponentials).

### P1 · `slddt_prep`
Grid (B, NB, NB), one CTA per 128 x 128 tile of the ground truth: counts the pairs inside the radius per row (integer atomics: the sum is
exact and order independent) and writes "any pair in this tile" as a flag. Done once per call for all augments.

### F1 · forward `slddt_fwd`
Grid (A * B, NB). Per pair: distances from the coordinates, `x = |d_pred - d_gt|`, one `exp(x)` shared by the four thresholds
(`sigmoid(t - x) = 1 / (1 + exp(x) exp(-t))`), a masked sum. Tiles flagged empty are skipped by the whole CTA. Per-row-block partial sums are
added on the host side of the op.

### B1 · backward `slddt_bwd`
The same tiles recomputed (nothing is saved but the inputs). The score is symmetric in the pair's distance, but the radius test is not
(row-wise `cut_i`), so every atom collects the gradient of **both** of its roles (first and second index of a pair, weights
`[d_gt < cut_i] + [d_gt < cut_j]`) from the tile pair (ib, jb) and (jb, ib). That removes every float atomic: the gradient is deterministic.

## Measurements (2026-10-12)

One B200 (sm_100a) shared with a training job, torch 2.13.0+cu129; forward + backward, A = 48, B = 1, 73-100 % of the atoms resolved:

| N atoms | this op (ms) | Triton version of the same algorithm (ms) |
|---|---|---|
| 3072 | 3.2 | 5.5 |
| 4096 | 3.5-4.3 | 7.9 |
| 8192 | 5.2 | 6.6 |

`ncu --set full`, SM clock locked at 1.15 GHz (durations are longer than at boost clock), A = 48, N = 4096:

| kernel | duration | issue slots busy | active threads / warp | note |
|---|---|---|---|---|
| `slddt_prep` (as first written: grid (B, NB)) | 365 us | 2.9 % | 28.8 | 32 CTAs on 148 SMs, occupancy 6 % |
| `slddt_prep` (grid (B, NB, NB)) | 26.7 us | 41 % | 28.8 | 14x faster |
| `slddt_fwd` | 944 us | 52.6 % | 17.2 | occupancy 35 % of 75 % theoretical |
| `slddt_bwd` | 1.34 ms | 52.9 % | 18.6 | 40 registers |

Two things were tried and rejected. Sharing the ground-truth work between augments (`-DGA=2/4/8`: one thread owns a row for GA augments,
the distance / radius test is done once per pair) cuts instructions but also the number of warps, and the latency of the exponentials
and shared-memory reads stops being hidden: 3.7 / 5.3 / 6.5 / 11.9 ms for GA = 1 / 2 / 4 / 8 at N = 4096. GA = 1 stays the default. The
17-18 active threads per warp are the pairs outside the radius and unresolved atoms; a compacted neighbour list (per row, the atoms inside
the radius and a two-bit role weight) would remove them at the price of a gather and a second pass; not built (the term is about 2 % of
the diffusion step).

## Tests

`tests/integrations/test_smooth_lddt_gpu.py` compares the value and the coordinate gradient with the dense fp32 reference (exact
`cdist`: the matmul form of `torch.cdist` cancels about 1e-4 A at structure scale) for A up to 48, ragged N, masks and nucleotide radii;
checks the zero-resolved case, the bf16 input, bitwise determinism of the backward and a CUDA-graph replay.
