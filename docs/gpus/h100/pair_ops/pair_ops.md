# La-Proteina pair builders and token-pair initialisation on H100 (sm90)

Kernel-level status of `kernels/pair_embed_geom` (the autoencoder encoder's backbone pair and the decoder's CA-distance pair), `kernels/pair_embed_adaln` (the denoiser's
initial pair) and `kernels/token_pair_init` (AF3-style input pair initialisation); the module-level summary is in [h100.md](../h100.md). These ops are not in the shape registry
and have no bench target, so there is no **성능 확인** to give (the maintainer's call; the cells stay empty). **cache build ✓** for the CUDA kernels (no autotune: fixed
launch shapes, extensions in the prebuilt bundle); the portable kernels below have `default/` and `grid/` config sets and no committed H100 cache (they matter only for A/B runs here).

## Scope and dispatch

| op | calls | dtype | served by |
|---|---|---|---|
| `pair_embed_geom` | forward only (frozen encoder) | bf16 pair out, 256 channels | CUDA sm_90a; portable kernels under the A/B switch and on a recorded card |
| `pair_embed_geom_train`, `pair_embed_ca` (+ `ln_eps`) | forward + backward in the table (autoencoder training) | same | same |
| `pair_embed_adaln` | forward + backward in w, ln_w, ln_b, gamma, beta | same | same |
| `token_pair_init` | forward + backward in left, right, w_rel, w_bond | fp32, 128 channels | CUDA sm_90a / sm_100a; portable kernels as above |

H100 runs its CUDA kernels. The routing (`kernels/_pair_route.py`, [dispatch.md](../../../dispatch.md#the-la-proteina-pair-ops)) sends any other card to the portable kernels
once the engine holds a record for it and otherwise to the PyTorch reference with one warning; under the internal A/B switch every card, H100 included, runs the portable kernels. That
switch is what produced the numbers below. `refusal(...)` / `refusal_ca(...)` read metadata only and never warn.

## Portable kernels

| id | kernel | grid / role |
|---|---|---|
| G1 | `_geom_bins`, `_ca_bins` | (j block, row i, batch): the 16-byte code of every pair (bins, live bit, chain product), identical to the CUDA kernel's |
| G2 | `_geom_fwd_gather` | pair tiles: sum of the bf16 table rows the code names (encoder 317 rows, decoder 157), optional LayerNorm (`ln_eps`), bf16 out |
| G3 | `_geom_dw`, `_geom_dw_reduce` | (channel slice, 128-row block, pair slice): one-hot x d pair on the tensor cores, 192 fixed pair slices, fixed-order reduce, one bf16 rounding |
| G4 | `_geom_ln_bwd` | LayerNorm backward with the pair recomputed from the codes |
| A1 | `_adaln_fwd` | (j block, row, batch): bins, three fp32 table rows, both LayerNorms, gamma / beta, bf16 out + bin words |
| A2 | `_adaln_bwd`, `_adaln_dw`, `_adaln_bwd_reduce` | recompute, bf16 d(table sum) in chunks, per-(sample, 8 rows) parameter partials, table gradient as G3, fixed-order reduces |
| T1 | `_tpi_fwd` | (j block, row, batch): left + right + class rows + bond row, fp32 in the CUDA kernel's order |
| T2 | `_tpi_dx`, `_tpi_dw`, `_tpi_dw_reduce` | row / column sums in a fixed order; table gradient from a one-hot product with dz in three bf16 terms; fixed-order reduce |

No kernel uses atomics for a parameter gradient; every backward is bitwise reproducible for a given launch configuration, and the table-gradient kernels give the same bits in every
configuration. Details, parity and the full speed / memory tables: [records/portable_pair_ops_20261025.md](../../../records/portable_pair_ops_20261025.md).

## Measurements (2026-10-25, after the CUDA kernel rework)

One H100, torch 2.13.0+cu129, triton 3.7.1, CUDA graph; ranges over B = 16 / 32 / 48 at L = 128 .. 512, reference = the dense PyTorch builder. The last column is the CUDA time
relative to the portable kernels (> 1: CUDA faster); the rework itself, its profile evidence and the full per-shape tables: [records/pair_ops_cuda_vs_portable_20261025.md](../../../records/pair_ops_cuda_vs_portable_20261025.md).

| op | portable vs PyTorch reference | peak memory vs reference | CUDA vs portable (before the rework -> now) |
|---|---|---|---|
| `pair_embed_geom` forward | 25-33x faster | 7.9x less | 0.84-0.97x -> **1.17-1.38x** |
| `pair_embed_geom_train` + `ln_eps`, forward + backward | 8.5-12.8x | 3.3-4.0x | 0.74-0.87x -> **1.27-1.44x** |
| `pair_embed_ca` + `ln_eps`, forward + backward | 9.5-15x | 2.8-3.1x | 0.57-0.76x -> **1.11-1.36x** |
| `pair_embed_adaln` forward | 16.6-17.9x | 10.5-10.9x | 1.50-2.72x (unchanged) |
| `pair_embed_adaln`, forward + backward | 6.3-8.0x | 6.7-11x | 0.88-1.06x -> **1.78-2.20x** |
| `token_pair_init`, forward + backward | 4.1-5.8x | 4.8-5.3x | 1.37-1.62x (unchanged) |

The reference runs out of memory at B = 48, L = 512 (and for `pair_embed_adaln` training from B = 32, L = 512); the CUDA and portable kernels do not.

## CUDA kernel structure (sm90)

| kernel | role |
|---|---|
| `pg_fwd`, `pg_fwd_ca` | persistent CTAs, 16 / 32 warps, one warp per (b, i) row; bins as torch computes them (bucket by binary search over the limits in shared memory), up to ten bf16 table rows from shared memory summed in fp32, optional LayerNorm, 16 B stores; the 16-byte codes for the backward |
| `pg_ln_bwd` | LayerNorm backward with the pair recomputed from the codes, `UL` pairs per warp step, loads pipelined two steps deep |
| `pg_bwd_mma4` | table gradient: the one-hot operand scattered into a swizzled shared-memory tile (two extra warps), `wgmma.m64n64k16` with both operands from shared memory, one warpgroup per 64 table rows, fixed pair slices and tile order, fixed-order reduce: bitwise deterministic. `MINIWORLD_PAIR_EMBED_BWD_IMPL=2` selects the previous `mma.sync` kernel, `0` the shared-memory kernel |
| `pe_bwd_tbl4` | the same for `pair_embed_adaln` (fp32 partials); `MINIWORLD_PAIR_EMBED_BWD_IMPL=0` selects the previous kernel |
