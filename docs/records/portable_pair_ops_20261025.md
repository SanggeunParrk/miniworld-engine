# Portable kernels for the La-Proteina pair builders and the token-pair initialisation (2026-10-25)

Branch `feat/portable-pair-ops` (base: engine `main` at `fdb91b50`). Three ops had a PyTorch reference and H100 CUDA kernels and nothing for the other cards:
`pair_embed_geom` (+ the decoder's `pair_embed_ca`), `pair_embed_adaln` and `token_pair_init`. This change adds portable kernels for all of them (forward,
and the backward wherever the CUDA op has one) with the CUDA contract, and routes the three ops by card (`kernels/_pair_route.py`, [dispatch.md](../dispatch.md)).

Everything below was measured on **one H100 80GB HBM3** (torch 2.13.0+cu129, triton 3.7.1) with `settings.force_portable_kernels=True` (the internal A/B switch: the portable kernels on every card,
H100 included); the CUDA column is the same call without the switch (what H100 runs). **No other card was run:** the A100 / A5000 / A6000
manifests hold no row for these kernels, so by the routing rule those cards warn (`UnrecordedCardWarning`) and run the PyTorch reference until
`python -m miniworld_engine.autotune.run_all` has been run on them.

## What was built

| op | kernels (jit symbols) | what they do |
|---|---|---|
| `pair_embed_geom` / `pair_embed_ca` | `_geom_bins`, `_ca_bins`, `_geom_fwd_gather` | the bins of every pair (sequence separation, four CA distances, five orientation angles, chain product; the geometry is computed as torch does, fp32, IEEE division and square root, `atan2` from libdevice) are written as the CUDA kernel's 16-byte code; a second kernel sums up to ten bf16 table rows per pair in fp32, rounds to bf16 (`ln_eps`: LayerNorm of the rounded sum, no affine). One kernel doing both was 5-6x slower: the per-pair geometry is replicated over the threads that hold the 256-channel tile |
| | `_geom_dw`, `_geom_dw_reduce` | table gradient `onehot^T @ d pair` on the tensor cores: bf16 one-hots built from the codes in registers, fp32 accumulation, 192 fixed contiguous pair slices, tiles in order, then a fixed-order reduce and one bf16 rounding. No atomics |
| | `_geom_ln_bwd` | LayerNorm backward with the pair recomputed from the codes |
| `pair_embed_adaln` | `_adaln_fwd` | bins, gather of three fp32 table rows, both LayerNorms, per-sample gamma / beta, bf16 out; writes the packed bin word of every pair |
| | `_adaln_bwd`, `_adaln_dw`, `_adaln_bwd_reduce` | recompute the pair, bf16 d(table sum) (in chunks of at most 512 MiB) and per-(sample, 8 rows) fp32 partials of dgamma / dbeta / dln_w / dln_b; table gradient as above; fixed-order reduces |
| `token_pair_init` | `_tpi_fwd` | `left[i] + right[j]` + up to four table rows + the bond row, in the CUDA kernel's order: bit-identical output |
| | `_tpi_dx`, `_tpi_dw`, `_tpi_dw_reduce` | dleft / dright as fixed-order sums over j / over i; table gradient as a one-hot product with dz split into three bf16 terms (fp32-accurate), 192 slices, fixed-order reduce |

Registry rows `pair_embed_geom_{bins,fwd_gather,dw,bwd_reduce,bwd_ln}_triton`, `pair_embed_ca_bins_triton`, `pair_embed_adaln_{fwd,bwd,dw,bwd_reduce}_triton`,
`token_pair_init_{fwd,bwd_dx,dw,dw_reduce}_triton` (drivers / checkers in `drivers|checks/{pair_embed_geom,pair_embed_adaln,token_pair_init}.py`, config sets `default/` and
`grid/`, `rtol` = 4x the error measured here, aligned and ragged shapes). Records: `OP_KERNELS` entries `pair_embed_geom`, `pair_embed_ca`, `pair_embed_adaln`, `token_pair_init`.

## Parity (H100, B = 4, L = 256, ragged lengths 60 .. 100 %, 3 chains, missing CA / CB)

Against the dense fp32 PyTorch reference (the one-hot Linear). The portable kernels and the CUDA kernels give the same numbers at every entry: the geometry bins are
bit-identical (the 16-byte codes compare equal), so the pairs whose bin the fp32 reference puts one bin away (the documented fp32-vs-autocast flips: 4e-4 of the pairs
here, 1e-3 at L = 100, 4e-3 at B = 48, L = 24) are the same pairs, and they are the whole of the encoder's "error" in the table.

| op | portable vs reference | CUDA vs reference | bitwise reruns of the backward (3 runs) |
|---|---|---|---|
| `pair_embed_geom` forward | 4.1e-4 of the pairs differ (bin flips); padded pairs exactly 0 | same | - |
| `pair_embed_geom_train` (+ `ln_eps`) | out 1.4e-2 / 1.3e-2, dW 1.2e-2 (flips) | same | yes / yes |
| `pair_embed_ca` | out 1.8e-3, dW 0 | same | yes |
| `pair_embed_ca` + `ln_eps` | out 1.6e-3, dW 1.3e-4 | out 1.6e-3, dW 9.4e-5 | yes |
| `pair_embed_adaln` | out 3.0e-3; dW 3.7e-3, dln_w 3.5e-3, dln_b 1.7e-3, dgamma 3.1e-3, dbeta 1.6e-3 | identical to 2 digits | yes |
| `token_pair_init` | z 5.6e-8; dleft 1.3e-7, dright 3.1e-7, dW_rel 7.7e-7, dW_bond 1.1e-6 | z 5.6e-8; 1.8e-7, 1.5e-7, 8.2e-7, 1.1e-6 | yes |

Portable against CUDA (same inputs): geometry codes bit-identical; `pair_embed_geom` forward bit-identical (with `ln_eps` at most one bf16 rounding step apart on a handful of values); table
gradients bit-identical for the same codes (geom) / relative 1e-5 (adaln, whose bf16 d(table sum) differs in the last bit where the LayerNorm statistics do); `token_pair_init`
forward bit-identical, gradients 1e-7 relative. Every candidate launch configuration of every kernel was run alone: the gradients of the table-gradient kernels are bitwise
equal across configurations (their summation order does not depend on the tile), the others give the right numbers and the same bits twice.

## Speed and peak memory (H100, CUDA graph, B = 16 / 32 / 48, L = 128 .. 512)

CUDA-graph timing of the whole step (median of 15 replays after warm-up), peak extra memory of one eager step in GiB (`reference ms / GiB` etc.). The reference is
the dense PyTorch builder of the op (`*_reference`): fp32 one-hot Linear, a bf16 round and, where the op has one, the LayerNorm. "OOM" = more than 80 GB.
Training rows are forward + backward of the op alone (the weight gradient / all parameter gradients), with a random upstream gradient.

#### pair_embed_geom, forward (frozen encoder)

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 3.59 / 1.03 | 0.14 / 0.13 | 0.17 / 0.13 | 25.3x (7.9x mem) | 0.86x |
| 16 | 256 | 13.95 / 4.09 | 0.46 / 0.52 | 0.54 / 0.50 | 30.5x (7.9x mem) | 0.85x |
| 16 | 384 | 31.51 / 9.21 | 0.99 / 1.16 | 1.17 / 1.13 | 31.7x (7.9x mem) | 0.85x |
| 16 | 512 | 56.23 / 16.36 | 1.72 / 2.06 | 1.97 / 2.00 | 32.7x (7.9x mem) | 0.87x |
| 32 | 128 | 7.09 / 2.04 | 0.25 / 0.26 | 0.29 / 0.25 | 28.6x (7.8x mem) | 0.84x |
| 32 | 256 | 27.95 / 8.18 | 0.88 / 1.03 | 1.00 / 1.00 | 31.8x (7.9x mem) | 0.88x |
| 32 | 384 | 63.29 / 18.40 | 1.93 / 2.32 | 2.08 / 2.25 | 32.8x (7.9x mem) | 0.93x |
| 32 | 512 | 107.99 / 32.72 | 3.40 / 4.13 | 3.57 / 4.00 | 31.8x (7.9x mem) | 0.95x |
| 48 | 128 | 10.71 / 3.07 | 0.35 / 0.39 | 0.38 / 0.38 | 30.3x (7.9x mem) | 0.92x |
| 48 | 256 | 42.15 / 12.27 | 1.30 / 1.55 | 1.44 / 1.50 | 32.5x (7.9x mem) | 0.90x |
| 48 | 384 | 91.16 / 27.61 | 2.85 / 3.48 | 2.94 / 3.38 | 32.0x (7.9x mem) | 0.97x |
| 48 | 512 | OOM | 5.06 / 6.19 | 5.22 / 6.00 | reference OOM | 0.97x |

#### pair_embed_geom_train + ln_eps, forward + backward (w grad)

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 5.62 / 1.03 | 0.66 / 0.31 | 0.76 / 0.26 | 8.5x (3.3x mem) | 0.87x |
| 16 | 256 | 25.35 / 4.09 | 2.15 / 1.07 | 2.80 / 1.03 | 11.8x (3.8x mem) | 0.77x |
| 16 | 384 | 57.15 / 9.21 | 4.62 / 2.34 | 6.22 / 2.30 | 12.4x (3.9x mem) | 0.74x |
| 16 | 512 | 101.78 / 16.36 | 7.97 / 4.12 | 10.66 / 4.07 | 12.8x (4.0x mem) | 0.75x |
| 32 | 128 | 12.74 / 2.04 | 1.15 / 0.57 | 1.44 / 0.52 | 11.1x (3.6x mem) | 0.80x |
| 32 | 256 | 50.95 / 8.18 | 4.13 / 2.09 | 5.41 / 2.04 | 12.3x (3.9x mem) | 0.76x |
| 32 | 384 | 115.03 / 18.40 | 8.97 / 4.63 | 11.92 / 4.58 | 12.8x (4.0x mem) | 0.75x |
| 32 | 512 | 199.08 / 32.72 | 15.76 / 8.18 | 20.89 / 8.14 | 12.6x (4.0x mem) | 0.75x |
| 48 | 128 | 19.13 / 3.07 | 1.64 / 0.82 | 2.09 / 0.77 | 11.7x (3.7x mem) | 0.78x |
| 48 | 256 | 76.65 / 12.27 | 6.01 / 3.11 | 7.98 / 3.06 | 12.8x (3.9x mem) | 0.75x |
| 48 | 384 | 168.07 / 27.61 | 13.31 / 6.91 | 17.59 / 6.87 | 12.6x (4.0x mem) | 0.76x |
| 48 | 512 | OOM | 23.32 / 12.25 | 30.93 / 12.20 | reference OOM | 0.75x |

#### pair_embed_ca + ln_eps, forward + backward (w grad)

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 3.21 / 0.78 | 0.34 / 0.28 | 0.44 / 0.26 | 9.5x (2.8x mem) | 0.76x |
| 16 | 256 | 13.67 / 3.12 | 1.02 / 1.04 | 1.65 / 1.02 | 13.4x (3.0x mem) | 0.62x |
| 16 | 384 | 30.73 / 7.01 | 2.13 / 2.31 | 3.65 / 2.29 | 14.4x (3.0x mem) | 0.58x |
| 16 | 512 | 54.62 / 12.47 | 3.72 / 4.09 | 6.33 / 4.07 | 14.7x (3.0x mem) | 0.59x |
| 32 | 128 | 6.28 / 1.56 | 0.56 / 0.54 | 0.84 / 0.51 | 11.1x (2.9x mem) | 0.67x |
| 32 | 256 | 27.30 / 6.23 | 1.90 / 2.06 | 3.18 / 2.04 | 14.3x (3.0x mem) | 0.60x |
| 32 | 384 | 61.52 / 14.03 | 4.18 / 4.60 | 7.23 / 4.58 | 14.7x (3.1x mem) | 0.58x |
| 32 | 512 | 109.68 / 24.94 | 7.48 / 8.15 | 12.63 / 8.13 | 14.7x (3.1x mem) | 0.59x |
| 48 | 128 | 9.59 / 2.34 | 0.78 / 0.79 | 1.24 / 0.77 | 12.3x (3.0x mem) | 0.63x |
| 48 | 256 | 41.13 / 9.35 | 2.73 / 3.08 | 4.72 / 3.05 | 15.0x (3.0x mem) | 0.58x |
| 48 | 384 | 92.23 / 21.04 | 6.13 / 6.88 | 10.68 / 6.86 | 15.0x (3.1x mem) | 0.57x |
| 48 | 512 | 164.07 / 37.41 | 10.90 / 12.22 | 18.67 / 12.19 | 15.0x (3.1x mem) | 0.58x |

#### pair_embed_adaln, forward

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 3.49 / 1.37 | 0.21 / 0.13 | 0.14 / 0.13 | 16.6x (10.5x mem) | 1.50x |
| 16 | 256 | 13.72 / 5.47 | 0.79 / 0.50 | 0.41 / 0.50 | 17.5x (10.9x mem) | 1.90x |
| 16 | 384 | 30.87 / 12.31 | 1.74 / 1.13 | 0.99 / 1.13 | 17.7x (10.9x mem) | 1.76x |
| 16 | 512 | 54.92 / 21.88 | 3.06 / 2.02 | 1.24 / 2.02 | 17.9x (10.8x mem) | 2.48x |
| 32 | 128 | 6.90 / 2.73 | 0.40 / 0.25 | 0.21 / 0.25 | 17.2x (10.9x mem) | 1.93x |
| 32 | 256 | 27.41 / 10.94 | 1.55 / 1.01 | 0.73 / 1.01 | 17.7x (10.8x mem) | 2.11x |
| 32 | 384 | 61.79 / 24.61 | 3.45 / 2.27 | 1.46 / 2.27 | 17.9x (10.8x mem) | 2.36x |
| 32 | 512 | 109.80 / 43.75 | 6.12 / 4.03 | 2.36 / 4.03 | 17.9x (10.9x mem) | 2.59x |
| 48 | 128 | 10.48 / 4.10 | 0.59 / 0.38 | 0.32 / 0.38 | 17.7x (10.8x mem) | 1.84x |
| 48 | 256 | 41.15 / 16.41 | 2.31 / 1.51 | 0.99 / 1.51 | 17.8x (10.9x mem) | 2.34x |
| 48 | 384 | 92.64 / 36.92 | 5.17 / 3.40 | 2.08 / 3.40 | 17.9x (10.9x mem) | 2.48x |
| 48 | 512 | OOM | 9.17 / 6.05 | 3.37 / 6.05 | reference OOM | 2.72x |

#### pair_embed_adaln, forward + backward (w, ln_w, ln_b, gamma, beta grads)

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 6.50 / 1.94 | 1.04 / 0.29 | 0.98 / 0.26 | 6.3x (6.7x mem) | 1.05x |
| 16 | 256 | 25.44 / 7.38 | 3.27 / 1.04 | 3.71 / 1.01 | 7.8x (7.1x mem) | 0.88x |
| 16 | 384 | 57.15 / 16.45 | 8.81 / 1.73 | 8.74 / 1.65 | 6.5x (9.5x mem) | 1.01x |
| 16 | 512 | 101.59 / 29.25 | 14.67 / 2.66 | 13.99 / 2.54 | 6.9x (11.0x mem) | 1.05x |
| 32 | 128 | 12.81 / 3.75 | 1.71 / 0.54 | 1.87 / 0.51 | 7.5x (6.9x mem) | 0.92x |
| 32 | 256 | 50.73 / 14.75 | 6.42 / 1.58 | 7.28 / 1.52 | 7.9x (9.3x mem) | 0.88x |
| 32 | 384 | 114.31 / 32.77 | 16.45 / 2.94 | 16.13 / 2.80 | 7.0x (11.1x mem) | 1.02x |
| 32 | 512 | OOM | 29.39 / 4.81 | 27.98 / 4.59 | reference OOM | 1.05x |
| 48 | 128 | 19.12 / 5.57 | 2.81 / 0.79 | 2.65 / 0.76 | 6.8x (7.1x mem) | 1.06x |
| 48 | 256 | 76.37 / 21.94 | 9.58 / 2.12 | 10.69 / 2.04 | 8.0x (10.3x mem) | 0.90x |
| 48 | 384 | OOM | 24.24 / 4.14 | 24.14 / 3.94 | reference OOM | 1.00x |
| 48 | 512 | OOM | 43.95 / 6.97 | 41.67 / 6.63 | reference OOM | 1.05x |

#### token_pair_init, forward + backward (left, right, w_rel, w_bond grads)

| B | L | reference ms / GiB | portable ms / GiB | CUDA ms / GiB | portable vs reference | portable vs CUDA |
|---|---|---|---|---|---|---|
| 16 | 128 | 2.04 / 0.67 | 0.50 / 0.14 | 0.31 / 0.16 | 4.1x (4.8x mem) | 1.63x |
| 16 | 256 | 7.65 / 2.69 | 1.47 / 0.52 | 1.01 / 0.60 | 5.2x (5.2x mem) | 1.46x |
| 16 | 384 | 17.05 / 6.05 | 3.07 / 1.15 | 2.15 / 1.33 | 5.6x (5.3x mem) | 1.43x |
| 16 | 512 | 30.27 / 10.76 | 5.32 / 2.02 | 3.69 / 2.33 | 5.7x (5.3x mem) | 1.44x |
| 32 | 128 | 3.89 / 1.35 | 0.82 / 0.27 | 0.58 / 0.32 | 4.7x (5.0x mem) | 1.41x |
| 32 | 256 | 15.18 / 5.38 | 2.74 / 1.02 | 1.95 / 1.20 | 5.5x (5.3x mem) | 1.40x |
| 32 | 384 | 34.16 / 12.11 | 5.94 / 2.28 | 4.24 / 2.65 | 5.8x (5.3x mem) | 1.40x |
| 32 | 512 | 60.46 / 21.52 | 10.75 / 4.04 | 7.39 / 4.66 | 5.6x (5.3x mem) | 1.46x |
| 48 | 128 | 5.75 / 2.02 | 1.14 / 0.39 | 0.85 / 0.48 | 5.1x (5.2x mem) | 1.34x |
| 48 | 256 | 22.77 / 8.07 | 4.03 / 1.53 | 2.93 / 1.81 | 5.7x (5.3x mem) | 1.38x |
| 48 | 384 | 50.96 / 18.16 | 8.78 / 3.41 | 6.21 / 3.98 | 5.8x (5.3x mem) | 1.41x |
| 48 | 512 | 90.80 / 32.29 | 16.00 / 6.05 | 11.04 / 6.99 | 5.7x (5.3x mem) | 1.45x |


Reading: the portable kernels beat the reference by 25-33x (`pair_embed_geom` forward), 8-13x (training with the LayerNorm folded in), 9-15x (`pair_embed_ca`),
17-18x (`pair_embed_adaln` forward), 6-8x (`pair_embed_adaln` training), 4-6x (`token_pair_init` forward + backward), at 3-11x less peak memory, and they run
the shapes at which the reference runs out of memory (B = 48, L = 512). Against the CUDA kernels they are on par or faster for the two encoder / decoder
pair embeddings (0.57-0.97x of the CUDA time: the portable training step is 1.3x faster than the CUDA op's for the encoder and 1.7x for the decoder, the forward 3-15 % faster), on par for `pair_embed_adaln` training, 1.5-2.7x slower for `pair_embed_adaln` forward and 1.4-1.6x slower for `token_pair_init`. H100 and B200 keep running their CUDA kernels.
