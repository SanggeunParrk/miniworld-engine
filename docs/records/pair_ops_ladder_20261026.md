# Pair-op / sampler ladder fixes on H100 (2026-10-26)

Branch `perf/ladder-fix-pair` (on `d9be45ad`, which holds the pair-op CUDA rework `88265370`). Goal: no CUDA-vs-portable violation and no near-miss (ratio >= 0.95) in the ladder rows of
`pair_embed_geom`, `pair_embed_ca`, `pair_embed_adaln`, `token_pair_init`, `smooth_lddt` and the sampler ops, inference and training. Harness: `benchmarks/runners/ladder.py` (one H100, 3 repeats,
do_bench over CUDA-graph replay, JIT cache private to this checkout). "before" = the same harness at `d9be45ad` (plus only the one-line adaln fix below, which is what makes the training rows measurable);
the committed record `performance_ladder_20261025.md` holds the earlier run (identical within noise).

## What was wrong and what changed

| row(s) | cause | change |
|---|---|---|
| `pair_embed_adaln` training: 5 n.a. rows | the portable backward returned outputs that alias each other (`dl[0, :C].contiguous()` of an already contiguous view; `dgb[:, :C].contiguous()` is a view for B = 1): `torch.library` refuses it, so the portable rung never ran under the default `compile_wrap=custom_op` (also `test_adaln_matches_the_cuda_path[1-257-True]`, `test_matches_the_dense_builder[1-33-False]`, `[1-1-True]`) | `triton/main.py portable_backward` returns clones. Training rows now measure: CUDA 0.50-0.57 of portable |
| `token_pair_init` inference 1.00-1.13x | forward kernel: one CTA per row re-filled a 72 KB table (plain loads) and re-read `right[j]` for every row; plus the glue cast the bool bond adjacency twice | `cuda/sm90/token_pair_init.cu` forward: a CTA writes R rows (4 up to L256, else 2), the table arrives by one TMA bulk copy overlapped with the bin packing, `right[j]` is loaded once per R rows (bitwise equal to the old kernel); `interface.py`: a bool bond is viewed as uint8 (no `!= 0` / cast passes) |
| `pool_atoms` 1.03-1.25x | 8-byte loads per thread, 32-wide shuffled atom window although a token has 24 atoms | wide kernel (16-byte loads, 4 atom rows in flight, real atom count), bitwise equal to the old one |
| `smooth_lddt` A48 N1024 inference tie 1.002 | 49 k rows = 11 warps per SM on 96 CTAs; the kernel is issue-latency bound (IPC 2.0 of 4) | forward `whole-structure` kernel deals each row to 2 threads (round-robin over the 4-word groups), CTA size picked from the grid (`_fwd_variant`); backward CTA size picked from the grid (`_fblk`); B200 keeps its 512 x 1 launch |
| `pair_embed_ca` inference B16 L128 0.965 | 13 us of launches around a 49 us kernel (linspace, arange, casts) | `_constant` caches the bin limits / default residue index per device (never created inside a graph capture), a bool mask is viewed as uint8 |

Tried and dropped: prefetching the first work item of `pg_fwd_ca` ahead of the table fill and changing its grid (no change, kernel sits 8-19 % above the write roofline); other store policies in the
token-pair-init kernel (`__stcs` / plain / `.cg` / `.wt`: same within 1 %).

## Before / after (ms, one H100, CUDA-graph, `ladder.py`; c/p = CUDA / portable)

| op | mode | B | L | portable before | CUDA before | c/p before | portable after | CUDA after | c/p after |
|---|---|---|---|---|---|---|---|---|---|
| pair_embed_geom | inference | B16 | 128 |  0.137  |  0.116  |  0.846  |  0.137  |  0.115  |  0.840  |
| pair_embed_geom | inference | B16 | 256 |  0.454  |  0.371  |  0.817  |  0.453  |  0.369  |  0.814  |
| pair_embed_geom | inference | B16 | 384 |  0.989  |  0.808  |  0.817  |  0.989  |  0.807  |  0.817  |
| pair_embed_geom | training | B16 | 128 |  0.648  |  0.450  |  0.695  |  0.653  |  0.454  |  0.695  |
| pair_embed_geom | training | B16 | 256 |  2.103  |  1.619  |  0.770  |  2.097  |  1.619  |  0.772  |
| pair_embed_geom | training | B16 | 384 |  4.505  |  3.588  |  0.796  |  4.512  |  3.643  |  0.807  |
| pair_embed_geom | inference | B48 | 128 |  0.345  |  0.271  |  0.786  |  0.346  |  0.271  |  0.784  |
| pair_embed_geom | inference | B48 | 256 |  1.279  |  0.967  |  0.756  |  1.289  |  0.972  |  0.754  |
| pair_embed_geom | training | B48 | 128 |  1.613  |  1.211  |  0.751  |  1.614  |  1.211  |  0.750  |
| pair_embed_geom | training | B48 | 256 |  5.930  |  4.512  |  0.761  |  5.928  |  4.546  |  0.767  |
| pair_embed_ca | inference | B16 | 128 |  0.066  |  0.064  |  0.965  |  0.060  |  0.057  |  0.953  |
| pair_embed_ca | inference | B16 | 256 |  0.214  |  0.196  |  0.917  |  0.210  |  0.190  |  0.905  |
| pair_embed_ca | inference | B16 | 384 |  0.462  |  0.415  |  0.899  |  0.458  |  0.407  |  0.890  |
| pair_embed_ca | training | B16 | 128 |  0.334  |  0.245  |  0.734  |  0.328  |  0.239  |  0.729  |
| pair_embed_ca | training | B16 | 256 |  1.019  |  0.854  |  0.838  |  1.012  |  0.847  |  0.837  |
| pair_embed_ca | training | B16 | 384 |  2.118  |  1.867  |  0.882  |  2.111  |  1.860  |  0.881  |
| pair_embed_ca | inference | B48 | 128 |  0.166  |  0.153  |  0.922  |  0.160  |  0.146  |  0.914  |
| pair_embed_ca | inference | B48 | 256 |  0.610  |  0.554  |  0.909  |  0.606  |  0.546  |  0.901  |
| pair_embed_ca | training | B48 | 128 |  0.757  |  0.648  |  0.857  |  0.750  |  0.641  |  0.855  |
| pair_embed_ca | training | B48 | 256 |  2.720  |  2.429  |  0.893  |  2.717  |  2.421  |  0.891  |
| pair_embed_adaln | inference | B16 | 128 |  0.209  |  0.139  |  0.663  |  0.209  |  0.139  |  0.664  |
| pair_embed_adaln | inference | B16 | 256 |  0.788  |  0.411  |  0.521  |  0.789  |  0.412  |  0.522  |
| pair_embed_adaln | inference | B16 | 384 |  1.736  |  0.980  |  0.564  |  1.749  |  0.987  |  0.564  |
| pair_embed_adaln | training | B16 | 128 |  1.023  |  0.524  |  0.512  |  1.025  |  0.523  |  0.510  |
| pair_embed_adaln | training | B16 | 256 |  3.230  |  1.851  |  0.573  |  3.230  |  1.845  |  0.571  |
| pair_embed_adaln | training | B16 | 384 |  8.756  |  4.647  |  0.531  |  8.763  |  4.677  |  0.534  |
| pair_embed_adaln | inference | B48 | 128 |  0.593  |  0.323  |  0.545  |  0.588  |  0.321  |  0.546  |
| pair_embed_adaln | inference | B48 | 256 |  2.324  |  0.989  |  0.426  |  2.307  |  0.983  |  0.426  |
| pair_embed_adaln | training | B48 | 128 |  2.767  |  1.385  |  0.501  |  2.774  |  1.376  |  0.496  |
| pair_embed_adaln | training | B48 | 256 |  9.513  |  5.197  |  0.546  |  9.514  |  5.227  |  0.549  |
| token_pair_init | inference | B16 | 128 |  0.079  |  0.089  |  1.130  |  0.072  |  0.067  |  0.926  |
| token_pair_init | inference | B16 | 256 |  0.222  |  0.233  |  1.050  |  0.216  |  0.207  |  0.956  |
| token_pair_init | inference | B16 | 384 |  0.464  |  0.476  |  1.026  |  0.453  |  0.441  |  0.974  |
| token_pair_init | training | B16 | 128 |  0.496  |  0.303  |  0.612  |  0.488  |  0.283  |  0.579  |
| token_pair_init | training | B16 | 256 |  1.463  |  1.011  |  0.691  |  1.454  |  0.985  |  0.678  |
| token_pair_init | training | B16 | 384 |  3.031  |  2.153  |  0.710  |  3.038  |  2.118  |  0.697  |
| token_pair_init | inference | B48 | 128 |  0.185  |  0.205  |  1.105  |  0.179  |  0.164  |  0.914  |
| token_pair_init | inference | B48 | 256 |  0.623  |  0.626  |  1.004  |  0.623  |  0.572  |  0.918  |
| token_pair_init | training | B48 | 128 |  1.121  |  0.846  |  0.755  |  1.106  |  0.804  |  0.727  |
| token_pair_init | training | B48 | 256 |  3.960  |  2.943  |  0.743  |  3.947  |  2.870  |  0.727  |
| smooth_lddt | inference | A48 B1 | 1024 |  0.160  |  0.160  |  1.003  |  0.161  |  0.131  |  0.813  |
| smooth_lddt | inference | A48 B1 | 2048 |  0.549  |  0.489  |  0.890  |  0.549  |  0.380  |  0.692  |
| smooth_lddt | training | A48 B1 | 1024 |  0.406  |  0.318  |  0.783  |  0.408  |  0.277  |  0.679  |
| smooth_lddt | training | A48 B1 | 2048 |  1.507  |  1.059  |  0.703  |  1.508  |  0.846  |  0.561  |
| smooth_lddt | inference | A16 B1 | 4096 |  0.821  |  0.598  |  0.728  |  0.815  |  0.593  |  0.728  |
| smooth_lddt | training | A16 B1 | 4096 |  2.101  |  1.280  |  0.609  |  2.102  |  1.279  |  0.609  |
| sampler augment_noise | inference | B5 | 4096 |  0.009  |  0.008  |  0.885  |  0.009  |  0.008  |  0.869  |
| sampler augment_noise | inference | B5 | 12288 |  0.011  |  0.009  |  0.821  |  0.010  |  0.008  |  0.838  |
| sampler augment_noise | inference | B5 | 24576 |  0.012  |  0.010  |  0.840  |  0.011  |  0.010  |  0.849  |
| sampler align_euler | inference | B5 | 4096 |  0.017  |  0.012  |  0.682  |  0.017  |  0.012  |  0.685  |
| sampler align_euler | inference | B5 | 12288 |  0.019  |  0.015  |  0.758  |  0.019  |  0.014  |  0.755  |
| sampler align_euler | inference | B5 | 24576 |  0.022  |  0.018  |  0.807  |  0.022  |  0.017  |  0.806  |
| sampler pool_atoms | inference | B5 c768 | 4096 |  0.018  |  0.018  |  1.015  |  0.017  |  0.015  |  0.876  |
| sampler pool_atoms | inference | B5 c768 | 12288 |  0.034  |  0.039  |  1.144  |  0.034  |  0.031  |  0.925  |
| sampler pool_atoms | inference | B5 c768 | 24576 |  0.056  |  0.070  |  1.247  |  0.056  |  0.051  |  0.907  |

Every row is OK (CUDA < portable < PyTorch). The 5 `pair_embed_adaln` training rows were n.a. in `performance_ladder_20261025.md`; their "before" is the harness at `d9be45ad` with the aliasing fix.
Raw: `~/miniworld-engine-scratch/lf-pair/out/{base,final}{.md,.csv}`.

## Rows still close to 1 (not fixed, floor-bound)

* `token_pair_init` inference B16 L256 0.956 and B16 L384 0.974: the op writes 0.54 / 1.2 GB of fp32 `z`; under the harness (`do_bench` zeroes the L2 before each call, so ~50 MB of dirty lines drain during
  the kernel) both kernels sit within 8-16 % of the DRAM floor (memset of the same size), CUDA 428 us against portable 459 us (L384, kernel only). No store policy or CTA shape moved it; the rest is glue shared by both rungs.
* `pair_embed_ca` inference B16 L128 0.953: `pg_fwd_ca` 49 us against a 41 us roofline; 13 us (now ~6 us) of launches around it are shared with the portable rung.

## Tests

`git grep -l -i -E 'pair_embed|token_pair_init|smooth_lddt|sampler_ops' tests` GPU files (any, sm90, sm100/test_smooth_lddt_gpu.py): 293 passed. Whole CPU suite: 4363 passed, 54 skipped, 2 failed + 2 errors
(`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins` x2): the same four fail on the base commit `d9be45ad` (stale Ampere caches). `dev derive --arch sm86`
re-run: `registry_kernel.units.json` regenerated (it was stale at the base: `test_the_sweep_page_matches_the_repo` and `test_compact_search_space` failed until then).
Parity: the new `token_pair_init` forward and `pool_atoms` are bitwise equal to the kernels they replace (probe, 5 + 3 shapes); `smooth_lddt` loss / gradient pass the existing 1e-6 / 5e-5 checks. No tolerance was changed.

## Merge notes

`pair_embed_*` `.cu` files were not touched. `sampler_ops.cu`, `token_pair_init.cu`, `smooth_lddt.cu` and the `smooth_lddt` / `token_pair_init` Python wrappers changed; a concurrent edit of those on another branch will conflict.
