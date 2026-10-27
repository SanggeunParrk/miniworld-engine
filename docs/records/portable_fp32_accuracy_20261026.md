# Portable fp32 (TF32) kernels: accuracy against the PyTorch TF32 statements (2026-10-26)

Problem: Triton's `tl.dot(..., input_precision="tf32")` TRUNCATES fp32 operands to the 10-bit TF32 mantissa; cuBLAS rounds to nearest. The portable
fp32 kernels were 2.2-5.6x the PyTorch TF32 error against an fp64 run (up to 20x on small bias gradients). Yardstick: engine error <= 1.5x the PyTorch
statements' error, output and every gradient.

Fix: round the operands to nearest (add 0x1000 to the fp32 bit pattern) before the MMA. ONE helper, `kernels/_tf32.py`: `tf32_dot` (a `tl.dot` that rounds
fp32 operands when the precision is "tf32"; bf16 compiles to the plain dot), `tf32_round`, `tf32_grid_tile` (exact grid value of a register tile),
`tf32_grid` / `tf32_grid_many` (host, whole tensors, up to six per launch), `tf32_cached` (inference: weights rounded once). Hand-CUDA kernels and bf16 untouched.
Where operands are rounded (cost measured on H100, CT inference d768, 102 us before: rounding both operands in the kernel +57 us, weights in a launcher pass +17, weights cached +4):
weights on the host (cached while grad is off), activations in the kernel, SwiGLU `h` on the grid at its store; Transition rounds its activation on the host
too (in-kernel rounding of the A tile gave wrong h at K = 192 with the ragged K tile; test `test_ragged_widths_stay_within_1p5x`).

## Error ratio (engine / PyTorch TF32, vs fp64), module level, H100, `tests/gpu/any/test_portable_fp32_accuracy_gpu.py` (`PARITY_REPORT=1 TF32ACC_VERBOSE=1`)

| family / shape | out before | out after | worst (out or grad) before | worst after |
|---|---|---|---|---|
| Transition d128/d256/d64 n4/n2, 96x96..40x40 | 3.2x | 1.00 | 11.6 / 6.3 / 20.0 | 1.24 / 1.28 / 1.18 |
| ConditionedTransition d768 (A3 L128), atom d128 | 5.6 / 1.0 | 1.00 | 5.6 / 2.4 | 1.00 / 1.00 |
| AdaLN d768 / d128 | 2.6 / 1.0 | 1.00 | 21.3 / 2.6 | 1.00 / 1.00 |
| AugmentedAttentionPairBias d768 | 3.5 | 0.99 | 4.1 | 1.05 |
| AttentionPairBias | 2.2 | 1.14 | 3.1 | 1.34 |
| DiT block d768 | 5.4 | 1.00 | 5.4 | 1.01 |
| BiasOnlyDiT block | 5.6 | 1.00 | 5.6 | 1.04 |
| LocalDiT AF3 | 1.1 | 1.01 | 2.8 | 1.19 |
| Boltz-2 token stack / atom stack | 1.05 | 1.00 | 1.8 / 2.0 | 1.03 / 1.11 |
| TriMul out/in, TriAttn start/end, OPM, PWA (fp32) | 1.00 | 1.00 | 1.00 | 1.00 (fp32 is served by the PyTorch statements; no Triton fp32 path) |

Not measured here: SWA DiT (its fp32 kernels already use a measured mix with tf32x3 for the FFN; its test holds the fused block to 1.5x of the per-op path).
No family remains above 1.5x.

## Ladder (ms, CUDA graph, 3 repeats, H100) -- base e6a635d7 vs this branch merged with integ/ladder-all2 (portable column)

| group | mode | L | portable before | portable after | after/before | portable/PyTorch after |
|---|---|---|---|---|---|---|
| tr_fp32_D128 | inf | 384 / 768 | 0.4152 / 1.5785 | 0.4710 / 1.7609 | 1.13 / 1.12 | 0.56 / 0.54 |
| tr_fp32_D128 | train | 384 / 768 | 1.7913 / 6.8564 | 1.8659 / 7.1501 | 1.04 / 1.04 | 0.79 / 0.78 |
| tr_fp32_D256 | inf / train | 384 | 1.1034 / 4.8643 | 1.1997 / 5.0654 | 1.09 / 1.04 | 0.71 / **1.026 (violation)** |
| aat_fp32 | inf / train | 384 | 0.2715 / 5.1110 | 0.2712 / 5.1767 | 1.00 / 1.01 | 0.80 / 0.84 |
| dit_fp32 | inf / train | 384 | 0.3742 / 8.0158 | 0.3837 / 8.0908 | 1.03 / 1.01 | 0.81 / 0.88 |
| dit_xbias_fp32 | inf / train | 256 | 0.1764 / 0.6314 | 0.1845 / 0.6616 | 1.05 / 1.05 | 0.90 / 0.96 |
| local_fp32 | inf / train | 512 | 0.3624 / 10.524 | 0.3738 / 10.494 | 1.03 / 1.00 | 0.70 / 0.84 |
| bodit_fp32 | inf 256 / 512 | | 0.1393 / 0.2340 | 0.1498 / 0.2479 | 1.08 / 1.06 | 0.70 / 0.57 |
| bodit_fp32 | train 256 / 512 | | 3.7016 / 7.1520 | 3.7438 / 7.1797 | 1.01 / 1.00 | 0.97 / 0.99 |
| adaln_fp32 | inf / train | 384 | 0.02438 / 0.5873 | 0.02554 / 0.5958 | 1.05 / 1.02 | 0.74 / 0.93 |
| ct_fp32 | inf / train | 384 | 0.10835 / 2.7992 | 0.11206 / 2.8206 | 1.03 / 1.01 | 0.82 / 0.98 |

Cost of correct rounding: 0-13 % on the portable rung (largest on the Transition inference GEMMs, which the extra activation pass and the rounded `h` store lengthen), 0-5 % elsewhere.
Rule check (hand-CUDA < portable < PyTorch): every row passes except
* `tr_fp32_D256` training: portable 5.065 vs PyTorch 4.939 ms (ratio 1.026). Base measured 4.864 vs 4.944 (0.985, and 1.03 in the Transition record's own earlier run): the row is at the noise edge and the rounding adds 4 %. OPEN.
* `dit_xbias_fp32` inference L256: CUDA 0.196 vs portable 0.185 (cuda_vs_portable, present on the base too: 0.216 vs 0.176).
Portable inference ladder rows that were far below PyTorch got slower by at most 13 % and stay below it.

## Tests
`tests/gpu/any/test_portable_fp32_accuracy_gpu.py` (new, 37 cases, with `tests/tf32_accuracy.py`): all pass. Related GPU files (tests/gpu/any + sm90 for Transition, CT, AdaLN, AugAttn, DiT, BiasOnlyDiT, LocalDiT, SWA DiT,
token DiT, window attention, test_numerical; 28 files finished before the job's time limit): all pass except (all also failing on the base, same causes as in portable_transition_ladder_20261025.md):
`test_numerical` x10 (three `*_sm90_cuda` transition rows -- cuBLASDx headers; seven mpnn kernels), `test_dit_variant_train_h100_gpu` x30 (`dit_variants_train_rows_sm90a` fails to build in this environment, hand-CUDA).
Not reached by the time limit: test_h100_token_dit_layouts, test_token_dit_psb, test_transition_*_sm90a (4), test_attention_pair_fused, test_pair_embed_adaln_gpu (sm90, hand-CUDA files; Transition/CT/AdaLN Triton files above all pass).
`test_transition_fp32_portable_gpu` failed twice (d192, d256 inference) in that run: the in-kernel A rounding bug above; fixed, 61 passed on re-run together with the residual tests and the accuracy file.
`dev derive --arch sm86` (units.json) and the sweep page regenerated; `_grid_many_kernel` is listed as an un-autotuned launch (it has no tiles to tune). Autotune CSV/JSON caches not rebuilt: the H100 entries of the changed kernels are stale
by source identity (cache-miss path: bounded search, `autotune_miss_cap`); the ladder numbers above include that.
