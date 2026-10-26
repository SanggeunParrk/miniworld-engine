# Near-miss ladder rows on H100: before / after (2026-10-26)

Branch `perf/final-nearmiss` (on `integ/ladder-all` e6a635d7). Ladder harness `benchmarks/runners/ladder.py run --repeats 3`, H100 80GB HBM3, torch 2.13, CUDA-graph replay of the compiled / uncompiled step (faster regime per rung). Ratio = CUDA / portable.
Before = out_base (this tree, e6a635d7, with the DiT build fix applied); after = out_final (commit 46f56f99).

| row | CUDA before ms | portable ms | ratio before | CUDA after ms | ratio after |
|---|---|---|---|---|---|
| Transition training bf16 D512 L384 | 5.920 | 5.935 | 0.997 | 5.607 / 5.622 / 5.789 (3 runs) | 0.942 / 0.951 / 0.953 |
| pair_embed_ca inference B16 L128 | 0.057 | 0.060 | 0.958 | 0.055 | 0.934 |
| pair_embed_ca inference B16 L256 / L384 | 0.190 / 0.408 | 0.209 / 0.456 | 0.908 / 0.895 | 0.183 / 0.395 | 0.882 / 0.868 |
| DiT Chai-1 token fp32 training | not measurable (CUDA rung failed to build) | 6.310 | n.a. | 5.847 | 0.929 (already < 0.95; no kernel change) |
| token_pair_init inference B16 L256 / L384 | 0.207 / 0.441 | 0.216 / 0.448 | 0.958 / 0.985 | 0.207 / 0.442 | 0.957 / 0.970 (unchanged, noise) |
| pool_atoms 24576 | 0.051 | 0.056 | 0.906 | 0.051 | 0.905 (already fine) |
| AttentionPairBias pair256 training | 0.335 | 0.369 (portable / PyTorch 0.872) | 0.908 | 0.336 | 0.912 (already fine in the ladder; see below) |

## What changed
* `transition/cuda/fused_wide_sm90a.py`: D512 n=4 training retains the forward's a / b (split forward + dh-only gate backward, previously only the 768x768 pair) also at the 384x384 pair. Kernel-time probe 5.34 -> 5.18 ms. Errors vs fp64 (module level, bf16): out 2.8e-3, dx 3.8e-3, dW 3.4-4.3e-3 against PyTorch bf16 4.1-5.8e-3 (ratio <= 0.89; recompute path was 3.3-3.9e-3). Test added. The tma / compact / split sub-switches were A/B'd: all-on is fastest (others 5.4-6.2 ms).
* `pair_embed_geom/interface.py`: `pair_embed_ca` without a needed gradient calls a new forward-only op (no codes written, table transpose + cast in one kernel instead of two). Bitwise equal to the training forward (test added).
* `dit_variants/cuda/train_rows.cu`: local `transpose_k` renamed `dv_transpose_k`. In the integrated tree `dit_variants_train_rows_sm90a` did not build (ambiguous with `tdr::transpose_k`), so every CUDA DiT-variant rung failed. The lead has an equivalent fix in `integ/ladder-all2`: take theirs on merge.

## Not moved
* Transition D512 margin is ~0.95 and run-to-run noise (portable 5.91-6.07 ms) is about as large as the margin: not a clear <= 0.95. Remaining kernel time is expand_saved 1.25 ms and gate_saved 1.22 ms (HBM-bound by the a/b/h traffic); a deeper epilogue rewrite was not attempted.
* token_pair_init L256 0.957 / L384 0.970: kernel 180 us vs ~165 us floor at HBM peak; glue (7 small launches, ~15 us) is shared by both rungs so removing it moves the ratio by 0.003 only. Not changed.
* APB pair256 training: in this tree the ladder (faster regime per rung) gives portable / PyTorch 0.870; only the compiled-only regime (0.955 in the earlier record) stays near 0.95, not a ladder row. Not changed.
* DiT Chai-1 fp32 training 0.925-0.929 is already below 0.95; the SwiGLU epilogue fusion was not needed.

## Tests (H100, all green)
transition (9 files), pair_embed_geom / adaln (4 + 2 files), pair_ops_portable_vs_cuda, dit_variant (3 files), token_pair_init: 0 failures. `tests/cpu/registry`: 2 failed + 2 errors identical to the base tree (stale committed caches of other kernels).
