# Portable RMSNorm and pair LayerNorm vs PyTorch on H100 (2026-10-25)

Branch `perf/portable-vs-pytorch-c`, based on `perf/ladder-audit` (ladder run at 12ae4ac8). Portable = `kernels/<family>/triton`, reached on H100 through
`settings.force_portable_kernels`. Ratio = portable ms / best PyTorch ms (graph-eager or `torch.compile`); must be < 1. Times in microseconds, CUDA-graph replay,
median of 3 (ladder harness, `ladder.py run --group '^(ln_|rms_)'`). Before = out_full_B; after = three runs of the final code (r1 earlier code, r2, r3 = committed code).

## Before / after (us)

| op | mode | shape | PyTorch best | portable before | ratio before | portable after (r3) | ratio after |
|---|---|---|---|---|---|---|---|
| RMSNorm bf16 | inf | x[48,384,768] | 24.4 | 28.5 | 1.165 | 23.4 | 0.959 |
| RMSNorm bf16 | inf | x[48,1024,768] | 56.2 | 79.9 | 1.421 | 56.0 | 0.995 |
| RMSNorm bf16 | train | x[48,384,768] | 68.6 | 84.4 | 1.230 | 60.8 | 0.887 |
| RMSNorm bf16 | train | x[48,1024,768] | 165 | 211.8 | 1.281 | 139.8 | 0.848 |
| LN pair bf16 D128 | inf | L384 | 30.5 | 31.7 | 1.036 | 30.7 | 1.005 (not fixed) |
| LN pair bf16 D128 | inf | L768 | 106.3 | 107.2 | 1.010 | 104.3 | 0.981 |
| LN pair bf16 D128 | train | L384 | 93.5 | 87.6 | 0.935 | 82.5 | 0.882 |
| LN pair bf16 D128 | train | L768 | 300.5 | 295.6 | 0.984 | 270.1 | 0.899 |
| LN pair fp32 D128 | inf | L384 | 55.5 | 57.4 | 1.034 | 55.1 | 0.994 |
| LN pair fp32 D128 | train | L384 | 156.3 | 137.5 | 0.878 | 140.3 | 0.898 |
| LN pair bf16 D256 | inf | L384 | 56.2 | 58.6 | 1.040 | 55.8 | 0.993 |
| LN pair bf16 D256 | train | L384 | 162.7 | 174.9 | 1.072 | 144.4 | 0.887 |

r2 (same code except pipelining guard) gave the same verdicts except RMSNorm inf L1024 = 1.0006 (56.2 vs 56.2): that row and LN D128 L384 inf are ties at the HBM floor.
LN D256 inf (0.993), fp32 inf (0.994), RMS L1024 inf (0.995) win by under 1%, which is inside run-to-run noise (PyTorch itself moves 24.5-25.2 us between processes).

## What changed

- `layer_norm_bwd_dx_fused`, `rmsnorm_bwd_kernel`: persistent grid (SMs x 8 programs, capped at the tile count), tile loop software-pipelined (`tl.range(num_stages)`,
  1 stage when the x/dy tile exceeds 8192 elements, to stay inside shared memory), dweight/dbias accumulated in registers and flushed with one atomic per program
  instead of one (two for LN) per row tile. Old launches (one program per tile, still used by `compile_native.py`, `layernorm_linear/autograd.py`) stay correct.
- `rmsnorm_fwd_kernel`: covering tile is persistent + pipelined.
- `layer_norm_fwd_fused`: `STORE_STATS` constexpr (default True). The no-grad forward (`ctx.needs_input_grad` all False) skips the mean/rstd stores (1.5% of the bytes at D=128);
  `_ln_fwd` returns empty (0,) stats then. Not in the autotune key, recorded in `build/key_gaps_allowed.csv`.
- `rsqrt` instead of `1/sqrt` in the forwards (what Inductor emits).
- Default config CSVs of the four ops: at most 24 rows. Above the miss cap (`autotune_miss_cap` = 24) a stale or missing cache gets `heuristic_subset`, which prefers warps 4/8 and
  middle tiles and dropped the winners (1-2 warps, 2-16 rows). Stage-duplicate rows (3-6 stages; the loops have no global-stage dependence) and rows with BLOCK_M1*BLOCK_K > 16384 were
  removed, the covering small-row tiles added. No kernel code was deleted.
- `registry_kernel.units.json` re-derived (`MINIWORLD_COMPILE_WRAP=disable dev derive --arch sm86`; `registry_kernel.csv` unchanged).

Why the inference rows tie: single-kernel row norms are bandwidth bound and PyTorch's Inductor kernel is already at the copy floor (`y.copy_(x)`: 31.2 us at LN D128 L384, 55.3 us
at RMS L1024; PyTorch 30.9 / 56.0). Cache hints (evict_first, .cg, .cs), unmasked loads, row counts 1-32, warps 1-8 and persistent pipelining were swept in probes outside the checkout;
none moved the kernel more than 1% off the floor.

## Tests

- CPU suite: 4330 passed, 2 failed, 2 errors, all four also fail on the unmodified tree (`key_flags_are_built_on_both_sides` mpnn DROPOUT, `no_stale_caches`, two `a_ladder_offers_what_wins` setup errors).
  `no_stale_caches` now additionally lists the H100 caches of `layernorm_fwd_saveact_triton`, `layernorm_bwd_atomic_triton`, `rmsnorm_fwd_triton`, `rmsnorm_bwd_triton` (op_identity changed) and the sm86 `rmsnorm_fwd_triton`: rebuild + `dev merge` is pending (not run).
- GPU (H100): layernorm/rmsnorm parity files (backward tiles, tiled offset, atomic gradients, rmsnorm checker, linear stats, mixed fallback, force-portable, dispatch cache, op contracts, whole-op) pass; the 4
  `test_adaln_matches_the_cuda_path` failures (`pair_embed_adaln_bwd` custom-op aliasing) also fail on the unmodified tree. No rtol was loosened.
  First version failed `test_atomic_backward_mask_and_residual[256-...-fp32]` with OutOfResources (pipelined 64x256 fp32 tile > 227 KB smem); fixed by the 8192-element guard.

## Finding for the CUDA owner (not changed here)

Training LN pair, bf16 D128: the H100 CUDA rung runs `layer_norm_fwd_fused` (the same Triton forward) plus `layer_norm_bwd_main_kernel` + reduce:
67+6 us at L384 and 244+6 us at L768, against 46 us / 160 us for the portable persistent backward (ladder r2 profile). CUDA/portable = 1.25 / 1.32. D256 L384 training: 163 vs 144 us.
RMSNorm: the CUDA rung (`mwln::ln_fwd_kernel`) is now slower than portable in inference (31.1 vs 23.4 us, 67.9 vs 56.0) and in training at L384 (72.0 vs 60.8) and L1024 (155.4 vs 139.8), so these rows now show `cuda_vs_portable` violations that did not exist when portable was slower.

## Open

- LN pair bf16 D128 L384 inference: 30.7 vs 30.5 us (1.005), not fixed.
- H100 tuned-cache rebuild for the four ops.
