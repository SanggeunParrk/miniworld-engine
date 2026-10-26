# H100 hand-CUDA LayerNorm (pair) and RMSNorm against the portable rung (2026-10-26)

Branch `perf/ladder-fix-norms-cuda`, from `perf/portable-vs-pytorch-c` (da8655ab, the faster portable kernels). Ladder harness
(`ladder.py run --group '^(ln_|rms_)' --repeats 3`, CUDA-graph replay, median of 3), H100 80GB HBM3, torch 2.13 + cu129. Before = da8655ab (run `base`),
after = this branch (run `after3`). Times in us. c/p = CUDA / portable, must be < 1.

## Before / after

| op | mode | shape | PyTorch best | portable | CUDA before | c/p before | CUDA after | c/p after |
|---|---|---|---|---|---|---|---|---|
| LN pair bf16 D128 | inf | L384 | 30 | 31 | 32 | 1.035 | 32 | 1.034 (unchanged: the Triton forward, see below) |
| LN pair bf16 D128 | inf | L768 | 106 | 104 | 107 | 1.028 | 107 | 1.028 (same) |
| LN pair bf16 D128 | train | L384 | 94 | 82 | 103 | 1.261 | 81 | 0.995 |
| LN pair bf16 D128 | train | L768 | 299 | 271 | 358 | 1.320 | 281 | 1.039 |
| LN pair fp32 D128 | inf | L384 | 56 | 55 | 56 | 1.008 | 56 | 1.010 (same) |
| LN pair fp32 D128 | train | L384 | 157 | 140 | 267 | 1.905 | 145 | 1.031 |
| LN pair bf16 D256 | inf | L384 | 56 | 56 | 56 | 1.008 | 57 | 1.018 (same) |
| LN pair bf16 D256 | train | L384 | 163 | 143 | 163 | 1.139 | 150 | 1.041 |
| RMSNorm bf16 | inf | x[48,384,768] | 24 | 24 | 31 | 1.322 | 25 | 1.062 |
| RMSNorm bf16 | inf | x[48,1024,768] | 56 | 56 | 68 | 1.213 | 57 | 1.013 |
| RMSNorm bf16 | train | x[48,384,768] | 69 | 61 | 73 | 1.192 | 66 | 1.090 |
| RMSNorm bf16 | train | x[48,1024,768] | 167 | 140 | 156 | 1.113 | 147 | 1.044 |

Floor reference (amortized, 10 launches per graph, same GPU): plain `copy_` of the input = 27.1 us (LN D128 bf16 L384), 100.9 (L768), 51.6 (fp32 D128 L384 and D256 bf16 L384), 19.7 (RMS L384), 51.5 (RMS L1024); the Triton forwards measure 27.2 / 100.7 / 52.0 / 52.4 / 19.2 / 55.5, i.e. the inference rows are floor-limited (the ladder adds ~4 us of graph launch to all of them).

(Run-to-run: a repeat of the whole set (`after2`, same code but the cp.async forward also in LN inference) gave the same c/p within 0.01 on every training / RMSNorm
row.) **Not fixed: no row reached c/p <= 0.95, and 11 of the 12 rows are still above 1.0** (one is 0.995, a tie). Every row improved or stayed equal against the CUDA
rung before; the large ones are LN training (1.26 -> 0.995 at L384, 1.32 -> 1.04 at L768, fp32 1.91 -> 1.03) and RMSNorm (1.32 -> 1.06, 1.21 -> 1.01 inference).

## What changed

New `kernels/layernorm/cuda/pipe/` (`ln_pipe.cu`, `__init__.py`; one extension, bf16 / fp16 / fp32, rows of 144-2048 bytes in 16-byte chunks), reached from
`layernorm_kernel` (training only) and `rmsnorm_cuda` on sm90, not under `force_portable_kernels` and not under `torch.use_deterministic_algorithms`:

- a warp owns a row (TPR 16 or 32 lanes x CH 16-byte chunks) and streams its rows through a per-warp cp.async ring (3-4 stages): every lane reads back what it copied, no
  CTA barrier; persistent grid; one atomic per column per CTA, 4 accumulator copies, the last CTA converts to the weight dtype and clears the workspace:
  **one backward kernel** (the old rung ran the Triton forward + `layer_norm_bwd_main_kernel` + two column-reduce kernels + `.to()` casts, or the general `mwln::` kernels).
- the training forward stores (mean, rstd) per row (8 bytes) so the backward is one reduction pair per row (a first version that recomputed them was 10-20% slower in
  the backward: three dependent shuffle reductions per row at 12-25% occupancy).
- weights / biases are read in their own dtype (no cast kernel); single-chunk rows (D128 / D256 bf16, D128 fp32) use a non-persistent forward (the ring is 5-8% slower
  there: all warps in lock step).
- the weight-gradient atomics are not bit-reproducible between replays (fp32 rounding); `torch.use_deterministic_algorithms(True)` keeps the general bit-reproducible kernels.

## Where the time goes (amortized probes, 10 launches per graph, outside the checkout `~/miniworld-engine-scratch/lf-normscuda/probe`)

- Backward without the weight gradient (no atomics, no last-CTA) is **at the HBM floor and faster than the portable backward**: 40.7 / 82.7 / 83.0 / 30.5 / 79.6 us
  (LN D128 bf16 L384, fp32, D256, RMS L384, L1024). With the weight gradient 45.8 / 85.7 / 88.4 / 37.3 / 85.9. The 5-7 us difference is the epilogue
  (reduce over warps in shared memory, atomics, fence, ticket, last CTA, a chain of ~4 global round trips after the slowest CTA); accumulator copies took it from 18+ us (all CTAs on
  the same 256-768 addresses) to this; 4 copies beat 8 / 16 / 32 / 64. This tail is what keeps training at c/p 1.03-1.09, and it is the thing to attack next
  (a PDL-chained reduce kernel, or a cluster / DSMEM reduction).
- Forward: a TMA-bulk-load + STG copy and a pure TMA load/store copy both sit at 55 us for 151 MB, a plain `copy_` at 51.5 (it is not an SM kernel); the Triton LN forwards reach
  52-54 us at the same bytes, so they are at the practical floor and the cp.async / direct CUDA forwards (28.6-29 us at D128 L384 against 27.2, 52.0 fp32 = tie, 54 D256 against 52.4)
  cannot beat them by more than the 1-5% they currently lose. For that reason the no-grad LayerNorm forward stays on the Triton kernel (as before) and the CUDA rung is the portable
  kernel there; RMSNorm no-grad uses the cp.async forward (20.3 vs 19.2 us amortized, ladder 25 vs 24), which is 20-25% faster than the `mwln::ln_fwd_kernel` it replaces but not at parity.
- Tried and not kept: PU > 1 passes per stage, 4 / 16 warps per CTA, fewer CTAs per SM, deeper rings (16 stages), a non-persistent direct kernel for RMS rows (26 us), recomputed statistics.

## Tests

- New `tests/gpu/sm90/test_norm_pipe_gpu.py` (fp64 reference, bf16 / fp16 / fp32, LN and RMS, widths 72-1024 incl. padded 152, ragged row counts, fp32 weights under bf16 activations,
  partial `requires_grad`, backward twice, two forwards before one backward, CUDA graph, `torch.compile`, routing). 
- GPU (H100), whole files, run to the end: layernorm_linear stats, backward tiles, general cuda, force-portable, dispatch cache, rmsnorm general / checker, tiled offset, atomic gradients,
  determinism, op contracts, whole-op reachable, new pipe test: 351 passed, 3 failed in the first long run (`test_output_all_gradients[False-128-16-3d-dt0 / dt2]`, and
  `test_rmsnorm_general_cuda_gpu::test_cuda_graph_and_compile` which was a real bug of the first version: `data_ptr() % 16` under Dynamo, fixed). The two layernorm_linear cases passed
  in three later runs of the same file (isolated, with the rmsnorm / pipe files, and in the final run: 333 and 176 passed); they pass on the base commit; not reproduced, cause unknown (it was the first file of a
  cold-cache run that shared the node with a derive job). Final runs: 333 passed / 4 skipped, then 176 passed after the last routing edit.
- CPU suite: 4326 passed, 4 failures + 2 errors, all present on the base commit (`key_flags_are_built_on_both_sides` mpnn DROPOUT, `no_stale_caches`, two `a_ladder_offers_what_wins` errors), plus
  `test_compact_search_space` and the three `the_sweep_page` tests which fail when `registry_kernel.units.json` is stale: re-derived (`MINIWORLD_COMPILE_WRAP=disable dev derive --arch sm86`) and passing.
  (the sweep page tests were re-run on the final tree only through the cpu2 subset, which passed.)

## Open

- c/p stays 1.01-1.09 on every row but one: the weight-gradient epilogue (above) and, for the forwards, the HBM floor shared with Triton.
- The no-grad LayerNorm rung is the Triton forward (n.a. in spirit; unchanged from before).
- Portable LN D128 L384 training in `after1` failed once with a FileNotFoundError in the harness (jit cache); the `after3` run shows it normally.
- No H100 cache rebuild; rows under 144 bytes or over 2048 bytes keep the general kernels.
