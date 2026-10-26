# H100 hand-CUDA LayerNorm (pair) and RMSNorm against the portable rung (2026-10-26)

Branch `perf/ladder-fix-norms-cuda`, from `perf/portable-vs-pytorch-c` (da8655ab, the faster portable kernels). Ladder harness
(`ladder.py run --group '^(ln_|rms_)' --repeats 3`, CUDA-graph replay, median of 3), H100 80GB HBM3, torch 2.13 + cu129. Before = da8655ab (run `base`),
after = this branch (run `after4`). Times in us. c/p = CUDA / portable, must be < 1. The ladder adds ~4 us of graph launch to every row.

## Before / after

| op | mode | shape | PyTorch best | portable | CUDA before | c/p before | CUDA after | c/p after |
|---|---|---|---|---|---|---|---|---|
| LN pair bf16 D128 | train | L384 | 94 | 82 | 103 | 1.261 | 79 | **0.971** |
| LN pair bf16 D128 | train | L768 | 299 | 271 | 358 | 1.320 | 270 | 0.997 |
| LN pair fp32 D128 | train | L384 | 156 | 141 | 267 | 1.905 | 140 | 0.998 |
| LN pair bf16 D256 | train | L384 | 164 | 143 | 163 | 1.139 | 142 | 0.992 |
| RMSNorm bf16 | train | x[48,384,768] | 69 | 61 | 73 | 1.192 | 62 | 1.018 |
| RMSNorm bf16 | train | x[48,1024,768] | 167 | 141 | 156 | 1.113 | 142 | 1.009 |
| LN pair bf16 D128 | inf | L384 | 31 | 31 | 32 | 1.035 | 33 | 1.071 |
| LN pair bf16 D128 | inf | L768 | 106 | 104 | 107 | 1.028 | 109 | 1.042 |
| LN pair fp32 D128 | inf | L384 | 56 | 55 | 56 | 1.008 | 56 | 1.016 |
| LN pair bf16 D256 | inf | L384 | 56 | 56 | 56 | 1.008 | 58 | 1.030 |
| RMSNorm bf16 | inf | x[48,384,768] | 24 | 24 | 31 | 1.322 | 25 | 1.053 |
| RMSNorm bf16 | inf | x[48,1024,768] | 56 | 57 | 68 | 1.213 | 57 | 1.013 |

**Not met: no row is at c/p <= 0.95** (best 0.971). Training rows went from 1.11-1.91 to 0.97-1.02 (four of six below 1.0); RMSNorm inference from 1.21-1.32 to 1.01-1.05.
The inference LN rows are slightly worse than before (the CUDA rung used to be the Triton forward itself; it is now the cp.async / direct CUDA forward, 2-4 us slower in the ladder).

### Inference rows are floor-limited (numbers only, copy_ next to them)

Amortized probe (10 launches per graph, same GPU, eager+CUDA graph), us; the ladder numbers above include ~4 us of launch:

| row | `copy_` of the input | portable (Triton) | CUDA (this branch) |
|---|---|---|---|
| LN bf16 D128 L384 | 27.1 | 27.2 | 28.8 |
| LN bf16 D128 L768 | 100.9 | 100.7 | 102.7 |
| LN fp32 D128 L384 | 51.6 | 52.0 | 52.1 |
| LN bf16 D256 L384 | 51.7 | 52.4 | 54.0 |
| RMS L384 | 19.7 | 19.2 | 20.3 |
| RMS L1024 | 51.5 | 55.5 | 56.1 |

A pure TMA bulk load + bulk store copy takes 55 us for the 151 MB rows (`copy_` 51.5 is not an SM kernel), so ~55 us is what a kernel gets at that size; Triton reaches 52-54 at the LN shapes,
which the CUDA forwards did not match (I could not find why). The CUDA forward is not routed to the Triton kernel anywhere.

## What changed

New `kernels/layernorm/cuda/pipe/` (`ln_pipe.cu`, `__init__.py`; bf16 / fp16 / fp32, rows of 144-2048 bytes in 16-byte chunks), reached from `layernorm_kernel` and `rmsnorm_cuda` on sm90
(not under `force_portable_kernels`):

- a warp owns a row (16 or 32 lanes x 1-4 chunks) and streams rows through a per-warp cp.async ring (3-4 stages); persistent grid; no CTA barrier in the main loop. Single-chunk rows
  (D128 / D256 bf16, D128 fp32) use a non-persistent forward (the ring is 5-8% slower there: warps in lock step). The training forward stores (mean, rstd) per row (8 bytes).
- backward: one kernel for dx and one partial weight-gradient row per CTA (plain stores), then a small reduce kernel launched behind it with programmatic dependent launch
  (`griddepcontrol`), which adds the rows in a fixed order and converts to the weight dtype: **bit-reproducible replays** (an earlier version with atomics + last-CTA ticket had
  a 5-7 us tail; PDL + plain stores cut it to ~2-3 us). The backward launches 1 CTA per SM for single-chunk rows (2-3 CTAs per SM were 5-10% slower: contention on the tail).
- weights / biases are read in their own dtype (no cast kernel); the old rung ran Triton forward + `layer_norm_bwd_main_kernel` + two reduces + casts, or the general `mwln::` kernels.

## Where the time goes (amortized probes, outside the checkout `~/miniworld-engine-scratch/lf-normscuda/probe`)

| backward, us | LN D128 L384 | LN D128 L768 | fp32 D128 | LN D256 | RMS L384 | RMS L1024 |
|---|---|---|---|---|---|---|
| without weight gradient (HBM floor) | 41.0 | 161.6 | 78.3 | 79.0 | 30.3 | 79.6 |
| with it (kernel + PDL reduce) | 43.6 | 164.8 | 81.2 | 81.5 | 35.1 | 84.6 |
| kernel only (reduce skipped) | 41.5 | - | - | - | 32.7 | 81.6 |

The remaining 2.5-5 us is the partial-row epilogue of the kernel (shared-memory fold of the 8 warps) plus the reduce kernel (~2.5 us even with PDL: one more graph node).
Portable's backward is at the same floor (portable train - forward: 46 / 160 / ~83 / ~86 / 35 / 79 us), so the training margin cannot exceed ~5% by removing the epilogue completely
(RMS L1024: 134.8 portable against 56.1 + 79.6 = 135.7 at the CUDA floor). Not tried: cluster launch with a DSMEM reduction of the partials (it still needs the reduce kernel for the
cross-cluster sum) and fusing the reduce into the next consumer (the optimizer is outside this library).
Tried and not kept: atomics into 4-64 accumulator copies + last-CTA ticket (5-7 us tail), 4 / 16 warps per CTA, 2-3 CTAs per SM in the backward, deeper rings, passes-per-stage > 1,
a non-persistent RMS forward (26 us), recomputing statistics in the backward (10-20% slower).

## Tests

- New `tests/gpu/sm90/test_norm_pipe_gpu.py` (fp64 reference, bf16 / fp16 / fp32, LN and RMS, widths 72-1024 incl. padded 152, ragged row counts, fp32 weights under bf16 activations,
  partial `requires_grad`, backward twice, two forwards before one backward, CUDA graph, `torch.compile`, bit-identical replays, routing).
- GPU (H100), whole files run to the end on the final code: layernorm_linear stats, determinism, backward tiles, layernorm general cuda, force-portable, dispatch cache, op contracts,
  rmsnorm general / checker, tiled offset, atomic gradients, whole-op reachable, new pipe test: **355 passed, 5 skipped, 0 failed**. No rtol loosened.
  Earlier in the work two `test_layernorm_linear_stats_gpu::test_output_all_gradients[False-128-16-3d-dt0/dt2]` cases failed once in a cold-cache run and passed in every later run
  (also on the base commit); a real bug found by `test_rmsnorm_general_cuda_gpu::test_cuda_graph_and_compile` (`data_ptr() % 16` under Dynamo) is fixed.
- CPU suite: the four failures / errors of the base commit only (`key_flags_are_built_on_both_sides` mpnn DROPOUT, `no_stale_caches`, two `a_ladder_offers_what_wins` errors);
  `registry_kernel.units.json` re-derived (`MINIWORLD_COMPILE_WRAP=disable dev derive --arch sm86`).

## Open

- Training c/p 0.97-1.02, inference 1.01-1.07: nothing at 0.95; the floors are shared with the portable rung.
- The CUDA forwards are 1-7% above the Triton forwards on the LN / RMS inference shapes.
- No H100 tuned-cache rebuild.
