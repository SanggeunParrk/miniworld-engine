# LayerNorm / RMSNorm ladder rows: what is at the floor and what was not (2026-10-27)

Branch `perf/final-norms`, from `integ/ladder-all2` (23009c44). Ladder harness (`ladder.py run --group '^(ln_|rms_)' --repeats 5`, CUDA-graph replay, `triton.testing.do_bench` with its
L2 flush, median of 5), H100 80GB HBM3, torch 2.13 + cu129. Times in us (the ladder adds ~4 us of graph launch to every row). c/p = CUDA / portable, p/t = portable / best PyTorch
(`torch.compile` + graph, which is also the fastest of the three PyTorch statements on every row; eager `F.layer_norm` is 5x slower at these row widths). Before = run `base`
(integ/ladder-all2 as is), after = run `after2` (this branch, final code).

## Before / after

| op | mode | shape | PyTorch | portable | CUDA before | c/p before | CUDA after | c/p after | p/t |
|---|---|---|---|---|---|---|---|---|---|
| LN pair bf16 D128 | inf | L384 | 30.8 | 31.1 | 33.1 | 1.068 | 31.2 | 1.004 | 1.010 |
| LN pair bf16 D128 | inf | L768 | 106.5 | 105.0 | 108.9 | 1.042 | 105.4 | 1.004 | 0.986 |
| LN pair fp32 D128 | inf | L384 | 55.7 | 55.4 | 56.2 | 1.016 | 56.1 | 1.014 | 0.994 |
| LN pair bf16 D256 | inf | L384 | 56.5 | 56.2 | 57.6 | 1.029 | 56.3 | 1.003 | 0.995 |
| RMSNorm bf16 | inf | x[48,384,768] | 24.5 | 23.7 | 25.0 | 1.052 | 23.7 | 0.999 | 0.966 |
| RMSNorm bf16 | inf | x[48,1024,768] | 56.3 | 56.3 | 57.2 | 1.012 | 56.1 | 0.996 | 1.001 |
| LN pair bf16 D128 | train | L384 | 94.1 | 81.9 | 79.3 | 0.970 | 78.0 | 0.952 | 0.871 |
| LN pair bf16 D128 | train | L768 | 299.6 | 271.4 | 269.8 | 0.996 | 267.5 | 0.985 | 0.906 |
| LN pair fp32 D128 | train | L384 | 156.9 | 140.6 | 140.2 | 0.998 | 140.2 | 0.997 | 0.896 |
| LN pair bf16 D256 | train | L384 | 164.0 | 143.6 | 141.9 | 0.988 | 141.1 | 0.983 | 0.875 |
| RMSNorm bf16 | train | x[48,384,768] | 69.6 | 60.7 | 61.5 | 1.013 | 60.1 | 0.990 | 0.872 |
| RMSNorm bf16 | train | x[48,1024,768] | 167.5 | 140.3 | 141.9 | 1.012 | 140.5 | 1.001 | 0.837 |

Rule "CUDA < portable < PyTorch" (ratio < 1.0): after the change 7 of 12 rows have c/p < 1 (all four LN training rows, RMSNorm inference x2, RMSNorm training L384). Five rows are ties inside
the run-to-run spread (c/p 1.001 - 1.014: LN pair inference x4 and RMSNorm training L1024). Bar "clearly faster" (<= 0.95): no row reaches it (best: LN pair bf16 D128 L384 training 0.952). p/t: portable is below PyTorch on 10 of 12 rows,
below 0.95 only in training; inference p/t is 0.966 - 1.010 (RMSNorm L1024 1.001, LN D128 L384 1.010: ties at the floor).

## Why the inference rows cannot be made a win (measured, not argued)

The PyTorch row is not fused, cached or mis-timed: it is the compiled `triton_per_fused_native_layer_norm` / `..._rms_norm` kernel (one kernel node; the ladder's kernel list shows it), the harness
flushes L2 (256 MB) between replays, and the bandwidth is below the HBM peak (L384 D128: 75.5 MB in 30.8 us = 2.45 TB/s; L768: 302 MB in 105 us = 2.9 TB/s). Plain copies sit at the same
time (probe, same harness, 7 buffer placements, us): `copy_` 30.98 / 104.70 / 55.44 (D128 L384 / D128 L768 / D256), hand-written 16-byte copy kernels 30.71 - 31.07 / 104.7 - 105.5 / 55.5 - 55.6,
portable 30.97 / 104.69 / 55.68, compiled PyTorch 30.87 / 106.19 / 56.05. LayerNorm moves exactly the bytes of a copy, so the best any kernel can do is the copy time; every rung is within
1.5% of it, and the sub-1% orderings move with where the buffers land. Call-level fusion cannot change these rows (they time the standalone op); where a consumer is in this repo the fused
forms already exist (`layernorm_linear`, the trimul front, the transition b2b kernel, `rmsnorm_adamod`, the adaLN kernels) and are measured under their own ladder groups.

## What changed (`kernels/layernorm/cuda/pipe/ln_pipe.cu` only, plus a test and this record)

1. `direct2_fwd`: the single-chunk (rows of 256 / 512 bytes) forward for exact-fit widths whose weight / bias have the activation dtype. The existing `direct_fwd` carried per-element weight-dtype
   branches and needed 56 registers (`cuobjdump --dump-resource-usage`), so an SM held 4 resident 8-warp CTAs (50% occupancy); the new kernel is a plain vector-load / reduce / vector-store kernel at 40
   registers, 2 row passes per warp, 4 warps per CTA. Probe (interleaved with portable and PyTorch, 7 buffer placements): bf16 D128 L384 32.3 -> 31.4 us, L768 108.6 -> 105.3, D256 56.6 -> 55.95.
   The training forward (it stores mean / rstd) uses it too. Mixed dtypes (fp32 weights under bf16 activations), padded widths and absent affine keep the generic kernel.
2. Six cp.async stages for 96-chunk rows (`MWP_S3`, was 3): the persistent RMSNorm d768 bf16 forward goes 24.8 -> 23.8 us at L384 and 56.8 -> 56.0 at L1024; training step L384 60.8 -> 59.4 in the probe.

## Tried and not kept

- Register caps on the generic direct forward (`__launch_bounds__` min blocks 5-8: 42 registers and below spill, 2.5x slower), 2 / 4 warps per CTA, 1 / 2 / 4 row passes, vector affine loads: 1 - 3% at best,
  superseded by `direct2_fwd`.
- Load / store hints on `direct2_fwd`: `ld.global.nc.L1::no_allocate`, `L1::evict_first`, `st.global.cs`, L2 `evict_first` / `evict_last` policies on x and y. **L2 `evict_last` on x looked like a win
  (-1.0 .. -1.6% on all four LN rows) and is a benchmark artifact**: lines marked evict_last survive the harness's 256 MB flush, so the next replay reads part of x from L2. With six distinct inputs per graph
  (no x read twice within 6 ops; probe `rotsweep.py` in `~/miniworld-engine-scratch/norms/probe`, outside the checkout) the advantage is gone (1.003 / 0.999 / 1.000 / 0.994 of portable). Rejected; nothing in the tree uses cache policies.
- Persistent ring forward for the single-chunk widths (per_sm 1/2/4): 1 - 5% slower than the direct kernels. 4 / 16 warps and 2 - 8 stages on the 32-lane rings: no gain beyond the S3 change.
- Not done: removing the 2 - 2.5 us weight-gradient reduce launch of the training step. A last-CTA ticket was measured by the previous agent (5 - 7 us tail, not bit-reproducible); a cooperative
  grid barrier costs about the same as the PDL-chained reduce; a cluster / DSMEM pre-reduction still needs the cross-cluster kernel.

## Tests

- `tests/gpu/sm90/test_norm_pipe_gpu.py`: new `test_no_grad_forward_all_affine_forms` (no-grad forward, widths 128 / 256 / 64 / 72 / 768, bf16 / fp16 / fp32, LN and RMS, weights of the activation dtype, weight
  only, bias only, none, fp32 weights, row counts 1 .. 40000 incl. partial CTAs). File: 73 passed, 10 skipped.
- GPU (H100), whole files on the final code: layernorm_linear stats 52, determinism 12 (+1 skipped), backward tiles 56, layernorm general cuda 64, force-portable 8, norm dispatch cache 4, rmsnorm general 55,
  rmsnorm checker 16, tiled offset 18, atomic gradients 12, whole-op reachable 4, mixed fallback 4: all passed; `test_op_contracts_gpu.py` 5 passed (38 min, cold JIT cache). No rtol touched.
- CPU suite: 4368 passed, 2 failed, 2 errors, the same four as on the base commit (`key_flags_are_built_on_both_sides` mpnn DROPOUT, `no_stale_caches`, two `a_ladder_offers_what_wins` setup errors).
  The registry does not list the new kernel (no `registry_kernel.units.json` change).

## Open

- LN pair inference (four rows) and RMSNorm training L1024 are ties at the copy floor; the portable fp32 LN forward is 1% below every CUDA variant tried (no reason found).
- Portable RMSNorm L1024 inference vs PyTorch 1.001: the Triton tuner picks a 4-row config (57 us) on some runs where 1 - 4 row configs give 55.9; not touched (Triton side is another owner's).
