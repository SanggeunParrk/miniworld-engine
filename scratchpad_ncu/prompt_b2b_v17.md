# cuda-kernel-optimizer skill — transition_b2b ROUND v17 (FIX the fused-stats reduction: warp-parallel, not 1-thread-per-row serial)

Follow `scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100; YOU
implement + write v17.md; DO **NOT** commit. Branch `b2b-cutlass-opt`. PTX allowed.

## Context: v16 (fused LN stats) REGRESSED the module path
The fused-stats kernel `transition_b2b_fwd_fused_stats` is CORRECT (module cos 0.999989, no spill) but
MODULE INFERENCE got SLOWER, not faster: L=1024 = **2.05 ms vs 1.30 ms** for the previous
stats_triton-based path (and 1.57 ms Triton). Root cause is the reduction implementation at
`transition_b2b_kernel.cu:419-447`:
```
if (wg_tid < WG_M) {          // only 64 of 128 threads active; other 64 idle at the barrier
    for (k=0..127) sum += sXn(m,k);          // SERIAL 128-elem scan by ONE thread
    for (k=0..127) sq += (sXn(m,k)-mean)^2;  // SECOND serial 128-elem scan
}
```
i.e. 64 threads each do 256 serial smem reads while 64 idle — a slow serial prologue on the critical
path (1 CTA/SM = 8 warps, can't hide). This dwarfs the (cheap, full-occupancy) stats_triton pass it
replaced.

## v17 experiment — make the fused-stats reduction WARP-PARALLEL (all threads active)
Replace the 1-thread-per-row serial scan with a warp-cooperative reduction so ALL 128 threads of the
warpgroup participate and each row is reduced by a full warp via shuffles:
- The warpgroup has 4 warps; WG_M=64 rows -> assign each warp 16 rows (warp w handles rows
  `w*16 .. w*16+15`, `w = warp_id_in_wg = wg_tid/32`, `lane = wg_tid%32`).
- For each of its 16 rows m: the 32 lanes cooperatively read the K=128 columns from `sXn` (each lane
  reads cols `lane, lane+32, lane+64, lane+96` = 4 values), accumulate a local fp32 `sum`; warp-reduce
  with `__shfl_xor_sync`/`__shfl_down_sync` (5 steps) -> `mean = total/128` (broadcast to all lanes via
  `__shfl_sync(..., 0)` or reduce-to-all with xor). Then SECOND pass: each lane reads the same 4 cols,
  accumulates `(x-mean)^2`, warp-reduce -> `var`. Lane 0 (or all lanes) computes
  `rstd = rsqrtf(var+eps)`, `c1 = mean*rstd` and writes `pRstd[cta_m]`/`pC1[cta_m]`
  (`cta_m = wg_id*WG_M + m`). Keep the TWO-PASS centered variance (parity with `stats_triton`; cos must
  stay 0.999989 at module level).
- Keep the existing `NamedBarrier` before/after so `sXn` is fully staged before the reduction and
  `pRstd/pC1` are visible before the normalization loop reads them.
- Coalescing: lane reads `sXn(m, lane + 32*j)` — 32 consecutive lanes hit 32 consecutive columns =
  conflict-free/coalesced smem access. (Do NOT reintroduce the swizzle here; `sXn` holds raw x
  row-major.)

This should cut the reduction from ~256 serial reads/thread (64 threads) to ~8 reads + 2 warp-shuffles
per thread (all 128 threads) — a large speedup, hopefully bringing module inference at/below the
stats_triton path.

## Alternative if warp-per-row is awkward
You may instead accumulate per-row partials DURING the existing vectorized `smem_load_u128` normalize
pass is NOT possible (stats must precede normalize). But you MAY do a single vectorized pre-pass: each
thread `smem_load_u128` its 8-wide slices (reusing the u128 path), accumulate partial (sum, sumsq) per
row into a small smem scratch `[WG_M]` via a tree/atomic-free reduction, then finalize. Choose whichever
is cleaner and genuinely parallel across all 128 threads; state which in v17.md.

## Keep everything else v16/v14
Both entries (`transition_b2b_fwd` precomputed + `transition_b2b_fwd_fused_stats`), the FUSE_STATS
template, expand SS / squeeze RS, swizzled STG.128 epilogue, gamma/beta register load, 2-WG coop,
`__launch_bounds__(256,1)`, sm_90a, bf16 (NO precision reduction). cos>=0.999. Watch ptxas: expect
regs ~255, spill 0 (the shuffle reduction adds a few regs — if it spills, note it).

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: replace ONLY the FUSE_STATS reduction block (lines ~419-447) with
   the warp-parallel reduction. Nothing else changes. `kernels/transition/cuda/__init__.py` already
   calls the fused-stats entry — leave it.
2. Write `docs/kernel-optimization/transition_b2b/v17.md`: the v16 regression analysis (serial
   reduction), the warp-parallel design (row/warp/lane mapping, shuffle steps, two-pass), Validation +
   Perf as TODO (I fill from H100 — watch: module cos 0.999989, module inference ms vs 1.30 (stats_triton)
   and vs 2.05 (v16), ptxas regs/spill, standalone verify precomputed entry unchanged).
- No GPU. Final message: the exact reduction scheme (row->warp->lane mapping, shuffle intrinsics used,
  two-pass), register impact, and correctness risks. Note honestly that even a fast fused reduction may
  not beat the full-occupancy stats_triton pass at 1 CTA/SM — the H100 numbers decide.
