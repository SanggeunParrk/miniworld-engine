# cuda-kernel-optimizer skill — transition_b2b ROUND v16 (FUSE LN STATS into the CUDA b2b kernel)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste; YOU implement + write v16.md draft; DO **NOT** commit. Branch `b2b-cutlass-opt`. PTX allowed.
**BEST = v14 (`ea5aa3f`): kernel ~421us = 1.29x of Triton; wired into the Transition module inference
(cuda_transition_b2b) where it gives ~1.20x end-to-end.** Kernel file currently IS v14.

## Why: close the 1.29x (kernel) -> 1.20x (module) gap
The module path `kernels/transition/cuda/__init__.py::cuda_transition_b2b` currently computes LN stats
with a SEPARATE `stats_triton(x2, eps)` kernel (a full extra pass over [M,128]) BEFORE calling the b2b
kernel, because the kernel takes precomputed `rstd`/`c1`. That extra pass is the main reason the module
win (1.20x) is below the kernel win (1.29x). The b2b kernel ALREADY stages the full K=128 x row into
smem (`sXn`) in its prologue — so it can compute the LN row-stats ON-CHIP and the separate stats pass
can be dropped entirely.

## Current prologue (for reference, ~`transition_b2b_kernel.cu:390-430`)
- `stage_x_into_sxn()` cp.async-stages the raw x tile into `sXn` (per-WG, WG_M=64 rows x kK=128).
- Lines ~409-414 load `rstd[row]`/`c1[row]` from GLOBAL into `pRstd[cta_m]`/`pC1[cta_m]`.
- After `cp_async_wait<0>()`, each thread reads its x k-slice from `sXn` and computes
  `xn = (x*rstd - c1)*gamma + beta` (rstd/c1 from pRstd/pC1).

## v16 experiment — compute rstd/c1 on-chip from `sXn`, add a fused-stats entry point
1. **Add a new kernel + host entry** `transition_b2b_fwd_fused_stats(x, g, beta, wa, wb, ws, eps)`
   (NO rstd/c1 args). KEEP the existing `transition_b2b_fwd(x, rstd, c1, g, beta, wa, wb, ws)` intact
   (the microbench `verify_b2b_cuda.py` and any backward-stats-reuse still use it). Implement fused-stats
   as a template flag (`bool FUSE_STATS`) on the kernel so the two entries share one kernel body, OR a
   thin second kernel — your call; keep code duplication minimal.
2. **On-chip stats:** after `cp_async_wait<0>()` (x row resident in `sXn`), compute per-row
   `mean` and `var` over k=0..kK-1 for each of the WG_M rows, then
   `rstd = rsqrt(var + eps)`, `c1 = mean * rstd`, and write into `pRstd[cta_m]`/`pC1[cta_m]`
   (replacing the global load). Reduction: `sXn` is in smem so any thread can read any (m,k); a simple
   correct scheme is one row per active thread (WG_M=64 rows, 128 threads) reading the full 128-wide
   row from `sXn` and accumulating in fp32. **Match `stats_triton` numerics for cos parity:** it uses
   population variance via the TWO-PASS centered form — `mean = sum(x)/K`, then
   `var = sum((x-mean)^2)/K` (NOT the one-pass E[x^2]-mean^2, to avoid cancellation), fp32 accumulation,
   bf16 x cast to fp32. `eps` is a kernel arg (the module passes `ln_in.eps`). Guard `row < M`.
3. Keep the existing per-WG `NamedBarrier` before the normalized-xn reads (stats must be visible before
   xn uses rstd/c1). Everything downstream (expand SS / squeeze RS / swizzled STG.128 epilogue /
   gamma-beta register load) is UNCHANGED.

## Module wiring (do this too)
Update `kernels/transition/cuda/__init__.py::cuda_transition_b2b` to call the fused-stats entry and
**drop the `stats_triton` call** (no separate stats pass). Signature stays
`cuda_transition_b2b(x, ln_weight, ln_bias, wa, wb, ws, eps)`. The module dispatch
(`modules/transition/module.py`) is unchanged.

## Constraints
cos>=0.999 (module-level cos vs torch must stay ~0.99999, matching the current precomputed-stats path).
bf16, NO precision reduction. Fixed shapes (K=128, ND=512, D=128). Watch ptxas: the stats reduction adds
a little register/smem work in the prologue — expect regs ~255, spill 0; if it spills or drops <1 CTA,
note it (likely revert). One focused experiment (fuse stats), no other changes.

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu` (add fused-stats kernel/entry + on-chip reduction) and
   `kernels/transition/cuda/__init__.py` (call fused-stats, drop stats_triton).
2. Write `docs/kernel-optimization/transition_b2b/v16.md`: environment; the 1.29->1.20 gap analysis;
   the fused-stats design (reduction scheme + two-pass numerics matching stats_triton); code changes;
   Validation + Perf sections as TODO (I fill from H100 — watch: module cos vs torch ~0.99999,
   module inference ms vs the current 1.20x / whether the gap toward 1.29x closes, ptxas regs/spill,
   standalone `verify_b2b_cuda.py` still passes on the precomputed-stats entry).
- No GPU. Final message: the fused-stats reduction scheme, the two entries' signatures, the exact
  numerics used (two-pass var), register/smem impact, and correctness risks (stats visibility barrier,
  var formula parity with stats_triton, M-boundary rows).
