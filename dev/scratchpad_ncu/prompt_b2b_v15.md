# cuda-kernel-optimizer skill — transition_b2b ROUND v15 (software-pipeline the ND loop: hide SwiGLU gate + squeeze under wgmma)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste; YOU analyze + implement + write v15.md draft; DO **NOT** commit. Branch `b2b-cutlass-opt`.
PTX/inline-asm ALLOWED. **BEST = v14 (`ea5aa3f`): ~421us = 1.26-1.31x of Triton (~540us), cos 1.0.**
Kernel file currently IS v14.

## State after v14 — the only remaining stall is `wait`
lg_throttle 0, bank conflicts ~0 (14,854), SM 51%, DRAM 12.7% (not mem-bound). Dominant stall =
**`wait 1.34`** = warps stalled on WGMMA completion. Occupancy is fixed at 1 CTA/SM (8 warps, smem-capped),
so we CANNOT add warps to hide it. The lever is to keep the TENSOR CORE BUSY so warps wait less: hide
the non-tensor work (the SwiGLU gate compute) and the dependent squeeze under other chunks' wgmma.

## Current ND loop (around `transition_b2b_kernel.cu:472-530`) — the serialization
Per ND chunk (BN-wide), each WG does:
```
consumer_wait(weights[chunk])
expand: warpgroup_arrive; gemm(mmaE, tXnA, tWaB, a_acc); commit;
        warpgroup_arrive; gemm(mmaE, tXnA, tWbB, b_acc); commit;
warpgroup_wait<0>                       // <-- waits for expand FULLY; tensor then IDLE during gate
gate:   h = silu(a_acc)*b_acc  (fp32 MUFU) -> h_bf -> h_frag (RS A operand)
squeeze: warpgroup_arrive; gemm(mmaS, h_frag, tWsB, out_acc); commit;
consumer_release
```
Problem: after `warpgroup_wait<0>` the SwiGLU **gate runs with the tensor core idle**. The squeeze(n)
does overlap expand(n+1) somewhat (its wait is deferred), but the gate is exposed.

## v15 experiment — reorder into a software pipeline (NO accumulator double-buffering)
Restructure so (a) the SwiGLU **gate(n) overlaps the previous chunk's squeeze(n-1) wgmma**, and (b)
**squeeze(n) overlaps expand(n+1) wgmma**. Crucial trick to avoid register blowup: gate(n) CONSUMES
`a_acc`/`b_acc` into `h(n)` FIRST, which FREES `a_acc`/`b_acc` to be reused by expand(n+1) — so you do
**NOT** double-buffer the big `a_acc`/`b_acc` (that would be +128 regs and spill at 255).

Target schedule (steady state), per WG:
```
PROLOGUE:
  consumer_wait(0); issue expand(0): arrive;gemm(a_acc);commit; arrive;gemm(b_acc);commit;

LOOP n = 0 .. num_chunks-1:
  warpgroup_wait<K1>            // ensure expand(n) complete; keep squeeze(n-1) IN FLIGHT (do NOT wait it)
  gate(n): a_acc,b_acc -> h(n) // runs concurrently with squeeze(n-1) wgmma  ==> hides the gate
  if (n+1 < num_chunks):
    consumer_wait(n+1)
    issue expand(n+1) into the SAME a_acc,b_acc (safe: gate(n) already consumed them)  // arrive;gemm;commit x2
  issue squeeze(n): arrive; gemm(mmaS, h_frag(n), tWsB(n), out_acc); commit   // overlaps expand(n+1)
  consumer_release(n)

EPILOGUE:
  warpgroup_wait<0>            // drain last squeeze
  (existing vectorized STG.128 output epilogue, unchanged)
```
- Pick the `warpgroup_wait<K>` depths correctly for CuTe/GMMA: expand issues 2 wgmma groups, squeeze 1.
  You must ensure `a_acc`/`b_acc` for chunk n are complete BEFORE gate(n) reads them, while leaving
  squeeze(n-1) outstanding so it overlaps the gate. Document the exact commit-group accounting and the
  chosen wait depths. `warpgroup_fence_operand` the accumulators as needed for correctness.
- **h double-buffering:** squeeze(n) reads `h_frag(n)` while the tensor core runs; the NEXT iteration's
  gate(n+1) writes `h(n+1)`. If squeeze(n) is still in flight when gate(n+1) writes, single-buffer `h`
  would corrupt it. So either (i) double-buffer ONLY `h`/`h_frag` (small — the squeeze RS A operand is
  ~16-32 regs/thread, far cheaper than doubling a_acc/b_acc), OR (ii) if that spills, add a
  `warpgroup_wait` that drains squeeze(n) before gate(n+1) writes h (loses the squeeze∥expand overlap
  but keeps the gate∥squeeze overlap). Prefer (i); fall back to (ii) if ptxas shows spills. State which.
- **This is register-pressure-sensitive (we're at 255 regs / 1 CTA). If ptxas reports ANY spill
  (STACK>0 / LOCAL>0) or the register count forces <1 CTA, that's likely a net loss — note it; I will
  measure and we revert if it regresses.** Do NOT reduce precision to save registers.

## Keep everything else v14
LN prologue + gamma/beta register-load, expand SS / squeeze RS, TMA weight ring + pipeline mbarriers,
out_acc persistent across chunks, swizzled shuffle + vectorized STG.128 epilogue, v13/v14 per-WG
NamedBarriers, 2-WG coop (256 threads), `__launch_bounds__(256,1)`, host signature, current-stream,
sm_90a, bf16 (NO precision reduction). cos>=0.999.

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: reorder the ND loop into the software pipeline above. Change only
   the loop scheduling (+ minimal h buffering); keep the math identical.
2. Write `docs/kernel-optimization/transition_b2b/v15.md`: environment; v14 baseline (~421us/1.29x,
   wait 1.34); the serialization analysis; the pipelined schedule + exact wgmma commit-group / wait-depth
   accounting; the "consume-then-reuse a_acc/b_acc (no double-buffer)" argument; h buffering choice
   (i vs ii); Validation + Perf/NCU as TODO (I fill from H100 — watch: `wait` vs 1.34, SM% vs 51%,
   issue_active, cos 1.0, **ptxas regs/STACK/LOCAL — spill = likely revert**, runtime vs 421us / Triton 540us).
- No GPU. Final message: the exact pipelined schedule you emitted, the wait-depth accounting, whether
  you double-buffered h (i) or drained squeeze (ii), the expected register-pressure impact, and
  correctness risks (accumulator reuse ordering, h lifetime vs squeeze in-flight, out_acc accumulation
  order preserved).
