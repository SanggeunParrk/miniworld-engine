# cuda-kernel-optimizer skill — transition_b2b ROUND v11 (Triton-matched VECTORIZED STG.128 output epilogue; PTX now ALLOWED)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste results; YOU analyze + implement (edit the .cu) + write v11.md draft; DO **NOT** commit. Branch
`b2b-cutlass-opt`. BEST = v5 (`7528d0d`): **563.3us = 0.956x** of Triton (~543us), cos 1.0. Kernel file
currently IS v5. **NEW: the user has LIFTED the "no PTX / no inline-asm editing" constraint** — you MAY
now use inline PTX (`asm volatile`) for stores/loads where it lets us match Triton's instruction mix.

## Ground truth from Triton's own SASS (this is the TRITON->CUDA principled lever)
I disassembled the winning Triton `_transition_b2b_kernel` (`dev/scratchpad_ncu/b2b_dump/.../*.sass`).
Its memory-op mix:
- **8x `STG.E.128`** — the OUTPUT store is VECTORIZED 128-bit (8 bf16 per store instruction).
- **48x `LDGSTS.E.BYPASS.128`** — all inputs loaded via async cp.async, 128-bit, **L1 BYPASS**.
- **16x `STS.128` + 24x `LDSM.16.M88.4` + 16x `LDS.U16`** — Triton DOES stage to smem + ldmatrix
  (so it is NOT smem-store-free; v10's "eliminate STS" was the wrong target — do NOT pursue that).
- `HGMMA.64x64x16` x16 + `64x128x16` x4; `LDGDEPBAR`/`WARPGROUP.DEPBAR.LE`/`DEPBAR.LE` for async deps.

### The gap vs our v5
Our v5 stores the output **scalar, per accumulator element**: in the epilogue (around
`transition_b2b_kernel.cu:502-509`) it does `tGOut(i) = static_cast<BF>(out_acc(i))` for every
`i in size(out_acc)` (~64 elements/thread) — i.e. ~64 scalar `STG.U16` per thread instead of Triton's
~8 `STG.E.128`. This scalar store path is a dominant remaining `lg_throttle` source (regular global
stores hit the LSU pipe). v10 proved the xn STS is NOT the bottleneck; the OUTPUT store is the target.

## v11 experiment — vectorized STG.128 output epilogue via a SMEM-SHUFFLE (match Triton)
Replace the scalar output store with a smem-shuffle epilogue that emits **128-bit vectorized global
stores** (`STG.128` = 8 bf16 each), exactly like Triton:
1. After `warpgroup_wait<0>()` and casting `out_acc -> bf16`, **STS the per-WG output tile into smem**
   in the wgmma-C accumulator layout (use `st.shared.v4` / a u128 store; the acc already gives each
   thread contiguous-ish register runs). Then a WG barrier (`__syncwarp`-group / NamedBarrier or the
   existing 2-WG sync) so the tile is fully written.
2. **Reload the tile from smem in ROW-CONTIGUOUS order and `st.global.v4.b32` (STG.128) to
   `out[row, col..col+7]`** — each thread writes 8 contiguous bf16 columns of a row, warp writes a full
   coalesced 256-byte row segment. Predicate on `row < M` (M may not be a multiple of the tile; keep
   the existing boundary guard, applied per 8-wide chunk / per row).
3. Prefer inline PTX for the vectorized global store: `asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ...)`
   (and `ld.shared.v4`/`st.shared.v4` for the smem side). D=128 = 16 bf16-groups-of-8 = a whole row is
   two `STG.128`s; lay out threads so the WG covers `WG_M` rows x 128 cols with vectorized stores.

**Why this avoids v7's spill**: v7 vectorized IN-REGISTER and blew the 255-reg budget. The smem-shuffle
writes the accumulator to smem FIRST (freeing those registers), then reloads small vectorized chunks —
live register count stays ~v5. This is the standard CUTLASS/Triton epilogue and is exactly why Triton
can do STG.128 without spilling.

**SMEM reuse (no new smem)**: the `pXn` region (kWarpgroups * lXn = ~32KB) is DEAD by the epilogue
(xn only feeds the expand). Reuse `pXn` as the output shuffle tile (WG_M*kD*sizeof(BF) = 64*128*2 =
16KB per WG, 2 WGs = 32KB — fits). Do NOT grow `kDynamicSmemBytes` (keep the 230944 static_assert, or
adjust it only if the shuffle tile genuinely needs a few more bytes — document if so).

### Secondary (only if trivial, else skip to keep ONE focused experiment)
If the current x/weight cp.async is not already `.BYPASS` (L1 bypass, `cp.async.cg`/`ca`), matching
Triton's `LDGSTS.E.BYPASS.128` (use `cp.async.cg.shared.global` = bypass-L1) is a cheap aligned change
— but if it complicates the round, leave loads as-is and do ONLY the output epilogue.

## Keep everything else v5
LN prologue, expand `SM90_64x128x16_F32BF16BF16_SS` (xn in smem — KEEP, do not RS it), squeeze
`..._RS` with `convert_layout_acc_Aregs`, TMA weight ring (2-stage), out_acc persistent, 2-consumer-WG
coop (256 threads), `__launch_bounds__(256,1)`, host signature
`transition_b2b_fwd(x,rstd,c1,g,beta,wa,wb,ws)->out`, current-stream, sm_90a, bf16 (NO precision
reduction). cos>=0.999. Expect regs 255 / 1 CTA / smem 230944 unchanged.

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: replace ONLY the scalar output store (lines ~502-509) with the
   smem-shuffle vectorized STG.128 epilogue. Keep the `#else` scalar-fallback branch intact.
2. Write `docs/kernel-optimization/transition_b2b/v11.md`: environment; v5 baseline; the Triton SASS
   evidence (8x STG.128 vs our ~64 scalar STG); v11 hypothesis (vectorized store epilogue cuts the
   dominant remaining lg_throttle LSU source); exact code changes; the smem-reuse-of-pXn plan; and
   Validation + Perf/NCU sections as TODO (I fill from H100 — watch: STG count/width [expect few
   STG.128], `lg_throttle` vs 1.58, `sm__inst_executed_pipe_lsu` vs v5, regs 255, spill 0, runtime vs
   563.3us / Triton 543us).
- No GPU. Final message: the exact epilogue design (smem tile layout, the (thread->row,col) mapping for
  the vectorized reload, the PTX store instr used), confirmation registers/smem unchanged and no spill
  expected, and correctness risks (esp. the M-boundary predication for partial tiles + the WG sync
  before reload).
