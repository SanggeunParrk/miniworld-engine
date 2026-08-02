# cuda-kernel-optimizer skill — transition_b2b ROUND v13 (WG-local barriers: cut CTA __syncthreads -> per-WG NamedBarrier)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste; YOU analyze + implement + write v13.md draft; DO **NOT** commit. Branch `b2b-cutlass-opt`.
PTX/inline-asm ALLOWED. **BEST = v12 (`8b820c6`): ~442us = 1.20-1.23x of Triton (~540us), cos 1.0**
(vectorized STG.128 swizzled shuffle epilogue). Kernel file currently IS v12.

## v12 ncu (M=262144, 277us) — barrier is now the top addressable stall
- `lg_throttle` 0.00 ✓, bank conflicts down to 2.1M, SM 48%, issue_active 39.6%.
- Stall ranking: `wait 1.38` (wgmma completion — occupancy-bound, HARD, leave it), **`barrier 1.01`**
  (the CTA-wide `__syncthreads`), `mio 0.24`, `short_sb 0.24`, `long_sb 0.26`.
- `barrier 1.01` is the best remaining LEVER we can address without changing occupancy.

## Root-cause of the barrier stall
The kernel has (I believe) TWO `__syncthreads()`, both CTA-wide (all 256 threads = both warpgroups):
1. **xn-visibility sync** — after the LN prologue stores `xn` into `sXn` (smem) and before the expand
   reads `xn` via the SS wgmma descriptor. This guards `sXn`.
2. **epilogue reload sync** — after the C-fragment is written into the shuffle tile and before the
   `ld.shared.v4` vectorized reload. This guards the output shuffle tile.

**Both `sXn` and the shuffle tile are PER-WARPGROUP** (`pXnWg` = the per-WG region; each WG writes and
reads only its own 64 rows). Neither sync guards cross-WG data. The only genuinely SHARED smem is the
TMA weight ring (`pWa/pWb/pWs`), and that is synchronized by the `PipelineTmaAsync` mbarriers
(producer/consumer), NOT by these `__syncthreads`. So both `__syncthreads` can be replaced by a
**per-WG barrier over just this WG's 128 threads** — halving the participating-thread count and
removing the cross-WG arrival-skew that inflates the barrier stall.

## v13 experiment — replace CTA `__syncthreads()` with per-WG NamedBarrier(128)
- For EACH of the two `__syncthreads()` that guard per-WG smem, use a warpgroup-scoped barrier over
  the 128 threads of THIS warpgroup only. Options (pick one, state which in v13.md):
  - `cutlass::arch::NamedBarrier(128, /*id=*/ wg_id)` with a distinct barrier id per WG (wg_id = 0/1),
    `.arrive_and_wait()` — the CUTLASS Hopper idiom (search the bundle
    `.../cutlass/include/cutlass/arch/barrier.h`).
  - or raw PTX `asm volatile("bar.sync %0, 128;" :: "r"(barrier_id));` with `barrier_id = wg_id + 1`
    (avoid id 0 which `__syncthreads` uses) and 128 as the thread count.
- FIRST verify (and state in v13.md) that NO `__syncthreads` in the kernel guards the SHARED weight-ring
  smem or any cross-WG communication. If one does, leave THAT one CTA-wide and convert only the
  per-WG ones. Correctness (cos) must not break.
- Named-barrier ids: ensure the two per-WG barriers (xn-visibility, epilogue) don't collide destructively
  if both WGs are in flight — using `id = wg_id + 1` gives WG0->id1, WG1->id2; if you need the two
  DIFFERENT sync points to not alias within a WG, that's fine because they're temporally separated
  (prologue vs epilogue) so the same id can be reused across the two points within a WG. Document the
  id assignment.

## Keep everything else v12
LN prologue, expand SS / squeeze RS, TMA weight ring + its pipeline mbarriers (DO NOT touch pipeline
sync), out_acc persistent, 2-WG coop (256 threads), swizzled shuffle tile + vectorized
`st.global.v4.b32`, `__launch_bounds__(256,1)`, host signature, current-stream, sm_90a, bf16 (NO
precision reduction). cos>=0.999. Expect regs ~255, spill 0, smem 230944 unchanged.

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: replace the per-WG-guarding CTA `__syncthreads()` calls with per-WG
   NamedBarrier(128). Change NOTHING else.
2. Write `docs/kernel-optimization/transition_b2b/v13.md`: environment; v12 baseline (~442us/1.22x,
   barrier 1.01); which `__syncthreads` exist and which guard per-WG vs shared smem (the safety
   argument); v13 hypothesis (WG-local 128-thread barrier -> barrier stall down -> issue_active up ->
   faster); the barrier mechanism + id assignment; Validation + Perf/NCU as TODO (I fill from H100 —
   watch: `barrier` stall vs 1.01, issue_active vs 39.6%, cos 1.0, regs/spill/smem, runtime vs 442us /
   Triton 540us).
- No GPU. Final message: how many `__syncthreads` there were, which you converted (per-WG) vs kept
  (shared, if any), the barrier primitive + ids used, and correctness risks (esp. any hidden cross-WG
  smem dependence you had to preserve).
