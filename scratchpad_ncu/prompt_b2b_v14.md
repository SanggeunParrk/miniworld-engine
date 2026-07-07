# cuda-kernel-optimizer skill — transition_b2b ROUND v14 (remove the LN-param-staging CTA sync)

Follow `scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste; YOU analyze + implement + write v14.md draft; DO **NOT** commit. Branch `b2b-cutlass-opt`.
PTX/inline-asm ALLOWED. **BEST = v13 (`5669483`): ~433us = 1.22-1.25x of Triton (~540us), cos 1.0.**
Kernel file currently IS v13.

## State after v13
lg_throttle 0, bank conflicts ~2.1M, SM 49%, issue_active 40%. Remaining stalls:
- `wait 1.31` — wgmma completion. Occupancy-bound (1 CTA/SM = 8 warps; 2 CTA is smem-blocked at 230KB).
  This is the HARD floor — do NOT attack it this round.
- `barrier 0.88` — v13 converted 2 of 4 CTA `__syncthreads` to per-WG NamedBarrier. The 2 remaining
  CTA-wide syncs are: (a) pipeline init (one-time, leave it) and (b) **LN param staging** — WG0 stages
  `pG`/`pBeta` (gamma/beta, K=128 bf16 each) into shared smem and BOTH WGs read them, forcing a CTA-wide
  sync. THIS is the addressable one.

## v14 experiment — eliminate the cross-WG LN-param dependency
Goal: make the gamma/beta (`pG`/`pBeta`) availability NOT require a CTA-wide sync, so that sync becomes
WG-local (or disappears). Pick whichever is cleaner and state which in v14.md:

- **Preferred: per-thread register load from global (no smem stage, no sync).** In the LN prologue each
  thread computes `xn = (x*rstd - c1)*gamma[k] + beta[k]`. If, for the SS-expand `xn` tiling, each
  thread only ever needs a SMALL fixed set of `k` columns of gamma/beta, load those `gamma[k]`/`beta[k]`
  values DIRECTLY from global into registers once (coalesced across the warp) and use them — removing
  the smem staging of pG/pBeta AND its CTA sync entirely. gamma/beta are read-only, tiny (128 each),
  and L2-resident; a one-time per-thread global load is cheap. ONLY do this if the per-thread k-set is
  small/static; if a thread needs all K of gamma/beta, this bloats registers — then use the fallback.
- **Fallback: redundant per-WG staging.** Have EACH warpgroup stage its OWN copy of gamma/beta into its
  OWN per-WG smem region (duplicate the 2*128 bf16 = 512 B/WG), so no WG reads the other WG's copy. The
  staging sync then guards only per-WG smem -> convert it to `cutlass::arch::NamedBarrier(128, wg_id+1)`
  like v13 did for the other per-WG syncs. Costs 512 B extra smem per WG (~1 KB/CTA) — confirm it still
  fits within the 227 KB opt-in (v13 is at 230944 B; +1 KB is fine, but UPDATE the static_assert if the
  dynamic smem total changes, and document the new value).

Do NOT touch: the pipeline-init CTA sync, the wgmma path, the vectorized/swizzled epilogue, the two
v13 per-WG NamedBarriers, or the weight-ring pipeline mbarriers.

## Keep everything else v13
expand SS / squeeze RS, TMA weight ring + mbarriers, out_acc persistent, 2-WG coop (256 threads),
swizzled shuffle + vectorized `st.global.v4.b32`, v13 per-WG NamedBarriers, `__launch_bounds__(256,1)`,
host signature, current-stream, sm_90a, bf16 (NO precision reduction). cos>=0.999. regs ~255, spill 0.

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: remove the cross-WG LN-param dependency (register-load or per-WG
   staging) and make its sync WG-local or gone. Change nothing else.
2. Write `docs/kernel-optimization/transition_b2b/v14.md`: environment; v13 baseline (~433us/1.25x,
   barrier 0.88); which approach (register-load vs per-WG staging) + why; smem accounting if it changed;
   hypothesis (barrier down -> issue_active up -> faster); Validation + Perf/NCU as TODO (I fill from
   H100 — watch: `barrier` vs 0.88, issue_active vs 40.1%, cos 1.0, regs/spill/smem, runtime vs 433us /
   Triton 540us). Be honest if the expected gain is small (the dominant `wait 1.31` is untouched).
- No GPU. Final message: the approach taken, whether the LN-param sync is now WG-local or fully removed,
  any smem-size change, and correctness risks (esp. gamma/beta indexing per thread if you did the
  register-load path).
