# cuda-kernel-optimizer skill — transition_b2b ROUND v10 (RS expand from SMEM, not global — isolate the xn STS removal)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste results; YOU analyze + implement (edit the .cu) + write v10.md draft; DO **NOT** commit (commit
happens after I paste measured results). Branch `b2b-cutlass-opt`. BEST = v5 (`7528d0d`): **563.3us =
0.956x** of Triton (~538us), cos 1.0. The kernel file currently IS v5.

## What we know (v6–v9 all failed to cut lg_throttle 1.58; here's the crucial v9 lesson)
We are at Triton's occupancy (8 warps/SM, 1 CTA reg-capped at 255), no spill (local ld/st = 0), loads
hidden (long_scoreboard 0.13). The ONLY remaining gap vs Triton is `lg_throttle 1.58` (Triton 0) —
pure LSU-pipe throughput.

**v9** tried RS-expand to kill the 163,840 xn shared-stores (STS). It DID eliminate the STS
(163840→4, LDSM→0), cos 1.0, no spill — but came out **571.6us = 0.950x (a wash/tiny regression)**.
ROOT CAUSE (from v9.md): v9 **removed the resident `xn`/`x` smem staging entirely**, so the RS
A-fragment was filled by **direct global LDG of `x[row,k]`**. It merely TRADED the xn-STS for scalar
global x-loads — the LSU traffic just moved. It never isolated the STS removal.

## v10 experiment — the clean isolation (differs from v9 in exactly ONE way)
Keep v5's `x` path **entirely**: `stage_x_into_sxn()` still cp.async-stages `x` into the `sXn` smem
buffer, `cp_async_fence`/`cp_async_wait<0>` as in v5. **Do NOT drop the smem x staging and do NOT add
any global LDG of `x`.** Then, exactly like v9, switch the expand atom to
`SM90_64x128x16_F32BF16BF16_RS` and build the `xn` A-register fragment via the MMA's own A-layout
(`thrE.partition_fragment_A` + `make_identity_tensor((WG_M,K))` + `thrE.partition_A` for the logical
`(m,k)` of each slot) — BUT fill each slot by reading **`x` from the `sXn` smem buffer (LDS)** at
`(row,k)` and applying the LN transform `(x*rstd - c1)*g[k] + beta[k]` in registers, casting to bf16.

Net effect vs v5: we DELETE the `smem_store_u128(&sXn, xn_vec)` xn-STS (the 163,840 stores) and the
expand's SS smem read of xn, while `x` still arrives via cp.async→smem and is read via cheap coalesced
**LDS** (NOT global LDG). This is v5 **minus the xn STS**, nothing else — the pure test of whether the
xn shared-store is the lg_throttle source. If lg_throttle drops and runtime beats v5, the STS *was*
the cost; if it's again a wash, v9's global-LDG was NOT the confound and the STS is genuinely free →
strong evidence 0.956x is the hand-CUDA ceiling (document that honestly).

### Implementation notes
- REUSE the `sXn` shared buffer and the v5 `stage_x_into_sxn` cp.async prologue verbatim. `sXn` must
  still hold the raw `x` tile (bf16) after the cp.async wait. Do NOT overwrite `sXn` with xn.
- Expand: `using MmaExpandOp = SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>;`
- Build `xn_frag` once (xn is reused by every ND n-tile for both `a` and `b`): partition the A
  fragment, iterate its slots, for each slot get logical `(m=row, k)` via the identity tensor, read
  `bf16_to_float(sXn(row,k))`, apply `(xv*row_rstd - row_c1)*gamma[k] + bias[k]`, cast bf16, store
  into `xn_frag`. Keep `xn_frag` live across the whole ND loop.
- Expand gemms: `cute::gemm(mmaE, xn_frag, tWaB, a_acc)` and `... tWbB, b_acc)`.
- Keep EVERYTHING else v5: TMA weight ring (wa/wb/ws, 2-stage, 230944B dynamic smem — note smem does
  NOT shrink here since sXn is retained), RS squeeze (`SM90_64x128x16_F32BF16BF16_RS` + h via
  `convert_layout_acc_Aregs`), out_acc persistent, 2-consumer-WG coop (256 threads), LN param staging
  (g/beta/rstd/c1), `__launch_bounds__(256,1)`, host signature
  `transition_b2b_fwd(x,rstd,c1,g,beta,wa,wb,ws)->out`, current-stream, sm_90a, bf16 (NO precision
  reduction).

## Constraints
cos>=0.999, bf16 only, h off HBM, 255-reg / 1-CTA occupancy expected to be unchanged, no local
mem/spill (if the fragment-fill adds spill, that's a fail → note it). One focused experiment only.

## Deliverable THIS call (NO commit)
1. Edit `src/miniworld_kernels/kernels/transition/cuda/transition_b2b_kernel.cu` per v10.
2. Write `docs/kernel-optimization/transition_b2b/v10.md`: environment; the v5 baseline + v9 lesson
   (v9 confounded STS-removal with global-LDG); v10 hypothesis (isolate STS removal, keep cp.async x
   + LDS fragment fill); exact code changes (diff vs v5); the single-variable-vs-v9 framing; and
   Validation + Perf/NCU sections as TODO (I fill from H100 — watch: STS→~4, LDSM→0, **no new global
   LDG of x**, lg_throttle vs 1.58, regs 255, spill 0, runtime vs 563.3us / Triton 538us).
- No GPU. Final message: the exact edit (which lines changed vs v5), how the A-fragment slots are
  filled from `sXn` (the (m,k)→smem index mapping + any bank-conflict risk), confirmation the smem x
  staging is RETAINED (no global x LDG), smem/reg/occupancy expectation, and correctness risks.
