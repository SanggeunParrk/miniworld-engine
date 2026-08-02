# cuda-kernel-optimizer skill — transition_b2b ROUND v12 (kill the v11 shuffle-tile SHARED BANK CONFLICTS)

Follow `dev/scratchpad_ncu/SKILL_cuda_kernel_optimizer.md`. No-GPU: I build/validate/profile on H100 and
paste; YOU analyze + implement + write v12.md draft; DO **NOT** commit. Branch `b2b-cutlass-opt`.
PTX/inline-asm is ALLOWED (unlocked). **BEST = v11 (`1f7025b`): ~448us = 1.18-1.21x of Triton (~540us),
cos 1.0** — the first version to beat Triton, via the vectorized STG.128 smem-shuffle output epilogue.
Kernel file currently IS v11.

## v11 ncu (M=262144, 283us) — the NEW bottleneck the shuffle introduced
v11 killed `lg_throttle` (1.58 -> 0.00, great). But its output epilogue smem-shuffle introduced heavy
overhead that now caps us at SM 47% / issue_active 36% (NOT memory-bound: DRAM 12.7%; tensor pipe 47%,
"should not be a bottleneck"):
- **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` = 5,771,641** (v5 was 908!). The dominant new
  cost. Almost certainly the **scalar bf16 C-fragment store into the row-major shuffle tile**
  `pOutShuffle[m*128 + n]`: 2-byte (sub-word) scalar stores where adjacent threads/n collide on the
  same 4-byte smem banks, and the row stride 128 (a power of 2) aligns every row to the same banks.
- Stalls: `short_scoreboard 0.34` + `mio_throttle 0.36` are the smem side-effects of those conflicts.
- (`barrier 1.10` from the CTA `__syncthreads` is a SEPARATE issue — leave it for v13; do NOT touch the
  sync in this round. ONE focused experiment: bank conflicts only.)

## v12 experiment — make the shuffle tile BANK-CONFLICT-FREE (target 5.77M -> ~0)
Keep the v11 epilogue STRUCTURE (out_acc -> smem tile -> `__syncthreads` -> `ld.shared.v4` ->
predicated `st.global.v4.b32`), change ONLY the smem tile LAYOUT + the store into it so there are no
shared bank conflicts:
- **Preferred: pad the row stride.** Change the shuffle tile from `[m*128 + n]` to a padded stride
  `[m*STRIDE + n]` with `STRIDE = 128 + PAD` (bf16 elements) chosen so successive rows land on
  different bank phases and the 128-bit reload stays conflict-free. Typical choice: pad so the row
  stride in BYTES is not a multiple of 128 (the 32-bank * 4-byte line) — e.g. `PAD = 8` bf16 (16 B) is
  a common swizzle-free fix; verify the 8-element (16 B) `ld.shared.v4` alignment still holds
  (`n` multiple of 8, base 16 B-aligned). Ensure the padded tile still fits the reused `pXn` smem
  region (pXn = kWarpgroups * lXn; lXn = WG_M*kK = 64*128 bf16 = 16 KB/WG = 32 KB total; a padded
  64 x (128+8) tile = 64*136*2 = 17408 B/WG = 34816 B total — this EXCEEDS the 32 KB pXn region by
  ~2 KB). **If padding overflows pXn**, either (a) also reuse an adjacent dead smem region for the
  epilogue tile (the TMA weight ring `pWa/pWb/pWs` is DEAD by the epilogue — the last squeeze already
  consumed the weights — so the whole 192 KB weight-ring smem is free to alias for the output tile;
  document the aliasing carefully so it can't clobber in-flight weights), or (b) use a swizzle instead
  of padding (below).
- **Alternative: XOR swizzle** the tile index (CUTLASS-style `smem_idx = (m*128 + n) ^ swizzle(m)`),
  which removes conflicts WITHOUT growing the tile — apply the SAME swizzle to both the fragment store
  and the `ld.shared.v4` reload so they stay consistent. Prefer this if padding overflows the smem
  region.
- Also consider: instead of scalar bf16 stores from the C-fragment, pack the accumulator into wider
  `st.shared` writes if the C-fragment gives each thread ≥2 adjacent-n bf16 (wgmma C fragments own
  pairs of columns) — packing 2 bf16 -> `st.shared.u32` halves the sub-word conflict pressure. Use
  this if it composes cleanly with the chosen layout; else the layout fix alone is the experiment.

State clearly in v12.md which approach you took (pad vs swizzle vs pack) and the exact smem index math.

## Keep everything else v11 / v5
LN prologue, expand `SM90_64x128x16_F32BF16BF16_SS` (xn in smem), squeeze `..._RS`, TMA weight ring
(2-stage), out_acc persistent, 2-WG coop (256 threads), the `__syncthreads` before reload (v13 target,
NOT this round), `global_store_u128`/`st.global.v4.b32` vectorized store, `__launch_bounds__(256,1)`,
host signature, current-stream, sm_90a, bf16 (NO precision reduction). cos>=0.999. Expect regs ~255,
spill 0, smem 230944 (or document the exact new value if the padded/aliased tile changes it).

## Deliverable THIS call (NO commit)
1. Edit `transition_b2b_kernel.cu`: change ONLY the shuffle-tile layout + the store/reload indexing to
   eliminate bank conflicts. Do not touch the barrier, the expand/squeeze, or the load path.
2. Write `docs/kernel-optimization/transition_b2b/v12.md`: environment; v11 baseline (~448us/1.20x,
   bank conflicts 5.77M); the ncu evidence; v12 hypothesis (conflict-free tile -> short_sb/mio down ->
   issue_active up -> faster); exact layout math (pad/swizzle) + smem accounting; Validation + Perf/NCU
   sections as TODO (I fill from H100 — watch: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`
   -> ~0, short_scoreboard/mio_throttle down, cos 1.0, regs/spill/smem, runtime vs 448us / Triton 540us).
- No GPU. Final message: the chosen approach (pad/swizzle/pack), the exact smem index formula for BOTH
  the fragment store and the vectorized reload, the smem-size accounting (does it still fit pXn, or did
  you alias the dead weight-ring region?), and correctness risks.
