# TODO

## Cute kernels bypass autotune entirely (the real "hardcoding") — 2026-08-04
**Root finding:** NO cute/CUTLASS kernel uses the autotune system. Each hardcodes ONE
hand-picked `GemmConfig` in its `if config is None:` branch (`grep -L select_config` over
`kernels/*/cute/*` = all of them). The `select_config` cache mechanism exists but zero cute
kernels call it. THIS is what "only tile_m / hardcoding everywhere / not brute-forced" means —
not the config *space* (quack's `GemmConfig` is rich: tile_m/n, cluster_m/n, pingpong/coop,
swap_ab, max_swizzle; `_get_sm90_configs("gated")` already sweeps 18 configs, plain 44).
Note on Triton-style knobs: on SM90 `num_warps` is NOT a free knob (WGMMA warps are
warpgroup-bound; the analogue is pingpong-vs-coop, already swept), and `num_stages`
(ab/epi) is auto-MAXIMIZED from smem (strictly better than Triton's manual num_stages).

**Fix built (2026-08-04):** `autotune/cute_config.py` — brute-force sweep of the FULL sm90
config space + cache-select, the cute counterpart of the Triton grid capture:
- `gated_sm90_candidates()` / `plain_sm90_candidates()` = `_get_sm90_configs(epilogue)` (sm90).
- `resolve_config(op, candidates, dtype, bucket, default)` — cache-select fastest, else default.
- `sweep_and_cache(op, dtype, cases, candidates)` — `do_bench` every candidate per shape bucket,
  write ranked cache (reuses `store_ranked_configs`; config space hash guards staleness).
- WIRED: `transition/cute/gemm_transition_swiglu.py` (fwd swiglu) — reference pattern; resolves
  by `(gpu, dtype, bucket_mixed(M)|kK)`, falls back to the K-aware default on miss. Verified.

**RESOLVED (2026-08-04) — gated cute epilogue fixed; the "second drift" was a STALE CACHE:**
The GATED cute paths produced all-zero (then garbage) output. TWO real fixes, both now verified:
- **FIX 1 — gated postact field rename (`mPostAct` -> `mAuxOut`).** quack 0.5.0 renamed the gated
  postact field and its attrs (`postact_dtype/postact_layout/cta_tile_shape_postact_mn` ->
  `aux_out_dtype/aux_out_layout/cta_tile_shape_aux_out_mn`). Our 3 gated kernels used the old names,
  so `GemmGatedMixin._epi_ops`' `TileStore("mAuxOut")` resolved to None -> the postact store was
  SKIPPED -> zeros. Fixed in `gemm_transition_swiglu.py`, `backward_gatebwd.py`, `gemm_gated_ln.py`.
- **FIX 2 (CRITICAL infra) — register our source in `quack.cache.EXTRA_SOURCE_DIRS`
  (`kernels/_quack_compat.py`).** quack's jit disk-cache keys `.o` by `(qualname, *args)` + a hash of
  QUACK's source, NOT ours. So editing a `.cute` kernel does NOT invalidate its cached `.o`: a stale
  (broken) binary from `/tmp/<user>/quack_cache` is silently reused. This masked FIX 1 for an entire
  debug session — every "still broken / garbage / tile_m=256 corrupts" result was a stale `.o`, not a
  real bug. (So the earlier `#3` "tm256 corrupts h" and "atom_layout_m=2 dual-store" theories were
  ALL stale-cache artifacts — disregard them.) Registering our pkg root makes edits bust the key.
- **VERIFIED with clean/enabled cache:** forward `transition_expand_swiglu_cute` = cos 1.0 vs torch
  for ALL 18 gated configs (incl. tm256 coop); `transition_gate_bwd` h + dAB = cos 1.0; plain
  `layernorm_linear` = cos 1.0 vs torch. Config is performance-only across the gated space.
- Restores the recorded transition-forward CuTe win (~1.1x K=128 → 2.6x K=512 vs triton) that the
  MINIWORLD route selects for large d_pair. → re-capture + ship swiglu_fwd / gate_bwd caches.

**TODO to finish the sweep-all-cute-kernels effort:**
- [x] `layernorm_linear` M1 — wired + swept + cache SHIPPED (verified config-invariant, plain path).
- [~] `transition/cute/gemm_transition_swiglu.py` (fwd), `backward_gatebwd.py` — wired (pattern
      ready) but DORMANT: gated postact broken (zeros), caches withheld until the store is fixed.
- [ ] Wire remaining cute GEMMs to `resolve_config` + their candidate space, dropping the
      hardcoded single-config default (keep it only as the cache-miss fallback):
      `backward_gatebwd.py`, `trimul_inproj/cute/gemm_gated_ln.py` (front, gated),
      `layernorm_linear/cute/*` (replace the hand-baked `_tuned.py` table with a swept cache),
      `back_split.py`, `tm1/tm2` (custom-CuTe: expose their tile params or document the atom limit).
- [ ] Build a capture driver that calls `sweep_and_cache` for each op across representative
      shapes on H100, writes `data/.../autotune/<op>/<gpu>.json`, commit.
- [ ] `dgrad_lnbwd.py` #2: to make tile_m brute-forceable (drop the tile_m=64 pin) the epilogue
      must load x̂ from gmem (frees the epi-C smem so a cooperative atom_layout works) — the
      single-subtile full-N LN reduction genuinely needs atom 1×1 at tile_m=64 today. Deep
      reimpl, then add it to the swept candidate space. (Not just a wiring change.)

## Config fix — eliminate correctness-pinned constants

**Goal:** an autotune config must be *performance-only* — like the Triton kernels, where any
launchable config gives the correct result and the tuner is free to pick the fastest. Today
several CuTe/CUTLASS GEMM kernels **pin `tile_m` / `cluster_m` / `pingpong` for correctness**:
some configs produce numerically wrong output (races, half-writes, corrupt epilogues), so the
implementation hard-codes a "safe" value. That is an implementation bug, not a tuning limit.
Fix the implementations so **no config affects numerics** (a genuinely algorithmic constraint
is fine *only if a comment states why* — e.g. LN-backward needs a full-N reduction subtile,
SwiGLU gate needs `tile_n % 32`). What we do NOT want is `tile_m` (and friends) pinned because
the kernel is buggy at other values.

> Note: brute-force Triton retuning showed ~no speedup (optimal config saturates by L~256), so
> the payoff here is correctness hygiene + freeing the config space, not raw speed.

### sm90 (H100) — fixable on this cluster
- [x] **#1 layernorm_linear — `cluster_m=1` / `pingpong=True` pin** (`cute/_tuned.py`,
      `gemm_layernorm_linear_fused.py` `_FUSED_CONFIG`). Was: `cluster_m=2` / non-pingpong
      *coop* reported cos 0.96–0.999 (timing-dependent). **VERIFIED 2026-08-04: does NOT
      reproduce** — 480 cos runs across 32 config×shape combos all cos=1.0, and
      `compute-sanitizer racecheck` = 0 hazards on `cluster_m=2 + coop`. Warning is **stale
      (already fixed)** → remove the safe-subset restriction and include `cluster_m=2`/coop in
      the config space. (Caveat: racecheck doesn't fully cover async TMA/mbarrier hazards.)
- [x] **dgrad + dab LN-backward — were BROKEN on quack 0.5.0, FIXED 2026-08-04** (`cute/dgrad_lnbwd.py`,
      `transition/cute/dab_lnbwd.py`). Three API drifts in their custom epilogue overrides (both used
      in production backward): (1) `_compute_stages` gained `warp_shape_mnk=None`; (2)
      `epi_smem_bytes_per_stage`→`epi_smem_bytes(...).{unstaged,d_stage,c_stage}`; (3) the tile-shape
      override `_sm90_compute_tile_shape_or_override`→`_compute_tile_shape_or_override` (old name never
      called → partial LN reduction, dx cos 0.48). Fixed all → dgrad cos=1.0, dab cos=1.0 vs torch. [a4dee06]
- [x] **#2 layernorm_linear dgrad — `tile_m=64` / atom-1×1** (`cute/dgrad_lnbwd.py`).
      NOT a correctness bug (reviewed 2026-08-04): `dgrad_lnbwd_cute` takes **no config** — it
      hardcodes tile_m=64 and *asserts* the atom_layout-1×1 invariant
      (`_sm90_compute_tile_shape_or_override`), so no caller config can yield wrong numerics. The
      `tile_n = K` single-subtile is a genuine algorithmic constraint (LN-bwd single-pass full-N
      reduction), already commented (module docstring + the assert). Compliant with the
      "constraint OK if commented" rule → no correctness fix needed.
      **Optional perf follow-up (not correctness):** the tile_m=64/atom-1×1 pin is only a d=256
      *speed* ceiling (can't use the cooperative tile_m=128). To rescue d=256, load x̂ from gmem in
      the epilogue (M2's per-element pattern) instead of as the C operand → frees ~64KB epi-C smem
      → cooperative tile. Low priority (configs saturate → ~no speedup; d=128 already wins/ties).
- [x] **#3 transition gate-bwd — postact `h` — FIXED 2026-08-04.** Root cause = the gated postact
      field rename (`mPostAct`->`mAuxOut`, FIX 1 above) + stale jit cache (FIX 2 above), NOT tile_m.
      With the rename + a clean/enabled cache, `h` cos=1.0 and `dAB` cos=1.0 vs torch across all
      gated configs (incl. tm256 coop). The earlier "tm256 corrupts h / garbage / denormals" reports
      were ALL stale `.o` (the edit never recompiled). Config is performance-only; the removed
      `_safe_gated_bwd_config` clamp stays removed. dswiglu math + the dual D=[dA|dB]+postact=h
      epilogue are correct.
- [ ] **transition swiglu / dab_lnbwd — hardcoded per-K configs** (`cute/gemm_transition_swiglu.py`,
      `cute/dab_lnbwd.py`). These are mostly *perf* hardcodes (not wrong) → replaced by proper
      tuning, not a correctness fix. Keep the `tile_n % 32` gate constraint (algorithmic) with a
      comment.
- [ ] side: **M2 fused path is currently BROKEN** — `layernorm_linear_cute_fused` raises
      `AttributeError: 'GemmLNLFusedSm90' object has no attribute 'load_AB'` (quack version
      drift?). `layernorm_linear` dispatches to it for N<=256. Fix or re-pin the quack GEMM base.

### sm100 (B200) — BLOCKED: no B200 GPU on this cluster (H100-only)
These pin config for correctness on sm100 and **cannot be reproduced / fixed / verified here** —
they need a Blackwell (B200) node. Do when B200 access exists:
- [ ] **#5 trimul front — M-major bdll TMA store half-writes** (`cute/front_sm100.py`,
      `front_train_sm100.py`, `v6_training_merged_sm100.py`). The M-major bdll postact store
      writes only half of each tile (cos ~0.05–0.5); only the `[M, 2D]` N-major layout is
      bit-correct. Pins `cluster_m=2,cluster_n=1,pingpong=False` + a per-shape swap_ab table.
      Fix the store atom / layout so the store is correct for any tile → free the config.
- [ ] `layernorm_linear/cute/dgrad_lnbwd_sm100.py` — `tile_m=128` pin (4-subtile → cos~0.5).
- [ ] `transition/cute/{b2b_fused_sm100,b2b_fwd_sm100,gatebwd_sm100}.py`
- [ ] `trimul_inproj/cute/{front_sm100_fused,front_fused_gemm_sm100,back_split_sm100,`
      `bidirectional_sm100,bidir_training_sm100,gatebwd_sm100,training_b200}.py`
- [ ] `tm1/cute/sm100_gate_gemm_collective.py`
- [ ] `ln_linear_sm100.py`

### Audited 2026-08-04 — no sm90 correctness exposure beyond #3
- [x] `tm2/cute/tm2_cute_kernel.py` — `tile_m=64` is **hard-asserted** (`assert tile_m == 64,
      "currently only TILE_M=64 (single m64 atom) is supported"`), so no config can select a
      wrong value. It's a custom-CuTe implementation-scope limit (one m64 MMA atom), reason in the
      assert msg → no correctness hazard. (Nice-to-have: widen to multi-atom for larger tiles — perf only.)
- [x] `trimul_inproj/cute/back_split.py` — delegates to the layernorm_linear (`lnl`) plain-D GEMM
      (`default_lnl_config`), NOT a gated dual-store; same family as #1 (already stale/clean). No exposure.
- [x] `tm1/cute/*` — not gated (no `GemmGated`/postact), plain GEMM. No dual-store exposure.
- [x] `transition/cute/gemm_transition_swiglu.py` (fwd) & `trimul_inproj/cute/gemm_gated_ln.py`
      (front) — gated but **postact-ONLY** (no D operand: `make_fake_gemm_tensors(...,None,None)`),
      so no D+postact dual-store interaction; the fwd swiglu is verified correct at tile_m=256.
      Their `tile_n % 32` gate is algorithmic + asserted. No exposure.
> Conclusion: on sm90 the D + gated-postact dual store (the #3 hazard) exists ONLY in
> `backward_gatebwd.py`, now clamped to the proven-correct tile_m set. Every other config-accepting
> sm90 wrapper is either plain-D, postact-only, or hard-asserts its tile_m. sm100 items below remain
> (B200-blocked). Remaining sm90 work is perf-only (transition swiglu/dab_lnbwd hardcodes → tuning).

## Follow-ups from the gated-postact fix (2026-08-04)
- [ ] **Verify `trimul_inproj/cute/dualgemm_kernel.py`** (used by tm2/trimul): it defines its OWN
      `TileStore("mPostAct")` (self-named, not the base `mAuxOut`). The base epilogue's
      `epi_setup_aux_out` stores the op named `mAuxOut` — a `mPostAct`-named TileStore may not be
      stored (same zeros symptom). Test vs torch on H100 (cache OFF); if broken, rename to `mAuxOut`.
- [ ] Re-check `dab_lnbwd` / other cute kernels for the same `mPostAct`/`postact_*` drift.
- [ ] Now that the CUTE transition forward works (cos=1.0 end-to-end incl. `cute_transition_fused`),
      re-evaluate the `implementation=triton` pin advice in `docs/quack-0.5.0-cute-port-plan.md` for
      the MINIWORLD/large-d_pair route on sm90.
