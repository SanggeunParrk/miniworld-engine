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

**MAJOR FINDING (2026-08-04) — the gated cute epilogue is broken (outputs ZEROS):**
While building the sweep I verified each wired kernel's config-invariance. The GATED cute paths
produce all-zero / garbage output for EVERY config (not a tuning issue — the store itself):
- `transition_expand_swiglu_cute` / `cute_transition_fused` (fwd): output all zeros (min=max=0),
  cos_vs_torch=0 for all 18 gated configs. `transition_gate_bwd` postact `h`: garbage (see #3).
- Same family as the M2-fused `AttributeError: 'GemmLNLFusedSm90' object has no attribute
  'load_AB'` — a quack version drift broke the gated/fused SM90 epilogues.
- NOT a production regression: the transition module defaults to TRITON; `KernelBackend.CUTE`
  is opt-in benchmarking only (module.py:191). Training is unaffected.
- The PLAIN cute path (`layernorm_linear` M1, `GemmDefaultEpiMixin` standard D store) is FINE —
  verified config-invariant (0/44 candidates diverge vs default). Only the GATED aux_out /
  postact STSM store is broken.
- **Real fix (supersedes config work for these):** repair the gated-postact store for the current
  quack (see `_bdll_patch`-style ownership, `permute_gated_Cregs_b16`, `GemmGatedMixin`/M2 fused).
  Until then the swiglu_fwd / gate_bwd sweeps are meaningless (their caches were withheld/removed).

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
- [ ] **#3 transition gate-bwd — postact `h` output broken for ALL configs** (`cute/backward_gatebwd.py`).
      **PRIOR DIAGNOSIS WAS WRONG.** Re-tested rigorously 2026-08-04 (M=8192 K=128 N=256, cute vs
      **triton** `_transition_expand_gatebwd_savedxn` vs torch): triton h == torch (cos=1.0), but
      the **cute `h`/PostAct is garbage for EVERY config** — cos(h,torch)=0.0, values ~1e-22/1e-38
      denormals — for tm192-pp (the "reference"), tm256-coop, tm64-pp alike, with or without
      `_bdll_patch`. `dAB`=[dA|dB] is CORRECT (cos=1.0) for **every** config incl. tm256-coop. So:
      (a) it is NOT tile_m-dependent — the earlier "tm256 corrupts h, tm192 bit-exact" was a
      garbage-vs-garbage kernel-vs-kernel compare; (b) config here is genuinely **performance-only**
      (dAB correct across the whole space) → NO clamp/pin. The bogus `_safe_gated_bwd_config` clamp
      was REMOVED 2026-08-04.
      **Why it never mattered:** `transition/cute/fused.py` defaults `backward_backend="triton"`
      (fused.py:92,205); the cute gatebwd (the only consumer of this `h`) is off by default, so the
      broken postact store has never affected training.
      **Real (open) bug:** the cute gatebwd postact `h` TMA store writes nothing/garbage
      (config-independent). To make the cute backend usable, fix the postact store in
      `GemmDLnGatedMixin` (dual D=[dA|dB] + postact=h epilogue) — likely the `tRS_rPostAct` build /
      `epi_convert_postact` permute / n-major TMA setup, since dAB works but h doesn't. Verify
      cos(h)=1.0 vs triton on H100. Independent of config — do NOT reintroduce a tile_m pin.
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
