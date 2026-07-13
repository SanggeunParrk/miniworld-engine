# Plan: port the CuTeDSL (cute) backend to quack 0.5.0

**Status:** planned, not started.
**Scope:** miniworld-kernels only. team-gm / MiniWorld are unaffected — they consume
`miniworld_kernels.ops.*`, and the cute-vs-triton backend choice happens *inside* the
ops whole-op (Phase 6). No consumer code changes.

## Why (root cause)

The cute kernels are pinned to **quack 0.3.11** (`[cute]` extra:
`quack-kernels==0.3.11`, `nvidia-cutlass-dsl==4.4.2`) and use its internal API
(`quack.cache_utils`, `quack.gemm_interface`, `quack.gemm_sm90/sm100`,
`quack.epi_ops`, …). The MiniWorld/team-gm consumer env ships **quack 0.5.0**
because **FlashAttention-4 requires it** (`flash-attn-4` →
`Requires-Dist: quack-kernels>=0.5.0`; FA4's own CuTeDSL kernels import
`quack.copy_utils` / `layout_utils` / `cute_dsl_utils`). So:

- **cute** needs a quack with a working `gemm_interface` → only **0.3.11** works.
- **FA4** needs **quack ≥ 0.5.0**.
- No released quack satisfies both (verified through **0.6.1**, still broken; and
  upgrading quack drags torch 2.10→2.13, breaking the cu130 stack — do NOT upgrade
  quack in the consumer env).

Because FA4 (quack ≥ 0.5.0) is a hard consumer requirement, the fix is to make the
**cute backend work on quack 0.5.0** (+ torch 2.10 + cutlass-dsl 4.5.2), so cute
trimul and FA4 coexist. Then `ops.*` dispatches cute on sm100 (B200), triton
elsewhere. Until this lands, the ops run the **triton** backend (quack-free, correct,
but not the tcgen05-tuned path).

## Constraints (learned the hard way)

1. **Never mutate the shared consumer env.** Upgrading its quack once pulled
   torch 2.10→2.13 and broke cu130; it was reverted. Do all porting in an
   **isolated quack-0.5.0 dev env**.
2. **Never patch `site-packages` quack.** Fix everything in miniworld-kernels source.
3. Two moving dimensions: **quack 0.3.11 → 0.5.0** *and* **cutlass-dsl 4.4.2 → 4.5.2**
   (the CuTeDSL API itself may have drifted, not just quack).

## Known breakage (from a static import scan on quack 0.5.0/0.6.1)

- `quack.gemm_interface` — **module import fails** on torch 2.10: its `gemm_out`
  `torch.library.custom_op` declares `SymInt rounding_mode=RoundingMode.RN`
  (an IntEnum default), which torch's `parse_schema` rejects
  (`invalid numeric default value`). cute needs `gemm` / `gemm_act` from this module.
  **This is the crux.**
- `quack.cache_utils` (`jit_cache`, `COMPILE_ONLY`) — **module removed**; the symbols
  moved into `quack.autotuner` / `quack.gemm` in 0.5.0.
- The other ~30 imported symbols (`epi_ops`, `gemm_sm100`, `gemm_config`,
  `cute_dsl_utils`, `activation`, `rmsnorm`, `rounding`, `compile_utils`,
  `gemm_default_epi`, `GemmGatedMixin`, …) **exist** in 0.5.0 — but signatures/behavior
  are not yet verified.

## Phases

### Phase 0 — isolated dev env
Scratch env mirroring the consumer stack: `torch 2.10.0+cu130` + `quack-kernels==0.5.0`
+ `nvidia-cutlass-dsl==4.5.2` + `triton==3.6.0`. A miniworld-kernels git worktree with
its own pixi env (or venv) pinned to these. All porting/testing here — never the shared
consumer env.

### Phase 1 — full breakage map
Import + run the cute trimul/bidir entry points on quack 0.5.0 + cutlass 4.5.2 and
collect *every* error (not just the import-statement scan): `bidirectional_trimul_sm100`,
`back_split_sm100`, `layernorm_linear` cute (`dgrad_lnbwd_sm100`, `ln_linear_sm100`,
`gemm_layernorm_linear_fused`), tm1/tm2 cute, front kernels, gate patches. Catalogue
gemm_interface + cache_utils + any runtime API drift + cutlass-4.5.2 DSL drift.

### Phase 2 — resolve `gemm_interface` (biggest unknown)
Only the `gemm_out` custom-op registration is broken; the `gemm` / `gemm_act` functions
cute needs live in the same module (so they can't be imported past the failing
registration). Decide between:
- **(A) use `GemmSm100` / `GemmSm90` classes directly** (they import fine on 0.5.0) and
  reimplement the thin `gemm` / `gemm_act` dispatch/launch in miniworld-kernels. Clean,
  larger.
- **(B) compat shim** that exposes `gemm` / `gemm_act` while neutralizing the broken
  `gemm_out` registration (e.g. make the `RoundingMode` default serialize to an int
  before importing the module). Fast, depends on torch/quack internals.
Pick after Phase 1 measures the difficulty.

### Phase 3 — `cache_utils` relocation
Replace `from quack.cache_utils import jit_cache, COMPILE_ONLY` with the 0.5.0 location
(`quack.autotuner` / `quack.gemm`) or a small miniworld compat shim. ~2 files
(`layernorm_linear/cute/dgrad_lnbwd.py`, `dgrad_lnbwd_sm100.py`).

### Phase 4 — remaining API + cutlass drift
Fix signature/behavior changes surfaced in Phase 1 across `epi_ops` / `gemm_config` /
`cute_dsl_utils`, and any `nvidia-cutlass-dsl` 4.4.2→4.5.2 CuTeDSL API changes.

### Phase 5 — verify (isolated env)
cute trimul + bidir correctness vs the pytorch reference (masked + unmasked, fwd + bwd,
cos ≥ 0.999 with randomised non-zero weights), then **cute vs triton performance** on
B200 (the payoff — confirm the tcgen05 win that motivates this).

### Phase 6 — integrate
- Update the `[cute]` extra pin to a quack-0.5.0-compatible set.
- Make the `ops.*` whole-ops **dispatch cute on sm100** (B200), triton fallback
  otherwise — the original goal.
- Bump the chain (miniworld-kernels → team-gm → MiniWorld); install the cute deps in the
  consumer env (coexisting with FA4 on quack 0.5.0); verify cute runs there.

## Risks
- **Phase 2** is the make-or-break unknown (gemm_interface workaround difficulty).
- **cutlass-dsl 4.5.2** CuTeDSL drift is a separate axis from quack and may add work.
- Likely a multi-session effort given the CuTeDSL depth.

## Recommended start
Phase 0 + Phase 1 → measure Phase 2 difficulty (A vs B) → then commit to the full scope.
