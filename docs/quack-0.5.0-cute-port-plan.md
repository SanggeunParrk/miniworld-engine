# Plan: port the CuTeDSL (cute) backend to quack 0.5.0

**Status:** IN PROGRESS. Phases 0–3 DONE (`a1aea37`); Phase 4 is the remaining blocker
(a cutlass-dsl launch/stream ABI migration — larger than the quack shim).
**Scope:** miniworld-engine only.

## Progress (2026-07-13)

- **Phase 0–1 done.** Dev env = the MiniWorld consumer env's python (quack 0.5.0 /
  torch 2.10 / cutlass-dsl 4.5.2 / B200) + `PYTHONPATH=<my>/src` — never mutates the
  shared env. A full import probe found exactly **two** quack breaks (below) and **zero**
  cutlass-4.5.2 drift *at import*.
- **Phase 2–3 done** (`a1aea37`, `kernels/_quack_compat.py`): the `RoundingMode.__str__`
  polyfill fixes the `gemm_interface` custom_op schema; `jit_cache`/`is_compile_only`
  replace the removed `cache_utils`. **52/52 cute modules import** on quack 0.5.0 (was
  25/52). triton + pytorch paths untouched — **trimul triton = 0.284 ms vs pytorch
  1.14 ms (4×) at L=384 on B200**, correctness intact.
- **Phase 4 = the real remaining blocker (cutlass-dsl 4.4.2→4.5.2 launch/stream ABI).**
  cute modules import but cute *execution* crashes: `_forward_cute` →
  `tm1/cute/sm100_gate_gemm_collective.py:640` `_CACHE[key](mA,mBp,mBg,mC,mac,strm)` →
  cutlass-dsl 4.5.2 `jit_executor` `DSLRuntimeError: cannot be converted to pointer`.
  Root cause: the last runtime arg `strm = cuda.bindings.driver.CUstream(...)` is no
  longer pointer-packable by the 4.5.2 executor (`ctypes.c_void_p(CUstream)` fails).
  In 4.5.2 the launch stream is NOT a packed runtime arg — quack 0.5.0 traces the launch
  with a real stream but passes `stream=None` at the compiled-fn *call* (→ ambient/current
  stream). Naively passing `None` at our call segfaults `cuLaunchKernelEx`, because our
  kernels use the *old* `cute.compile(op, …, strm)` + call-with-same-args shape; the whole
  compile+launch path must move to the 4.5.2 convention. This affects **all ~7 hand-written
  sm100 cute launch sites** (`grep -n CUstream src/**/cute/**.py`): tm1 gate-gemm, the
  front fused/train GEMMs, transition b2b, gatebwd, ln_linear_sm100. Each needs migrating
  to quack 0.5.0's `compile_gemm_kernel`/`.launch` stream convention — genuine per-kernel
  CuTeDSL work, likely multi-session. `make_trivial_tiled_mma(ab_dtype=…)` is also
  deprecated (→ separate `a_dtype`/`b_dtype`) but only warns, not fatal.
 team-gm / MiniWorld are unaffected — they consume
`miniworld_engine.ops.*`, and the cute-vs-triton backend choice happens *inside* the
ops whole-op (Phase 6). No consumer code changes.

> **Update (autotune batch, `f2a8529`).** The module-layer dispatch this plan's
> Phase 6 called for now exists: `modules/dispatch.py` `_MINIWORLD_KNOWN_BEST`
> already routes `triangle_multiplication` → **CUTE on sm90+ (Hopper/Blackwell)**,
> Triton pre-Hopper (`_trimul_known_best`). And the new backend-agnostic autotune
> cache (`src/miniworld_engine/autotune/`) exposes a **CuTe `select_config` hook**
> our ported cute kernels plug their tile/cluster configs into (worked example:
> `trimul_front_cute`). So Phase 6 shrinks to *making cute actually run on quack
> 0.5.0* + adding bidir to the table. **Latent consequence:** on B200 with quack
> 0.5.0, selecting `MINIWORLD` (auto — the doc's "what production should select")
> for trimul resolves to CUTE, i.e. the currently-broken path; consumers must pin
> `implementation=triton` on B200 until this port lands. This raises the port's
> priority. Note `bidirectional_triangle_multiplication` is NOT in the table (falls
> to Triton) — Phase 6 should add its cute routing.

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
2. **Never patch `site-packages` quack.** Fix everything in miniworld-engine source.
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
+ `nvidia-cutlass-dsl==4.5.2` + `triton==3.6.0`. A miniworld-engine git worktree with
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
  reimplement the thin `gemm` / `gemm_act` dispatch/launch in miniworld-engine. Clean,
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

### Phase 6 — integrate (mostly pre-wired by the autotune batch)
- Update the `[cute]` extra pin to a quack-0.5.0-compatible set.
- Module-layer dispatch is **already done** for unidirectional trimul
  (`_MINIWORLD_KNOWN_BEST` → CUTE on sm90+). Remaining: add
  `bidirectional_triangle_multiplication` to the table (cute on sm90+), and
  confirm each other op's family choice once its cute path works.
- Wire the ported cute kernels' build-time configs into the new autotune cache via
  `select_config(op, dtype, bucket, candidates)` (falls back to `default_config`
  on a miss); build + ship the B200 (sm100) cute config JSONs.
- Bump the chain (miniworld-engine → team-gm → MiniWorld); install the cute deps in the
  consumer env (coexisting with FA4 on quack 0.5.0); verify cute runs there — and that
  `MINIWORLD` (auto) trimul on B200 no longer hits the broken path.

## Risks
- **Phase 2** is the make-or-break unknown (gemm_interface workaround difficulty).
- **cutlass-dsl 4.5.2** CuTeDSL drift is a separate axis from quack and may add work.
- Likely a multi-session effort given the CuTeDSL depth.

## Recommended start
Phase 0 + Phase 1 → measure Phase 2 difficulty (A vs B) → then commit to the full scope.
