"""A constexpr flag in an op's autotune key must be TUNED on both sides, or say why not.

`build all` runs the per-op pass (`cli.py`'s `module_pass` is false for `all`), and `OpUnit` is
`(op, dtype, side, length, width)` -- there is **no switch axis**. `builder.SWITCHES`, which does
sweep settings-driven switches, is read only by the module pass. So under the documented default
build, a flag value is covered if and only if the op's registry DRIVER calls it.

That is easy to get wrong silently, because a one-sided flag looks exactly like a full cache:
`build all` reports success, `dev cache-status` reports OK (the fingerprints are all about *what
the code is*, never about *what the cache covers*), and the miss only appears as a heuristic
fallback at run time. Measured cases when this test was written -- every one of them a kernel the
model runs in production:

* `trimul_gemm_gate_mmajor_triton` had `SAVE_PREACT=1` only. Both trimul INFERENCE paths pass
  `save_preact=False`, so the entire inference side of the front GEMM was untuned.
* `gate_elem`'s `ADD_RESIDUAL` / `USE_DROPOUT` / `SAVE_GATE` were all `0`, while the production
  call sites pass a residual on every launch and `return_gate=True` in training.
* `transition_fwd_b2b_triton` had `FUSE_STATS=0, ADD_RESIDUAL=0, SAVE_XN=0` only.
* `layernorm_bwd_foldstats_triton` had `PRIVATIZE_DGDB=1` only -- the kernel's own comment claimed
  "the autotune builder sweeps the off-default False side (builder.SWITCHES)", which is true of the
  module pass and false of `build all`.

So: every boolean flag in a committed cache's keys must appear with both values, or be listed in
:data:`ONE_SIDED` with the launch site that proves the other value never runs.

A flag is recognised as BOOLEAN by its observed values being a subset of {0, 1}; key entries that
carry a shape instead (``K=128``, ``N=512``) are not flags and are not checked here.
"""
from __future__ import annotations

import json
from collections import defaultdict

from paths import ROOT

DATA = ROOT / "src" / "miniworld_engine" / "autotune" / "data"

#: (op, flag) -> the launch site proving the missing value never runs in production.
#: An entry here is a claim about the CODE, not a to-do: it has to name where the flag is fixed.
ONE_SIDED: dict[tuple[str, str], str] = {
    ("layernorm_fwd_strided_triton", "HAS_W"):
        "adaln/triton/inference.py:142 passes HAS_W=True as a literal; `_cond_affine` is the only "
        "launcher of fused3's _ln_kernel and its whole contract is 'LayerNorm(cond) * lnw'",
    ("layernorm_linear_fwd_triton", "HAS_BIAS"):
        "layernorm_linear/triton/fused.py:47 states it, and it still holds: every in-repo launch "
        "site passes bias=None. The =1 program exists for an out-of-tree caller",
    # The four below belong to `modules/triangle_multiplication/baseline_dtv1.py` -- the DTv1
    # reference implementation kept for comparison, not a miniworld production path (it was
    # dropped from the trimul benchmark). Its cache exists so the baseline is measured fairly in
    # the configuration it actually runs; the far side of each flag is a baseline variant nothing
    # in this repo launches any more, and tuning it would spend build time on a reference.
    ("trimul_gemm_gate_saveact_triton", "ALLOW_TF32"):
        "baseline_dtv1.py:693 passes `torch.backends.cuda.matmul.allow_tf32`, the process global, "
        "which is False for the bf16 baseline this row declares (registry dtypes: bf16)",
    ("trimul_gemm_gate_saveact_triton", "APPLY_MASK"):
        "baseline_dtv1.py:684 `apply_mask = mask is not None`; the retired DTv1 baseline is no "
        "longer benched, so only the unmasked form it is tuned at is ever launched here",
    ("trimul_gemm_gate_saveact_triton", "TRANSPOSE_OUT"):
        "baseline_dtv1.py:692 -- the fused forward writes (N, M) and passes transpose_out=True; "
        "the =0 form is the un-fused variant the baseline does not take",
    ("trimul_outproj_gemm_gate_saveact_triton", "ALLOW_TF32"):
        "baseline_dtv1.py:733, same process-global as above on the output GEMM",
    # ---- the unconditional-residual family -------------------------------------------------
    # `_ADD_RESIDUAL = True` is a module-level constant whose own comment says to EDIT the line to
    # turn it off for raw-op benchmarking. So the =0 program is a bench toggle, never a path.
    ("trimul_outproj_layernorm_gemm_gate_triton", "ADD_RESIDUAL"):
        "modules/triangle_multiplication/module.py:212 `_ADD_RESIDUAL = True`, unconditional; the "
        "sole sm80 launcher (unidirectional.py:193) therefore always has a residual. =0 is only "
        "cute/inference.py:75. Replay: 12 lookups, all =1",
    # ---- settings whose off-default side `build all` can never reach -------------------------
    ("transition_fwd_b2b_triton", "FUSE_STATS"):
        "settings.py:201 `transition_fuse_stats: bool = False`; read only at fused.py:1412, and "
        "its only setter is builder.SWITCHES, which the per-op `build all` pass never consults",
    ("layernorm_bwd_foldstats_triton", "PRIVATIZE_DGDB"):
        "settings.py:213 defaults True and nothing in production sets it; =0 is reachable only "
        "from a build-harness pin (builder.SWITCHES, module pass only). NOTE fused.py:1141-1143 "
        "still claims the builder sweeps the False side -- true of that pass, not of `build all`",
    # ---- flags with no caller for the other value --------------------------------------------
    ("transition_fwd_b2b_triton", "SAVE_XN"):
        "every caller passes save_xn=False: modules/transition/module.py:265, :326, "
        "kernels/transition/whole_op.py:79",
    ("transition_layernorm_expand_swiglu_triton", "SAVE_XN"): "same three call sites",
    ("transition_bwd_swiglu_recompute_triton", "STORE_H"):
        "fused.py:1052 defaults store_h=True and both callers take it -- cute/fused.py:216 "
        "explicitly, fused.py:1573 by default. No caller anywhere passes False",
    ("adaln_epilogue_saveact_triton", "HAS_SB"):
        "training.py:206 keys on `scale_bias is not None`, and the sole launcher (training.py:699) "
        "always passes to_scale.bias -- built at modules/adaptive_layernorm/module.py:42 with "
        "primitives' default bias=True. Replay: 9 lookups, all =1",
    # ---- reachable only on another architecture ----------------------------------------------
}

#: Ops whose driver NOW drives both values but whose committed cache predates that build.
#: This is not a free pass: :func:`test_awaiting_rebuild_is_actually_pending` requires each one to
#: be reported STALE by the fingerprint scanner, i.e. the driver really did change and the cache
#: really is due a rebuild. When the rebuild lands the op stops being STALE and the entry has to
#: go -- at which point the flag must genuinely be two-sided or the test above fails. So an entry
#: here expires on its own rather than quietly becoming permanent.
AWAITING_REBUILD: frozenset[str] = frozenset({
    # Restored after a review found their "unreachable" side IS launched: the public `ops` facade
    # (`kernels/**/whole_op.py`) calls these kernels without `add_residual`, and the cute paths at
    # sm90+/sm100 pass `row_scale` / `from_preact`. Their caches predate the corrected drivers.
    "transition_fwd_b2b_triton",
    "gated_projection_gate_dropres_triton",
    "layernorm_fwd_saveact_triton",
    "layernorm_bwd_atomic_triton",
    # The TWO ops that are genuinely two-sided in production and whose committed cache predates
    # the driver that now drives both. Everything that was here moved to ONE_SIDED instead: the
    # second side turned out to be a benchmark toggle (`_ADD_RESIDUAL = True` is unconditional),
    # an off-default setting `build all` never flips, a flag with no caller, or another
    # architecture's path. Driving "both because it is in the key" was the wrong rule.
    "trimul_gemm_gate_mmajor_triton",          # SAVE_PREACT: replay asked 21x at 0, 12x at 1
    "transition_bwd_swiglu_recompute_triton",  # NORMALIZE=0 is the saved-xn backward, 6 misses
    # USE_DROPOUT: pairformer runs p_drop=0.25 in training (modules/pairformer/module.py:49) and
    # 0 at inference, so both sides are live. Its cache predates the corrected driver -- the
    # rebuild is in flight.
    "gated_projection_bwd_gate_dropres_triton",
})


def _flag_values() -> dict[tuple[str, str], set[str]]:
    """(op, flag) -> the values that appear in any committed cache for that op."""
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for op_dir in sorted(DATA.iterdir()):
        if not op_dir.is_dir():
            continue
        for cache in sorted(op_dir.glob("*.json")):
            try:
                entries = json.loads(cache.read_text()).get("entries", {})
            except (OSError, ValueError):
                continue
            for key in entries:
                _, _, rest = key.partition("|")
                for part in rest.split(","):
                    name, sep, value = part.partition("=")
                    if sep and name != "shape_key":
                        seen[(op_dir.name, name)].add(value)
    return seen


def test_there_are_caches_with_key_flags() -> None:
    """Guard the guard: a moved data root would make the sweep below find nothing to check."""
    assert _flag_values(), f"no keyed flags found under {DATA}"


def test_every_boolean_key_flag_is_tuned_on_both_sides() -> None:
    booleans = {k: v for k, v in _flag_values().items() if v <= {"0", "1"}}
    assert booleans, "no boolean key flags found -- the parser stopped matching"
    one_sided = sorted(k for k, v in booleans.items()
                       if len(v) == 1 and k not in ONE_SIDED and k[0] not in AWAITING_REBUILD)
    detail = "\n".join(f"    {op}: {flag}={next(iter(booleans[(op, flag)]))} only"
                       for op, flag in one_sided)
    assert not one_sided, (
        f"boolean key flags tuned on ONE side only, with no entry in ONE_SIDED:\n{detail}\n\n"
        f"`build all` has no switch axis -- a flag value is covered only if the op's registry "
        f"driver CALLS it. Either drive both values in kernels/drivers/<family>.py and rebuild "
        f"the op, or add the (op, flag) to ONE_SIDED naming the launch site that pins it.")


def _flag_values_per_gpu() -> dict[tuple[str, str, str], set[str]]:
    """(op, gpu, flag) -> values in THAT card's cache.

    The staleness check below has to be per-card. Caches for cards the project no longer builds on
    are frozen history: `trimul_outproj_layernorm_gemm_gate_triton` holds `ADD_RESIDUAL=0` on the
    two sm86 cards (from a driver that has since been corrected) and `=1` on the A100. Unioning
    those reads as "both values are built" and would retract a declaration that is true of every
    cache the current code produces.
    """
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for op_dir in sorted(DATA.iterdir()):
        if not op_dir.is_dir():
            continue
        for cache in sorted(op_dir.glob("*.json")):
            try:
                entries = json.loads(cache.read_text()).get("entries", {})
            except (OSError, ValueError):
                continue
            for key in entries:
                _, _, rest = key.partition("|")
                for part in rest.split(","):
                    name, sep, value = part.partition("=")
                    if sep and name != "shape_key":
                        seen[(op_dir.name, cache.stem, name)].add(value)
    return seen


def test_one_sided_entries_are_still_one_sided() -> None:
    """A declaration that has been overtaken by a rebuild is stale documentation; drop it."""
    per_gpu = _flag_values_per_gpu()
    booleans: dict[tuple[str, str], set[str]] = {}
    for (op, _gpu, flag), vals in per_gpu.items():
        if len(vals) > 1:
            booleans.setdefault((op, flag), set()).update(vals)
    stale = sorted(k for k, why in ONE_SIDED.items()
                   if len(booleans.get(k, set())) > 1)
    assert not stale, (
        f"ONE_SIDED still claims these are pinned, but the cache now holds both values: {stale}. "
        f"Remove the entry -- the claim is no longer true of the code.")


def test_awaiting_rebuild_is_actually_pending() -> None:
    """Every AWAITING_REBUILD op must be one the scanner agrees is due a rebuild.

    Without this the set is just a mute list. With it, an entry is a checkable claim -- "this op's
    build driver changed and the committed cache predates it" -- and it becomes a test failure the
    moment the rebuild lands and the claim stops being true.
    """
    import pytest

    from miniworld_engine.autotune import cache_status

    rows = [r for r in cache_status.scan() if r.op in AWAITING_REBUILD]
    if not rows:
        pytest.skip("no caches for these ops on this checkout")
    # A driver edit is no longer a STALE verdict -- it does not void a measurement, it changes
    # which buckets get built (see cache.build_rev). The scanner still REPORTS it, and that report
    # is what "a rebuild is owed" means here, so match on the reason rather than the verdict.
    pending = {r.op for r in rows if "driver" in (r.reason or "")}
    fresh = sorted(set(AWAITING_REBUILD) & {r.op for r in rows} - pending)
    assert not fresh, (
        f"AWAITING_REBUILD names ops whose driver drift the scanner no longer reports: {fresh}. "
        f"Either the rebuild landed -- drop them from AWAITING_REBUILD, and the both-sides test "
        f"now applies -- or the driver was never actually changed and the flag gap is still open.")
