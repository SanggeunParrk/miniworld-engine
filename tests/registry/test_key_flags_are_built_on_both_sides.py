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
    # `row_scale` folds the AF pair-mask into the LN epilogue and every producer of it is a cute
    # path, which `dispatch` selects only at sm90+. The driver therefore branches on the card
    # (`drivers/layernorm.py:_sm90plus`), so the sm80/sm86 caches are legitimately =0 only and an
    # H100 cache will hold both. When that lands, `test_one_sided_entries_are_still_one_sided`
    # retires these two entries on its own.
    ("layernorm_fwd_saveact_triton", "HAS_ROWSCALE"):
        "drivers/layernorm.py:91-94 drives =1 only when `_sm90plus()`; below sm90 the triton "
        "trimul deliberately does not fold the mask into LN_in (unidirectional.py:225-227), and "
        "the =1 lookups an A100 replay shows come from `builder.cases()` forcing "
        "implementation='cute', which then aborts with `Gemm Sm80 is not implemented yet`",
    ("gated_projection_bwd_gate_dropres_triton", "FROM_PREACT"):
        "drivers/trimul_inproj.py gates the =1 probe on `_sm100()`. The preact form is passed only "
        "by the sm100 merged-training paths (cute/bidir_training_sm100.py, "
        "cute/v6_training_merged_sm100.py), which `dispatch` selects only on B200; below sm90 it "
        "is a program nothing can launch. This entry retires itself when a B200 cache lands",
    ("layernorm_bwd_atomic_triton", "HAS_ROWSCALE"):
        "drivers/layernorm.py:111-121, the backward of the same forward and the same sm90+ gate; "
        "`_bwd_atomic_impl` pins =0 (compile_native.py:154) and the =1 launcher (main.py:419) is "
        "reached only through the cute paths",
}

#: Ops whose driver NOW drives both values but whose committed cache predates that build.
#: This is not a free pass: :func:`test_awaiting_rebuild_is_actually_pending` requires each one to
#: be reported STALE by the fingerprint scanner, i.e. the driver really did change and the cache
#: really is due a rebuild. When the rebuild lands the op stops being STALE and the entry has to
#: go -- at which point the flag must genuinely be two-sided or the test above fails. So an entry
#: here expires on its own rather than quietly becoming permanent.
AWAITING_REBUILD: frozenset[str] = frozenset({
    # The A100 rebuild has landed for every op that was here, and the caches show the drivers did
    # what they were corrected to do: trimul_gemm_gate_mmajor now holds SAVE_PREACT 0 AND 1 (the
    # inference side had never been built) and transition_bwd_swiglu_recompute holds NORMALIZE 0
    # and 1. They are covered by the both-sides test directly now. (transition_fwd_b2b was here
    # for ADD_RESIDUAL, which no longer exists: the residual follows from HAS_LN, and HAS_LN is
    # driven on both sides.)
    #
    # gated_projection_bwd_gate_dropres_triton was here for USE_DROPOUT, which is gone: that
    # kernel is the TRAINING backward and every training launch now carries a drop scale (ones
    # when p_drop is 0), so the flag had one live value and keying it would have made every
    # backward miss the cache. Nothing is awaiting a rebuild.
})


SRC = ROOT / "src" / "miniworld_engine"


def _live_key_flags() -> dict[str, set[str]]:
    """op -> the flag names its CURRENT ``@triton.autotune(key=[...])`` lists.

    A committed cache is a record of the key the kernel had WHEN IT WAS BUILT, so a frozen cache
    for a card the project no longer builds on can carry names the live key dropped -- the sm86
    `transition_bwd_swiglu_recompute_triton` entries still spell `K=`, `ND=` and `STACK_DAB=`,
    all three of which left `fused.py`'s key. Checking a dead name for two-sidedness asks the
    driver to build a bucket that no longer exists; that staleness is the fingerprint scanner's
    job (those files are STALE on `key_scheme`), not this test's.

    Parsed from the source rather than by importing, so this stays a CPU test on a box with no
    triton: `configs_for("<op>")` and `key=[...]` sit in the same decorator, and `@triton.jit`
    always terminates it. The sweep is over all of `miniworld_engine`, not just `kernels/` --
    the DTv1 trimul baseline keeps its `@triton.autotune` in
    `modules/triangle_multiplication/baseline_dtv1.py`, and four ONE_SIDED entries are about it.
    """
    import re

    live: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        for m in re.finditer(r'configs_for\(\s*"([A-Za-z0-9_]+)"', text):
            tail = text[m.end():]
            stop = tail.find("@triton.jit")
            decorator = tail[:stop] if stop != -1 else tail
            k = re.search(r"key\s*=\s*\[(.*?)\]", decorator, re.DOTALL)
            if k is None:
                continue
            names = set(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", k.group(1)))
            live.setdefault(m.group(1), set()).update(names - {"shape_key"})
    return live


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


def test_the_live_key_parser_still_matches() -> None:
    """Guard the guard: a decorator rewrite that broke the parse would mute every check below."""
    live = _live_key_flags()
    assert live.get("transition_bwd_swiglu_recompute_triton") == {"NORMALIZE", "STORE_H"}, live.get(
        "transition_bwd_swiglu_recompute_triton")
    # gate_elem is now two flagless kernels (inference / training), so its keys are bare.
    assert live.get("gated_projection_gate_dropres_triton") == set(), live.get(
        "gated_projection_gate_dropres_triton")
    assert live.get("gated_projection_gate_res_triton") == set()
    assert live.get("gated_projection_bwd_gate_dropres_triton") == {"FROM_PREACT"}, live.get(
        "gated_projection_bwd_gate_dropres_triton")
    assert live.get("trimul_gemm_gate_saveact_triton") == {
        "K", "ALLOW_TF32", "APPLY_MASK", "TRANSPOSE_OUT"}, "the sweep no longer reaches modules/"
    # Not a round number: every op that carries a `configs_for(...)` autotune decorator. A parse
    # that silently stopped matching would drop well below this and mute both checks above.
    assert len(live) >= 60, f"the key parse found only {len(live)} autotuned ops"


def test_every_boolean_key_flag_is_tuned_on_both_sides() -> None:
    live = _live_key_flags()
    booleans = {(op, f): v for (op, f), v in _flag_values().items()
                if v <= {"0", "1"} and f in live.get(op, set())}
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
    are frozen history -- they were written by drivers and key lists that have since changed, so
    unioning them with a current card's cache can read as "both values are built" and retract a
    declaration that is true of every cache the current code produces.
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
    live = _live_key_flags()
    dead = sorted(k for k in ONE_SIDED if k[1] not in live.get(k[0], set()))
    assert not dead, (
        f"ONE_SIDED names flags that are no longer in their kernel's autotune key: {dead}. "
        f"The declaration explains why a bucket is not built; with the flag out of the key there "
        f"is no such bucket. Drop the entry.")


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
