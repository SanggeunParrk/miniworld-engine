"""registry.csv's `dtypes` must be what the kernel can actually run, not what it might.

29 of 91 rows declared `bf16|fp32` for a kernel that is fixed-precision by construction, and the
work list took them at their word. So `build all` ran both halves, both measured the same thing,
and the second wrote its result under a dtype the kernel never saw:

    op-cond_transition_swiglu_triton-bfloat16-L256.json  ->  {"float32|ND=512,shape_key=256": ...}

192 of the 922 units were that duplicate, and `audit` reported the 192 declared-but-absent pairs
as holes -- against a cache that was complete. Nothing was wrong with the build; the declaration
was. The evidence was in the shards the whole time: what a unit RECORDS is what the kernel ran.

drivers/conditioned_transition.py's docstring already said so for one family ("The conditioned_transition family
runs fp32: every file in it states 'fp32 io with TF32 tensor cores'"), and
gated_projection/triton/main.py casts its operands with `.to(torch.bfloat16)` before the launch
for the other. Neither fact had reached the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache

DATA = Path(cache.__file__).parent / "data"
PKG = Path(cache.__file__).resolve().parents[1]
from paths import registry_rows

ALIAS = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
#: The card whose build is complete and committed; the completeness half is scoped to it.
COMPLETE_GPU = "NVIDIA RTX A6000 (sm86)"
DISPATCH_DIRS = {"ln_bwd_dispatch", "bias_only_dispatch"}


def _declared() -> dict[str, set[str]]:
    out = {}
    for r in registry_rows():
        if r["backend"] == "triton" and (r["driver"] or "").strip():
            out[r["kernel"]] = {ALIAS.get(x, x) for x in (r["dtypes"] or "").split("|") if x}
    return out


def _recorded(gpu: str | None = None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in DATA.glob("*/*.json"):
        if f.parent.name in DISPATCH_DIRS:
            continue
        d = json.loads(f.read_text())
        if gpu is not None and d.get("gpu") != gpu:
            continue
        for k in d.get("entries", {}):
            for t in k.split("|", 1)[0].split("+"):
                out.setdefault(d["op"], set()).add(t)
    return out


def test_every_declared_op_has_at_least_one_dtype():
    empty = sorted(op for op, ts in _declared().items() if not ts)
    assert not empty, f"{len(empty)} row(s) declare no dtype: {empty[:5]}"


def test_a_single_precision_kernel_declares_exactly_that_precision():
    """The bug, and its mirror. Scoped to the one card whose build is complete.

    A bucket key records the dtypes the kernel SAW, so `bfloat16+float32` means mixed operands in
    one run -- bf16 activations against fp32 statistics, say -- not two precisions. Those say
    nothing about what the kernel can be tuned at, so they are excluded here. An op whose every
    entry records exactly ONE dtype is unambiguous: that is the only precision it runs, and the
    declaration has to say so and nothing more.
    """
    recorded = _recorded(COMPLETE_GPU)
    if len(recorded) < 80:
        pytest.skip(f"{COMPLETE_GPU} cache is not complete enough to judge ({len(recorded)} ops)")
    declared, wrong = _declared(), []
    for op, got in recorded.items():
        if len(got) != 1 or op not in declared:
            continue
        if declared[op] != got:
            wrong.append((op, sorted(declared[op]), sorted(got)))
    assert not wrong, (
        f"{len(wrong)} op(s) declare a precision set their kernel does not run. Over-declaring "
        f"costs a duplicate unit in every build and a phantom hole in every audit; "
        f"under-declaring leaves a precision untuned: {wrong[:5]}")


def test_the_two_families_the_bug_was_found_in_are_single_precision():
    """Named, because both are documented in the source and neither had reached the registry."""
    declared = _declared()
    #: Empty, and it took a measurement to get here. cond_transition_fwd_b2b_saveact used to be
    #: the exception: ConditionedTransitionTailFunction reroutes its bf16 calls to the cuBLAS
    #: split, so the argument ran that a bf16 unit would measure a path nothing takes, and its
    #: driver named torch.float32 to match. But the driver calls `_b2b_fwd_train` BELOW the
    #: autograd Function, where the reroute does not apply -- so a bf16 unit measures the kernel,
    #: which is exactly the number `training.py` asks for when it says lifting the reroute "needs
    #: a measurement, not an argument". The pin was what made that number impossible to take: the
    #: reroute justified the pin and the pin protected the reroute. Both halves of the original
    #: break are now accounted for -- the dtype half is fixed (3.13e-03 at bf16, the same as the
    #: inference twin the model already runs there) and the spill half is what the build will
    #: report. The reroute stays in production until it does.
    FP32_ONLY: set[str] = set()
    for op, want in declared.items():
        if op.startswith("cond_transition_"):
            # The family used to be fp32 EVERYWHERE, in the driver and in this column together --
            # two true halves that added up to never tuning krystal's own precision, which is
            # bf16. The driver was fixed to build at the overridable `BF16` name; this column was
            # the half left behind, so all eight rows still said fp32 and the build made fp32
            # units for kernels the model only ever calls in bf16.
            expect = {"float32"} if op in FP32_ONLY else {"bfloat16", "float32"}
            assert want == expect, (
                f"{op}: conditioned_transition builds at drivers.BF16 and krystal runs bf16, so "
                f"the row declares both -- except {sorted(FP32_ONLY)}, whose bf16 calls are "
                f"rerouted before they reach the kernel")
        if op.startswith(("gated_projection_gate", "gated_projection_bwd_gate")):
            if op.endswith("lowp_triton"):
                continue
            assert want == {"bfloat16"}, (
                f"{op}: gated_projection/triton/main.py casts its operands with "
                f".to(torch.bfloat16) before the launch")


def test_a_kernel_behind_the_dtype_guard_does_not_declare_fp32():
    """`guard_dtype` is bf16-only, so an fp32 cell behind it is a duplicate work list.

    `dispatch._FAST_KERNEL_DTYPES` is `frozenset({torch.bfloat16})` and `guard_dtype` sends
    anything else to the pytorch reference BEFORE a backend is chosen
    (`modules/transition/module.py:141`). A kernel reached only through a module that calls it
    therefore never sees fp32, whatever the column says, and every unit built at that precision
    measures a path production cannot take.

    The cache says the same thing in its own words: transition_fwd_b2b_ktiled records
    `bfloat16+float32` in all 22 of its buckets and plain `float32` in none. That key is the
    MIXED-OPERAND label of the bf16 run -- bf16 activations against the fp32-pinned norm affine
    params (`modules/primitives.py`, "Pin this module's floating-point params/buffers to fp32") --
    one launch wearing both operand types. `test_a_single_precision_kernel_declares_exactly_that_precision`
    excludes mixed keys for exactly that reason and so cannot catch this; the dispatch source can.

    Scoped to families whose module calls the guard, and to triton rows, which are the ones a build
    spends units on. conditioned_transition is NOT one of them -- krystal reaches it through
    `miniworld_engine.ops` rather than through a module -- and it declares both precisions.
    """
    guarded = {f.parent.name for f in (PKG / "modules").rglob("*.py")
               if f.name != "dispatch.py" and "guard_dtype(" in f.read_text()}
    assert guarded, "no module calls guard_dtype; this test has lost its subject"
    src = (PKG / "modules" / "dispatch.py").read_text()
    assert "_FAST_KERNEL_DTYPES = frozenset({torch.bfloat16})" in src, (
        "dispatch._FAST_KERNEL_DTYPES is no longer bf16-only, so a guarded module CAN be handed "
        "another precision and this test's premise is gone")
    bad = [f"{r['kernel']} ({r['family']}) declares {r['dtypes']}"
           for r in registry_rows()
           if r["family"] in guarded and r["backend"] == "triton"
           and "fp32" in (r["dtypes"] or "").split("|") and "_fp32_" not in r["kernel"]]
    assert not bad, ("kernels behind guard_dtype declaring a precision it never lets through -- "
                     "each doubles its share of every build:\n  " + "\n  ".join(bad))
