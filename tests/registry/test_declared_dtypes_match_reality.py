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

import csv
import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache

DATA = Path(cache.__file__).parent / "data"
REG = Path(cache.__file__).resolve().parents[1] / "kernels" / "registry.csv"
ALIAS = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
#: The card whose build is complete and committed; the completeness half is scoped to it.
COMPLETE_GPU = "NVIDIA RTX A6000 (sm86)"
DISPATCH_DIRS = {"ln_bwd_dispatch", "bias_only_dispatch"}


def _declared() -> dict[str, set[str]]:
    out = {}
    for r in csv.DictReader(REG.open()):
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
    #: The one conditioned_transition kernel that really is fp32-only, and why: the training
    #: autograd Function reroutes bf16 away from it ("use cuBLAS split"), so no bf16 unit would
    #: measure the kernel -- it would measure a path nothing takes. Its driver names
    #: torch.float32 outright rather than the overridable BF16 name, for the same reason.
    FP32_ONLY = {"cond_transition_fwd_b2b_saveact_triton"}
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
