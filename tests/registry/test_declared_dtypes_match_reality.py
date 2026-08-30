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
PKG = Path(cache.__file__).resolve().parents[1]
REG = PKG / "kernels" / "registry.csv"
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
    #: The whole family. "fp32 io with TF32 tensor cores" is what every file in it says, the
    #: module is `dtype: torch.dtype = torch.float32`, and every bucket the shipped cache holds
    #: for it is float32. The column carried bf16 as well for a while, on the argument that
    #: krystal reaches these kernels through `miniworld_engine.ops` with the model's own bf16
    #: tensors. That was true and beside the point: the path is being removed, and MiniWorld runs
    #: this family in fp32. A declared precision nobody runs is a duplicate of the entire work
    #: list -- here, half of the two kernels that were a quarter of the sweep.
    FP32_ONLY = {op for op in declared if op.startswith("cond_transition_")}
    for op, want in declared.items():
        if op.startswith("cond_transition_"):
            # The family used to be fp32 EVERYWHERE, in the driver and in this column together --
            # two true halves that added up to never tuning krystal's own precision, which is
            # bf16. The driver was fixed to build at the overridable `BF16` name; this column was
            # the half left behind, so all eight rows still said fp32 and the build made fp32
            # units for kernels the model only ever calls in bf16.
            assert want == {"float32"}, (
                f"{op}: this family is fp32 io with TF32 tensor cores, its module is "
                f"dtype=torch.float32, and every cached bucket it has is float32. A bf16 "
                f"declaration here builds a second copy of the work list for a precision "
                f"MiniWorld does not run it at")
        if op.startswith(("gated_projection_gate", "gated_projection_bwd_gate")):
            if op.endswith("lowp_triton"):
                continue
            assert want == {"bfloat16"}, (
                f"{op}: gated_projection/triton/main.py casts its operands with "
                f".to(torch.bfloat16) before the launch")


def test_a_dtype_guarded_module_declares_only_bf16():
    """A kernel behind `guard_dtype` cannot be reached at fp32, so it must not declare fp32.

    `dispatch._FAST_KERNEL_DTYPES` is `{torch.bfloat16}` and `guard_dtype` sends anything else to
    the pytorch reference before a kernel is chosen. So for a family whose module calls it, an
    `fp32` cell is not a capability claim -- it is a second copy of the whole work list for a
    precision production cannot hand it. transition_bwd_swiglu_recompute was built at both and its
    cache records only `bfloat16+float32`, the mixed-operand label of the BF16 run: bf16
    activations against the fp32-pinned norm affine params (`modules/primitives.py`, "Pin this
    module's floating-point params/buffers to fp32"). One launch, labelled with both operand types,
    not two precisions -- which is why test_a_single_precision_kernel_declares_exactly_that_precision
    excludes mixed keys and could never catch this. The dispatch source can.

    Scoped to the modules that actually call the guard. layernorm, adaln and augmented_attention do
    NOT -- they are reached from inside the diffusion blocks, which have their own precision -- so
    their fp32 declarations are not settled by this argument and are left alone.
    """
    import re

    guarded = set()
    for f in (PKG / "modules").rglob("*.py"):
        if "guard_dtype(" not in f.read_text() or f.name == "dispatch.py":
            continue
        guarded.add(f.parent.name)
    assert guarded, "no module calls guard_dtype; this test has lost its subject"

    src = (PKG / "modules" / "dispatch.py").read_text()
    m = re.search(r"_FAST_KERNEL_DTYPES\s*=\s*frozenset\(\{([^}]*)\}\)", src)
    assert m and m.group(1).strip() == "torch.bfloat16", (
        "dispatch._FAST_KERNEL_DTYPES is no longer bf16-only, so a guarded module CAN be handed "
        f"another precision and this test's premise is gone: {m.group(1) if m else '?'}")

    bad = []
    for r in csv.DictReader(REG.open()):
        if r["family"] not in guarded or r["backend"] != "triton":
            continue
        if "fp32" in (r["dtypes"] or "").split("|") and "_fp32_" not in r["kernel"]:
            bad.append(f"{r['kernel']} ({r['family']}) declares {r['dtypes']}")
    assert not bad, ("kernels behind guard_dtype declaring a precision it never lets through -- "
                     "each one doubles its share of every build:\n  " + "\n  ".join(bad))
