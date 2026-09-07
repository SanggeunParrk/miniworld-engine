"""A kernel that comes out of a whole build with no measurement is a broken driver, and says so.

`trimul_outproj_layernorm_gemm_gate_triton` called its own op without the required `eps`, so every
unit of it raised before a single config was timed -- on every card, for as long as the op had
existed. The build's exit rule was "fail only if nothing succeeded", which is right for the case it
was written for (a shape too big for the card is an ordinary skip, and a build that measured most
of the sweep should still ship what it measured) and blind to this one: 195 units succeeded, 12
died identically, the build exited 0, and the kernel shipped with no cache at all.

The distinction the exit code needs is not "did a unit fail" but "did a KERNEL come out empty".
"""
from __future__ import annotations

from miniworld_engine import cli


def _r(label, ops, rc=0, log="x.log"):
    return {"label": label, "ops": ops, "rc": rc, "log": log}


def test_a_kernel_with_no_measurement_anywhere_is_named() -> None:
    dead = cli._ops_that_measured_nothing([
        _r("good_triton[bfloat16] L=128", 3),
        _r("broken_triton[bfloat16] L=128", 0, rc=1, log="a.log"),
        _r("broken_triton[bfloat16] L=256", 0, rc=1, log="b.log"),
    ])
    assert set(dead) == {"broken_triton"}
    assert dead["broken_triton"] == "a.log", "the report must point at a log to read"


def test_a_kernel_that_measured_at_one_shape_is_not_named() -> None:
    """The ordinary case: a shape that does not fit the card. Not a driver failure."""
    dead = cli._ops_that_measured_nothing([
        _r("k_triton[bfloat16] L=128", 3),
        _r("k_triton[bfloat16] L=2048", 0, rc=1),
    ])
    assert dead == {}


def test_dtypes_of_one_kernel_are_one_kernel() -> None:
    """bf16 and fp32 are separate units of the SAME driver; only both empty is a failure."""
    assert cli._ops_that_measured_nothing([
        _r("k_triton[bfloat16] L=128", 0, rc=1),
        _r("k_triton[float32] L=128", 2),
    ]) == {}
    assert set(cli._ops_that_measured_nothing([
        _r("k_triton[bfloat16] L=128", 0, rc=1),
        _r("k_triton[float32] L=128", 0, rc=1),
    ])) == {"k_triton"}


def test_a_module_unit_is_not_a_kernel() -> None:
    """`Unit.label` carries no dtype bracket. A module unit tunes whatever its case touches, so
    'it measured nothing' names no kernel and must not be reported as a broken driver."""
    assert cli._ops_that_measured_nothing([_r("transition-miniworld-L256-eval", 0, rc=1)]) == {}


def test_an_all_empty_build_is_not_reported_kernel_by_kernel() -> None:
    """Every kernel empty is the existing 'nothing succeeded' failure, which already exits non-zero
    and prints the per-unit reason. Naming all 80 as broken drivers would bury it."""
    dead = cli._ops_that_measured_nothing([_r(f"k{i}_triton[bfloat16] L=128", 0, rc=1)
                                           for i in range(3)])
    assert len(dead) == 3, "grouping still works; the CALLER decides how loud to be"
