"""A kernel this card cannot run is a permanent, correct answer -- not a failure.

`run_all` had no notion of it. On an A6000 it reported six CuTeDSL kernels as `failed` because they
raise "expects arch to be sm_90a, but got sm_86" at every shape, which is the report being red for
hardware they were never written for. The build side already draws this distinction
(`is_bad_unit`, tests/test_permanent_skip_classification.py); this is the same rule for the
driver/checker side.

Two mechanisms, in this order:

1. **The declaration.** `registry.csv`'s `arch` column says the minimum, so a kernel that cannot
   run here is never launched and never compiled. Cheap, and it does not depend on matching the
   text of somebody else's exception.
2. **A runtime backstop**, for a row whose declaration is missing or wrong. A kernel that passes
   the declared gate and *then* refuses on arch grounds is a registry error, and `run_all` prints
   it as one instead of absorbing it into the skip count.

The predicate lives in `run_all`, not in a test. It was in `tests/test_numerical.py` -- which is
why `run_all` did not have it.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune.run_all import _sm, is_arch_gated, meets_arch


@pytest.mark.parametrize("detail", [
    "OpError: expects arch to be sm_90a, but got sm_86",
    "OpError: expects arch to be one of [Arch.sm_100a, Arch.sm_100f, ...]",
    "AssertionError: SM90 (H100) only",
    "RuntimeError: unsupported gpu architecture compute_120",
    "this kernel requires sm_100",
])
def test_an_arch_refusal_is_recognised(detail: str) -> None:
    assert is_arch_gated(detail)


@pytest.mark.parametrize("detail", [
    "WRONG NUMBERS: rel out=3.10e-01 (band 5e-02)",
    "NameError: name 'L' is not defined",
    "FileNotFoundError: [Errno 2] No such file or directory: '.../transition_cuda.cpp'",
    "CUDA out of memory",
    "checker returned nothing",
])
def test_a_real_failure_is_not_mistaken_for_an_arch_refusal(detail: str) -> None:
    """The important direction. Calling a genuine bug "wrong card" hides it, and the two bugs this
    suite found on its first run -- an undefined name and a missing source file -- both carry text
    a sloppier predicate could have swallowed."""
    assert not is_arch_gated(detail)


@pytest.mark.parametrize(("arch", "expected"), [
    ("sm86", 86), ("sm100", 100), ("SM90", 90), ("", -1), ("hopper", -1),
])
def test_arch_strings_order_numerically(arch: str, expected: int) -> None:
    """String comparison would put sm100 below sm86, which is the whole hazard."""
    assert _sm(arch) == expected


def test_string_order_would_have_been_wrong() -> None:
    """Guards the reason `_sm` exists: "sm100" < "sm86" lexically."""
    assert "sm100" < "sm86"
    assert _sm("sm100") > _sm("sm86")


@pytest.mark.parametrize(("declared", "device", "runnable"), [
    ("sm80", "sm86", True),
    ("sm86", "sm86", True),
    ("sm90", "sm86", False),
    ("sm100", "sm86", False),
    ("sm100", "sm100", True),
    ("sm80", "sm100", True),
    ("", "sm86", True),          # blank declares nothing, so never block a launch
])
def test_the_declared_minimum_decides(declared: str, device: str, runnable: bool) -> None:
    assert meets_arch({"kernel": "k", "arch": declared}, device) is runnable


def test_a_row_with_no_arch_column_still_runs() -> None:
    """The column is newer than the registry. A row without it must not become unrunnable."""
    assert meets_arch({"kernel": "k"}, "sm86") is True


def test_the_registry_and_the_gate_agree_on_what_this_repo_declares() -> None:
    """Every declared arch must be one `meets_arch` can actually order. A typo like `sm_90` would
    parse to -1 and silently let the kernel launch on any card."""
    import csv
    from pathlib import Path

    registry = Path(__file__).resolve().parents[1] / "src/miniworld_engine/kernels/registry.csv"
    with registry.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    unparseable = [r["kernel"] for r in rows if _sm(r.get("arch") or "sm80") < 0]
    assert not unparseable, f"unparseable arch values: {unparseable}"
