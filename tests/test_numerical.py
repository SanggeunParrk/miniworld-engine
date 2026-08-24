"""Every kernel the registry declares a checker for must match its torch reference.

This is the suite `pixi run test-gpu` and the CI comment have named all along -- it did not
exist. The machinery did: `kernels/checks/<family>.py` holds a reference implementation per kernel, 99 of 103
registry rows name one, and `autotune/run_all.py` can run them. Nothing called it from pytest,
so a wrong number was only ever caught by someone running a module by hand.

Marked `gpu` and excluded from the CPU suite: these launch real kernels.

The tolerance is `run_all`'s, not a new one -- max |a-e| / max |e| < 5e-2, the band bf16's ~3
decimal digits and the bench already work in. Two suites disagreeing about what "correct" means
is worse than one of them not existing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

REG = Path(__file__).resolve().parents[1] / "src/miniworld_engine/kernels/registry.csv"

pytestmark = pytest.mark.gpu


def _rows():
    return list(csv.DictReader(REG.open()))


def _declared():
    """(kernel, checker) for every row that names one."""
    return [(r["kernel"], r["check"].strip(), r) for r in _rows() if (r.get("check") or "").strip()]


@pytest.fixture(scope="session", autouse=True)
def _needs_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize(("kernel", "checker", "row"), _declared(),
                         ids=[k for k, _, _ in _declared()])
def test_kernel_matches_its_reference(kernel, checker, row):
    """One test per kernel, so a failure names the kernel instead of the suite.

    The band is the kernel's own (`registry.csv`'s `rtol`), not a constant in this file: one band
    for all 99 is the weakest kernel's tolerance applied to every other one.
    """
    from miniworld_engine.autotune.run_all import check_one, declared_rtol

    ok, detail = check_one(checker, declared_rtol(row))
    if not ok and _is_arch_gated(detail):
        pytest.skip(f"not runnable on this device: {detail}")
    assert ok, f"{kernel}: {detail}"


def _is_arch_gated(detail: str) -> bool:
    """A kernel that refuses to run on THIS card is not a wrong answer.

    sm100-only CuTe kernels raise on an A6000 at every shape; failing them here would make the
    suite red on hardware they were never meant for, and a suite that is always red is off.
    """
    d = detail.lower()
    return any(m in d for m in ("expects arch to be one of", "unsupported gpu architecture",
                                "requires sm", "not supported on this",
                                # checkers assert their own arch gate, e.g. "SM90 (H100) only"
                                "sm90", "sm100", "h100 only", "b200 only", "only on sm"))


ALLOWED = Path(__file__).with_name("numerical_gaps_allowed.csv")


def test_every_kernel_with_a_driver_declares_a_checker():
    """A driver proves a kernel launches; only a checker proves the number is right.

    Pinned so the gap cannot quietly grow: 56 kernels once had a driver and no reference, and
    'ok' meant nothing more than 'did not raise'. The four still open are recorded in
    ``numerical_gaps_allowed.csv`` with a reason each -- the same shape as the repo's other
    allowed-gap files. A NEW kernel without a checker fails here; closing a recorded one and
    forgetting to delete its row fails here too, so the file cannot rot.
    """
    allowed = {r["kernel"]: r["reason"] for r in csv.DictReader(ALLOWED.open())
               if not r["kernel"].startswith("#")}
    missing = {r["kernel"] for r in _rows()
               if (r.get("driver") or "").strip() and not (r.get("check") or "").strip()}
    new = sorted(missing - set(allowed))
    assert not new, f"{len(new)} kernel(s) gained a driver with no checker: {new}"
    stale = sorted(set(allowed) - missing)
    assert not stale, f"these now HAVE a checker; drop them from {ALLOWED.name}: {stale}"
