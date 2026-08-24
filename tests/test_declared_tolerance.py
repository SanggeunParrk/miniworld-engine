"""A kernel is held to ITS band, not to the weakest kernel's.

`run_all.check_one` applied one constant, `rel < 5e-2`, to all 99 declared checkers. That is bf16's
~3-decimal band, which is right for a fused attention backward and far too loose for a transpose, a
mask fold or a gate multiply -- several of which should be very nearly exact. A reduction-order
change costing 1e-3 passes silently under 5e-2, and 1e-3 on a residual accumulated over 48
pairformer blocks is not silent in the model.

`registry.csv` now carries an `rtol` column. Blank means "the default applies", never "unchecked".
These tests cover the mechanism with synthetic checkers -- no GPU, no kernel -- so the band logic is
verified independently of whether any particular kernel currently meets a tighter one. Calibrating
the per-kernel values is a separate step and needs a device (plan.md P2b).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from miniworld_engine.autotune.run_all import DEFAULT_RTOL, check_one, declared_rtol

REGISTRY = Path(__file__).resolve().parents[1] / "src/miniworld_engine/kernels/registry.csv"


def _rows() -> list[dict]:
    with REGISTRY.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_column_exists_and_every_value_parses() -> None:
    """A malformed band must be an error, not a silent fall back to the loose default -- a typo
    that widens a kernel's tolerance is what this column exists to prevent."""
    rows = _rows()
    assert rows, "registry.csv is empty"
    assert "rtol" in rows[0], "registry.csv has no rtol column"
    for row in rows:
        declared_rtol(row)          # raises with the kernel's name if unparseable


def test_a_blank_band_means_the_default() -> None:
    assert declared_rtol({"kernel": "k", "rtol": ""}) is None
    assert declared_rtol({"kernel": "k"}) is None
    assert declared_rtol({"kernel": "k", "rtol": "  "}) is None


def test_a_declared_band_is_read() -> None:
    assert declared_rtol({"kernel": "k", "rtol": "1e-3"}) == pytest.approx(1e-3)
    assert declared_rtol({"kernel": "k", "rtol": "0"}) == 0.0


@pytest.mark.parametrize("bad", ["one", "1e", "", " x ", "--3"])
def test_a_malformed_band_names_the_kernel(bad: str) -> None:
    if not bad.strip():
        pytest.skip("blank is the documented default, covered above")
    with pytest.raises(ValueError, match="my_kernel"):
        declared_rtol({"kernel": "my_kernel", "rtol": bad})


def test_a_negative_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        declared_rtol({"kernel": "k", "rtol": "-1e-3"})


# --- the comparison itself, against synthetic checkers ------------------------------------- #

def _checker(actual: float, expected: float):
    """A checker is a zero-argument callable returning (actual, expected)."""
    import torch

    def fn():
        return torch.tensor([actual]), torch.tensor([expected])
    return fn


def _register(monkeypatch, fn) -> str:
    """Give `check_one` a resolvable path to `fn` without touching the registry."""
    import miniworld_engine.autotune.run_all as run_all

    monkeypatch.setattr(run_all, "_resolve", lambda _path: fn)
    return "synthetic:checker"


def test_a_tighter_declared_band_rejects_what_the_default_accepts(monkeypatch) -> None:
    """The whole point: an error of 1e-2 is fine at 5e-2 and wrong at 1e-3."""
    path = _register(monkeypatch, _checker(1.0 + 1e-2, 1.0))
    ok_default, detail_default = check_one(path)
    assert ok_default, detail_default
    ok_tight, detail_tight = check_one(path, 1e-3)
    assert not ok_tight, detail_tight
    for expected in ("1e-03", "declared"):
        assert expected in detail_tight, f"{expected!r} missing from {detail_tight!r}"


def test_the_detail_names_the_band_it_was_measured_against(monkeypatch) -> None:
    """A failure that does not say the band leaves the reader hunting for a module constant."""
    path = _register(monkeypatch, _checker(2.0, 1.0))
    _, detail = check_one(path)
    assert f"{DEFAULT_RTOL:.0e}" in detail, detail


def test_a_nan_output_fails_every_band(monkeypatch) -> None:
    """NaN compares false against any threshold, so without the `worst == worst` guard a kernel
    writing NaN would pass -- including at a declared band of 0."""
    path = _register(monkeypatch, _checker(float("nan"), 1.0))
    for band in (None, 0.0, 1e-3, 1e9):
        ok, detail = check_one(path, band)
        assert not ok, f"band={band}: {detail}"


def test_an_exact_kernel_can_declare_zero(monkeypatch) -> None:
    """`rtol=0` must be usable and must mean exact -- it is the point of the column for the
    kernels that only move bytes around."""
    exact = _register(monkeypatch, _checker(1.0, 1.0))
    ok, detail = check_one(exact, 0.0)
    assert ok, detail
    off = _register(monkeypatch, _checker(1.0 + 1e-7, 1.0))
    ok, detail = check_one(off, 0.0)
    assert not ok, detail
