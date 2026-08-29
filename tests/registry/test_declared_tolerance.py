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

REGISTRY = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src/miniworld_engine/kernels/registry.csv"


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
        # At each precision the ROW declares. A per-precision band cannot be read without one --
        # deliberately, since handing an fp32 run a bf16 band is the hole that spelling closes --
        # so "does it parse" is a question per declared precision, not per row.
        for dt in (a.strip() for a in (row.get("dtypes") or "bf16").split("|")):
            if dt:
                declared_rtol(row, dt)   # raises with the kernel's name if unparseable


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


def test_a_declared_band_is_above_what_that_kernel_measured() -> None:
    """A band below the kernel's own measured error fails on a correct run; far above it catches
    nothing. Both are checked against `autotune/manifests/`, which run_all writes.

    The manifests are the only record of what each kernel actually costs, so a band that drifts
    away from them -- in either direction -- means the declaration and the measurement have stopped
    describing the same kernel.
    """
    import collections
    import math
    import re

    from miniworld_engine.autotune.run_all import DEFAULT_RTOL, RTOL_MARGIN

    manifests = REGISTRY.parent.parent / "autotune" / "manifests"
    # (kernel, precision) -> the worst rel each run recorded. Keyed by BOTH, because the same
    # kernel measures four orders apart in the two: `layernorm_fwd_saveact_triton` is 2.8e-03 in
    # bf16 and 1.7e-07 in fp32. Comparing a band against the wrong precision's rows is how a bf16
    # band gets called "far above what the kernel measures" -- or, worse, how an fp32 band gets
    # derived from bf16 rounding and passes anything.
    measured: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for f in sorted(manifests.glob("*.csv")):
        with f.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["status"] != "ok":
                    continue
                # A row written before the column existed is bf16: it was the only precision the
                # drivers ever ran. `devices.load_manifest` fills the same default.
                dt = (row.get("dtype") or "").strip() or "bf16"
                vals = [float(v) for v in re.findall(r"=([0-9.e+-]+)", row["detail"] or "")
                        if "e" in v or "." in v]
                if vals:
                    measured[(row["kernel"], dt)].append(max(vals))
    assert measured, f"no measurements in {manifests}; the check below would pass vacuously"

    too_tight, too_loose = [], []
    with REGISTRY.open(newline="") as fh:
        for row in csv.DictReader(fh):
            band = (row.get("rtol") or "").strip()
            if not band:
                continue
            # Every precision the ROW declares, against that precision's own rows. A precision the
            # manifests have not measured yet is skipped -- absence of evidence is not a band
            # error -- and one the row does not declare is not checked at all.
            for dt in (a.strip() for a in (row.get("dtypes") or "bf16").split("|")):
                worst = max(measured.get((row["kernel"], dt), [0.0]) or [0.0])
                if not dt or not worst:
                    continue
                b = declared_rtol(row, dt) if "=" in band else float(band)
                where = f"{row['kernel']} [{dt}]"
                if b < worst:
                    too_tight.append(f"{where}: band {b:.1e} < measured {worst:.1e}")
                elif b >= DEFAULT_RTOL:
                    continue                  # capped at the default, deliberately
                elif b > worst * RTOL_MARGIN * 1.5:
                    too_loose.append(f"{where}: band {b:.1e} vs measured {worst:.1e} "
                                     f"({b / worst:.0f}x, margin is {RTOL_MARGIN:.0f}x)")
    assert not too_tight, (
        "declared band below the kernel's own measured error -- a correct run would fail:\n  "
        + "\n  ".join(too_tight))
    assert not too_loose, (
        "declared band far above what the kernel measures, so it catches nothing:\n  "
        + "\n  ".join(too_loose)
        + f"\n  Re-derive from {manifests.name}/ at {RTOL_MARGIN:.0f}x, or say why in the row.")
    assert math.isfinite(DEFAULT_RTOL)
