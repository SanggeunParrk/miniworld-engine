"""A kernel this run is not going to judge must not be launched, and must not be called a failure.

Two reasons a run passes a kernel by, and they behave identically: the card cannot run it (an sm100
kernel on an A6000), or the row does not declare this precision (a bf16-only kernel in an fp32
run). Both mean ABSENT. Neither means the kernel is wrong.

They were two checks in two places and only one of them was early. The precision check sat inside
`if ok and chk` -- consulted only once the drive had already SUCCEEDED, which is exactly the case
where skipping costs nothing. So an fp32 `run_all` drove every bf16-only kernel anyway, paid the
compile, and recorded `trimul_outproj_layernorm_gemm_gate_triton` as FAILED on a CompilationError
that says nothing except that nobody asked it to build at fp32. The same run left 44 of 91 kernels
in neither `results` nor `skipped`; the accounting line said so on stderr and nothing acted on it.

One predicate now answers "does this run launch this kernel", and it answers before the driver.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from paths import registry_rows

from miniworld_engine.autotune.run_all import declines_this_run

REGISTRY = next(p for p in Path(__file__).resolve().parents
                if (p / "pyproject.toml").is_file()) / "src/miniworld_engine/kernels/registry.csv"


def _row(**kw):
    base = {"kernel": "k", "arch": "", "dtypes": "bf16"}
    base.update(kw)
    return base


def test_a_kernel_this_card_and_this_precision_both_allow_is_launched():
    assert declines_this_run(_row(arch="sm80", dtypes="bf16"), "sm86", "bf16") is None


def test_the_wrong_card_declines_it():
    why = declines_this_run(_row(arch="sm100"), "sm86", "bf16")
    assert why is not None
    assert "sm100" in why, why
    assert "sm86" in why, why


def test_a_precision_the_row_does_not_declare_declines_it():
    why = declines_this_run(_row(dtypes="bf16"), "sm86", "fp32")
    assert why is not None
    assert "bf16" in why, why
    assert "fp32" in why, why
    assert declines_this_run(_row(dtypes="bf16|fp32"), "sm86", "fp32") is None


def test_a_blank_dtypes_column_means_bf16():
    """The column's own default, stated in `op_units` too: a row that says nothing is bf16."""
    assert declines_this_run(_row(dtypes=""), "sm86", "bf16") is None
    assert declines_this_run(_row(dtypes=""), "sm86", "fp32") is not None


def test_the_card_is_checked_before_the_precision():
    """Order matters only for the message. A kernel wrong on both counts should say the card --
    the reason someone can act on -- rather than a precision they cannot reach anyway."""
    why = declines_this_run(_row(arch="sm100", dtypes="bf16"), "sm86", "fp32")
    assert "sm100" in why, why


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_the_real_registry_leaves_nothing_unclassified(dtype):
    """The accounting the fp32 run failed: every row with a driver is launched or declined, and
    the two sets must cover it. This is the check that would have named the 44."""
    rows = [r for r in registry_rows() if (r.get("driver") or "").strip()]
    assert rows, "no rows with drivers; this would pass vacuously"
    launched = [r for r in rows if declines_this_run(r, "sm86", dtype) is None]
    declined = [r for r in rows if declines_this_run(r, "sm86", dtype) is not None]
    assert len(launched) + len(declined) == len(rows)
    assert launched, f"nothing at all runs at {dtype} on an sm86 card"


def test_every_declared_precision_is_one_the_drivers_can_build():
    """`declines_this_run` compares against `drivers.DTYPE_MODE`, which is bf16 or fp32 and
    nothing else. A row declaring a third name would be declined by every run, silently."""
    for r in registry_rows():
        for dt in (a.strip() for a in (r.get("dtypes") or "bf16").split("|")):
            assert dt in ("bf16", "fp32"), f"{r['kernel']}: dtypes names {dt!r}"
