"""A unit declared fp32 must build fp32, and only the environment carries that.

registry.csv's `dtypes` column splits an op into one unit per precision, and the drivers read
`MINIWORLD_DRIVER_DTYPE` once at import -- `drivers.DTYPE_MODE`, which decides what
`drivers.BF16` actually is. Between those two facts sits `OpUnit.env()`, and for a long time it
did not carry the dtype: every unit built at the default, bf16, so a float32 unit repeated the
bf16 one and no fp32 cache entry was ever produced, on any card.

Nothing failed when that was true. The unit LIST already carried float32 (test_declared_dtype_
coverage pins that) and the child took `--dtype` on its command line (unused on the op path), so
the declaration looked honoured everywhere except where it mattered. This file pins the link
itself: delete the assignment in OpUnit.env and these fail.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune.builder import OpUnit, op_units


def test_a_float32_unit_asks_the_drivers_for_fp32() -> None:
    env = OpUnit(op="k", length=256, dtype="float32").env()
    assert env.get("MINIWORLD_DRIVER_DTYPE") == "fp32", (
        "a float32 unit must set MINIWORLD_DRIVER_DTYPE=fp32; without it the drivers build at "
        "their default and the unit is a duplicate of the bf16 one")


def test_a_bfloat16_unit_asks_for_bf16() -> None:
    assert OpUnit(op="k", length=256, dtype="bfloat16").env()["MINIWORLD_DRIVER_DTYPE"] == "bf16"


def test_the_two_precisions_differ_in_the_environment() -> None:
    """The property the bug violated: two units of one op must not be the same build."""
    a = OpUnit(op="k", length=256, dtype="bfloat16").env()
    b = OpUnit(op="k", length=256, dtype="float32").env()
    assert a != b, "a bf16 unit and an fp32 unit of the same op produce identical environments"


def test_the_name_is_the_one_the_drivers_read() -> None:
    """Spelling drift between builder and drivers would be silent."""
    from miniworld_engine.kernels import drivers

    assert drivers.DTYPE_MODE in ("bf16", "fp32")
    values = {OpUnit(op="k", length=256, dtype=d).env()["MINIWORLD_DRIVER_DTYPE"]
              for d in ("bfloat16", "float32")}
    assert values == {"bf16", "fp32"}, (
        f"the builder emits {values}; drivers.DTYPE_MODE accepts bf16/fp32 and raises on anything "
        f"else, so a third spelling would fail at driver import inside the build")


def test_a_dtype_the_drivers_cannot_build_says_so() -> None:
    with pytest.raises(ValueError, match="float16"):
        OpUnit(op="some_kernel", length=256, dtype="float16").env()


def test_every_real_unit_carries_a_precision() -> None:
    units = op_units()
    assert units
    missing = [u.label for u in units if "MINIWORLD_DRIVER_DTYPE" not in u.env()]
    assert not missing, f"units built without a declared precision: {missing[:5]}"
