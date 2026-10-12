"""The module plan preserves each declared dimensions/length/mode/augmentation tuple."""
from __future__ import annotations

from miniworld_engine.autotune import builder, derive, plan


def test_the_module_plan_covers_declared_rows_in_both_modes(monkeypatch):
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    cases = builder.cases()
    by_name = {case.name: case for case in cases}
    actual = [plan.label(unit, by_name[unit.case]) for unit in builder.units(cases)]
    declared = [unit.label for unit in derive.units(derive.module_rows(), arch="sm86")]
    assert len(actual) == len(set(actual))
    assert set(actual) == set(declared)
    assert any(" eval" in label for label in actual)
    assert any(" train" in label for label in actual)
    for case in cases:
        assert any(label.startswith(case.name + "[") for label in actual)
