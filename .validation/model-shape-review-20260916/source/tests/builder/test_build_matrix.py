"""The per-GPU kernel-cache build rules are data, so they need the checks code gets for free.

A typo in a CSV does not raise at import; it silently changes what a 20-hour build produces. The
two failure directions are not symmetric: building a unit the card cannot run wastes GPU-hours and
is obvious in the log that same day, while NOT building one leaves an empty cache bucket that
surfaces months later as a multi-minute full-grid stall inside a production forward. So the tests
below pin the second direction hardest -- the default is build, a skip must be explicit and
reasoned, and an unknown card must get everything rather than nothing.
"""

from __future__ import annotations

import pytest

from miniworld_engine import build as matrix
from miniworld_engine.autotune import builder

FLEET = ("sm_80", "sm_86", "sm_90", "sm_100")


@pytest.mark.parametrize("sm", FLEET)
def test_every_fleet_card_has_a_file(sm):
    """A card without a file still builds -- but silently, so the fleet is enumerated on purpose."""
    assert sm in matrix.known_gpus()


@pytest.mark.parametrize("sm", FLEET)
def test_rules_parse_and_every_skip_is_justified(sm):
    for rule in matrix.rules(sm):
        if not rule.build:
            assert rule.reason, f"unjustified skip in {rule.source}: {rule}"


@pytest.mark.parametrize("sm", FLEET)
def test_default_is_build(sm):
    """Anything no row matches must still be cached, on every card."""
    assert matrix.allows(sm, "pairformer_block", "miniworld", "bfloat16")
    assert matrix.allows(sm, "a_case_added_next_week", "miniworld", "bfloat16")


def test_unknown_card_builds_everything():
    """No file is not a deny -- the only safe default for an unrecognised GPU is the expensive one."""
    assert "sm_123" not in matrix.known_gpus()
    assert matrix.allows("sm_123", "triangle_multiplication", "cute", "bfloat16")


def test_common_rules_apply_to_every_card():
    """_common.csv is the reason the bf16-only misses are not pasted into four files."""
    for sm in FLEET:
        assert not matrix.allows(sm, "triangle_multiplication", "triton", "float32")
        assert matrix.allows(sm, "triangle_multiplication", "triton", "bfloat16")


def test_arch_specific_denies_are_arch_specific():
    """cute is the whole reason this directory exists: dead below sm_90, fastest path on it."""
    assert not matrix.allows("sm_80", "triangle_multiplication", "cute", "bfloat16")
    assert not matrix.allows("sm_86", "triangle_multiplication", "cute", "bfloat16")
    assert matrix.allows("sm_90", "triangle_multiplication", "cute", "bfloat16")


def test_sm_tag_is_the_file_stem():
    assert matrix.sm_tag((8, 6)) == "sm_86"
    assert matrix.sm_tag((10, 0)) == "sm_100"
    assert matrix.sm_tag((9, 0)) in matrix.known_gpus()


@pytest.mark.parametrize(("sm", "expected_impls"), [
    ("sm_86", {"miniworld", "triton"}),
    ("sm_90", {"miniworld", "triton", "cute"}),
])
def test_units_drop_only_the_unbuildable_impls(sm, expected_impls, monkeypatch):
    monkeypatch.setattr(builder, "device_sm", lambda: sm)
    cases = [c for c in builder.cases() if c.name == "triangle_multiplication"]
    assert {u.impl for u in builder.units(cases)} == expected_impls


def test_no_cuda_means_no_filtering(monkeypatch):
    """Off-GPU enumeration must report the FULL plan, not a card-shaped subset of it."""
    cases = builder.cases()
    monkeypatch.setattr(builder, "device_sm", lambda: None)
    unfiltered = len(builder.units(cases))
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    assert len(builder.units(cases)) < unfiltered
    assert builder.skipped_units(cases, None) == []


def test_skips_are_reported_with_reasons():
    """A dropped unit must be announced -- a silent skip reads as 'covered' in the build log."""
    skipped = builder.skipped_units(builder.cases(), "sm_86")
    assert any("cute" in label for label, _ in skipped)
    assert all(reason for _, reason in skipped)


# --------------------------------------------------------------------------- #
# NOTE: the "config exclusions" half of this file was removed.
#
# It pinned `_is_compile_monster`, a STATIC pre-filter that dropped num_warps>=16 / num_stages>=5
# before benching. fcd3c7a deleted the function and left the import, which made this whole module
# un-collectable -- so every test above has been silently absent since then too, not just the ones
# about the prune.
#
# The tests are not restored, because the prune should not be: a static rule shrinks whatever grid
# a tuning run declares, behind the caller's back, and its premise did not survive measurement
# (num_stages has no compiler maximum; the ceiling is smem_limit/operand_tile, per config, far
# above 5 for small tiles). The guard that remains is capture.py's per-config compile TIMEOUT,
# which judges each config by real compile time on the running card.
# --------------------------------------------------------------------------- #


def test_the_sweep_still_enumerates_every_declared_unit() -> None:
    """The 527-unit report, turned into a check.

    A consumer's clone predated `0854ac4` (*the sweep never drove fp32, and coverage could not see
    it missing*) and `build all` enumerated 527 units instead of 859 — half of every fp32 kernel's
    work, while coverage reported `missing 0` because it counted against what the build enumerated
    rather than against the registry. Ten hours went to it on another cluster.

    The count is derived from the registry here rather than written down, so adding a kernel does
    not fail this for the wrong reason. What it catches is the shape of that bug: units going
    missing while every other number still looks consistent.
    """
    import csv
    from pathlib import Path

    from miniworld_engine.autotune.builder import op_units
    from miniworld_engine.kernels import __file__ as kernels_init

    reg = Path(kernels_init).parent / "registry.csv"
    with reg.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("driver") or "").strip()]
    dtypes = sum(len([d for d in (r.get("dtypes") or "").split("|") if d]) for r in rows)
    got = len(op_units())
    assert got >= dtypes, (
        f"{got} units for {len(rows)} driven kernels declaring {dtypes} (kernel, dtype) pairs. "
        f"A unit is (op, dtype, shape bucket), so it cannot be fewer than the pairs -- this is the "
        f"527-vs-859 shape: work vanishing while every other count still agrees. "
        f"See docs/reproducing-a-report.md.")
    assert got > 800, f"only {got} units; the sweep has lost work (859 at the time of writing)"
