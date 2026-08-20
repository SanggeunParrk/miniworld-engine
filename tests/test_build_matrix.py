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
# config exclusions
# --------------------------------------------------------------------------- #
"""The pre-launch config bounds, pinned against the shapes they are meant to keep and drop.

These bounds are the only thing that stops a pathological config: a launch's duration is readable
only after it finishes, so nothing in-process can shorten one. The risk of a bound is therefore not
that it fires late, it is that it fires on a config that would have won -- which is why each case
below names a real winner from the shipped cache.
"""
import triton  # noqa: E402

from miniworld_engine.autotune.cache import _is_compile_monster  # noqa: E402


def _cfg(kwargs, warps, stages):
    return triton.Config(kwargs, num_warps=warps, num_stages=stages)


def test_every_kernel_declares_whether_it_is_a_matmul():
    """The classifier is a declared flag, not a guess from the config or the axis count.

    Axis count was tried and measured wrong: every layernorm/adaln reduction pins a second tile
    axis at the launch site, so counting real axes calls them 2-D and the warps==1 bound then
    deletes winners whose best alternative is 7.4% slower.
    """
    from miniworld_engine.build.audit import autotuners, import_all_kernels

    import_all_kernels()
    undeclared = [n for n, t in autotuners()
                  if getattr(t.early_config_prune, "_miniworld_op", None)
                  and not hasattr(t.early_config_prune, "_miniworld_matmul")]
    assert not undeclared, undeclared


def test_the_worst_measured_config_is_excluded():
    """{BM:16,BK:16,BN:16,warps:1,stages:1} on trimul ran 468 s -- one launch, 85% of the unit.

    Caught by the matmul num_warps==1 bound, not by any tile-size rule: the same tiles with a real
    warp count stay in the sweep.
    """
    assert _is_compile_monster(_cfg({"BM": 16, "BK": 16, "BN": 16}, 1, 1), matmul=True)
    assert not _is_compile_monster(_cfg({"BM": 16, "BK": 16, "BN": 16}, 4, 2), matmul=True)


def test_tile_size_is_bounded_by_the_grid_not_by_a_rule():
    """The candidate sets start at 16; nothing here restates that as an exclusion.

    A prune that also dropped the smallest tier would be a second definition of the sweep, and it
    would silently narrow any kernel that later adds a smaller candidate.
    """
    from miniworld_engine.autotune.grids import BLOCK_K, BLOCK_M, BLOCK_N

    assert min(BLOCK_M) == min(BLOCK_N) == min(BLOCK_K) == 16
    assert not _is_compile_monster(_cfg({"BM": 32, "BN": 16}, 4, 2), matmul=True)
    assert not _is_compile_monster(_cfg({"BLOCK_K": 32, "BLOCK_M": 32}, 4, 4), matmul=True)
    assert not _is_compile_monster(
        _cfg({"BLOCK_K": 32, "BLOCK_M": 32, "BLOCK_N": 64}, 4, 4), matmul=True)


def test_matmul_bound_does_not_touch_non_matmul_winners():
    """Real winners from the shipped cache: one warp is routine for a reduction.

    These have TWO tile axes once the launch-pinned one is counted, so an axis-count classifier
    would have excluded them.
    """
    assert not _is_compile_monster(_cfg({"BLOCK_M": 8}, 1, 3), matmul=False)   # fused_ln_mask
    assert not _is_compile_monster(_cfg({"BLOCK_M": 16}, 1, 4), matmul=False)  # aug_attn preproc
    assert not _is_compile_monster(_cfg({"BLOCK_M": 1}, 4, 2), matmul=False)


def test_stages_one_is_never_cut():
    """It wins 10% of gemm entries and 14% of elementwise ones -- on both sides of the split."""
    assert not _is_compile_monster(_cfg({"BM": 32, "BN": 64}, 4, 1), matmul=True)
    assert not _is_compile_monster(_cfg({"BLOCK_M": 32}, 4, 1), matmul=False)


def test_real_matmul_winners_survive():
    assert not _is_compile_monster(_cfg({"BM": 32, "BN": 64}, 4, 2), matmul=True)
    assert not _is_compile_monster(
        _cfg({"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 128}, 4, 3), matmul=True)
    assert not _is_compile_monster(_cfg({"BLOCK_M": 64, "BLOCK_N": 32}, 4, 2), matmul=True)


def test_compile_monsters_still_go():
    assert _is_compile_monster(_cfg({"BLOCK_M": 64}, 16, 2))
    assert _is_compile_monster(_cfg({"BLOCK_M": 64}, 4, 5))
