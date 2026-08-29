"""The cache-miss fallback must contain a config that FITS, checked against measured shared memory.

`heuristic_subset` is what a kernel runs when the cache has no answer for its shape: 24 configs
instead of the grid's thousands. Ranking them by distance from the middle of every block axis puts
every axis at its middle AT ONCE, and the offsets compound into the largest tile in the "reasonable"
region. When that corner does not fit in shared memory, NOTHING in the subset does, and the launch
dies with OutOfResources rather than being slow -- measured on an A5000, `adaln_bwd_dw_triton`'s
fallback asked for 294,912 B against a 101,376 B limit and all 24 candidates were over. 72c131b
reserved a quarter of the cap for the smallest tiles.

That fix was never walked by a test. `tests/autotune` exercises `heuristic_subset` on synthetic
configs; the three kernels that found the bug found it by dying. This walks it on a REAL grid
against REAL measurements -- `capture` logs the shared bytes of every config it compiles, and one
such log is kept as a fixture.

ONE KERNEL, and the file says so rather than implying more: it is the only fixture that carries
shared-memory readings (the others hold compile times and budget kills). The kernel the bug was
found on is not in the registry any more, so its numbers cannot be re-recorded here.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
FIXTURE = ROOT / "tests/autotune/compile_budget/a6000_one_grid/trimul_gemm_gate_a6000.smem.gz"

#: Shared memory per block on sm86 (A5000 / A6000), the cards this repo has manifests for. The
#: number the A5000 failure was measured against.
LIMIT = 101_376

OP = "trimul_gemm_gate_triton"


def _measured() -> tuple[dict[str, int], list[str]]:
    """{config signature: shared bytes} and the axes the signature names.

    Untagged rows only: `!` is a config the compile budget killed (no reading -- it never got far
    enough) and `~` is a compile time in milliseconds. Folding either in would put a 60 next to
    values in the hundreds of thousands.
    """
    smem: dict[str, int] = {}
    axes: list[str] = []
    for line in gzip.decompress(FIXTURE.read_bytes()).decode(errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0][:1] in ("!", "~"):
            continue
        smem[parts[1]] = int(parts[2])
        if not axes:
            axes = [a.split("=")[0] for a in parts[1].split(",")]
    return smem, axes


def _sig(config, axes: list[str]) -> str:
    """The config's signature over the axes the RECORDING names.

    The live grid has gained `GROUP_M` since these bytes were measured. It is the tile visit order
    -- index arithmetic, no buffer -- so a reading applies to every GROUP_M value of the same tile,
    warps and stages, and matching on the recorded axes is what lets old measurements stay usable.
    """
    d = dict(config.kwargs)
    d["num_warps"], d["num_stages"] = config.num_warps, config.num_stages
    return ",".join(f"{k}={d[k]}" for k in sorted(k for k in d if k in axes))


@pytest.fixture(scope="module")
def grid():
    from miniworld_engine.autotune.configs import (
        config_set,
        configs_for,
        use_config_dir,
    )

    use_config_dir(config_set("grid"))
    cfgs = configs_for(OP)
    assert cfgs, f"no configs for {OP} under the packaged `grid` set"
    return cfgs


def test_the_fixture_still_carries_shared_memory_readings() -> None:
    """Guard the guard. If the log format changes or the fixture is replaced by one holding only
    compile times, every check below passes on an empty dict."""
    smem, axes = _measured()
    assert len(smem) > 1000, f"only {len(smem)} readings; the fixture no longer covers a grid"
    assert axes, "no axes parsed out of the signatures"


def test_half_this_kernels_grid_cannot_launch_here() -> None:
    """The premise. If most of the grid fit, a fallback that picks badly would still work and this
    file would be testing nothing."""
    smem, _ = _measured()
    over = sum(1 for b in smem.values() if b > LIMIT)
    assert over / len(smem) > 0.25, (
        f"only {over}/{len(smem)} configs exceed {LIMIT} B on this kernel; the fallback's tile "
        f"choice no longer decides whether it can launch")


def test_the_fallback_contains_a_config_that_fits(grid) -> None:
    """The property the fix exists for: a miss is slow, never dead."""
    smem, axes = _measured()
    from miniworld_engine.autotune.cache import heuristic_subset

    subset = heuristic_subset(grid)
    known = [c for c in subset if _sig(c, axes) in smem]
    assert known, "no config in the subset has a measurement; the check would be vacuous"
    fits = [c for c in known if smem[_sig(c, axes)] <= LIMIT]
    assert fits, (
        f"every measured config in the miss fallback needs more than {LIMIT} B of shared memory "
        f"(smallest {min(smem[_sig(c, axes)] for c in known)} B). A miss would die with "
        f"OutOfResources instead of being slow.")


def test_the_reserved_floor_is_what_reaches_the_small_tiles(grid) -> None:
    """The fix, isolated: the subset must reach the SMALL END of the grid, not just the middle.

    Ranking alone puts every axis at its middle at once, and on this kernel the middle does not fit
    -- the median measured config is 104,448 B against a 101,376 B cap. The reserved quarter is
    what puts a genuinely small tile in the subset.

    Stated against the grid's own smallest measured tile (4,096 B here) rather than a byte
    constant, because that is the property: with the reserved quarter the subset reaches 5,120 B,
    1.25x the floor of the whole grid; ranking alone reaches 20,480 B, 5x. A constant threshold
    would have passed either way -- an earlier version of this assertion did.
    """
    smem, axes = _measured()
    from miniworld_engine.autotune.cache import heuristic_subset

    subset = heuristic_subset(grid)
    in_subset = [smem[_sig(c, axes)] for c in subset if _sig(c, axes) in smem]
    assert in_subset, "no config in the subset has a measurement"
    floor = min(smem.values())
    assert min(in_subset) <= 2 * floor, (
        f"the smallest tile the fallback offers needs {min(in_subset)} B, and the grid has one "
        f"needing {floor} B -- {min(in_subset) / floor:.1f}x. The subset is not reaching the small "
        f"end, which is what the reserved quarter of the cap is for, and on a kernel whose middle "
        f"does not fit that is the difference between slow and dead.")


def test_the_middle_of_this_grid_does_not_fit(grid) -> None:
    """Why the floor matters HERE and not only in principle. If the median config fit, ranking
    alone would be launchable and the previous test would pass without the fix being present."""
    smem, _ = _measured()
    ordered = sorted(smem.values())
    median = ordered[len(ordered) // 2]
    assert median > LIMIT, (
        f"the median measured config is {median} B, under the {LIMIT} B cap -- the fallback's "
        f"preference for middle tiles is no longer what decides whether it can launch")
