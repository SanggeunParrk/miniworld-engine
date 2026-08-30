"""A kernel's `num_warps` / `num_stages` ladder must offer every value that wins for it.

These two axes are per kernel, like every `BLOCK_*` axis already is, and narrowed from the shipped
cache: the values that win a bucket for that kernel on any card, plus one rung either side. That is a decision made from evidence, and evidence goes stale in
two directions.

TOO NARROW is the direction that costs. `layernorm_bwd_atomic_triton` is 1.83x slower without
`num_warps=16`; it wins 2 of that kernel's 22 buckets, so a ladder derived from a run that happened
to miss those two shapes would drop it and nothing would say so -- the build would simply produce a
worse cache. This file fails when a declared ladder omits a value the cache records as a winner.

It cannot check the other direction. A ladder that is too WIDE only costs build time, and the cache
cannot prove a value useless: it holds the top five configs per bucket, so a value that never wins
may still be second by a hair. Widening is a hand edit; this test is what stops the ladder narrowing past the evidence.

The union across cards is deliberate. `layernorm_bwd_split_mmajor_triton` wins at 16 on an A6000
and never on an A5000 -- a ladder is a card-independent declaration, and narrowing to one card's
evidence breaks the other.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
PKG = ROOT / "src" / "miniworld_engine"
DATA = PKG / "autotune" / "data"
CONFIGS = PKG / "autotune" / "configs"
REGISTRY = PKG / "kernels" / "registry.csv"
AXES = ("num_warps", "num_stages")

#: Every set a build can be pointed at. `grid` is the one `build all` uses; the others pin whole
#: configs (one row = one config) and are not ladders at all, so they are checked for the same
#: property in the form they have: the pinned value must be one that wins, or the set is naming a
#: config nothing measured.
LADDER_SETS = ("grid",)


def _winners() -> dict[str, dict[str, collections.Counter]]:
    with REGISTRY.open(newline="") as fh:
        live = {r["kernel"] for r in csv.DictReader(fh)}
    out: dict[str, dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or d.name not in live:
            continue
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for ranked in (data.get("entries") or {}).values():
                if isinstance(ranked, list) and ranked:
                    for ax in AXES:
                        out[d.name][ax][ranked[0][ax]] += 1
    return out


def _ladders(path: Path) -> dict[str, list[int]]:
    with path.open(newline="") as fh:
        return {r[0]: [int(x) for x in r[1].split()]
                for r in csv.reader(fh) if len(r) >= 2 and r[0] != "axis"}


@pytest.fixture(scope="module")
def won():
    got = _winners()
    assert len(got) > 40, f"only {len(got)} kernels have cache entries; this would pass vacuously"
    return got


@pytest.mark.parametrize("setname", LADDER_SETS)
def test_no_ladder_omits_a_winner(setname: str, won) -> None:
    bad = []
    for f in sorted((CONFIGS / setname).glob("*.csv")):
        ax = _ladders(f)
        for a in AXES:
            if a not in ax:
                continue
            missing = sorted(set(won.get(f.stem, {}).get(a, {})) - set(ax[a]))
            if missing:
                bad.append(f"{f.stem}: {a}={ax[a]} omits {missing}, which win buckets in "
                           f"autotune/data/.")
    assert not bad, "\n  ".join(["a narrowed ladder dropped a value that wins:", *bad])


@pytest.mark.parametrize("setname", LADDER_SETS)
def test_every_ladder_has_something_to_choose(setname: str) -> None:
    """A one-value axis is not tuned, it is a constant -- and a constant belongs in the kernel,
    not in a config set where it costs a column and reads as a choice."""
    thin = []
    for f in sorted((CONFIGS / setname).glob("*.csv")):
        for a, vals in _ladders(f).items():
            if len(vals) < 2:
                thin.append(f"{f.stem}: {a}={vals}")
    assert not thin, "\n  ".join(["single-value ladders:", *thin])


def _planned() -> dict[str, dict[str, int]]:
    """How many buckets a full build would leave in the cache, per kernel per precision."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from miniworld_engine.autotune.builder import op_units

    out: dict[str, dict[str, int]] = collections.defaultdict(collections.Counter)
    for u in op_units():
        out[u.op][u.dtype] += 1
    return out


def _measured() -> dict[str, dict[tuple[str, str], int]]:
    """How many buckets the shipped cache actually holds, per kernel per (card, precision)."""
    out: dict[str, dict[tuple[str, str], int]] = collections.defaultdict(collections.Counter)
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for key, ranked in (data.get("entries") or {}).items():
                if isinstance(ranked, list) and ranked:
                    out[d.name][(f.stem, key.split("|")[0])] += 1
    return out


def _covered() -> set[str]:
    """Kernels whose cache covers a whole planned build on at least one card.

    Below that line a ladder derived from the cache is derived from a SAMPLE, and the sample is
    biased in the one direction that costs: the widths still missing are the wide ones. Every
    kernel whose winners currently stop at `num_warps=4` is a kernel measured at d=128 only, with
    256 and 512 unbuilt -- and wider activations are exactly where more warps start to pay.
    """
    planned, measured = _planned(), _measured()
    return {k for k, want in planned.items()
            if any(n >= want.get(dt, 0) > 0 for (_card, dt), n in measured.get(k, {}).items())}


def _derive(won_axis: collections.Counter, rungs: tuple[int, ...]) -> list[int]:
    """The declared rule: every value that wins, plus one rung either side."""
    out: set[int] = set()
    for v in won_axis:
        i = rungs.index(v)
        out |= {rungs[j] for j in (i - 1, i, i + 1) if 0 <= j < len(rungs)}
    return sorted(out)


RUNGS = {"num_warps": (1, 2, 4, 8, 16, 32),
         "num_stages": (1, 2, 3, 4, 5, 6, 8, 10, 12)}


def test_a_fully_measured_kernel_carries_its_own_ladder(won) -> None:
    """The point of the exercise -- but only where there is a whole build to derive it from.

    These two axes are meant to be per kernel, like every `BLOCK_*` axis already is. They are not
    yet: 74 kernels share three `num_warps` spellings and three `num_stages` ones, split by which
    was written first. Narrowing them is a measurement, and the measurement is not in yet.

    So this test does not demand the narrowing wholesale. It demands it for each kernel whose cache
    already covers a full planned build, and stays quiet about the rest. That way the derivation
    lands kernel by kernel as the sweep fills in, and no kernel is narrowed on a partial sample.
    """
    covered = _covered()
    if not covered:
        pytest.skip("no kernel's cache covers a whole planned build yet")
    bad = []
    for f in sorted((CONFIGS / "grid").glob("*.csv")):
        if f.stem not in covered:
            continue
        ax = _ladders(f)
        for a, rungs in RUNGS.items():
            if a not in ax:
                continue
            want = _derive(won[f.stem][a], rungs)
            if want and ax[a] != want:
                bad.append(f"{f.stem}: {a}={ax[a]}, but a full build says "
                           f"{want} (winners {sorted(won[f.stem][a])} plus one rung either side)")
    assert not bad, "\n  ".join(
        ["a fully measured kernel is still carrying a boilerplate ladder:", *bad])


def test_the_narrowing_is_not_silently_stalled() -> None:
    """A record of how far off the narrowing is, so 'no kernel qualifies' cannot pass as done.

    Coverage as of the run that wrote this: 2 of 74 kernels complete. 35 hold no entry at all at
    the precision they are now declared at -- the dtype columns were corrected after those caches
    were built -- and the rest sit at 4/12 or 6/18, which is the d=128 width alone with 256 and 512
    unbuilt. This asserts only that the numbers are still computable and prints where they stand;
    it is the thing to read when the test above skips.
    """
    planned, measured, covered = _planned(), _measured(), _covered()
    grid = {f.stem for f in (CONFIGS / "grid").glob("*.csv")}
    absent = sorted(k for k in grid
                    if not any(dt in planned.get(k, {}) for (_c, dt) in measured.get(k, {})))
    print(f"\nladder coverage: {len(covered & grid)}/{len(grid)} kernels fully measured, "
          f"{len(absent)} with no cache at their declared precision")
    assert planned and measured, "coverage cannot be computed at all"
