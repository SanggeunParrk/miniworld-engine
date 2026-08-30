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


def test_the_two_axes_are_actually_per_kernel(won) -> None:
    """The point of the exercise. They used to be two boilerplate ladders pasted across 74
    kernels, split by which someone wrote first rather than by what the kernel needs -- if they
    collapse back to one or two shapes, the per-kernel derivation has been undone."""
    shapes = {a: collections.Counter() for a in AXES}
    for f in sorted((CONFIGS / "grid").glob("*.csv")):
        ax = _ladders(f)
        for a in AXES:
            if a in ax:
                shapes[a][tuple(ax[a])] += 1
    for a, c in shapes.items():
        assert len(c) >= 4, (
            f"{a} takes only {len(c)} distinct ladder(s) across the kernels: "
            f"{[list(k) for k in c]}. That is a boilerplate value, not a per-kernel decision.")
