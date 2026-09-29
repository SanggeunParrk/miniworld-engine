"""Scored against nine real kernels, because one was not enough and saying so cost a day.

Every conclusion in this module was first reached from a single GEMM-shaped kernel and then failed
on the others:

  * "shared memory is `(2*BK*BM + 4*BK*BN + 64) * (stages - 1)`" -- exact on that kernel, and on
    three others it discarded 168, 100 and 81 configs that would have run.
  * "a config larger on every axis needs at least as much shared memory" -- 12,377 comparable
    pairs with no violation on that kernel; 399 violations on `_stats_kernel`, whose shared memory
    goes DOWN as the tile grows because a wider tile leaves fewer partial sums per warp.

So the fixtures are `.smem.gz` logs from an actual build -- the `metadata.shared` triton reported
for every config it compiled, across a reduction, two elementwise kernels, three attention kernels
and three GEMM-shaped ones. The number that has to hold is FALSE POSITIVES: a config predicted
unusable that in fact runs is a config removed from the search, which is the mistake `cache.py`'s
old static `num_warps>=16` filter made and was reverted for.

Not one card's answer, within one architecture. The same nine kernels were re-measured on an
A6000 (job kcheck6, 2026-08-26) against these A5000 fixtures: **all 24,189 configs measured on
both cards reported the identical `metadata.shared`**, and the predictor scored the same 85.6%
caught with 0 false positives on each. That is what one would expect -- the requirement is the
compiler's, and both cards are sm86 with the same 101,376 B limit -- but it was worth checking
rather than assuming, and it is the reason the fixtures are not duplicated here: the second
card's log is byte-identical in what it says.

It says nothing about sm90 or sm100, where the limit is roughly 227 KB and the compiler makes
different choices. Every model in `viability.py` is fitted per kernel from probes taken on the
card being built for, so a different architecture refits rather than inherits -- but the SHARE it
catches there is unmeasured.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from miniworld_engine.autotune import smem_log, viability

LIMIT = 101376          # the A6000/A5000 these were measured on
DATA = Path(__file__).parent


def _kernels():
    return smem_log.read(DATA)


def _score(sigs: dict[str, int]):
    configs, shared = [], {}
    for sig, b in sigs.items():
        try:
            c = {k: int(v) for k, v in (p.split("=") for p in sig.split(","))}
        except ValueError:
            continue
        if "num_warps" not in c or "num_stages" not in c:
            continue
        configs.append(c)
        axes = sorted(k for k in c if k not in ("num_warps", "num_stages"))
        shared[(*(c[a] for a in axes), c["num_warps"], c["num_stages"])] = b
    if len(configs) < 150:
        return None
    axes = sorted(k for k in configs[0] if k not in ("num_warps", "num_stages"))

    def key(c):
        return (*(c[a] for a in axes), c["num_warps"], c["num_stages"])

    first = viability.choose_probes(configs)
    fits = viability.fit({key(p): shared[key(p)] for p in first if key(p) in shared}, configs)
    probes = first + viability.choose_anchor_probes(configs, fits)
    seen = {key(p): shared[key(p)] for p in probes if key(p) in shared}
    split = viability.classify(configs, fits, LIMIT,
                               measured_over=[p for p in probes if shared.get(key(p), 0) > LIMIT],
                               comparison_ok=viability.comparison_holds(seen, configs))
    skip = {key(c) for c in split["skip"]}
    over = {key(c) for c in configs if shared[key(c)] > LIMIT}
    return {"n": len(configs), "probes": len(probes),
            "tp": len(skip & over), "fp": len(skip - over), "over": len(over)}


@pytest.fixture(scope="module")
def scored():
    out = {k: s for k, sigs in _kernels().items() if (s := _score(sigs))}
    assert len(out) >= 6, f"only {len(out)} kernels have enough measured configs: {list(out)}"
    return out


def test_no_kernel_discards_a_config_that_would_have_run(scored) -> None:
    bad = {k: v["fp"] for k, v in scored.items() if v["fp"]}
    assert not bad, f"false positives: {bad}"


def test_the_kernels_disagree_enough_to_be_a_real_test(scored) -> None:
    """If they all behaved alike this would prove no more than the single-kernel test does."""
    rates = [v["over"] / v["n"] for v in scored.values()]
    assert min(rates) < 0.05 < max(rates), (
        f"every kernel has a similar share of unusable configs ({rates}); the fixtures no longer "
        f"span reduction-shaped and GEMM-shaped kernels")


def test_it_catches_a_useful_share_of_what_cannot_run(scored) -> None:
    tp = sum(v["tp"] for v in scored.values())
    over = sum(v["over"] for v in scored.values())
    assert over > 1000, over
    assert tp >= 0.75 * over, f"caught {tp} of {over} unusable configs"


def test_the_probe_is_cheap_where_it_matters(scored) -> None:
    """A probe is only worth paying for if it is much cheaper than what it saves.

    Sizing it purely by the model's column count compiled 77% of one small kernel's grid to
    predict the other 23%, which is worse than not predicting. The cap is a share of the grid, so
    the big grids -- the ones where compiling is the cost -- get the cheapest probes: 3% of a
    15,065-config kernel against 33% of a 240-config one.
    """
    big = {k: v for k, v in scored.items() if v["n"] > 3000}
    assert big, "no large-grid kernel in the fixtures"
    for k, v in big.items():
        assert v["probes"] <= v["n"] // 8, f"{k}: {v['probes']} probes for {v['n']} configs"


def test_the_probe_stays_a_small_fraction(scored) -> None:
    for k, v in scored.items():
        assert v["probes"] <= max(200, v["n"] // 4), f"{k}: {v['probes']} probes for {v['n']}"
