"""Scored against eight real kernels, on the probe round the shared-memory model already pays for.

The failure being predicted is not `viability`'s. There, triton compiles the kernel and reports
that it needs more shared memory than the card has; the config fails at LAUNCH. Here the compile
does not finish -- ptxas grinds on a register-spilling kernel and `capture` SIGKILLs it at the
budget. A killed config never reaches `metadata.shared`, so it is not in the shared-memory model's
training data and cannot be predicted by it.

It is worth predicting: on the A6000 rebuild, 13,875 of 869,844 configs were killed and those 1.6%
took 231 of the 429 compile CPU-hours.

The fixtures are the `~` (compile milliseconds) and `!` (killed) rows of eight units of one A6000
run, job paircost. The plain shared-memory rows are stripped -- this module needs the config list
and the times, and the config list is implied by them. They live in their own directory so the
nine-kernel shared-memory fixtures in the parent stay the sample that module was scored on.

What has to hold is FALSE POSITIVES, as it is there: a config predicted unusable that in fact
compiles is a config removed from the search.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from miniworld_engine.autotune import compile_budget, smem_log, viability

DATA = Path(__file__).parent / "compile_budget"
BUDGET = 60.0


def _parse(sig: str) -> dict | None:
    try:
        return {k: int(v) for k, v in (p.split("=") for p in sig.split(","))}
    except ValueError:
        return None


def _score(times: dict[str, int], kills: set[str]) -> dict | None:
    good = [(c, v / 1000) for s, v in times.items() if (c := _parse(s)) and "num_warps" in c]
    killed = [c for s in kills if (c := _parse(s)) and "num_warps" in c]
    if not good or not killed:
        return None
    configs = [c for c, _ in good] + killed
    axes = viability.tile_axes(configs)

    def key(c):
        return (*(c[a] for a in axes), c["num_warps"], c["num_stages"])

    # The BASE probe round, unchanged -- this rule asks for no probe of its own.
    probes = {key(p) for p in viability.choose_probes(configs)}
    seen = [c for c in killed if key(c) in probes]
    holds = compile_budget.dominance_holds(
        {key(c): v for c, v in good if key(c) in probes}, seen, configs, BUDGET)
    split = compile_budget.classify(configs, seen, holds)
    skip = {key(c) for c in split["skip"]}
    fp = [v for c, v in good if key(c) in skip]
    return {"n": len(configs), "killed": len(killed), "probe_kills": len(seen),
            "caught": sum(1 for c in killed if key(c) in skip),
            "fp": len(fp), "fp_seconds": fp, "holds": holds,
            "anchors": len(split["anchors"]), "probes": len(probes)}


@pytest.fixture(scope="module")
def scored():
    ms, killed = smem_log.compile_ms(DATA), smem_log.killed(DATA)
    out = {k: s for k in ms if (s := _score(ms[k], killed.get(k, set())))}
    assert len(out) >= 6, f"only {len(out)} kernels carry both times and kills: {list(out)}"
    return out


def test_it_almost_never_discards_a_config_that_would_have_compiled(scored) -> None:
    """The one error that costs something. 35 of 18,393 when this was written -- 0.19%."""
    fp = sum(v["fp"] for v in scored.values())
    n = sum(v["n"] for v in scored.values())
    assert fp / n < 0.005, f"{fp} false positives over {n} configs"


def test_the_configs_it_does_discard_wrongly_were_nearly_killed_anyway(scored) -> None:
    """Not a consolation -- it is the property that makes the rule's cost bounded. A config
    wrongly ruled out took a median of 53 s of a 60 s budget to compile: a register-spilling
    kernel that only just survived, which the bench was going to reject anyway.

    One of the 35 came in at 29 s, just under the line `dominance_holds` gates on. That is the
    gate's known blind spot and not a surprise: it can only check the PROBE points, so a
    violation the probe sample does not contain reaches `classify` unseen -- the same limitation
    `viability.comparison_holds` carries and documents.
    """
    fp = sorted(s for v in scored.values() for s in v["fp_seconds"])
    assert fp, "no false positives at all is a fixture that proves nothing about their cost"
    assert fp[len(fp) // 2] > BUDGET / 2, f"median wrongly-ruled-out compile was {fp[len(fp)//2]:.0f}s"
    cheap = sum(1 for x in fp if x < BUDGET / 2)
    assert cheap <= len(fp) // 20 + 1, (
        f"{cheap} of {len(fp)} wrongly-ruled-out configs compiled in under half the budget; the "
        f"rule is reaching past the tail")


def test_it_catches_most_of_the_kills(scored) -> None:
    caught = sum(v["caught"] for v in scored.values())
    killed = sum(v["killed"] for v in scored.values())
    assert killed > 300, killed
    assert caught >= 0.5 * killed, f"caught {caught} of {killed}"


def test_it_asks_for_no_probe_of_its_own(scored) -> None:
    """The whole reason this is cheap: the shared-memory probe round already compiles configs at
    the expensive end of every warp count, so some of them are already killed. If it needed its
    own round the round would cost more than the rule saves."""
    for k, v in scored.items():
        assert v["probe_kills"] >= 1, f"{k}: no killed config in the shared-memory probe round"


def test_the_kernels_disagree_enough_to_be_a_real_test(scored) -> None:
    rates = [v["caught"] / v["killed"] for v in scored.values()]
    assert min(rates) < 0.2 < max(rates), (
        f"every kernel behaves alike ({rates}); the fixtures no longer span a range")


def test_a_kernel_with_no_killed_probe_skips_nothing(scored) -> None:
    """The fallback is the behaviour the build had before any of this: compile everything."""
    split = compile_budget.classify([{"BLOCK_M": 32, "num_warps": 4, "num_stages": 2}], [], False)
    assert not split["skip"]
    assert len(split["keep"]) == 1


def test_the_gate_refuses_a_kernel_where_a_cheap_config_is_ruled_out() -> None:
    """`dominance_holds` is the per-kernel check, and it has to be able to say no. Compile time is
    not monotone by nature -- the shared-memory version of this claim is false for reductions --
    so a kernel where something cheap sits above a kill is a kernel this rule does not describe."""
    configs = [{"BLOCK_M": m, "num_warps": 1, "num_stages": 2} for m in (32, 64, 128, 256)]
    killed = [{"BLOCK_M": 64, "num_warps": 1, "num_stages": 2}]
    cheap = {(256, 1, 2): 1.0}          # bigger than the kill, yet compiled in a second
    assert not compile_budget.dominance_holds(cheap, killed, configs, BUDGET)
    slow = {(256, 1, 2): 58.0}
    assert compile_budget.dominance_holds(slow, killed, configs, BUDGET)


def test_an_anchor_that_another_anchor_covers_is_dropped() -> None:
    """A round can kill 261 of 2,592 configs; carrying every one as an anchor makes `classify`
    quadratic in kills for no added coverage."""
    killed = [{"BLOCK_M": 64, "num_warps": 1, "num_stages": 2},
              {"BLOCK_M": 256, "num_warps": 1, "num_stages": 4},
              {"BLOCK_M": 32, "num_warps": 2, "num_stages": 2}]
    got = compile_budget.anchors(killed)
    assert len(got) == 2, got
    assert {c["num_warps"] for c in got} == {1, 2}


def test_an_anchor_does_not_reach_across_warp_counts() -> None:
    """Register pressure per thread is the tile divided by the warps, so a tile that spills at one
    warp need not spill at eight. Reaching across the axis is how a static `num_warps` filter goes
    wrong, and `num_warps == 1` is the WINNER in 559 of the shipped cache's 1,244 buckets."""
    killed = [{"BLOCK_M": 64, "num_warps": 1, "num_stages": 2}]
    configs = [{"BLOCK_M": 256, "num_warps": w, "num_stages": 4} for w in (1, 8)]
    split = compile_budget.classify(configs, killed, holds=True)
    assert [c["num_warps"] for c in split["skip"]] == [1]
    assert [c["num_warps"] for c in split["keep"]] == [8]
