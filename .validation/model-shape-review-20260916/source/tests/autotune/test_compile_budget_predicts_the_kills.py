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

TWO ERROR RATES, and only the second one is the build's
------------------------------------------------------

"Ruled out but it compiled" is the error rate against what this rule CLAIMS -- that a config will
be killed -- and it is the one that can be checked without benching anything. It is reported and
asserted on below.

It is not what a build loses. A build's output is the top few configs per bucket; a config that
compiles in 53 s and then places 1300th of 1434 cost compile time and contributed nothing, so
ruling it out is right and counting it as an error measures the wrong thing. The error that costs
something is "ruled out but it would have been CHOSEN", and that needs bench times joined to the
same signatures.

Both are here. The first is asserted directly; the second is asserted through
`test_a_slow_compile_never_benches_well`, which measures the property the inference rests on --
every config compiling over 30 s placed in the bottom 7-16% of its bucket, 6.4x to 32.7x off the
fastest. What is NOT yet measured is those exact ruled-out configs: they belong to three kernels
(`_dgrad_epi`, `_dx_swiglubwd_kernel`, `fused_sigmoid_gate_fwd_kernel`) for which no bench data is
in these fixtures. The two kernels that do have bench data produced no false positives at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from miniworld_engine.autotune import compile_budget, smem_log, viability

DATA = Path(__file__).parent / "compile_budget"
BUDGET = 60.0
LIMIT = 101376   # the A6000 these were measured on


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
    """The error against what the rule CLAIMS: 35 of 18,393 when this was written -- 0.19%.

    Not the error that costs a build anything; see the module docstring. Kept as the tightest
    number that can be checked without benching, and because a rule whose stated claim drifts is
    a rule nobody can reason about.
    """
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


def test_tightening_the_rule_trades_away_the_coverage(scored) -> None:
    """Pins the reason the rule is where it is, so a future tightening has to beat this.

    Requiring equal `num_stages` removes the neighbour-straddling-the-line false positives -- and
    with them three quarters of the catches, because a kill at stages=3 is the evidence that
    stages=6 at the same tile will not compile either.
    """
    ms, killed = smem_log.compile_ms(DATA), smem_log.killed(DATA)
    caught = kills = 0
    for k in ms:
        good = [(c, v / 1000) for s, v in ms[k].items() if (c := _parse(s)) and "num_warps" in c]
        kc = [c for s in killed.get(k, set()) if (c := _parse(s)) and "num_warps" in c]
        if not good or not kc:
            continue
        configs = [c for c, _ in good] + kc
        axes = viability.tile_axes(configs)

        def key(c, axes=axes):
            return (*(c[a] for a in axes), c["num_warps"], c["num_stages"])

        probes = {key(p) for p in viability.choose_probes(configs)}
        anchors = compile_budget.anchors([c for c in kc if key(c) in probes])
        strict = [c for c in kc
                  if any(b["num_warps"] == c["num_warps"] and b["num_stages"] == c["num_stages"]
                         and all(b[a] <= c[a] for a in axes) for b in anchors)]
        caught += len(strict)
        kills += len(kc)
    loose = sum(v["caught"] for v in scored.values())
    assert caught < loose / 2, (
        f"stage-equality catches {caught} of {kills} against {loose} -- if it no longer costs "
        f"most of the coverage, take it: it removes the boundary false positives")


def test_every_config_it_wrongly_ruled_out_could_never_have_been_CHOSEN() -> None:
    """The direct check the module docstring said was missing, on the kernel it was missing for.

    `fused_sigmoid_gate_fwd_kernel`, re-run on an A6000 with shared-memory readings AND compile
    times AND bench results for the same grid (job fpcheck, 2,596 configs). The rule ruled out 177
    of them, 175 correctly; the four it got wrong all compiled -- 43, 44, 55 and 59 s -- and not
    one of them could have been chosen:

        59 s    49,152 B      fits the card, and produced no bench time at all
        55 s   368,640 B      3.6x the card's 101,376 -- cannot launch
        44 s   270,336 B      2.7x  -- cannot launch
        43 s   405,504 B      4.0x  -- cannot launch

    Three of the four are the shared-memory predictor's business and it rules them out
    independently. The fourth fits and still never benched: 64 threads for a 256x128 tile is more
    registers than a thread has, so it fails at launch for the same reason it took 59 s to
    compile.
    """
    # Its OWN directory. `smem_log.read` merges every log in a directory by kernel name, and this
    # kernel also appears in the eight-kernel fixture beside it from a different run -- merged,
    # the two grids make a kernel that never existed and the false-positive count triples.
    one = DATA / "a6000_one_grid"
    ms, killed, shared = (smem_log.compile_ms(one), smem_log.killed(one), smem_log.read(one))
    name = "fused_sigmoid_gate_fwd_kernel"
    assert shared.get(name), "the fixture no longer carries shared-memory readings for this kernel"
    good = [(c, v / 1000) for s, v in ms[name].items() if (c := _parse(s)) and "num_warps" in c]
    kills = [c for s in killed[name] if (c := _parse(s)) and "num_warps" in c]
    configs = [c for c, _ in good] + kills
    axes = viability.tile_axes(configs)

    def key(c):
        return (*(c[a] for a in axes), c["num_warps"], c["num_stages"])

    def sig(c):
        return ",".join(f"{a}={c[a]}" for a in sorted(c))

    probes = {key(p) for p in viability.choose_probes(configs)}
    seen = [c for c in kills if key(c) in probes]
    skip = {key(c) for c in compile_budget.classify(configs, seen, holds=True)["skip"]}
    fp = [(c, v) for c, v in good if key(c) in skip]
    assert fp, "no false positives left in the fixture; this test proves nothing"
    fits = [c for c, _ in fp if shared[name].get(sig(c), 0) <= LIMIT]
    assert len(fits) <= 1, (
        f"{len(fits)} of {len(fp)} wrongly-ruled-out configs fit the card; when this was written "
        f"it was one, and the other three needed 2.7-4.0x the card's shared memory")
