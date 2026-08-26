"""Predict which configs will not COMPILE inside the budget, from the probes already paid for.

A different failure from `viability`'s. There, triton compiles the kernel, reports that it wants
more shared memory than the card has, and the config fails at LAUNCH. Here the compile itself does
not finish: ptxas grinds on a register-spilling kernel for minutes, and `capture` SIGKILLs it at
`_COMPILE_BUDGET_S`. Both reach the bench as an undifferentiated `+inf`, and only the first leaves
a `metadata.shared` reading -- so a killed config is not even in the shared-memory model's
training data. It cannot be predicted by that model and is not.

It is worth predicting separately. Measured on the A6000 rebuild: 13,875 of 869,844 configs were
killed, and those 1.6% took 231 of the build's 429 compile CPU-hours. Concentrated, too -- 205 of
461 rounds killed nothing, while one op at three lengths burned 27.5 CPU-hours on 3.6% of its
grid.

WHAT SEPARATES THEM, and what does not
--------------------------------------

`num_warps` explains it almost perfectly. Fewer warps means fewer threads for the same tile, more
registers per thread, and spill. Across eight kernels, every one of 181 killed configs sat at
`num_warps <= 4`, and the median compile time falls monotonically with the axis -- 8.2 s at one
warp against 0.8 s at thirty-two.

And it is useless on its own. In the shipped cache `num_warps == 1` is the WINNER in 559 of 1,244
buckets, across 70 of 91 ops; `num_warps <= 4` wins 86.6%. A rule that dropped low-warp configs
would throw away the best config in nearly half the buckets, which is the mistake `cache.py`'s old
static `num_warps>=16` filter made and was reverted for.

The two are in different corners of the same warp count. A low-warp WINNER has a small tile
(median tile volume 256 at one warp, p90 16,384); a low-warp KILLED config has a large one
(smallest killer 32,768-262,144 depending on the kernel). So the rule is the ordering rule
`viability` already validated, anchored on the killed probes instead of on the over-limit ones: a
config at least as large on every tile axis, with at least as many pipeline stages, at the same
warp count, will not compile faster.

WHAT IT SCORES, on the probe round the shared-memory model already pays for
--------------------------------------------------------------------------

Eight kernels, 18,393 configs, 631 of them killed (`tests/autotune/compile_budget/`):

    killed configs ruled out                     395 / 631  = 63%
    configs ruled out that did compile            35 / 18,393 = 0.19%
    those 35 took to compile      min 29 s, median 53 s, max 60 s
    extra probes required                          none

The last line is the point. `viability.choose_probes` already compiles the largest tile at every
warp count, and at low warp counts those are already being killed -- one to twenty-seven of them
per kernel across all eight. So this rule rides on a probe round that is already paid for, and
adds a set membership test.

The cost is bounded by how close its mistakes are to the line: a config wrongly ruled out took a
median 53 s of a 60 s budget, which is a spilling kernel that only just survived and that the
bench was going to reject anyway. `dominance_holds` gates on exactly that, at half the budget.
It can only check the PROBE points, so a violation the sample does not contain reaches `classify`
unseen -- one of the 35 came in at 29 s. That is the same limitation `viability.comparison_holds`
carries, for the same reason.
"""
from __future__ import annotations

from miniworld_engine.autotune.viability import dominated_by, tile_axes


def anchors(killed: list[dict]) -> list[dict]:
    """The minimal killed configs -- the ones no other killed config is smaller than.

    Only these can anchor anything: if A and B were both killed and A <= B on every axis, then
    everything B rules out A rules out too. Trimming them keeps `classify` linear in the useful
    anchors rather than in every kill, which matters when a round kills 261 of 2,592 configs.
    """
    if not killed:
        return []
    axes = tile_axes(killed)
    out: list[dict] = []
    for c in sorted(killed, key=lambda c: (c["num_warps"], c["num_stages"],
                                           *(c.get(a, 0) for a in axes))):
        if not dominated_by(c, axes, out):
            out.append(c)
    return out


def dominance_holds(times: dict[tuple, float], killed: list[dict], configs: list[dict],
                    budget: float) -> bool:
    """Does "at least as large compiles no faster" hold for THIS kernel's probes?

    Checked rather than assumed, per kernel, exactly as `viability.comparison_holds` is -- because
    the shared-memory version of this claim is FALSE for some kernels, and there is no reason to
    expect compile time to be better behaved.

    The test is deliberately not "is compile time monotone": it is noisy, measured under a loaded
    node, and near-ties are meaningless. What has to hold is that no probe which compiled COMFORT-
    ABLY is ruled out by a killed probe. Half the budget is the line, and it is where the evidence
    put it: over eight kernels the fastest wrongly-ruled-out config took 34.4 s of a 60 s budget,
    so a kernel where something CHEAP is dominated by a kill is a kernel this rule does not
    describe.
    """
    if not killed or not times:
        return False
    axes = tile_axes(configs)
    keep = anchors(killed)
    for key, seconds in times.items():
        if seconds >= budget / 2:
            continue
        c = dict(zip([*axes, "num_warps", "num_stages"], key, strict=True))
        if dominated_by(c, axes, keep):
            return False
    return True


def classify(configs: list[dict], killed: list[dict], holds: bool) -> dict[str, list]:
    """Split the grid into what is worth compiling and what a probe has already shown it is not.

    `skip` is empty unless `holds` -- the fallback is to compile everything, which is what the
    build did before any of this existed. Nothing here needs a model or a fitted coefficient: the
    anchors ARE the measurement.
    """
    if not holds or not killed:
        return {"keep": list(configs), "skip": [], "anchors": []}
    axes = tile_axes(configs)
    keep_anchors = anchors(killed)
    keep, skip = [], []
    for c in configs:
        (skip if dominated_by(c, axes, keep_anchors) else keep).append(c)
    return {"keep": keep, "skip": skip, "anchors": keep_anchors}
