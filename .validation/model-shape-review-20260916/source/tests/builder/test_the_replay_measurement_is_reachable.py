"""`builder.audit` -- the only direct measure of cache coverage -- had no caller.

`cache.py` documents `_CACHE_MISSES` as "the only direct measure of whether the cache covers a
workload" and said `miniworld-engine audit` requires it to come back empty. It does not: the CLI's
`dev audit` is the STATIC check in `build/audit.py`, which never reads that set. The replay it
meant is `builder.audit`, and nothing in src, tests or benchmarks called it.

That matters because the static check and the replay do not measure the same thing. The static one
compares DECLARED work -- (op, dtype, shape bucket) -- against the cache; the cache key also
carries each kernel's constexprs. One pass of the GPU suite on an A6000 hit 14 (op, key) misses
against a cache the static check reports complete (missing_pairs=0).

So this pins the wiring, not the number: the measurement must stay reachable from a command.
"""
from __future__ import annotations

import inspect

from miniworld_engine import cli
from miniworld_engine.autotune import builder


def test_the_cli_exposes_the_replay() -> None:
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:                     # parser is built inside main(); read the source instead
        src = inspect.getsource(cli)
        assert "--replay" in src, "no command reaches builder.audit"
        return
    ns = parser.parse_args(["dev", "audit", "--replay"])
    assert ns.replay is True


def test_the_replay_calls_the_measurement() -> None:
    """Reachable is not enough -- it has to be the function that reads the miss set."""
    body = inspect.getsource(cli._replay_audit)
    assert "builder.audit" in body, body
    assert "cache_misses" in inspect.getsource(builder.audit), "the replay stopped measuring"


def test_the_miss_set_can_report_an_improvement() -> None:
    """It could only ever grow, so before/after over one process returned the before twice.

    Measured: a scoped fill of `gated_projection` covered its four missing keys, the shards merged,
    and the second `builder.audit` named the same four -- in 0 s, because nothing was re-measured.
    """
    from miniworld_engine.autotune import cache

    cache._CACHE_MISSES.add(("op", "gpu", "key"))
    assert cache.cache_misses()
    cache.clear_cache_misses()
    assert not cache.cache_misses(), "a filled cache can never be observed as filled"


def test_the_replay_clears_before_it_measures() -> None:
    body = inspect.getsource(builder.audit)
    assert "clear_cache_misses()" in body, (
        "audit must start from an empty miss set or it reports an earlier run's holes")
