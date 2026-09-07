"""`build all` must cost only what actually went stale.

The builder used to plan work from the config grid alone: every (op, dtype, shape bucket) the grid
declares became a unit, and `--resume` filtered them only by what THIS shard directory had already
produced. So a rebuild against a repo that already ships this card's cache -- a new shard dir, a
new machine, a second run -- re-benched all 2,079 units and hours of GPU for answers already
sitting in `autotune/data`.

The fingerprints that decide whether a cached answer is still good already exist; `dev
cache-status` reads them. These tests pin the builder to that same judgement: an op whose cache
passes is not work, an op whose cache is stale is, and `--rebuild-cached` turns the skip off.
"""
from __future__ import annotations

from miniworld_engine.autotune import builder


def _unit(op: str, length: int = 384) -> builder.OpUnit:
    return builder.OpUnit(op=op, length=length, dtype="bfloat16")


def test_a_non_stale_cache_with_entries_is_not_work(monkeypatch) -> None:
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load",
                        lambda op, gk: {"entries": {"bfloat16|shape_key=1": [{}]}})
    assert builder._cache_answers(_unit("some_op_triton"), {"some_op_triton"}) is True


def test_a_stale_op_is_rebuilt_even_though_its_file_is_there(monkeypatch) -> None:
    """Staleness is the ONLY reason to redo an op -- and it is decided per op, so every unit of a
    stale op comes back as work even though the JSON on disk is full."""
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load",
                        lambda op, gk: {"entries": {"bfloat16|shape_key=1": [{}]}})
    assert builder._cache_answers(_unit("some_op_triton"), ok_ops=set()) is False


def test_an_empty_cache_file_is_not_an_answer(monkeypatch) -> None:
    """A merge writes a file per op even when the sweep captured nothing; `entries` is what says
    the op was actually tuned."""
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load", lambda op, gk: {"entries": {}})
    assert builder._cache_answers(_unit("some_op_triton"), {"some_op_triton"}) is False

    monkeypatch.setattr("miniworld_engine.autotune.cache._load", lambda op, gk: None)
    assert builder._cache_answers(_unit("some_op_triton"), {"some_op_triton"}) is False


def test_ok_ops_come_from_cache_status_not_a_second_rulebook(monkeypatch) -> None:
    """The builder must not re-derive staleness: it asks `cache_status.scan`, so a rule added
    there (a new fingerprint, a new scheme) reaches the builder with no edit here."""
    seen: dict = {}

    class _Row:
        def __init__(self, op, verdict):
            self.op, self.verdict = op, verdict

    def _scan(gpu_substr=None):
        seen["gpu_substr"] = gpu_substr
        return [_Row("fresh_triton", "OK"), _Row("moved_triton", "STALE"),
                _Row("odd_triton", "UNKNOWN")]

    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache_status.scan", _scan)
    assert builder._cache_ok_ops() == {"fresh_triton"}
    assert seen["gpu_substr"] == "GPU (sm86)", "must scan THIS card, not every shipped cache"
