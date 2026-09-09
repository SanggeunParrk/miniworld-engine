"""`build all` must skip a unit only when the RUNTIME would actually serve its cached answer.

The builder used to plan every (op, dtype, shape bucket) the config grid declares and filter only
by `--resume`, which knows just what THIS shard dir produced. A rebuild therefore re-benched work
the repo already ships.

Skipping is the fix, but "already answered" is a sharper question than it looks, and the first
version of this got it wrong in two ways that these tests now pin:

  * `dev cache-status`'s verdict is not the question. Its "OK" means "not stale enough to fail
    CI", and two of its OK reasons -- `config grid changed -- incremental build pending` and
    `build driver changed` -- mean the op still owes a build. The runtime reader is stricter: a
    `config_space_hash` mismatch is a full miss. On this repo every one of the 51 OK A6000 rows
    carried the first reason, so trusting the verdict skipped 51 ops whose caches no launch can
    use.
  * dtype is a decidable axis of the entry key and was being ignored. 23 of this card's 74 caches
    hold no fp32 entry at all; skipping their fp32 units left a hole that the next build skipped
    again for the same reason, forever.
"""
from __future__ import annotations

import pytest

from miniworld_engine import cli
from miniworld_engine.autotune import builder


class _Row:
    """Shaped like `cache_status.CacheStatus` -- including the two fields the first version of
    this test left out, which is exactly where the bugs were."""

    def __init__(self, op, verdict, reason="", env_matches=True):
        self.op, self.verdict, self.reason, self.env_matches = op, verdict, reason, env_matches


def _rows(monkeypatch, rows):
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache_status.scan",
                        lambda gpu_substr=None: rows)


def test_an_ok_row_that_still_owes_a_build_is_not_done(monkeypatch) -> None:
    """The bug that mattered: `verdict == "OK"` with a reason means the op is NOT finished."""
    _rows(monkeypatch, [
        _Row("clean_triton", "OK"),
        _Row("pending_triton", "OK", "config grid changed -- incremental build pending"),
        _Row("driver_triton", "OK", "build driver changed -- coverage may differ; rebuild the op"),
    ])
    assert builder._cache_ok_ops() == {"clean_triton"}


def test_a_toolchain_mismatch_is_not_done(monkeypatch) -> None:
    """`env_identity` is kept out of the verdict on purpose (it must not fail CI elsewhere), but
    the runtime treats a mismatch as a miss -- so the builder has to read it itself."""
    _rows(monkeypatch, [_Row("other_toolchain_triton", "OK", "", env_matches=False),
                        _Row("no_env_recorded_triton", "OK", "", env_matches=None),
                        _Row("here_triton", "OK", "", env_matches=True)])
    assert builder._cache_ok_ops() == {"here_triton"}


def test_stale_and_unknown_are_not_done(monkeypatch) -> None:
    _rows(monkeypatch, [_Row("moved_triton", "STALE", "kernel source/key changed"),
                        _Row("odd_triton", "UNKNOWN", "unreadable JSON"),
                        _Row("fine_triton", "OK")])
    assert builder._cache_ok_ops() == {"fine_triton"}


def test_the_scan_is_asked_about_this_card(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")

    def _scan(gpu_substr=None):
        seen["gpu_substr"] = gpu_substr
        return []

    monkeypatch.setattr("miniworld_engine.autotune.cache_status.scan", _scan)
    builder._cache_ok_ops()
    assert seen["gpu_substr"] == "GPU (sm86)", "must scan THIS card, not every shipped cache"


@pytest.mark.parametrize(("label", "dtype", "serves"), [
    ("bfloat16", "bfloat16", True),
    # a bf16 launch whose norm affine is pinned fp32 records BOTH dtypes; it still covers bf16
    ("bfloat16+float32", "bfloat16", True),
    # ...and it must NOT be read as fp32 cover, which a substring test would do
    ("bfloat16+float32", "float32", False),
    ("float32", "float32", True),
    ("bfloat16", "float32", False),
])
def test_a_dtype_label_is_matched_by_rule_not_substring(label, dtype, serves) -> None:
    assert builder._label_serves_dtype(label, dtype) is serves


def test_a_bf16_only_cache_does_not_answer_an_fp32_unit(monkeypatch) -> None:
    """The self-perpetuating hole: skip the fp32 units of a bf16-only cache and no build ever
    fills them, because the next build sees the same non-empty cache and skips again."""
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load", lambda op, gk: {
        "entries": {"bfloat16|shape_key=1": [{}], "bfloat16|shape_key=2": [{}]}})
    ok = {"half_built_triton"}
    assert builder._cache_answers(
        builder.OpUnit(op="half_built_triton", length=384, dtype="bfloat16"), ok) is True
    assert builder._cache_answers(
        builder.OpUnit(op="half_built_triton", length=384, dtype="float32"), ok) is False


def test_an_op_outside_the_ok_set_is_work_whatever_its_file_says(monkeypatch) -> None:
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load",
                        lambda op, gk: {"entries": {"bfloat16|shape_key=1": [{}]}})
    unit = builder.OpUnit(op="moved_triton", length=384, dtype="bfloat16")
    assert builder._cache_answers(unit, ok_ops=set()) is False


def test_an_empty_or_missing_cache_is_not_an_answer(monkeypatch) -> None:
    """A merge writes a file per op even when the sweep captured nothing."""
    monkeypatch.setattr("miniworld_engine.autotune.cache.gpu_key", lambda: "GPU (sm86)")
    unit = builder.OpUnit(op="empty_triton", length=384, dtype="bfloat16")
    monkeypatch.setattr("miniworld_engine.autotune.cache._load", lambda op, gk: {"entries": {}})
    assert builder._cache_answers(unit, {"empty_triton"}) is False
    monkeypatch.setattr("miniworld_engine.autotune.cache._load", lambda op, gk: None)
    assert builder._cache_answers(unit, {"empty_triton"}) is False


def test_the_escape_hatch_is_spelled_the_way_the_code_reads_it() -> None:
    """`cmd_build` reads the flag with `getattr(args, "rebuild", False)`, so a rename or a typo
    would not raise -- it would silently mean "only ever fill gaps" and the hatch would be gone.

    The flag is `--rebuild` now. The op-level skip these tests are about is what `--rebuild-cached`
    used to turn off; the CLI no longer uses that skip at all (see
    `test_the_cli_no_longer_skips_a_unit_because_its_op_is_answered`), so the two questions the
    flag used to answer at once -- "run this unit?" and "re-measure keys it already has?" -- are
    one question now, and this is its name."""
    parsed = cli.build_parser().parse_args(["build", "all", "--rebuild"])
    assert parsed.rebuild is True
    assert cli.build_parser().parse_args(["build", "all"]).rebuild is False
    # the old spelling still works: job scripts and shell history carry it
    assert cli.build_parser().parse_args(["build", "all", "--rebuild-cached"]).rebuild is True


def test_the_cli_no_longer_skips_a_unit_because_its_op_is_answered() -> None:
    """The op-level skip and the new default cannot both be on.

    `skip_cached` drops a unit when the cache answers its op at ANY bucket -- which is exactly the
    unit that owes a NEW bucket after a ladder change. That was survivable while the fix was a
    separate flag; with "build what is missing" as the default it would cancel the default, so
    `cmd_build` passes `skip_cached=False` and lets `fill_gaps` decide what gets benched. The
    parameter stays on `build_all` for callers that plan whole ops rather than modules."""
    import inspect

    src = inspect.getsource(cli.cmd_build)
    assert "skip_cached=False" in src, (
        "cmd_build no longer forces skip_cached off; a unit whose op is answered at some other "
        "bucket would be dropped before fill_gaps could look at the bucket it owes")


def test_skipping_is_the_default_in_the_signature() -> None:
    """The point of the change: an ordinary `build all` skips. A default flip would be silent."""
    import inspect
    assert inspect.signature(builder.build_all).parameters["skip_cached"].default is True
