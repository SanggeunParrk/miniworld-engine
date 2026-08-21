"""Captured timings must be split by shape bucket, not lumped into ``any|any``.

The kernels used to carry a hand-written ``make_cache_prune(bucket_of=key_bucket_of("N","K","DT"))``
each; fcd3c7a removed them and nothing took over, so ``_record_one`` fell through to its
``"any"`` default. Every cache built since records exactly one entry, ``any|any`` -- one config per
op for every shape. Nothing failed: Triton still re-tunes per ``shape_key`` in-process, so runs
were fast, and only the PERSISTED cache lost the distinction. That is the failure mode this pins.

38 kernel files still import ``key_bucket_of``/``tensor_dtype_of`` and call neither; lint cannot
see it because ``kernels/**/triton/*.py`` is excluded from ruff entirely.
"""
from __future__ import annotations

import torch

from miniworld_engine.autotune import capture


class _Autotuner:
    """Enough of ``triton.runtime.autotuner.Autotuner`` for ``_record_one``."""

    def __init__(self, keys, nargs, configs):
        self.keys = keys
        self.nargs = nargs
        self.configs = configs
        self.early_config_prune = None


class _Cfg:
    def __init__(self, **kw):
        self.kwargs = kw
        self.num_warps = 4
        self.num_stages = 2
        self.maxnreg = None


def _capture(keys, nargs, op="op_probe"):
    capture._CAPTURE.clear()
    at = _Autotuner(keys, nargs, [_Cfg(BLOCK_M1=64)])
    # _op_name resolves through configs.op_of(list identity); stub it to name this fake op.
    orig = capture._op_name
    capture._op_name = lambda a: op
    try:
        capture._record_one(at, _Cfg(BLOCK_M1=64), {}, 1.0)
    finally:
        capture._op_name = orig
    return set(capture._CAPTURE[op]["entries"])


def test_bucket_comes_from_the_kernels_own_key_list():
    got = _capture(["shape_key", "K"], {"x": torch.empty(4, 4), "shape_key": 256, "K": 128})
    assert got == {("float32", "K=128,shape_key=256")}, got


def test_two_shape_keys_are_two_buckets():
    """The whole point of the shape key: 256 and 1024 must not share one cached winner."""
    capture._CAPTURE.clear()
    at = _Autotuner(["shape_key"], {"x": torch.empty(4, 4, dtype=torch.bfloat16), "shape_key": 256},
                    [_Cfg(BLOCK_M1=64)])
    orig = capture._op_name
    capture._op_name = lambda a: "op_probe"
    try:
        capture._record_one(at, _Cfg(BLOCK_M1=64), {}, 1.0)
        at.nargs = {"x": torch.empty(4, 4, dtype=torch.bfloat16), "shape_key": 1024}
        capture._record_one(at, _Cfg(BLOCK_M1=64), {}, 2.0)
    finally:
        capture._op_name = orig
    assert set(capture._CAPTURE["op_probe"]["entries"]) == {
        ("bfloat16", "shape_key=256"), ("bfloat16", "shape_key=1024")}


def test_dtype_is_read_off_the_operands():
    got = _capture(["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16), "shape_key": 512})
    assert got == {("bfloat16", "shape_key=512")}, got


def test_no_usable_key_still_records_rather_than_crashing():
    assert _capture([], {}) == {("any", "any")}


def test_a_measured_timing_that_is_not_a_number_is_not_recorded():
    """inf is how a FAILED config scores, and a NaN arriving from the bench is a broken reading.

    Not a blanket isfinite filter, which is what this started as and was wrong: NaN also reaches
    `_record_one` on purpose, from the single-config path (see below). Filtering it unconditionally
    stopped that path recording anything at all -- caught only by an end-to-end run that captured
    zero ops.
    """
    for bad in (float("inf"), float("nan")):
        capture._CAPTURE.clear()
        at = _Autotuner(["shape_key"], {"x": torch.empty(2, 2), "shape_key": 256}, [_Cfg(BLOCK_M1=64)])
        orig = capture._op_name
        capture._op_name = lambda a: "op_probe"
        try:
            capture._record_one(at, _Cfg(BLOCK_M1=64), {}, bad)
        finally:
            capture._op_name = orig
        assert not capture._CAPTURE, f"{bad} was recorded"


def test_a_finite_timing_is_recorded():
    """The guard must not reject real readings."""
    capture._CAPTURE.clear()
    at = _Autotuner(["shape_key"], {"x": torch.empty(2, 2), "shape_key": 256}, [_Cfg(BLOCK_M1=64)])
    orig = capture._op_name
    capture._op_name = lambda a: "op_probe"
    try:
        capture._record_one(at, _Cfg(BLOCK_M1=64), {}, 0.25)
    finally:
        capture._op_name = orig
    assert capture._CAPTURE["op_probe"]["entries"]


def test_the_sole_config_of_a_one_config_op_is_still_recorded():
    """An op with one config runs no tuning loop, so its config is the winner by default and there
    is no measurement. That record is wanted -- it is what tells a reader which config to use --
    and it is why the 37 caches built from the one-config sets hold `"ms": NaN`."""
    capture._CAPTURE.clear()
    at = _Autotuner(["shape_key"], {"x": torch.empty(2, 2), "shape_key": 256}, [_Cfg(BLOCK_M1=64)])
    orig = capture._op_name
    capture._op_name = lambda a: "op_probe"
    try:
        capture._record_one(at, _Cfg(BLOCK_M1=64), {}, float("nan"), unmeasured=True)
    finally:
        capture._op_name = orig
    assert capture._CAPTURE["op_probe"]["entries"], "the sole config must still be recorded"


def test_an_unmeasured_entry_never_outranks_a_measured_one():
    """`sorted` on a NaN key neither raises nor orders -- comparisons against NaN are all False --
    so an unmeasured entry can land at the head of the ranking, where store_ranked_configs reads
    it as the winner. It must sort last regardless of arrival order."""
    a, b, c = _Cfg(BLOCK_M1=32), _Cfg(BLOCK_M1=64), _Cfg(BLOCK_M1=128)
    for order in ([(a, float("nan")), (b, 2.0), (c, 1.0)],
                  [(c, 1.0), (a, float("nan")), (b, 2.0)],
                  [(b, 2.0), (c, 1.0), (a, float("nan"))]):
        ranked = capture._rank(order)
        assert [r[0].kwargs["BLOCK_M1"] for r in ranked] == [128, 64, 32], order
