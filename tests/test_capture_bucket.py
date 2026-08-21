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
