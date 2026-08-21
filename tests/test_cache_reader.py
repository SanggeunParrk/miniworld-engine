"""The shipped cache must actually narrow a Triton autotuner's grid, under the SAME key it was
written with.

Since fcd3c7a nothing read it. `select_config` is the only other reader and only the CuTe/CUDA
paths call it, so every process re-benched the full grid in-process while the committed
`data/*.json` was written and never read back. Nothing looked wrong because every shipped config
set holds ONE config per op, which makes a full sweep free -- and it stops being free the moment a
real search grid is installed (`configs/grid` is 205,266 configs, per shape bucket, inside a
production forward).

The other half of the contract is that read and write agree on the key. They now call one pair of
functions, `dtype_of_args` / `bucket_of_autotuner`; the tests below write through the capture path
and read through the reader path, so a divergence fails here rather than silently missing forever.
"""
from __future__ import annotations

import torch
import triton

from miniworld_engine import settings
from miniworld_engine.autotune import cache


class _Cfg:
    def __init__(self, bm, warps=4):
        self.kwargs = {"BLOCK_M1": bm}
        self.num_warps = warps
        self.num_stages = 2
        self.maxnreg = None


class _Autotuner:
    """Enough of triton's Autotuner for the reader: it reads .configs, .keys, .nargs."""

    def __init__(self, configs, keys, nargs):
        self.configs = configs
        self.keys = keys
        self.nargs = nargs


GRID = [_Cfg(m) for m in (32, 64, 128, 256)]


def _entry_for(at, keep, tmp_path, monkeypatch, *, op="op_probe", csh=None):
    """Write a cache file the way the BUILD writes one, then point the reader at it."""
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    cache._load_cache.clear()
    ranked = [(c, 0.1 * i) for i, c in enumerate(keep)]
    h = cache.config_space_hash(at.configs) if csh is None else csh
    dtype = cache.dtype_of_args(at.nargs)
    bucket = cache.bucket_of_autotuner(at, at.nargs, {})
    cache.store_ranked_configs(op, cache.gpu_key(), dtype, bucket, ranked, h)
    monkeypatch.setattr(cache, "op_of", lambda _c: op, raising=False)
    import miniworld_engine.autotune.configs as cfgmod
    monkeypatch.setattr(cfgmod, "op_of", lambda _c: op)
    return at


def test_the_reader_narrows_the_grid_to_the_cached_configs(tmp_path, monkeypatch):
    at = _Autotuner(GRID, ["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16),
                                          "shape_key": 256})
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    got = cache._cached_subset(at, at.configs, at.nargs, {})
    assert got is not None, "a written entry must be found again"
    assert [c.kwargs["BLOCK_M1"] for c in got] == [64, 128]


def test_a_different_shape_bucket_does_not_reuse_the_entry(tmp_path, monkeypatch):
    """The whole point of the shape key: 256's winner must not be served to 1024."""
    at = _Autotuner(GRID, ["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16),
                                          "shape_key": 256})
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    at.nargs = {"x": torch.empty(2, 2, dtype=torch.bfloat16), "shape_key": 1024}
    assert cache._cached_subset(at, at.configs, at.nargs, {}) is None


def test_a_stale_config_space_hash_falls_back_to_the_full_grid(tmp_path, monkeypatch):
    at = _Autotuner(GRID, ["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16),
                                          "shape_key": 256})
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch, csh="deadbeef")
    assert cache._cached_subset(at, at.configs, at.nargs, {}) is None


def test_the_cache_can_only_narrow_within_the_live_grid(tmp_path, monkeypatch):
    """A cached config the grid no longer offers must not be resurrected -- the cache names
    configs, the grid decides what is launchable."""
    at = _Autotuner(GRID, ["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16),
                                          "shape_key": 256})
    _entry_for(at, [_Cfg(4096)], tmp_path, monkeypatch)
    got = cache._cached_subset(at, at.configs, at.nargs, {})
    assert got is None, "an entry naming only configs outside the grid must not narrow it"


def test_a_build_ignores_the_cache(tmp_path, monkeypatch):
    """run_autotune=True means re-bench the whole grid on purpose; a reader that narrowed it
    would make every build a no-op that re-confirms its own previous answer."""
    at = _Autotuner(GRID, ["shape_key"], {"x": torch.empty(2, 2, dtype=torch.bfloat16),
                                          "shape_key": 256})
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    prev = settings.configure(run_autotune=True)
    try:
        @triton.autotune(configs=list(GRID), key=["shape_key"])
        @triton.jit
        def _k(x, shape_key, BLOCK_M1: triton.language.constexpr):  # noqa: ANN001, N803
            pass

        pruned = _k.early_config_prune(list(GRID), at.nargs)
        assert len(pruned) == len(GRID)
    finally:
        import dataclasses
        settings.configure(**dataclasses.asdict(prev))


def test_the_reader_is_installed_on_every_autotuner():
    """Installed by patching Autotuner.__init__, so a kernel cannot forget to wire it -- which is
    exactly how the per-kernel version was lost."""
    @triton.autotune(configs=[_Cfg(64)], key=["shape_key"])
    @triton.jit
    def _k2(x, shape_key, BLOCK_M1: triton.language.constexpr):  # noqa: ANN001, N803
        pass

    assert _k2.early_config_prune is not None


def test_installing_the_reader_does_not_stop_capture_recording(monkeypatch):
    """The reader and the recorder must coexist. They did not.

    `_record_one` used to read `_miniworld_dtype_of` / `_miniworld_bucket_of` off
    `early_config_prune`, guarded by `if ecp`. Once fcd3c7a deleted the prune OBJECTS that carried
    those attributes the guard was only ever false, so the dead branch was invisible -- until the
    cache reader began installing a prune FUNCTION on every autotuner, which made `ecp` truthy
    everywhere and turned every call into an AttributeError. The caller's `except: pass` swallowed
    it, so a build ran to completion, reported success, and wrote an empty shard.

    Two things stop that recurring: the key is always derived from the autotuner itself, and a
    recording failure is counted and warned about instead of vanishing.
    """
    from miniworld_engine.autotune import capture

    @triton.autotune(configs=[_Cfg(64), _Cfg(128)], key=["shape_key"])
    @triton.jit
    def _k3(x, shape_key, BLOCK_M1: triton.language.constexpr):  # noqa: ANN001, N803
        pass

    at = _k3
    assert at.early_config_prune is not None, "the reader must be installed on this autotuner"

    capture._CAPTURE.clear()
    capture._RECORD_ERRORS.clear()
    at.nargs = {"x": torch.empty(2, 2, dtype=torch.bfloat16), "shape_key": 256}
    monkeypatch.setattr(capture, "_op_name", lambda _a: "op_probe")

    capture._record_one(at, _Cfg(64), {}, 0.5)

    assert not capture.record_errors(), capture.record_errors()
    assert capture._CAPTURE["op_probe"]["entries"], "nothing was recorded"
    assert list(capture._CAPTURE["op_probe"]["entries"]) == [("bfloat16", "shape_key=256")]
