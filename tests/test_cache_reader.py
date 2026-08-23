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
    cache.store_ranked_configs(op, cache.gpu_key(), dtype, bucket, ranked, h,
                               op_id=cache.op_identity(at))
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


# --------------------------------------------------------------------------- #
# identity: what must invalidate an entry BESIDES the config grid
#
# The grid hash alone answered "are these the same candidate configs?". It never answered "was
# this measured by the same compiler, on the same device, against the same kernel?". Three edits
# left the hash untouched and the entry silently wrong: bumping triton/ptxas, editing the kernel
# body, and editing the autotuner's ``key=[...]`` (which re-partitions the buckets, so a stored
# bucket answers a question the runtime no longer asks). triton-dejavu keys its storage path on
# all of them; these tests pin that we do too.
# --------------------------------------------------------------------------- #

class _JitFn:
    def __init__(self, src):
        self.src = src


class _AutotunerWithFn(_Autotuner):
    def __init__(self, configs, keys, nargs, src):
        super().__init__(configs, keys, nargs)
        self.fn = _JitFn(src)


_SRC = "@triton.jit\ndef k(x, BLOCK_M1: tl.constexpr):\n    tl.store(x, 1)\n"


def _at(src=_SRC, keys=("shape_key",)):
    return _AutotunerWithFn(GRID, list(keys),
                            {"x": torch.empty(2, 2, dtype=torch.bfloat16), "shape_key": 256}, src)


def test_a_cache_built_by_a_different_toolchain_is_not_served(tmp_path, monkeypatch):
    """A tuned config is a claim about a compiler, not just about a grid."""
    at = _at()
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    assert cache._cached_subset(at, at.configs, at.nargs, {}) is not None, "sanity: it reads back"
    cache._load_cache.clear()
    monkeypatch.setattr(cache, "env_identity", lambda: "triton-4-0-0")
    assert cache._cached_subset(at, at.configs, at.nargs, {}) is None


def test_an_edited_kernel_body_invalidates_the_entry(tmp_path, monkeypatch):
    at = _at()
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    cache._load_cache.clear()
    edited = _at(src=_SRC.replace("tl.store(x, 1)", "tl.store(x, 2)"))
    assert cache._cached_subset(edited, edited.configs, edited.nargs, {}) is None


def test_an_edited_autotune_key_list_invalidates_the_entry(tmp_path, monkeypatch):
    """``key=[...]`` decides how entries are partitioned; changing it changes what a bucket means."""
    at = _at()
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    cache._load_cache.clear()
    rekeyed = _at(keys=("shape_key", "dtype"))
    assert cache._cached_subset(rekeyed, rekeyed.configs, rekeyed.nargs, {}) is None


def test_reformatting_a_kernel_does_not_invalidate_it(tmp_path, monkeypatch):
    """Blank lines and trailing whitespace are not semantics. dejavu hashes the JIT function with
    line numbers excluded for the same reason: a reformat must not throw away a measured cache."""
    at = _at()
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    cache._load_cache.clear()
    moved = _at(src="\n\n" + _SRC.replace("\n", "   \n") + "\n\n")
    assert cache.op_identity(moved) == cache.op_identity(at), "whitespace must not count"
    assert cache._cached_subset(moved, moved.configs, moved.nargs, {}) is not None


def test_an_entry_written_without_an_op_identity_still_reads(tmp_path, monkeypatch):
    """Caches committed before this field exists must degrade to the old behaviour, not to a
    permanent miss that silently re-benches 205k configs inside a production forward."""
    at = _at()
    _entry_for(at, GRID[1:3], tmp_path, monkeypatch)
    import json
    fp = next(tmp_path.rglob("*.json"))
    data = json.loads(fp.read_text())
    data.pop("op_identity")
    fp.write_text(json.dumps(data))
    cache._load_cache.clear()
    assert cache._cached_subset(at, at.configs, at.nargs, {}) is not None


# --------------------------------------------------------------------------- #
# a cache MISS must cost a bounded search, not the whole grid
#
# Returning None on a miss means "keep the full grid", and the shipped grid is 205,266 configs --
# so the first production forward on a GPU nobody has built a cache for runs a tuning sweep inside
# itself. No one in the ecosystem does that: triton-dejavu takes a user heuristic on miss
# (TRITON_DEJAVU_FORCE_FALLBACK), Liger-Kernel skips autotuning and derives num_warps from the row
# width, vLLM ships a default and warns. A build still gets the full space, on purpose.
# --------------------------------------------------------------------------- #

BIG = [_Cfg(m, warps=w) for m in (16, 32, 64, 128, 256) for w in (1, 2, 4, 8, 16, 32)]


def test_a_miss_returns_a_bounded_subset_not_the_whole_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    cache._load_cache.clear()
    at = _at()
    at.configs = BIG
    import miniworld_engine.autotune.configs as cfgmod
    monkeypatch.setattr(cfgmod, "op_of", lambda _c: "op_probe")
    got = cache._cached_subset(at, BIG, at.nargs, {})
    assert got is not None, "a miss must still narrow the grid"
    assert len(got) <= settings.current().autotune_miss_cap < len(BIG)


def test_a_build_still_gets_the_full_grid(tmp_path, monkeypatch):
    """run_autotune=True is a tuning run; capping it would make the build confirm its own guess."""
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    cache._load_cache.clear()
    at = _at(); at.configs = BIG
    import miniworld_engine.autotune.configs as cfgmod
    monkeypatch.setattr(cfgmod, "op_of", lambda _c: "op_probe")
    prev = settings.configure(run_autotune=True)
    try:
        assert cache._cached_subset(at, BIG, at.nargs, {}) is None
    finally:
        settings.restore(prev) if hasattr(settings, "restore") else settings.configure(
            run_autotune=prev.run_autotune if hasattr(prev, "run_autotune") else False)


def test_the_fallback_prefers_the_industry_centre_of_the_space():
    """warps in {4,8} and stages in {2,3,4} -- vLLM's whole fused_moe space -- come first."""
    got = cache.heuristic_subset(BIG, cap=8)
    assert all(c.num_warps in (4, 8) for c in got), [c.num_warps for c in got]


def test_the_fallback_never_invents_a_config():
    got = cache.heuristic_subset(BIG, cap=8)
    sigs = {cache._sig(c) for c in BIG}
    assert all(cache._sig(c) in sigs for c in got)


def test_a_small_grid_is_left_alone():
    """Capping a 4-config grid would throw away a search that is already cheap."""
    small = BIG[:4]
    assert cache.heuristic_subset(small, cap=24) == small


def test_the_toolchain_identity_actually_contains_ptxas():
    """`env_identity` exists to notice a compiler change; a probe that silently misses defeats it.

    The first version imported `_path_to_binary` from `triton.backends.nvidia.driver`, where it
    does not exist in triton 3.6 — so the except swallowed it and every identity ever produced
    recorded `ptxas=?`. The component the function is for was simply absent, and nothing said so.
    Found by `ty`, not by reading the code.
    """
    assert cache._ptxas_version() != "?", (
        "ptxas could not be located; env_identity is blind to a toolkit change")
    assert "release" in cache._ptxas_version().lower()


def test_the_identity_changes_when_ptxas_does(monkeypatch):
    cache._env_identity_cache = None
    monkeypatch.setattr(cache, "_ptxas_version", lambda: "release 1.0")
    a = cache.env_identity()
    cache._env_identity_cache = None
    monkeypatch.setattr(cache, "_ptxas_version", lambda: "release 2.0")
    b = cache.env_identity()
    cache._env_identity_cache = None
    assert a != b
