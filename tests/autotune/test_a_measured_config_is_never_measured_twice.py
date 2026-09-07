"""The rebuild must not re-search what a card has already searched. One invariant, enumerated.

``miniworld-engine build`` takes options, and none of them should have to be passed for this: the
plain command is the one people run, and the plain command re-measuring a kernel whose search was
already done is the single most expensive failure this repository has. It has happened repeatedly,
each time for a reason a test here now pins:

  * nothing recorded WHICH configs a build searched, so a narrowed ladder -- a change that removes
    work -- cost a full re-tune of every kernel it touched;
  * what was recorded, ``config_space``, was per FILE while a build sweeps per (dtype, bucket), so
    subtracting it would have told a never-visited bucket it was done;
  * the incremental subtraction existed in ``cache.configs_to_bench`` and had no caller anywhere,
    so none of it ran.

The two numbers every test below is really about: a bucket already swept with today's grid must
cost ZERO new measurements, and a bucket nobody has swept must cost the WHOLE grid. Everything
else is a way of getting one of those two wrong.
"""
from __future__ import annotations

import json

import pytest

from miniworld_engine.autotune import cache as C
from miniworld_engine.autotune import capture as X

OP = "adaln_gemm_gate_triton"      # real op: the policy reads its registry row and config grid
GK = "TESTGPU (sm00)"
KEY = "bfloat16|b1"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_CACHE_ROOT", tmp_path)
    C._load_cache.clear()
    return tmp_path


def _configs(n=None, skip=0):
    from miniworld_engine.autotune.configs import configs_for
    cfgs = configs_for(OP)[skip:]
    return cfgs[:n] if n else cfgs


def _write(cfgs, *, bucket="b1", dtype="bfloat16", op_id="opid-A", ranked=None):
    ranked = ranked if ranked is not None else [(c, 1.0 + i) for i, c in enumerate(cfgs)]
    C.store_ranked_configs(OP, GK, dtype, bucket, ranked,
                           C.config_space_hash(cfgs), configs=cfgs, op_id=op_id)
    C._load_cache.clear()


def _read(root):
    return json.loads((root / OP / f"{GK}.json").read_text())


def _todo(cfgs, *, key=KEY, op_id="opid-A"):
    return C.configs_to_bench(OP, GK, cfgs, entry_key=key, op_id=op_id)


# --------------------------------------------------------------------------- #
# the two numbers
# --------------------------------------------------------------------------- #
def test_a_bucket_already_swept_with_this_grid_costs_nothing(root):
    grid = _configs(20)
    _write(grid)
    assert _todo(grid) == [], "a build would re-measure a grid this bucket has already been swept with"


def test_a_bucket_nobody_swept_costs_the_whole_grid(root):
    grid = _configs(20)
    _write(grid, bucket="b1")
    assert len(_todo(grid, key="bfloat16|b2")) == len(grid), (
        "a bucket with no entry was told it had been searched -- the per-file config_space bug, "
        "which hands a fresh shape the delta and calls it tuned")


def test_a_dtype_nobody_swept_costs_the_whole_grid(root):
    """The entry key is dtype AND bucket; an fp32 run is not answered by a bf16 sweep."""
    grid = _configs(20)
    _write(grid, dtype="bfloat16")
    assert len(_todo(grid, key="float32|b1")) == len(grid)


# --------------------------------------------------------------------------- #
# grid edits
# --------------------------------------------------------------------------- #
def test_narrowing_the_grid_costs_nothing(root):
    """The edit this whole path exists for: removing configs cannot create work."""
    _write(_configs(20))
    assert _todo(_configs(8)) == [], "narrowing a ladder re-bought measurements it only removed"


def test_widening_the_grid_costs_exactly_what_was_added(root):
    _write(_configs(8))
    wider = _configs(20)
    todo = _todo(wider)
    assert len(todo) == 12
    known = {C._sig(c) for c in _configs(8)}
    assert not ({C._sig(c) for c in todo} & known), "an already-measured config was re-listed"


def test_a_grid_that_moved_sideways_costs_the_whole_grid(root):
    """Disjoint grids share no measurement, and this is the case that must NOT be optimised.

    A kernel that gains a tunable (`GROUP_M`) makes every stored config a launch that can no longer
    happen. 190 of the 497 measured A6000 buckets are in exactly this state, and calling them done
    would pin configs nobody has ever timed."""
    _write(_configs(8))
    other = _configs(8, skip=8)
    assert not ({C._sig(c) for c in other} & {C._sig(c) for c in _configs(8)}), "slices overlap"
    assert len(_todo(other)) == len(other)


def test_a_config_measured_under_an_older_wider_grid_stays_measured(root):
    """Narrow, then widen back. The second widening must not re-buy the first grid's work."""
    _write(_configs(20))
    _write(_configs(8))                      # narrowed: nothing new to measure
    assert _todo(_configs(20)) == [], "a config measured under the wider grid was forgotten"


# --------------------------------------------------------------------------- #
# what must still invalidate
# --------------------------------------------------------------------------- #
def test_another_kernel_costs_the_whole_grid(root):
    grid = _configs(20)
    _write(grid, op_id="opid-A")
    assert len(_todo(grid, op_id="opid-B")) == len(grid), (
        "a cache measured against different kernel source was treated as already searched")


def test_another_toolchain_costs_the_whole_grid(root):
    grid = _configs(20)
    _write(grid)
    d = _read(root)
    d["env_identity"] = "some-other-triton"
    (root / OP / f"{GK}.json").write_text(json.dumps(d))
    C._load_cache.clear()
    assert len(_todo(grid)) == len(grid)


def test_a_dangling_grid_reference_costs_the_whole_grid(root):
    """A reference with no space behind it is not a smaller space; it is no information."""
    grid = _configs(20)
    _write(grid)
    d = _read(root)
    d["grids"] = {}
    (root / OP / f"{GK}.json").write_text(json.dumps(d))
    C._load_cache.clear()
    assert len(_todo(grid)) == len(grid)


def test_a_file_that_predates_the_field_costs_the_whole_grid(root):
    """Fail closed. Every shipped cache was in this state, and guessing for them is what would
    silently pin configs nobody timed."""
    grid = _configs(20)
    _write(grid)
    d = _read(root)
    del d["entry_grids"]
    (root / OP / f"{GK}.json").write_text(json.dumps(d))
    C._load_cache.clear()
    assert len(_todo(grid)) == len(grid)


def test_the_file_level_space_is_not_read_per_entry(root):
    """`config_space` says which grid the LAST build held, not which buckets it visited."""
    grid = _configs(20)
    _write(grid)
    d = _read(root)
    assert d.get("config_space"), "the file-level space stopped being recorded"
    d["entry_grids"] = {}
    (root / OP / f"{GK}.json").write_text(json.dumps(d))
    C._load_cache.clear()
    assert len(_todo(grid)) == len(grid)


# --------------------------------------------------------------------------- #
# storage: the record has to stay cheap enough to keep
# --------------------------------------------------------------------------- #
def test_the_space_is_stored_once_and_referenced(root):
    """Per entry, this field was 4.3 million lines of JSON across the corpus; by reference it is
    one list per grid."""
    grid = _configs(20)
    for b in ("b1", "b2", "b3"):
        _write(grid, bucket=b)
    d = _read(root)
    assert len(d["grids"]) == 1, "the same grid was stored once per entry"
    assert set(d["entry_grids"]) == {f"bfloat16|{b}" for b in ("b1", "b2", "b3")}
    assert all(v == [C.config_space_hash(grid)] for v in d["entry_grids"].values())


def test_a_grid_nothing_points_at_is_dropped(root):
    """Otherwise a file that has seen six grid edits carries six copies of the space forever."""
    _write(_configs(20))
    _write(_configs(20), op_id="opid-B")     # a reset: the entries go, their provenance goes too
    d = _read(root)
    assert list(d["grids"]) == [C.config_space_hash(_configs(20))]
    assert len(d["grids"]) == 1


# --------------------------------------------------------------------------- #
# the hook that has to call it, and the one flag that turns it off
# --------------------------------------------------------------------------- #
def test_the_build_hook_subtracts_by_default():
    assert X._INCREMENTAL is True, (
        "the subtraction is opt-out, not opt-in: `miniworld-engine build` with no options is the "
        "command people run, and it is the one that must not re-measure")


def test_the_escape_hatch_is_spelled_the_way_the_builder_reads_it():
    import inspect

    from miniworld_engine.autotune import builder
    src = inspect.getsource(builder)
    assert "capture.set_incremental(not args.rebuild_cached)" in src
    assert '"--rebuild-cached"' in src


def test_the_hook_never_hands_triton_an_empty_list(root):
    """`Autotuner.run` does `min(timings, ...)`, and an empty dict makes that a ValueError that
    takes the shard down. Nothing new to measure means ONE config, not zero."""
    grid = _configs(20)
    _write(grid)
    assert _todo(grid) == []
    best = X._cheapest_known(OP, GK, KEY, grid)
    assert len(best) == 1
    assert best[0] in grid


# --------------------------------------------------------------------------- #
# the two sides of the cache must key a launch identically
# --------------------------------------------------------------------------- #
def test_the_prune_and_the_recorder_derive_one_key() -> None:
    """The failure this forecloses is silent: a subtraction against the wrong entry is not a
    crash, it is a winner measured for another shape. Both sides call one function."""
    import inspect

    src = inspect.getsource(X)
    assert src.count("_bucket_of(") == 1, (
        "the bucket is derived in more than one place; prune and record can now disagree")
    assert src.count("_dtype_of(") == 1, (
        "the dtype is derived in more than one place; prune and record can now disagree")
    assert "key = _entry_key(autotuner, kwargs)" in src
    assert "dtype, bucket = _entry_parts(autotuner, meta, nargs)" in src


def test_the_key_is_the_one_a_cache_file_is_written_with(root) -> None:
    """`_entry_key` joins with '|', which is what `store_ranked_configs` files an entry under."""
    _write(_configs(4), dtype="bfloat16", bucket="b1")
    assert KEY == "bfloat16|b1", "the key this file tests is not the one the cache is keyed by"
    assert KEY in _read(root)["entries"]
