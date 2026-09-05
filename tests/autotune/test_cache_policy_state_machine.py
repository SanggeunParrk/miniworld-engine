"""The cache invalidation policy, enumerated. Every transition, on CPU, with no GPU anywhere.

The policy decides one thing -- given a cache file and a write, are the stored measurements kept or
discarded -- and it decides it from six recorded fields. That is a finite state machine over pure
functions of files in the repo, so it is checkable exhaustively rather than by smoke test. It has
been got wrong four times in one day, each time in a way a test like this would have caught:

  * resetting on ``driver_identity`` deleted 32 of 38 tuned buckets for an edit that changed only
    WHICH buckets get built;
  * demoting ``op_identity`` without a reset made the reader refuse a rebuild's own fresh entries
    (stamp kept) or serve a pre-edit winner as current (stamp rewritten);
  * dropping the grid from the reset predicate let the cute reader hand back a config the live
    grid no longer contains;
  * ``int(x or 1)`` read a stored ``build_rev`` of 0 as 1, so revision 0 and 1 compared equal.

Each test below is one of those, plus the transitions nobody has broken yet.
"""
from __future__ import annotations

import json

import pytest

from miniworld_engine.autotune import cache as C

OP = "adaln_gemm_gate_triton"      # real op: the policy reads its registry row and config grid
GK = "TESTGPU (sm00)"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_CACHE_ROOT", tmp_path)
    C._load_cache.clear()
    return tmp_path


def _configs(n=None):
    from miniworld_engine.autotune.configs import configs_for
    cfgs = configs_for(OP)
    return cfgs[:n] if n else cfgs


def _write(cfgs, *, bucket="b1", ranked=None, op_id="opid-A"):
    """One build's worth of writing, through the real entry point."""
    ranked = ranked if ranked is not None else [(c, 1.0 + i) for i, c in enumerate(cfgs)]
    C.store_ranked_configs(OP, GK, "bfloat16", bucket, ranked,
                           C.config_space_hash(cfgs), configs=cfgs, op_id=op_id)
    C._load_cache.clear()


def _read(root):
    return json.loads((root / OP / f"{GK}.json").read_text())


def _mutate(root, **fields):
    d = _read(root)
    d.update(fields)
    (root / OP / f"{GK}.json").write_text(json.dumps(d))
    C._load_cache.clear()


# --------------------------------------------------------------------------- keeps

def test_a_repeated_build_keeps_its_entries(root):
    _write(_configs(20))
    before = _read(root)["entries"]
    _write(_configs(20), bucket="b2")
    after = _read(root)["entries"]
    assert set(before) <= set(after), "an identical rebuild discarded a bucket"


def test_a_narrowed_grid_keeps_the_winners_it_still_contains(root):
    """The case that cost 25 GPU-hours to apply under the old rule."""
    _write(_configs(20))
    kept_before = _read(root)["entries"]["bfloat16|b1"]
    _write(_configs(12), bucket="b2")
    entry = _read(root)["entries"].get("bfloat16|b1")
    assert entry, "narrowing the grid discarded an untouched bucket"
    live = {C._sig(c) for c in _configs(12)}
    assert all(C._sig_from_dict(c) in live for c in entry), (
        "a stored config survived that the narrowed grid no longer contains")
    assert len(entry) <= len(kept_before)


def test_a_widened_grid_keeps_the_winners(root):
    _write(_configs(20))
    _write(_configs(30), bucket="b2")
    assert _read(root)["entries"].get("bfloat16|b1"), "widening the grid discarded a bucket"


def test_a_changed_build_driver_keeps_the_entries(root):
    """`driver_identity` says which buckets get BUILT, never whether a winner is right."""
    _write(_configs(20))
    _mutate(root, driver_identity="deadbeef0000", driver_id_scheme=C.DRIVER_ID_SCHEME)
    _write(_configs(20), bucket="b2")
    assert _read(root)["entries"].get("bfloat16|b1"), "a driver edit discarded a measured bucket"


# --------------------------------------------------------------------------- resets

@pytest.mark.parametrize("field,value,why", [
    ("build_rev", 0, "a person declared the measurement method changed"),
    ("env_identity", "deadbeef0000", "another triton/cuda/ptxas measured it"),
    ("op_identity", "deadbeef0000", "a different kernel body"),
])
def test_an_invalidating_field_resets(root, field, value, why):
    _write(_configs(20))
    _mutate(root, **{field: value})
    _write(_configs(20), bucket="b2")
    assert "bfloat16|b1" not in _read(root)["entries"], f"kept entries although {why}"


def test_build_rev_zero_is_not_read_as_one(root):
    """`int(x or 1)` made revision 0 compare equal to 1; 0 is falsy and revisions start at 1."""
    _write(_configs(20))
    _mutate(root, build_rev=0)
    assert C._stored_rev(_read(root)) == 0
    _write(_configs(20), bucket="b2")
    assert "bfloat16|b1" not in _read(root)["entries"]


# --------------------------------------------------------------------------- the reader

def test_the_reader_refuses_when_every_tuned_config_was_cut(root, monkeypatch):
    """Triton path, driven through the real reader.

    Store winners, then narrow the grid so that NONE of them survive. Before the grid stopped
    resetting the file this state was unreachable; now it is the normal outcome of a narrowing, and
    the reader must say so rather than silently hand back the full grid -- which this module's own
    docstring calls ruinous inside a forward.
    """
    import torch

    from miniworld_engine.autotune import configs as cfgmod

    class _Cfg:
        def __init__(self, m):
            self.kwargs, self.num_warps, self.num_stages = {"BLOCK_M": m}, 4, 2
        def __repr__(self): return f"_Cfg({self.kwargs['BLOCK_M']})"

    class _At:
        def __init__(self, configs, nargs):
            self.configs, self.keys, self.nargs = configs, ["shape_key"], nargs

    old_grid = [_Cfg(m) for m in (32, 64)]
    new_grid = [_Cfg(m) for m in (128, 256)]          # disjoint from what was tuned
    nargs = {"x": torch.empty(2, 2, dtype=torch.bfloat16), "shape_key": 128}
    at_old = _At(old_grid, nargs)
    monkeypatch.setattr(cfgmod, "op_of", lambda cfgs: "op_probe")
    C.store_ranked_configs("op_probe", C.gpu_key(), C.dtype_of_args(nargs),
                           C.bucket_of_autotuner(at_old, nargs, {}),
                           [(c, 1.0) for c in old_grid],
                           C.config_space_hash(new_grid),   # written under the NEW space
                           configs=new_grid)
    C._load_cache.clear()
    at_new = _At(new_grid, nargs)
    got = C._cached_subset(at_new, new_grid, nargs, {})
    assert got is None, f"reader returned {got!r} from an entry with no live config"
    misses = C.cache_misses()
    assert misses, "the reader fell back to the full grid without recording a miss"


def test_select_config_refuses_an_entry_the_candidate_space_lost(root):
    """CuTe path -- the hole the review found: it returned entry[0] with no membership check.

    Constructed so the cut is certain rather than incidental: the stored winners and the live
    candidates are disjoint.
    """
    tuned = _configs(4)
    live_cfgs = _configs(12)[8:]                      # disjoint slice
    assert not ({C._sig(c) for c in tuned} & {C._sig(c) for c in live_cfgs}), "slices overlap"
    _write(tuned)
    live = [C.config_to_dict(c) for c in live_cfgs]
    got = C.select_config(OP, dtype="bfloat16", bucket="b1", candidates=live)
    assert got is None, f"select_config returned {got!r}, which is not a live candidate"


def test_config_space_is_recorded_whenever_entries_are(root):
    """`configs_to_bench` trusts this field; a write that updates the hash without it is a lie."""
    _write(_configs(20))
    d = _read(root)
    assert d.get("config_space"), "entries written with no config_space recorded"
    assert len(d["config_space"]) == 20
    assert d["config_space_hash"] == C.config_space_hash(_configs(20))
