"""A `level=both` kernel keys on its ROW COUNT, because length collides across its two sides.

A pair activation (B, L, L, D) at L=1024 and an atom activation (B, A, D) at A=1024 both have
`shape[-2] == 1024`, so `both_key(length)` put them in one bucket while the first launches
1,048,576 rows and the second launches 1,024. It is visible in the shipped A6000 cache for
transition_layernorm_expand_swiglu_triton:

    shape_key=384    0.6103 ms     pair  L=384,  M = 147,456
    shape_key=1024   0.0215 ms     atom  A=1024, M =   1,024

The bucket grows and the time falls 28x. The module bench then ran the pair side at L=1024
against a config tuned at 1,024 rows and measured 9.50 ms where the same card in the same
configuration had measured 5.50 ms a month earlier.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache
from miniworld_engine.autotune.builder import op_units
from miniworld_engine.autotune.shape_key import (
    ATOM_SHAPES,
    BOTH_ROWS,
    TOKEN_SHAPES,
    both_key,
    rows_of,
)

REG = Path(cache.__file__).resolve().parents[1] / "kernels" / "registry.csv"


def test_a_pair_and_an_atom_of_the_same_length_are_different_buckets():
    """The regression, stated directly."""
    pair = both_key(rows_of((1, 1024, 1024, 128)))
    atom = both_key(rows_of((1, 1024, 128)))
    assert pair != atom, "a 1,048,576-row launch and a 1,024-row launch share a bucket"
    assert atom == 1024
    assert pair == max(BOTH_ROWS)


@pytest.mark.parametrize("length", TOKEN_SHAPES)
def test_a_pair_launch_buckets_on_its_row_count(length):
    assert both_key(rows_of((1, length, length, 128))) == length * length


@pytest.mark.parametrize("length", ATOM_SHAPES)
def test_an_atom_launch_buckets_on_its_row_count(length):
    assert both_key(rows_of((1, length, 128))) == length


def test_rows_of_refuses_a_flattened_shape():
    with pytest.raises(ValueError, match="already flattened"):
        rows_of((262144, 128))


def test_the_bucket_set_is_exactly_what_the_work_list_drives():
    """Every BOTH_ROWS bucket has a unit, and every unit lands on one. Neither had been true.

    The old list drove one length axis and picked a side per length, so it built 8 of the 10
    buckets and two of those 8 were the other side's.
    """
    both_ops = {r["kernel"] for r in csv.DictReader(REG.open())
                if r["level"] == "both" and r["backend"] == "triton" and (r["driver"] or "").strip()}
    assert both_ops, "no level=both triton kernels in the registry"
    op = sorted(both_ops)[0]
    units = [u for u in op_units(only={op}) if u.dtype == "bfloat16"]
    got = set()
    for u in units:
        assert u.side in ("pair", "atom"), f"{u.stem}: a both-level unit must name its side"
        got.add(u.bucket)
    assert got == set(BOTH_ROWS), f"missing {sorted(set(BOTH_ROWS) - got)}, extra {sorted(got - set(BOTH_ROWS))}"


def test_a_units_side_reaches_the_driver():
    units = [u for u in op_units() if u.side]
    assert units
    assert units[0].env()["MINIWORLD_DRIVER_SIDE"] in ("pair", "atom")
    assert "--side" in units[0].cmd_args()


def test_the_two_sides_get_different_shards():
    """Same op, same length, both sides -- they must not overwrite one another."""
    units = {u.stem for u in op_units() if u.side and u.length == 256 and u.dtype == "bfloat16"}
    pair = {s for s in units if "-pair-" in s}
    atom = {s for s in units if "-atom-" in s}
    assert pair
    assert atom
    assert not (pair & atom)


def test_an_entry_from_the_old_scheme_is_not_served_to_a_both_level_kernel():
    assert cache._scheme_stale("transition_expand_swiglu_triton", None) is True
    assert cache._scheme_stale("transition_expand_swiglu_triton", cache.KEY_SCHEME) is False


def test_a_token_level_kernel_keeps_its_entries():
    """Scheme 2 re-based only both-level keys; invalidating the rest would discard a good build."""
    token = next(r["kernel"] for r in csv.DictReader(REG.open()) if r["level"] == "token")
    assert cache._scheme_stale(token, None) is False


def test_an_unknown_op_is_treated_as_stale():
    assert cache._scheme_stale("not_a_kernel_triton", None) is True


def test_every_write_stamps_the_key_scheme(tmp_path, monkeypatch):
    """A file has to say which scheme its keys are in, not leave it to be reconstructed."""
    import json

    import triton

    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path)
    cfg = triton.Config({"BLOCK_M1": 64}, num_warps=4, num_stages=2)
    fp = cache.store_ranked_configs("op_probe", "gpu0", "bfloat16", "shape_key=256",
                                    [(cfg, 1.0)], "hash0", op_id="op0", env_id="env0")
    assert json.loads(fp.read_text())["key_scheme"] == cache.KEY_SCHEME

    # ... and again on an APPEND to a file that is not being reset.
    fp = cache.store_ranked_configs("op_probe", "gpu0", "bfloat16", "shape_key=512",
                                    [(cfg, 1.0)], "hash0", op_id="op0", env_id="env0")
    data = json.loads(fp.read_text())
    assert data["key_scheme"] == cache.KEY_SCHEME
    assert len(data["entries"]) == 2, "the second write reset the file instead of appending"


def test_the_coverage_check_compares_buckets_not_lengths():
    """`want` used `u.length`; a both-level unit's key is its row count, so every one would miss."""
    units = [u for u in op_units() if u.side == "pair"]
    assert units
    u = units[0]
    assert u.bucket == both_key(u.length * u.length)
    assert u.bucket != u.length, "a pair unit's bucket must not be its length"


def test_a_token_or_atom_unit_buckets_on_its_length():
    units = [u for u in op_units() if not u.side]
    assert units
    assert all(u.bucket == u.length for u in units)


def test_a_shard_records_the_scheme_its_buckets_are_in(tmp_path, monkeypatch):
    """Without this a merge after a bump folds old and new keys into one file."""
    import json

    from miniworld_engine.autotune import capture

    monkeypatch.setattr(capture, "_CAPTURE", {})
    p = tmp_path / "unit.json"
    capture.dump_shard(str(p))
    assert json.loads(p.read_text())["_key_scheme"] == cache.KEY_SCHEME


def test_the_merge_skips_a_shard_from_an_older_scheme(tmp_path, monkeypatch):
    """An old pair measurement and a new atom one both write `shape_key=256`.

    Merging both means the file's 256 bucket is whichever shard the merge reached last -- a
    65,536-row measurement served to a 256-row launch, or the reverse, by file order.
    """
    import json

    from miniworld_engine.autotune import capture

    op = "transition_expand_swiglu_triton"                     # level=both
    entry = [{"kwargs": {"BLOCK_M1": 64}, "num_warps": 4, "num_stages": 2, "ms": 1.0}]
    old = tmp_path / "old.json"
    old.write_text(json.dumps({op: {"grid": entry, "entries": {"bfloat16|shape_key=256": entry},
                                    "op_id": "x"}}))            # no _key_scheme = scheme 1
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"_key_scheme": cache.KEY_SCHEME,
                               op: {"grid": entry, "entries": {"bfloat16|shape_key=256": entry},
                                    "op_id": "x"}}))

    written = []
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "data")
    monkeypatch.setattr(capture, "store_ranked_configs",
                        lambda *a, **k: written.append(a) or (tmp_path / "f.json"))
    capture.merge_shards([str(old), str(new)], gpu="g")
    assert any("old.json" in s for s, _ in capture._MERGE_SKIPPED), capture._MERGE_SKIPPED
    assert not any("new.json" in s for s, _ in capture._MERGE_SKIPPED)


def test_the_merge_keeps_an_old_shard_for_an_op_the_bump_did_not_touch():
    """Scheme 2 re-based both-level keys only; a token op's shard is still good."""
    from miniworld_engine.autotune import cache as c

    token = next(r["kernel"] for r in csv.DictReader(REG.open()) if r["level"] == "token")
    assert c._scheme_stale(token, None) is False
