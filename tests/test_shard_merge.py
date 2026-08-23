"""``merge_shards`` must hash the WHOLE config space, not one shard's slice.

Shards that split by SHAPE all carry the same full grid, so the old code -- take the grid from
whichever shard was read first -- happened to be right and nothing caught it. Splitting the CONFIG
SET, which is the only way to divide a 200k-config grid across jobs, gives each shard a different
slice. Then the merged cache records a ``config_space_hash`` no full-grid run ever reproduces, and
``store_ranked_configs`` responds to the mismatch by RESETTING every entry: the next build silently
discards the whole tuning run. It was also order-dependent, so the recorded hash was not even
stable between two merges of the same shards.
"""
from __future__ import annotations

import json


from miniworld_engine.autotune import cache, capture


class _Cfg:
    """Minimal stand-in for ``triton.Config`` -- ``config_space_hash`` only reads these three."""

    def __init__(self, d):
        self.kwargs = d["kwargs"]
        self.num_warps = d["num_warps"]
        self.num_stages = d["num_stages"]
        self.maxnreg = None


def _cfg(bm, warps):
    return {"kwargs": {"BLOCK_M1": bm}, "num_warps": warps, "num_stages": 2}


FULL = [_cfg(m, w) for m in (32, 64, 128, 256) for w in (4, 8)]

#: A REAL registry op, not OP. The merge applies the cache reader's per-op staleness rule, so
#: a synthetic name has no `level` and is treated as stale -- correct for the reader, and it would
#: make these tests silently measure the skip path instead of the merge.
OP = "triangle_attention_fwd_triton"


def _write_shards(tmp_path, slices):
    paths = []
    for i, sl in enumerate(slices):
        p = tmp_path / f"shard{i}.json"
        p.write_text(json.dumps({"_key_scheme": cache.KEY_SCHEME, OP: {
            "grid": sl,
            "entries": {"bfloat16|N=128": [dict(c, ms=1.0 + i) for c in sl]},
        }}))
        paths.append(str(p))
    return paths


def test_merged_hash_covers_the_union_of_every_shard(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(capture, "store_ranked_configs",
                        lambda op, gk, d, b, ranked, csh, top_k=5, **kw: seen.update(csh=csh, n=len(ranked)))
    paths = _write_shards(tmp_path, [FULL[:4], FULL[4:]])

    capture.merge_shards(paths, gpu="TEST")

    assert seen["csh"] == cache.config_space_hash([_Cfg(c) for c in FULL]), (
        "merged cache must hash the full grid, else the next full-grid run resets every entry")
    assert seen["n"] == len(FULL), "every shard's measurements must survive the merge"


def test_merged_hash_does_not_depend_on_shard_order(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(capture, "store_ranked_configs",
                        lambda op, gk, d, b, ranked, csh, top_k=5, **kw: seen.append(csh))
    paths = _write_shards(tmp_path, [FULL[:4], FULL[4:]])

    capture.merge_shards(paths, gpu="TEST")
    capture.merge_shards(list(reversed(paths)), gpu="TEST")

    assert seen[0] == seen[1], f"hash depends on shard read order: {seen}"


def test_shape_sharding_is_unaffected(tmp_path, monkeypatch):
    """The pre-existing style -- every shard carries the identical full grid -- must be unchanged."""
    seen = {}
    monkeypatch.setattr(capture, "store_ranked_configs",
                        lambda op, gk, d, b, ranked, csh, top_k=5, **kw: seen.update(csh=csh))
    paths = _write_shards(tmp_path, [FULL, FULL])

    capture.merge_shards(paths, gpu="TEST")

    assert seen["csh"] == cache.config_space_hash([_Cfg(c) for c in FULL])


# --------------------------------------------------------------------------- #
# a shard that cannot be parsed must be LOUD
#
# `dump_shard` used a bare write_text: truncate, then write. Two processes on one shard interleave
# -- the shorter write lands inside the longer one and the first write's tail survives past its
# end. That is "Extra data: line 1 column 359431", and it happened for real on a --reclaim restart
# that handed a live unit's shard to a second worker. merge_shards then dropped the file silently,
# so a whole unit's measurements vanished from the cache and looked exactly like a unit never run.
# --------------------------------------------------------------------------- #

def test_dump_shard_is_atomic(tmp_path, monkeypatch):
    """A reader must never observe a half-written shard."""
    from miniworld_engine.autotune import capture
    monkeypatch.setattr(capture, "_CAPTURE", {OP: {"grid": [], "op_id": "", "entries": {}}})
    p = tmp_path / "u.json"
    capture.dump_shard(str(p))
    import json
    json.loads(p.read_text())                       # parses
    assert not list(tmp_path.glob("*.tmp")), "the temp file must be renamed away, not left behind"


def test_an_unparseable_shard_is_reported_not_swallowed(tmp_path, monkeypatch):
    from miniworld_engine.autotune import cache, capture
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "data")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"_key_scheme": cache.KEY_SCHEME,
                                OP: {"grid": [], "entries": {}, "op_id": ""}}))
    bad = tmp_path / "bad.json"
    bad.write_text('{"a": {}}{"a": {}}')          # two documents, as the real corruption was
    capture.merge_shards([str(good), str(bad)])
    assert len(capture._MERGE_SKIPPED) == 1
    assert "bad.json" in capture._MERGE_SKIPPED[0][0]
    assert "Extra data" in capture._MERGE_SKIPPED[0][1]


def test_merge_skipped_resets_between_runs(tmp_path, monkeypatch):
    """Stale entries would make a clean merge look broken."""
    from miniworld_engine.autotune import cache, capture
    monkeypatch.setattr(cache, "_CACHE_ROOT", tmp_path / "data")
    bad = tmp_path / "bad.json"; bad.write_text("{{{")
    capture.merge_shards([str(bad)])
    assert capture._MERGE_SKIPPED
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"_key_scheme": cache.KEY_SCHEME,
                                OP: {"grid": [], "entries": {}, "op_id": ""}}))
    capture.merge_shards([str(good)])
    assert not capture._MERGE_SKIPPED
