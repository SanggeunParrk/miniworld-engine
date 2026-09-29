"""Every committed cache file has to be readable, attributed, and ranked.

`data/**/*.json` is the shipped product: it is what a consumer gets on `pip install`, and the
reader consults it on the first launch of every autotune kernel. Nothing checked what was in it.

Two of this repo's bugs put bad content there and nothing noticed. A non-atomic `write_text` let
two workers interleave on one shard and produce unparseable JSON, which the merge then dropped in
silence. And `env_identity` recorded `ptxas=?` for months because its probe imported a function
triton had moved -- a staleness key that cannot go stale is the same as no key.

These are cheap structural checks over the whole tree. They do not judge whether a config is fast;
they judge whether the file says what it claims to.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache

DATA = Path(cache.__file__).parent / "data"
#: `data/` holds two kinds of cache under one roof. These two directories are the per-shape
#: BACKEND-PATH calibrations (`{"<d>|<M>": {"ms": {...}, "path": "atomic"}}`), written by
#: `kernels/layernorm/dispatch.py` and its bias-only counterpart, not the autotune config cache.
#: Different schema, different writer; they get their own check below.
DISPATCH_DIRS = {"ln_bwd_dispatch", "bias_only_dispatch"}
FILES = sorted(p for p in DATA.glob("*/*.json") if p.parent.name not in DISPATCH_DIRS)
DISPATCH_FILES = sorted(p for p in DATA.glob("*/*.json") if p.parent.name in DISPATCH_DIRS)
REQUIRED = {"config_space_hash", "entries", "env_identity", "gpu", "op", "op_identity",
            "provenance", "schema"}


def _id(p: Path) -> str:
    return f"{p.parent.name}/{p.stem}"


def test_there_is_a_shipped_cache():
    """Otherwise every parametrized test below is vacuous."""
    assert len(FILES) > 50, f"only {len(FILES)} cache files under {DATA}"


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_a_cache_file_is_wellformed(path: Path):
    data = json.loads(path.read_text())          # unparseable JSON fails here, loudly
    missing = REQUIRED - set(data)
    assert not missing, f"missing key(s): {sorted(missing)}"
    assert data["schema"] == 1
    assert data["op"] == path.parent.name, "op does not match the directory it lives in"
    assert data["gpu"] == path.stem, "gpu does not match the filename"
    assert data["entries"], "no entries: an empty cache file is a miss with extra steps"


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_the_identities_are_real(path: Path):
    """A staleness key that is empty, or a constant like `?`, cannot detect staleness."""
    data = json.loads(path.read_text())
    for key in ("config_space_hash", "env_identity", "op_identity"):
        value = data[key]
        assert isinstance(value, str), f"{key} is {type(value).__name__}, not a string"
        assert value, f"{key} is empty"
        assert "?" not in value, f"{key}={value!r} -- a probe failed and was swallowed"
    prov = data["provenance"]
    assert prov.get("torch")
    assert prov.get("triton")
    assert prov.get("built_utc")


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_entries_are_ranked_fastest_first(path: Path):
    data = json.loads(path.read_text())
    for bucket, ranked in data["entries"].items():
        assert ranked, f"{bucket}: empty ranking"
        times = [c["ms"] for c in ranked]
        assert all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times), \
            f"{bucket}: non-finite or non-positive ms in {times}"
        assert times == sorted(times), f"{bucket}: not fastest-first: {times}"
        for c in ranked:
            assert isinstance(c.get("kwargs"), dict), f"{bucket}: entry has no kwargs"
            assert isinstance(c.get("num_warps"), int)
            assert c["num_warps"] > 0
            assert isinstance(c.get("num_stages"), int)
            assert c["num_stages"] > 0


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_a_bucket_key_names_a_dtype(path: Path):
    """`"<dtype>|<bucket>"` -- the reader splits on the first `|`, so both halves must be there."""
    data = json.loads(path.read_text())
    for key in data["entries"]:
        assert "|" in key, f"{key!r} has no dtype prefix"
        dtype, bucket = key.split("|", 1)
        assert dtype
        assert bucket
        for one in dtype.split("+"):
            assert one in {"bfloat16", "float32", "float16", "int32", "int64", "int8",
                           "uint8", "float64", "bool"}, f"{key!r}: unknown dtype {one!r}"


# --- the backend-path calibration caches ------------------------------------------------------ #


def test_there_are_dispatch_caches():
    assert DISPATCH_FILES, f"no calibration caches under {sorted(DISPATCH_DIRS)}"


@pytest.mark.parametrize("path", DISPATCH_FILES, ids=_id)
def test_a_dispatch_cache_is_wellformed(path: Path):
    """`{"<key>": {"ms": {path: ms, ...}, "path": <the winner>}}`.

    The key is `|`-joined and ends in the shape it was calibrated at -- `"128|1048576"` for
    layernorm (d, M-bucket), `"gate|1024|512|1048576"` for bias-only, which also names the
    epilogue. What matters here is that the recorded winner IS the fastest of the times recorded
    next to it. A file naming a slower path is not a wrong answer -- path choice is
    performance-only -- but it means the calibration and what it wrote down disagree, and nothing
    else in the repo would say so.
    """
    data = json.loads(path.read_text())
    assert data, "empty calibration cache"
    for key, entry in data.items():
        fields = key.split("|")
        assert len(fields) >= 2, f"{key!r}: expected '|'-joined fields, got {fields}"
        assert all(fields), f"{key!r}: an empty field in {fields}"
        assert fields[-1].isdigit(), f"{key!r}: the last field must be the shape it was tuned at"
        times = entry["ms"]
        # The two writers spell the winner differently -- layernorm calls it `path`, bias-only
        # calls it `choice`. Their readers each know their own name, so this accepts both rather
        # than renaming a field in a committed cache.
        won = entry.get("path", entry.get("choice"))
        assert won is not None, f"{key}: entry names no winner ({sorted(entry)})"
        if not times:
            # A documented marker, not a hole: when calibration raises (OOM at the top shape on a
            # 24 GB card, say) the writer caches the STATIC choice with no timings so the failing
            # do_bench is not retried on every forward. It has to still name a choice, which the
            # assert above covers -- an empty `ms` is the file saying "not measured", and that is
            # the honest thing for it to say.
            continue
        assert all(math.isfinite(v) and v > 0 for v in times.values()), f"{key}: {times}"
        assert won in times, f"{key}: winner {won!r} is not among {list(times)}"
        best = min(times, key=lambda k: times[k])
        assert won == best, f"{key}: recorded {won!r} ({times[won]}ms) but {best!r} is faster"
