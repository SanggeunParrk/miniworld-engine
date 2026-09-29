"""The triton cache is a build artifact, and one rebuild made 40 GB of it on a shared filesystem.

221,487 entries at 187 KB each. Sampled over 400 of them, 62% was intermediate IR -- ttir, ttgir,
llir, ptx -- which nothing launches a kernel from: a cache HIT reads the metadata json and the
cubin. `TRITON_STORE_BINARY_ONLY` writes only those, 71 KB an entry, 15 GB instead of 40.

It is safe to make the default because the knob is NOT one of triton's cache-invalidating
environment variables: verified on an A6000, the same config compiled to the same hash either way,
and the warm hit came back with its metadata and launcher intact.

What ships from a build is the JSON under `autotune/data/`, which names configs. Nothing reads the
triton cache afterwards, so once the shards are merged it can go. These tests are mostly about the
ways it must REFUSE to go: emptying the wrong directory is not recoverable.
"""
from __future__ import annotations

import json

import pytest

from miniworld_engine.autotune import triton_cache

DIGEST = "XU5DT2AO5BD5AEHEYGLPP5LRDFHHCUEJT4LGDVLB4STXUGVGHFPA"


def _entry(root, name=DIGEST, kernel="k"):
    d = root / name
    d.mkdir()
    (d / f"{kernel}.json").write_text(json.dumps({"shared": 1024}))
    (d / f"{kernel}.cubin").write_bytes(b"\x00" * 4096)
    return d


def test_the_default_writes_only_what_a_launch_needs():
    env = {}
    triton_cache.store_binary_only_env(env)
    assert env["TRITON_STORE_BINARY_ONLY"] == "1"


def test_keeping_the_ir_is_one_flag_away():
    """A build debugging a kernel wants the ttgir; it must not have to unset a global."""
    env = {"TRITON_STORE_BINARY_ONLY": "1"}
    triton_cache.store_binary_only_env(env, keep_ir=True)
    assert "TRITON_STORE_BINARY_ONLY" not in env


def test_an_environment_that_already_asked_for_something_is_left_alone():
    env = {"TRITON_STORE_BINARY_ONLY": "0"}
    triton_cache.store_binary_only_env(env)
    assert env["TRITON_STORE_BINARY_ONLY"] == "0"


def test_a_real_cache_is_recognised_even_with_one_entry(tmp_path):
    _entry(tmp_path)
    assert triton_cache.looks_like_a_triton_cache(tmp_path)


def test_the_loose_files_triton_leaves_beside_the_entries_are_fine(tmp_path):
    _entry(tmp_path)
    (tmp_path / "cuda_utils.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x00")
    (tmp_path / "some.lock").write_text("")
    assert triton_cache.looks_like_a_triton_cache(tmp_path)


def test_one_foreign_file_is_enough_to_refuse(tmp_path):
    """The failure mode this exists for: a path that is a triton cache AND something else."""
    _entry(tmp_path)
    (tmp_path / "results.csv").write_text("do not delete me")
    assert not triton_cache.looks_like_a_triton_cache(tmp_path)


def test_a_short_named_subdirectory_is_enough_to_refuse(tmp_path):
    _entry(tmp_path)
    (tmp_path / "shards").mkdir()
    assert not triton_cache.looks_like_a_triton_cache(tmp_path)


def test_long_named_directories_holding_no_metadata_are_not_a_cache(tmp_path):
    (tmp_path / DIGEST).mkdir()
    (tmp_path / DIGEST.replace("X", "Y", 1)).mkdir()
    assert not triton_cache.looks_like_a_triton_cache(tmp_path)


def test_an_empty_directory_is_not_a_cache(tmp_path):
    """Nothing to gain by emptying one, everything to lose by being wrong about which it is."""
    assert not triton_cache.looks_like_a_triton_cache(tmp_path)


def test_a_file_is_not_a_cache(tmp_path):
    f = tmp_path / "cache"
    f.write_text("")
    assert not triton_cache.looks_like_a_triton_cache(f)


def test_clearing_removes_the_entries_and_keeps_the_directory(tmp_path):
    """A build still pointed at the directory must keep working after it is emptied."""
    _entry(tmp_path)
    _entry(tmp_path, DIGEST.replace("X", "Y", 1))
    entries, total = triton_cache.clear(tmp_path)
    assert entries == 2
    assert total > 8000
    assert tmp_path.is_dir()
    assert not list(tmp_path.iterdir())


def test_a_dry_run_removes_nothing_and_still_reports_the_size(tmp_path):
    _entry(tmp_path)
    entries, total = triton_cache.clear(tmp_path, dry_run=True)
    assert entries == 1
    assert total > 4000
    assert (tmp_path / DIGEST).is_dir()


def test_clearing_refuses_a_directory_that_is_not_a_cache(tmp_path):
    (tmp_path / "notes.md").write_text("important")
    with pytest.raises(ValueError, match="does not look like a triton cache"):
        triton_cache.clear(tmp_path)
    assert (tmp_path / "notes.md").exists()
