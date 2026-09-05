"""The cache fingerprint scanner must actually FLAG each kind of change.

``test_no_stale_caches`` asserts the shipped caches are fresh; it says nothing about whether the
scanner would notice if they were not. These tests drive the scanner against a synthetic cache
directory whose fingerprints we control, so each detector is exercised on purpose:

Which changes are STALE is a POLICY, and the policy is deliberate (see ``cache.build_rev``):

* fingerprints match                                 -> ``OK``
* ``build_rev`` bumped in registry.csv               -> ``STALE``  (a person declared the method changed)
* ``key_scheme`` bumped                              -> ``STALE``  (the bucket string means something else)
* ``env_identity`` differs                           -> reported, never a CI failure
* config grid changed                                -> ``OK``, reported: served incrementally
* ``op_identity`` changed                            -> ``STALE`` (a different kernel body)
* ``driver_identity`` changed                        -> ``OK``, reported: coverage moved, numbers did not

``driver_identity`` used to be STALE and it cost real measurements: correcting which SHAPES a
driver builds deleted 32 of 38 tuned buckets from ``cond_transition_expand_swiglu``, an edit that
said nothing against the numbers. ``op_identity`` stays STALE -- a different kernel body really
does void a recorded time, and demoting it has no safe form: without a reset the reader either
refuses the rebuild's own fresh entries (if the stamp is kept) or serves the pre-edit winner as
current (if it is not).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniworld_engine.autotune import cache_status
from miniworld_engine.autotune.cache import (
    DRIVER_ID_SCHEME,
    config_space_hash,
    driver_identity,
    env_identity,
)
from miniworld_engine.autotune.configs import configs_for

#: A real op: the scanner recomputes fingerprints from the live code, so a made-up name would have
#: no grid and no driver and every check would be skipped as UNKNOWN.
OP = "adaln_gemm_gate_triton"
GPU = "NVIDIA A100 80GB PCIe (sm80)"


def _write_cache(root, **overrides):
    """A synthetic cache file whose fingerprints match the CURRENT code unless overridden."""
    data = {
        "schema": 1,
        "key_scheme": 3,
        "gpu": GPU,
        "op": OP,
        "config_space_hash": config_space_hash(configs_for(OP)),
        "op_identity": cache_status._current_op_identity(OP),
        "driver_identity": driver_identity(OP),
        "driver_id_scheme": DRIVER_ID_SCHEME,
        "env_identity": env_identity(),
        "build_rev": 1,
        "entries": {},
    }
    data.update(overrides)
    d = root / OP
    d.mkdir(parents=True)
    (d / f"{GPU}.json").write_text(json.dumps(data))
    return data


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_status, "_DATA", tmp_path)
    return tmp_path


def _verdict(rows):
    assert len(rows) == 1, rows
    return rows[0]


def test_matching_fingerprints_are_ok(fake_data):
    _write_cache(fake_data)
    row = _verdict(cache_status.scan())
    assert row.verdict == "OK", row.reason


def test_changed_config_grid_is_reported_but_served_incrementally(fake_data):
    """A grid edit is a top-up, not a reset: `configs_to_bench` benches what the grid ADDED and
    `store_ranked_configs` drops what it removed. Failing CI on it is what made narrowing a ladder
    -- which the ladder tests themselves demand -- cost a full rebuild."""
    _write_cache(fake_data, config_space_hash="deadbeef0000")
    row = _verdict(cache_status.scan())
    assert row.verdict == "OK", row.reason
    assert "grid" in row.reason and "incremental" in row.reason


def test_bumped_build_rev_is_stale(fake_data):
    """The one invalidator a person writes. Everything else here is automatic."""
    _write_cache(fake_data, build_rev=0)          # registry declares 1; 0 is "built under another"
    row = _verdict(cache_status.scan())
    assert row.verdict == "STALE"
    assert "build_rev" in row.reason


def test_changed_build_driver_is_reported_not_stale(fake_data):
    """The driver decides which buckets get BUILT, never whether a measured winner is right."""
    if driver_identity(OP) is None:
        pytest.skip(f"{OP} has no resolvable registry driver on this machine")
    _write_cache(fake_data, driver_identity="deadbeef0000")
    row = _verdict(cache_status.scan())
    assert row.verdict == "OK", row.reason
    assert "driver" in row.reason


def test_changed_kernel_source_is_stale(fake_data):
    """A different kernel body voids the measurement -- there is no safe way to keep the entries."""
    if cache_status._current_op_identity(OP) is None:
        pytest.skip(f"{OP} autotuner is not importable on this machine")
    _write_cache(fake_data, op_identity="deadbeef0000")
    row = _verdict(cache_status.scan())
    assert row.verdict == "STALE"
    assert "kernel source" in row.reason


def test_env_mismatch_is_reported_but_not_a_stale_verdict(fake_data):
    """A cache built under another triton/cuda is legitimately "not this machine's" -- surfaced,
    but never a CI failure, or every checkout on a different toolchain would fail."""
    _write_cache(fake_data, env_identity="deadbeef0000")
    row = _verdict(cache_status.scan())
    assert row.verdict == "OK"
    assert row.env_matches is False


def test_driver_stamp_from_another_scheme_is_skipped_not_failed(fake_data):
    """A stamp computed by a different version of the hashing scope is not comparable. Treating the
    disagreement as drift would be a false STALE that no rebuild-free action can clear -- which is
    exactly what happened when the scope was narrowed mid-development."""
    _write_cache(fake_data, driver_identity="deadbeef0000",
                 driver_id_scheme=DRIVER_ID_SCHEME + 1)
    row = _verdict(cache_status.scan())
    assert row.verdict == "OK", row.reason


def test_driver_identity_follows_cross_module_imports():
    """The blindness that let a cache-destroying driver edit ship while the guard said OK.

    ``drivers/adaln.py`` takes its shapes from ``drivers/conditioned_transition.py``:

        from .conditioned_transition import _D, _DC, _M, _SHAPE_KEY

    ``_M`` is the row count every adaln bucket is built at, so editing it rewrites what adaln's
    cache covers. Under ``DRIVER_ID_SCHEME`` 1 the hash saw only the driver module's own text --
    and the IMPORT STATEMENT is byte-identical across the edit, so adaln's stamp did not move and
    ``cache-status`` called the now-junk cache fresh. It took a full rebuild and an entry-level
    diff against HEAD to notice.

    This drives the real edit through the real hash: the sibling changes, adaln's fingerprint must
    move. Asserting on the sibling too would pass under either scheme -- it is the IMPORTER that
    was blind.
    """
    import sys

    src = (Path(cache_status.__file__).resolve().parents[1]
           / "kernels" / "drivers" / "conditioned_transition.py")
    if not src.is_file():
        pytest.skip("driver package not laid out as expected")
    importer = "adaln_gemm_gate_triton"
    if driver_identity(importer) is None:
        pytest.skip(f"{importer} has no resolvable registry driver on this machine")

    def _flush():
        for name in [m for m in sys.modules if "kernels.drivers" in m]:
            del sys.modules[name]

    original = src.read_text()
    edited = original.replace("_M = ragged(driver_length(512))",
                              "_M = ragged(48 * driver_length(512))")
    assert edited != original, "the `_M` definition this test edits has moved"
    before = driver_identity(importer)
    try:
        src.write_text(edited)
        _flush()
        after = driver_identity(importer)
    finally:
        src.write_text(original)
        _flush()
    assert before != after, (
        "adaln's driver_identity did not move when the sibling module's `_M` changed -- the "
        "cross-module scope is not being hashed")
