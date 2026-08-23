"""A shape this GPU cannot hold is a correct answer, not a bad unit.

`augmented_attention_bwd_split_triton[float32] L=4096` wants 153,600 bytes of shared memory and
an A6000 has 101,376. The child says so -- `[unit] SKIPPED-PERMANENT ... shape does not fit this
GPU` -- and returns rc=1 because the op it was told to drive never ran.

The parent read that line only `if not ops`. A unit is `--op <one kernel>`, but its driver fires
the neighbouring kernels too, so the shard held entries for three ops while the driven one was
skipped: `ops=3`, `skipped` never set, and the merge reported

    build produced 1 bad unit(s) of 168; their (op, bucket) entries will be MISSING from the cache

against a card that had answered correctly. `--strict` would have refused to merge the other 167.
"""
from __future__ import annotations

import json
from pathlib import Path

from miniworld_engine.autotune import builder
from miniworld_engine.cli import is_bad_unit

SKIP_LINE = "  [unit] SKIPPED-PERMANENT augmented_attention_bwd_split_triton: shape does not fit"


#: The real thing, not a stub: `stem`, `label`, `env()` and `cmd_args()` are what
#: `_run_unit_subprocess` reads off a unit, and a stub that drifts from them would keep passing
#: while the code it stands in for changed.
UNIT = builder.OpUnit(op="augmented_attention_bwd_split_triton", length=4096, dtype="float32")


def _run(tmp_path: Path, monkeypatch, *, rc: int, shard_ops: int, log_text: str) -> dict:
    """Drive `_run_unit_subprocess` with the child replaced by a canned outcome."""
    shard_dir = tmp_path / "shards"
    (shard_dir / "logs").mkdir(parents=True)
    unit = UNIT
    shard = shard_dir / f"{unit.stem}.json"
    log = shard_dir / "logs" / f"gpu0-{unit.stem}.log"

    class _Proc:
        returncode = rc

    def fake_run(cmd, **kw):
        if shard_ops:
            shard.write_text(json.dumps({f"op{i}": {"entries": {"bfloat16|k": []}}
                                         for i in range(shard_ops)}))
        log.write_text(log_text)
        return _Proc()

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    return builder._run_unit_subprocess(unit, 0, shard_dir, tmp_path, compile_jobs=1)


def test_a_permanent_skip_with_entries_on_disk_is_not_a_failure(tmp_path, monkeypatch):
    """The regression. rc=1 and ops=3, because the driver's neighbours did run."""
    r = _run(tmp_path, monkeypatch, rc=1, shard_ops=3, log_text=SKIP_LINE)
    assert r["skipped"] is True
    assert r["ops"] == 3
    assert not is_bad_unit(r)


def test_a_permanent_skip_with_nothing_on_disk_is_still_not_a_failure(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, rc=1, shard_ops=0, log_text=SKIP_LINE)
    assert r["skipped"] is True
    assert not is_bad_unit(r)


def test_a_real_failure_is_still_a_failure(tmp_path, monkeypatch):
    r = _run(tmp_path, monkeypatch, rc=1, shard_ops=0, log_text="Traceback: something broke")
    assert r["skipped"] is False
    assert is_bad_unit(r)


def test_a_permanent_skip_keeps_its_claim(tmp_path, monkeypatch):
    """Releasing it made every resumed job re-claim the same shapes and produce nothing."""
    r = _run(tmp_path, monkeypatch, rc=1, shard_ops=0, log_text=SKIP_LINE)
    assert r["skipped"]
    assert (tmp_path / "shards" / f"{UNIT.stem}.claim").exists()


def test_a_real_failure_releases_its_claim(tmp_path, monkeypatch):
    """Nothing produced and no permanent reason: a later run must be able to retry it."""
    _run(tmp_path, monkeypatch, rc=1, shard_ops=0, log_text="Traceback: something broke")
    assert not (tmp_path / "shards" / f"{UNIT.stem}.claim").exists()
