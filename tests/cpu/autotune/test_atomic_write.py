"""Concurrent writers must not be able to produce a file that will not parse.

`store_ranked_configs`, `dump_shard` and the two backend-path calibrations each had the same
three lines: write `<name>.json.tmp`, then rename it over the target. The rename is atomic; the
SHARED temp name is not. Two writers truncate and fill the same temp path, the shorter write
lands inside the longer one, and whichever mixture is on disk at rename time becomes the file --
the same `Extra data: line 1 column 359431` the `.tmp` was introduced to prevent, moved one file
over. Concurrency here is the normal case: eight GPU workers per node calibrate at once, and a
cache build runs as three Slurm jobs merging into the same `data/` tree.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

from miniworld_engine._atomic import write_json


def _writer(args) -> None:
    path, payload = args
    for _ in range(40):
        write_json(Path(path), payload)


def test_a_lone_write_lands(tmp_path):
    p = tmp_path / "cache.json"
    write_json(p, {"a": 1}, indent=2, sort_keys=True)
    assert json.loads(p.read_text()) == {"a": 1}


def test_the_temp_file_does_not_survive(tmp_path):
    p = tmp_path / "cache.json"
    write_json(p, {"a": 1})
    assert list(tmp_path.iterdir()) == [p], "a .tmp was left behind"


def test_the_temp_name_is_unique_per_writer(tmp_path, monkeypatch):
    """The whole fix. Two writers must not choose the same temporary path."""
    seen = []
    real = Path.write_text

    def spy(self, *a, **k):
        seen.append(self.name)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", spy)
    write_json(tmp_path / "cache.json", {"a": 1})
    monkeypatch.setattr("os.getpid", lambda: 999999)
    write_json(tmp_path / "cache.json", {"a": 2})
    assert len(seen) == 2
    assert seen[0] != seen[1], f"both writers used {seen[0]!r}"
    assert all(n.endswith(".tmp") and n.startswith("cache.json.") for n in seen)


def test_concurrent_writers_never_leave_it_unparseable(tmp_path):
    """Four processes, two payload sizes, 40 writes each. Every read must parse."""
    p = tmp_path / "cache.json"
    small = {"k": "x"}
    large = {"k": "y" * 200_000}
    write_json(p, small)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=((str(p), payload),))
             for payload in (small, large, small, large)]
    for proc in procs:
        proc.start()
    reads, bad = 0, []
    while any(proc.is_alive() for proc in procs):
        try:
            json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            bad.append(str(exc))
        except FileNotFoundError:
            bad.append("target vanished between writes")
        reads += 1
    for proc in procs:
        proc.join()
    assert reads > 0, "the writers finished before a single read -- test proves nothing"
    assert not bad, f"{len(bad)} of {reads} reads did not parse: {bad[:3]}"
    # With a SHARED temp name this is the loud failure mode: the second writer renames the name
    # away while the first still holds it, and the first's `replace` raises FileNotFoundError.
    assert [proc.exitcode for proc in procs] == [0, 0, 0, 0], "a writer crashed"
    assert json.loads(p.read_text()) in (small, large)
