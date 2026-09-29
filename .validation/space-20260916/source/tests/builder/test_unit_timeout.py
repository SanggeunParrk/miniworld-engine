"""A wedged compile cannot monopolize a build slot or orphan its subprocesses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from miniworld_engine.autotune import builder


@pytest.mark.parametrize("detached", [False, True])
def test_timeout_kills_thread_spawned_descendant_even_in_private_session(tmp_path, detached):
    pid_file = tmp_path / "grandchild.pid"
    # The child ignores SIGTERM; the leader takes the default exit on SIGTERM.
    child = ("import os,signal,time; from pathlib import Path; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)")
    leader = ("import subprocess,sys,time,threading; "
              f"threading.Thread(target=lambda: (subprocess.Popen([sys.executable, '-c', {child!r}], "
              f"start_new_session={detached!r}), time.sleep(60))).start(); time.sleep(60)")
    with (tmp_path / "log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        builder._run_unit_process([sys.executable, "-c", leader], cwd=tmp_path,
                                  stdout=log, stderr=subprocess.STDOUT, check=False,
                                  env=dict(os.environ), timeout=2)
    assert pid_file.exists(), "descendant must have started before timeout"
    pid = int(pid_file.read_text())
    for _ in range(100):
        stat = Path(f"/proc/{pid}/stat")
        try:
            if stat.read_text().split(") ", 1)[1].startswith("Z"):
                break
        except (FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.01)
    else:
        pytest.fail("compiler descendant survived unit timeout")


def test_timeout_preserves_shards_and_other_claims_and_releases_own(tmp_path, monkeypatch):
    unit = builder.OpUnit("example_triton", 128)
    shard = tmp_path / f"{unit.stem}.json"
    payload = json.dumps({"_unit_complete": False, "op": {"entries": {"bf16|128": [1]}}})
    shard.write_text(payload)
    other = tmp_path / "healthy.claim"
    other.write_text("live worker")
    log = tmp_path / "logs" / f"gpu0-{unit.stem}.log"
    log.parent.mkdir()
    log.write_text("prior attempt\n[unit] SKIPPED-PERMANENT\n")
    def timeout(cmd, **kw):
        assert kw["timeout"] == 3
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(builder, "_run_unit_process", timeout)
    monkeypatch.setattr(builder, "visible_device", lambda _: "GPU-allocated")
    result = builder._run_unit_subprocess(unit, 0, tmp_path, tmp_path, 1, unit_timeout_seconds=3)
    assert result["rc"] == 124
    assert result["timed_out"]
    assert not result["skipped"]
    assert result["ops"] == 1
    assert not shard.with_suffix(".claim").exists()
    assert shard.read_text() == payload
    assert other.read_text() == "live worker"
    assert log.read_text().startswith("prior attempt\n")
    assert "TIMEOUT" in log.read_text()
    # A later unit on the same slot is still runnable.
    monkeypatch.setattr(builder, "_run_unit_process",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    next_unit = builder.OpUnit("next_triton", 128)
    following = builder._run_unit_subprocess(next_unit, 0, tmp_path, tmp_path, 1)
    assert following["rc"] == 0
    assert not following["timed_out"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_does_not_claim_work(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        builder._run_unit_subprocess(builder.OpUnit("example_triton", 128),
                                     0, tmp_path, tmp_path, 1, unit_timeout_seconds=timeout)
    assert not list(tmp_path.iterdir())


def test_launch_failure_releases_claim(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot launch")
    monkeypatch.setattr(builder, "_run_unit_process", fail)
    monkeypatch.setattr(builder, "visible_device", lambda _: "0")
    unit = builder.OpUnit("example_triton", 128)
    result = builder._run_unit_subprocess(unit, 0, tmp_path, tmp_path, 1)
    assert result["rc"] == 127
    assert not (tmp_path / f"{unit.stem}.claim").exists()


def test_failed_attempt_cannot_reuse_old_complete_shard(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"_unit_complete": True, "op": {"entries": {"key": [1]}}}))
    path.with_suffix(".failed").write_text("rc=124")
    assert not builder._shard_reusable(path)


@pytest.mark.parametrize("argv", [["build", "all"], ["bench_module", "triangle_multiplication"]])
def test_cli_exposes_unit_deadline(argv):
    from miniworld_engine import cli
    args = cli.build_parser().parse_args([*argv, "--unit-timeout-seconds", "45"])
    assert args.unit_timeout_seconds == 45


def test_worker_moves_to_next_unit_after_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "validate_build_gpus", lambda _: None)
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_80")
    monkeypatch.setattr(builder, "visible_device", lambda _: "0")
    monkeypatch.setattr(builder, "_generation_for_work", lambda _: "test")
    seen = []
    def run(cmd, **kwargs):
        seen.append(cmd)
        if len(seen) == 1:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(builder, "_run_unit_process", run)
    results = builder.build_all([builder.OpUnit("stuck", 128), builder.OpUnit("next", 128)],
                                tmp_path, [0], 1, skip_cached=False, unit_timeout_seconds=3)
    assert [result["rc"] for result in results] == [124, 0]
    assert [result["gpu"] for result in results] == [0, 0]


def test_partial_merge_does_not_hide_prebench_timeout(tmp_path, monkeypatch):
    import argparse

    from miniworld_engine import cli

    received = []
    def build(*args, **kwargs):
        received.append(kwargs["unit_timeout_seconds"])
        return [{"rc": 124, "ops": 1, "timed_out": True}]
    monkeypatch.setattr(builder, "build_all", build)
    monkeypatch.setattr(builder, "cases", list)
    monkeypatch.setattr(cli, "apply_config_dir", lambda _: 0)
    monkeypatch.setattr(cli, "resolve_config_dir", lambda *args: tmp_path)
    monkeypatch.setattr(cli, "_resolve_gpus", lambda _: [0])
    merged = []
    monkeypatch.setattr(cli, "_merge_built_shards", lambda *args: merged.append(True) or 0)
    args = argparse.Namespace(no_build=False, shards=str(tmp_path), gpus="1",
                              compile_jobs=1, resume=False, unit_timeout_seconds=45)
    rc = cli._bench_build_first(args, ("transition",), tmp_path,
                                {"transition": ("transition",)}, "MODULE_TARGETS")
    assert rc == 1
    assert received == [45]
    assert merged == [True]
