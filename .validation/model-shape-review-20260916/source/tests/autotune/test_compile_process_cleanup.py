"""A compile deadline must stop its assembler descendants without killing other workers."""
from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import time

import pytest

from miniworld_engine.autotune import capture

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux fork/session cleanup")


@pytest.fixture
def descendants(tmp_path):
    # Adopt the fake assembler so this test can reap it rather than leaving a zombie to PID 1.
    libc = ctypes.CDLL(None, use_errno=True)
    old = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(old), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    record = tmp_path / "compiler-pids"
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield record, sibling
    finally:
        if record.exists():
            for pid in map(int, record.read_text().split()[:2]):
                # waitpid first: never signal a PID already reaped by the tested guard.
                try:
                    done, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    continue
                if not done:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
        sibling.kill()
        sibling.wait()
        assert libc.prctl(36, old.value, 0, 0, 0) == 0


def _stalled_compiler(record):
    assembler = os.fork()
    if assembler == 0:
        time.sleep(60)
        os._exit(0)
    record.write_text(f"{os.getpid()} {assembler} {os.getpgrp()}")
    time.sleep(60)


def _assert_assembler_killed(record, sibling):
    compiler, assembler, group = map(int, record.read_text().split())
    assert group == compiler
    assert group != os.getpgrp()
    deadline = time.monotonic() + 3
    while True:
        done, status = os.waitpid(assembler, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
            break
        assert time.monotonic() < deadline, "assembler survived compiler timeout"
        time.sleep(0.01)
    assert sibling.poll() is None, "cleanup killed an unrelated worker in the parent group"


@pytest.mark.parametrize("path", ["chunk", "serial"])
def test_timeout_kills_assembler_and_preserves_other_workers(path, monkeypatch, descendants):
    record, sibling = descendants
    monkeypatch.setattr(capture, "_COMPILE_BUDGET_S", 1)
    if path == "chunk":
        monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)

        def compile_payload(payload, pre=None):
            if payload[0] == "stall":
                _stalled_compiler(record)

        monkeypatch.setattr(capture, "_compile_payload", compile_payload)
        payloads = [(tag, "fn", {}, {}, {}, ("cuda", 86, 32), {})
                    for tag in ("ok", "stall", "retry")]
        assert [ok for ok, _ in capture._compile_chunk(payloads)] == [True, False, True]
    else:
        import triton.compiler.compiler as compiler

        assert capture._orig_bench is None
        monkeypatch.setattr(compiler, "compile", lambda *a, **kw: _stalled_compiler(record))
        monkeypatch.setattr(capture, "_install_launch_probes", lambda: None)
        monkeypatch.setattr(capture, "_CURRENT", {})
        capture.install()
        try:
            with pytest.raises(RuntimeError, match="compile exceeded"):
                compiler.compile(None)
        finally:
            capture.uninstall()
    _assert_assembler_killed(record, sibling)


def test_timeout_before_setsid_does_not_signal_parent_group(descendants):
    _, sibling = descendants
    child = os.fork()
    if child == 0:
        time.sleep(60)  # deadline wins before the child establishes its own session
        os._exit(0)
    assert os.getpgid(child) == os.getpgrp()
    capture._kill_compile_process_group(child)
    with pytest.raises(ChildProcessError):
        os.waitpid(child, os.WNOHANG)
    assert sibling.poll() is None


def test_setsid_racing_with_group_lookup_still_kills_assembler(monkeypatch, descendants):
    record, sibling = descendants
    rfd, wfd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(wfd)
        os.read(rfd, 1)
        os.close(rfd)
        os.setsid()
        _stalled_compiler(record)
        os._exit(0)
    os.close(rfd)
    real_killpg = os.killpg
    first = True

    def raced_lookup(pgid, sig):
        nonlocal first
        if first:
            first = False
            # The group does not exist at lookup, then appears before ESRCH is handled.
            assert pgid == child
            assert os.getpgid(child) == os.getpgrp()
            os.write(wfd, b"1")
            deadline = time.monotonic() + 3
            while not record.exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            raise ProcessLookupError()
        return real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", raced_lookup)
    try:
        capture._kill_compile_process_group(child)
    finally:
        os.close(wfd)
    _assert_assembler_killed(record, sibling)


@pytest.mark.parametrize("pid", [0, -1, os.getpid(), os.getpgrp()])
def test_cleanup_rejects_parent_and_broadcast_targets(pid):
    with pytest.raises(ValueError, match="isolated compile"):
        capture._kill_compile_process_group(pid)
