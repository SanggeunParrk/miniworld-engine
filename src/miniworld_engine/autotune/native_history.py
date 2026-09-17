"""Build-only native candidate journal; runtime winners remain in versioned data/."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
from pathlib import Path

from miniworld_engine._atomic import write_json


def reusable(record):
    if not isinstance(record, dict):
        return False
    if record.get("status") == "observed_failure":
        return True
    ms = record.get("ms")
    return (record.get("status") == "ok" and isinstance(ms, (int, float))
            and math.isfinite(ms) and ms > 0)


class Journal:
    def __init__(self, path=None):
        self.path = path
        try:
            data = json.loads(path.read_text()) if path else {}
        except (OSError, ValueError):
            data = {}
        self.records = data.get("records", {}) if isinstance(data, dict) else {}
        if not isinstance(self.records, dict):
            self.records = {}

    def record(self, signature, result):
        self.records[signature] = result
        # Per-candidate atomic checkpoint survives interruption later in the round.
        if self.path:
            write_json(self.path, {"schema": 1, "records": self.records})


@contextlib.contextmanager
def session(directory, identity):
    if not directory:
        yield Journal()
        return
    root = Path(directory) / "native"
    root.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    with (root / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield Journal(root / f"{name}.json")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def compile_failure_status(result):
    """Only deterministic compiler rejections exclude a candidate permanently.

    Timeout, killed compiler, filesystem errors and unknown failures are retryable.
    They never count as searched coverage in the published cache.
    """
    if result.get("status") == "timeout" or result.get("returncode", 1) < 0:
        return "retryable_failure"
    try:
        log = Path(result["log"]).read_text(errors="replace")[-16000:]
    except (OSError, KeyError):
        return "retryable_failure"
    deterministic = ("OutOfResources:", "ValueError:", "AssertionError:",
                     "error: static assertion failed", "exceeds shared memory limit")
    return ("observed_failure" if any(marker in log for marker in deterministic)
            else "retryable_failure")
