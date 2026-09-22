"""Serialize and reuse tuning rounds shared by different module processes."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
from pathlib import Path

from miniworld_engine._atomic import write_json


def _timings_identity(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def reusable_timings(data: dict, outcomes: dict, *, predict: bool) -> tuple[dict, set[str]]:
    """Keep finite results; a verification pass retries unproven failures.

    Old round files mix actual failures and prediction-only skips as bare infinity. Keep that
    distinction unknown rather than inventing provenance. Prediction-enabled builds may reuse
    those exclusions, but they must not call them measured. A no-predict pass retries them even
    when an old committed entry incorrectly lists them as searched.
    """
    reused = {}
    retry = set()
    for sig, ms in data.items():
        if not isinstance(ms, (int, float)) or math.isnan(ms):
            continue
        if math.isfinite(ms) or outcomes.get(sig) == "observed_failure" or predict:
            reused[sig] = ms
        else:
            retry.add(sig)
    return reused, retry


@contextlib.contextmanager
def transaction(directory: str, identity: tuple, *, outcomes: dict | None = None):
    """A killed process releases its lock; only completed writes become visible."""
    if not directory:
        yield {}
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    path = root / f"{name}.json"
    provenance = root / f"{name}.provenance"
    with (root / f"{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = {}
            if outcomes is not None:
                outcomes.clear()
                try:
                    recorded = json.loads(provenance.read_text())
                    if (isinstance(recorded, dict)
                            and recorded.get("timings_identity") == _timings_identity(data)
                            and isinstance(recorded.get("outcomes"), dict)):
                        outcomes.update({sig: kind for sig, kind in recorded["outcomes"].items()
                                         if kind in ("observed_failure", "predicted_skip")})
                except (OSError, ValueError):
                    pass
            before = dict(data)
            before_outcomes = dict(outcomes) if outcomes is not None else None
            yield data
            if data != before:
                write_json(path, data)
            # Keep the numeric file readable by workers already running the previous version.
            # An old writer does not update the sidecar; its changed numeric hash makes the
            # sidecar untrusted next time. A crash between these writes has the same safe result.
            if outcomes is not None and (data != before or outcomes != before_outcomes):
                write_json(provenance, {"timings_identity": _timings_identity(data),
                                        "outcomes": outcomes})
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
