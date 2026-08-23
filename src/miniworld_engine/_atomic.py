"""One atomic JSON writer, because four hand-rolled copies of it shared one bug.

Every cache this package writes -- the autotune config cache, the capture shards, and the two
backend-path calibrations -- is read by other processes while it is being written, and written by
several at once: eight GPU workers per node calibrate concurrently, and a cache build runs as
three Slurm jobs merging into the same ``data/`` tree.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def write_json(path: Path, obj: Any, **dumps: Any) -> None:
    """Write ``obj`` to ``path`` so a reader sees either the old file or the new one, never a mix.

    The temporary name carries the writing PROCESS and THREAD. A shared ``<name>.json.tmp`` --
    which is what all four call sites used -- does not remove the interleave, it moves it one file
    over: two writers truncate and fill the same temp path, the shorter write lands inside the
    longer one, and whichever mixture happens to be on disk at rename time becomes the file. The
    symptom is a cache that will not parse (``Extra data: line 1 column 359431``), and
    ``merge_shards`` drops a shard it cannot parse, so a whole unit's measurements disappear with
    nothing said.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident():x}.tmp")
    try:
        tmp.write_text(json.dumps(obj, **dumps))
        tmp.replace(path)          # atomic within a filesystem; tmp is gone afterwards
    finally:
        tmp.unlink(missing_ok=True)
