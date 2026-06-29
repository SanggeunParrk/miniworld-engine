"""Compatibility wrapper for the benchmark runner.

The active implementation lives in ``benchmarks/runners/bench.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(_ROOT / "benchmarks" / "runners" / "bench.py"), run_name="__main__")
