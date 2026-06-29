"""Compatibility wrapper for the benchmark report renderer."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(
    str(_ROOT / "benchmarks" / "runners" / "plot_bench.py"),
    run_name="__main__",
)
