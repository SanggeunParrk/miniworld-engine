"""Compatibility wrapper for the sweep report renderer."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(
    str(_ROOT / "benchmarks" / "runners" / "plot_sweep.py"),
    run_name="__main__",
)
