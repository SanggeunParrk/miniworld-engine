"""Compatibility wrapper for the bidirectional triangle-multiplication suite."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(
    str(_ROOT / "benchmarks" / "suites" / "triangle_multiplication_bidirectional.py"),
    run_name="__main__",
)
