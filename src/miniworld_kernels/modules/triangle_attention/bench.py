"""Compatibility wrapper for the triangle-attention benchmark suite."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(
    str(_ROOT / "benchmarks" / "suites" / "triangle_attention.py"),
    run_name="__main__",
)
