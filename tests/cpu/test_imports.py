"""Every source file parses as Python 3.10 (the oldest supported version) and every module of the package imports on a CPU-only node."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SRC / "miniworld_engine"
FILES = sorted(PACKAGE.rglob("*.py"))

#: Build-time scripts, never imported at run time (pyproject excludes them from the wheel).
NOT_MODULES = ("miniworld_engine.kernels.swa_dit.cuda.gen",)
#: Optional dependencies (pyproject extras ``flash`` / ``baselines``) a module may import at module level: an ImportError naming one of
#: these is a missing extra, not a defect.
OPTIONAL = (
    "flash_attn",
    "flash_attn_4",
    "cuequivariance",
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "transformers",
    "transformer_engine",
    "cutlass",
    "cuda.bindings",
    "nvidia_cutlass_dsl",
    "quack",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


MODULES = [m for m in (_module_name(p) for p in FILES) if not m.startswith(NOT_MODULES) and not m.endswith(".__main__")]


def test_found_the_package():
    assert len(FILES) > 100
    assert "miniworld_engine.modules" in MODULES


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(SRC)) for p in FILES])
def test_parses_as_python_3_10(path: Path):
    """``requires-python = ">=3.10"``: no 3.11+ syntax (``except*``, PEP 695 generics, ...) anywhere in the package."""
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))


@pytest.mark.parametrize("name", MODULES)
def test_imports(name: str):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing.startswith(OPTIONAL) and missing not in sys.modules:
            pytest.skip(f"optional dependency not installed: {missing}")
        raise
