"""Every kernel family has the same shape on disk.

There are thirteen of them and they had drifted: one was not a package at all
(``bias_only_attention`` had no ``__init__.py`` and worked only by PEP 420 accident), two backend
directories were missing theirs, and ``interface.py`` existed for four families out of thirteen.
Nothing said what the shape was supposed to be, so each new kernel copied whichever neighbour its
author happened to open.

The shape, and what each part is for:

``reference.py``   the torch definition. The checkers in ``kernels/checks_*.py`` compare against
                   it, so it is what "correct" means for that family. Every family has one.
``interface.py``   the family's ONE public door: the names the rest of the repo may import, and
                   nothing about which backend serves them. ``kernels/__init__.py`` reaches only
                   here -- its docstring has always claimed it resolves names "without knowing the
                   per-op / per-backend folder layout", and for 13 of 16 entries it did exactly
                   that until this rule was enforced.
``dispatch.py``    a CHOICE among implementations, at whatever level the choice lives: per-GPU
                   calibration (``layernorm``, ``bias_only_attention``), a d-aware pick between
                   triton variants (``conditioned_transition/triton``), cuBLAS-vs-quack
                   (``trimul_inproj/cute``). Optional, and NOT the same thing as an interface --
                   the two were briefly the same filename, which is why this rule is written down.
``triton/`` ``cute/`` ``cuda/`` ``cutlass/``   backends. Optional, but each is a package.
``whole_op.py``    a whole model-layer op with weights as arguments (LN -> ... -> gate in one
                   call). Only some families expose one; it is a property of the layer, not of
                   the folder.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine" / "kernels"
BACKENDS = {"triton", "cute", "cuda", "cutlass"}


def _families() -> list[Path]:
    return sorted(d for d in KERNELS.iterdir()
                  if d.is_dir() and d.name != "__pycache__" and (d / "reference.py").exists())


def test_there_are_families_to_check() -> None:
    """Guard the guard: a discovery bug here would make every test below vacuously pass."""
    assert len(_families()) >= 12, f"only found {len(_families())} kernel families"


@pytest.mark.parametrize("part", ["__init__.py", "reference.py", "interface.py"])
def test_every_family_has(part: str) -> None:
    missing = [d.name for d in _families() if not (d / part).exists()]
    assert not missing, f"kernel families with no {part}: {', '.join(missing)}"


def test_every_backend_directory_is_a_package() -> None:
    """A directory that imports only by PEP 420 accident is a directory nobody meant to ship."""
    missing = [f"{d.name}/{b.name}" for d in _families() for b in sorted(d.iterdir())
               if b.is_dir() and b.name in BACKENDS and not (b / "__init__.py").exists()]
    assert not missing, f"backend dirs that are not packages: {', '.join(missing)}"


def test_no_unexpected_directory_under_a_family() -> None:
    """A new backend is a deliberate act; a stray directory is not."""
    stray = [f"{d.name}/{s.name}" for d in _families() for s in sorted(d.iterdir())
             if s.is_dir() and s.name not in BACKENDS and s.name != "__pycache__"]
    assert not stray, (f"unrecognised directories under a kernel family: {', '.join(stray)} "
                       f"(known backends: {', '.join(sorted(BACKENDS))})")


def test_interface_lives_only_at_family_level() -> None:
    """``interface.py`` means the family's public door. A choice among implementations is
    ``dispatch.py`` -- ``conditioned_transition/triton/`` used the first name for the second job."""
    deep = [str(p.relative_to(KERNELS)) for p in KERNELS.rglob("interface.py")
            if p.parent.parent != KERNELS.parent and p.parent not in _families()]
    assert not deep, ("interface.py below family level (rename to dispatch.py if it picks an "
                      f"implementation): {', '.join(deep)}")


def test_the_flat_surface_reaches_only_family_interfaces() -> None:
    """``kernels/__init__.py`` must not know which backend file holds a name.

    This is the rule its own docstring states. Before it was enforced, 13 of 16 lazy entries named
    a backend module directly, so moving a kernel between backends silently broke the public
    surface.
    """
    tree = ast.parse((KERNELS / "__init__.py").read_text())
    targets: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in node.targets)):
            continue
        assert isinstance(node.value, ast.Dict), "_LAZY_EXPORTS is no longer a dict literal"
        for value in node.value.values:
            assert isinstance(value, ast.Tuple), "_LAZY_EXPORTS values are (module, attr) tuples"
            module = value.elts[0]
            assert isinstance(module, ast.Constant)
            targets.append(str(module.value))
    assert targets, "_LAZY_EXPORTS not found -- this test is checking the wrong thing"
    bad = [t for t in targets if not t.endswith(".interface")]
    assert not bad, ("these lazy exports name a backend module instead of a family interface:\n  "
                     + "\n  ".join(bad))
