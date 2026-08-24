"""Every kernel family has the same shape on disk.

There are thirteen of them and they had drifted: one was not a package at all
(``bias_only_attention`` had no ``__init__.py`` and worked only by PEP 420 accident), two backend
directories were missing theirs, and ``interface.py`` existed for four families out of thirteen.
Nothing said what the shape was supposed to be, so each new kernel copied whichever neighbour its
author happened to open.

The shape, and what each part is for:

``reference.py``   the torch definition. The checkers in ``kernels/checks/<family>.py`` compare against
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

Two directories under ``kernels/`` are NOT families: ``drivers/`` and ``checks/``, the
autotune-capture harness. They hold one module per registry family -- ``drivers/<family>.py``
and ``checks/<family>.py``, named by registry.csv's ``family`` column, which is what
``test_registry_complete.py::test_the_harness_is_one_module_per_family`` enforces. They are
listed here so a third one cannot appear without a decision.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine" / "kernels"
BACKENDS = {"triton", "cute", "cuda", "cutlass"}

#: The only imports one ``drivers/<family>.py`` may take from another, and the shape group each
#: one shares. The harness is one module per family, but the SHAPES are per group -- the three
#: attention families drive the same [1, H, L, L, d] tensors, adaln and conditioned_transition the
#: same (M, D, DC) rows -- and a group's extents must be ONE constant or the families drift onto
#: different shapes and tune different buckets under one bench.
#:
#: They cannot all move to ``drivers/__init__.py``: four names (`_M`, `_D`, `L`, `D`) are defined
#: differently by two groups each (attention's `L = ragged(driver_length(128))` against trimul's
#: `ragged(driver_length(64))`), so one shared module cannot hold both under one name, and
#: renaming them would mean rewriting the driver bodies that read them. So each block stays whole
#: in the family that DEFINES the shape and the rest of its group imports it -- declared here, so
#: that a new one is a decision rather than a habit.
DRIVER_SHAPE_OWNERS = {
    "augmented_attention": "triangle_attention",
    "bias_only_attention": "triangle_attention",
    "adaln": "conditioned_transition",
    "fused_ln_mask": "layernorm_linear",
    "layernorm": "layernorm_linear",
    "gated_projection": "trimul_inproj",
    "tm1": "trimul_inproj",
    "tm2": "trimul_inproj",
}


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


HARNESS_DIRS = {"drivers", "checks"}


def test_the_only_non_family_directories_are_the_harness() -> None:
    """``drivers/`` and ``checks/`` are the autotune-capture harness, one module per registry
    family. Nothing else may sit beside the families: a directory with no ``reference.py`` is
    either a family someone forgot to finish or a place kernels will quietly accumulate."""
    stray = sorted(d.name for d in KERNELS.iterdir()
                   if d.is_dir() and d.name != "__pycache__"
                   and d not in _families() and d.name not in HARNESS_DIRS)
    assert not stray, (f"directories under kernels/ that are neither a family nor the harness "
                       f"({', '.join(sorted(HARNESS_DIRS))}): {', '.join(stray)}")


def test_the_harness_directories_exist() -> None:
    missing = [d for d in sorted(HARNESS_DIRS) if not (KERNELS / d / "__init__.py").is_file()]
    assert not missing, f"the autotune-capture harness is missing: {missing}"


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


def test_a_driver_module_imports_only_its_declared_shape_owner() -> None:
    """One module per family is only worth having if the modules stay peers.

    A cross-family import that is not in :data:`DRIVER_SHAPE_OWNERS` means either a shape group
    grew a second host -- two constants for one group, which is the drift this layout exists to
    stop -- or a helper that belongs in ``drivers/__init__.py`` was reached for sideways instead.
    """
    drivers = KERNELS / "drivers"
    prefix = "miniworld_engine.kernels.drivers."
    unexpected = []
    for path in sorted(drivers.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.startswith(prefix)):
                continue
            other = node.module[len(prefix):]
            if DRIVER_SHAPE_OWNERS.get(path.stem) != other:
                unexpected.append(f"{path.stem} imports from {other} (line {node.lineno})")
    assert not unexpected, unexpected
    stale = sorted(f for f in DRIVER_SHAPE_OWNERS if not (drivers / f"{f}.py").is_file())
    assert not stale, f"DRIVER_SHAPE_OWNERS names families with no driver module: {stale}"


def test_no_check_module_reaches_sideways() -> None:
    """``checks/`` has no shape blocks to share -- its helpers all fit in ``checks/__init__.py``,
    and they are all there. Nothing may quietly start importing across families here."""
    prefix = "miniworld_engine.kernels.checks."
    offenders = []
    for path in sorted((KERNELS / "checks").glob("*.py")):
        if path.name == "__init__.py":
            continue
        offenders.extend(
            f"{path.stem} imports from {node.module} (line {node.lineno})"
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module
            and node.module.startswith(prefix))
    assert not offenders, offenders
