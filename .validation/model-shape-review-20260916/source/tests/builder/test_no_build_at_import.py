"""Importing a kernel package must not compile anything.

`docs/library-standards.md` A2: an import must not compile a kernel, touch a GPU, read a cache, or
spend seconds. The package as a whole satisfied that only because nothing imported the two CUDA
subpackages eagerly -- each of them called `torch.utils.cpp_extension.load` at module scope, and
`transition/cuda` built THREE extensions that way, one of them compiled for `sm_90a`.

So importing `kernels.transition.cuda` on an sm_86 card raised "Error building extension" outright,
and anything that walks the package tree hit it. `dev audit`'s import check did exactly that and
reported `import: 0 OK, 2 not OK` on every A6000 run -- the audit command exiting 1 for a reason
having nothing to do with what it audits.

Both are PEP 562 lazy now. This is what keeps a third from appearing: the check is on the source, so
it runs on any machine, with or without CUDA, and does not depend on catching a build in the act.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

KERNELS = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "miniworld_engine" / "kernels"

#: The call that compiles. `load_extension` is this repo's wrapper (kernels/_nvcc.py); torch's
#: `load` is the thing it wraps. Either at module scope means a build at import.
BUILDERS = {"load_extension", "load"}


def _module_level_build_calls(tree: ast.Module) -> list[int]:
    """Lines where a build is called at module scope -- not inside a def, not inside a class."""
    lines = []
    for node in tree.body:                      # top level ONLY; a call inside a def is fine
        for inner in ast.walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                break                           # everything under a def is deferred by definition
            if isinstance(inner, ast.Call):
                fn = inner.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in BUILDERS:
                    lines.append(inner.lineno)
    return lines


def _cuda_packages() -> list[Path]:
    return sorted(KERNELS.glob("*/cuda/__init__.py"))


def test_there_are_cuda_packages_to_check() -> None:
    """Guard the guard: if the glob stops matching, everything below passes by doing nothing."""
    found = _cuda_packages()
    assert len(found) >= 2, f"expected the layernorm and transition CUDA packages, found {found}"


@pytest.mark.parametrize("init", _cuda_packages(), ids=lambda p: p.parent.parent.name)
def test_no_extension_is_built_at_module_scope(init: Path) -> None:
    lines = _module_level_build_calls(ast.parse(init.read_text()))
    assert not lines, (
        f"{init.relative_to(KERNELS.parent)} compiles at import (line(s) {lines}). Move the call "
        f"into a function and expose the name through a module-level `__getattr__`, as the other "
        f"CUDA packages do -- importing must not run nvcc.")


@pytest.mark.parametrize("init", _cuda_packages(), ids=lambda p: p.parent.parent.name)
def test_the_lazy_names_are_reachable(init: Path) -> None:
    """A `__getattr__` that raises for its own exports would trade a build for a broken import."""
    import importlib

    module = importlib.import_module(
        f"miniworld_engine.kernels.{init.parent.parent.name}.cuda")
    assert hasattr(module, "__getattr__"), f"{module.__name__} has no lazy accessor"
    with pytest.raises(AttributeError):
        module.definitely_not_an_export        # noqa: B018 -- the raise IS the assertion


def test_importing_the_cuda_packages_runs_no_compiler() -> None:
    """The property itself, not a proxy for it. No CUDA needed: a build would raise or take
    seconds, and on a machine without nvcc it would raise."""
    import importlib
    import time

    for init in _cuda_packages():
        name = f"miniworld_engine.kernels.{init.parent.parent.name}.cuda"
        started = time.monotonic()
        importlib.import_module(name)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"importing {name} took {elapsed:.1f}s -- something is building"


# --- scripts that live inside the importable package ---------------------------------------- #

def _setup_scripts() -> list[Path]:
    return sorted(KERNELS.rglob("setup.py"))


def test_there_are_setup_scripts_to_check() -> None:
    assert _setup_scripts(), "no setup.py under kernels/; this check now covers nothing"


@pytest.mark.parametrize("script", _setup_scripts(), ids=lambda p: str(p.parent.relative_to(KERNELS)))
def test_a_setup_script_does_nothing_when_imported(script: Path) -> None:
    """`setup()` at module scope means importing the module RUNS setuptools.

    These are standalone build scripts (`python setup.py build_ext --inplace`) that happen to sit
    inside the importable package, so anything walking the tree imports them. `dev audit`'s import
    sweep did, and got `SystemExit: usage: cli.py [global_opts] ...` -- reporting 2 not OK on every
    run, for a reason having nothing to do with what it audits.

    A __main__ guard costs nothing and leaves running the script directly unchanged.
    """
    tree = ast.parse(script.read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = getattr(node.value.func, "id", None) or getattr(node.value.func, "attr", None)
            assert name != "setup", (
                f"{script.relative_to(KERNELS.parent)}:{node.lineno} calls setup() at module "
                f"scope, so importing it runs setuptools and raises SystemExit. Put it under "
                f'`if __name__ == "__main__":`.')
