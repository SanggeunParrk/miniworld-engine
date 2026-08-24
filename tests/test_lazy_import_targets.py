"""Every deferred import inside a public lazy wrapper must name something that exists.

`miniworld_engine.kernels` keeps its import cheap by wrapping each backend entry in a function
that imports the implementation on first call. `test_every_public_name_resolves` checks the
wrapper is callable -- it always is. Nothing checked the *body*.

So `kernels.cuda_transition` shipped as a pinned public name whose body reads
`from .transition.cuda import cuda_transition`, and that symbol has never existed in this repo's
history. `Transition(implementation="cuda")` resolves to KernelBackend.CUDA and calls it in
forward. The path has never been able to run, the frozen-surface test passed, and the contract
test passed.

This resolves each deferred import statically -- no kernel is built, no GPU is touched -- so it
runs in the CPU suite alongside the other contract tests.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src/miniworld_engine"
KERNELS_INIT = PKG / "kernels/__init__.py"


PKG_PREFIX = "miniworld_engine.kernels."


def _deferred_imports(path: Path):
    """(function, module, name) for every kernels-package import written INSIDE a function.

    Both spellings are matched. They used to be relative (`from .transition.interface import x`);
    the repo now bans relative imports (ruff TID252, `ban-relative-imports = "all"`), so the same
    statements are absolute. Matching only the relative form is how this file quietly found zero
    wrappers and checked nothing -- which is what `test_there_are_lazy_wrappers_to_check` is for.
    """
    tree = ast.parse(path.read_text())
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.ImportFrom) and node.module):
                continue
            if node.level:
                mod = "." * node.level + node.module
            elif node.module.startswith(PKG_PREFIX):
                mod = node.module[len(PKG_PREFIX):]
            else:
                continue
            out.extend((fn.name, mod, a.name) for a in node.names)
    return out


CASES = _deferred_imports(KERNELS_INIT)


def test_there_are_lazy_wrappers_to_check():
    """If this file stops finding any, the check below has quietly become a no-op."""
    assert CASES, f"no deferred imports found in {KERNELS_INIT}; has the lazy pattern changed?"


def _module_file(mod: str) -> Path | None:
    """Resolve a relative module name to its file WITHOUT importing it.

    Importing is not an option: `.transition.cuda` builds a CUDA extension on import, so an
    import-based check skips on CPU -- skipping the exact case this test exists for.
    """
    rel = mod.lstrip(".").replace(".", "/")   # both spellings land on the same path
    for cand in (PKG / "kernels" / f"{rel}.py", PKG / "kernels" / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _top_level_names(path: Path) -> set[str]:
    """Names a module binds at import: defs, classes, assignments, imports, and __all__."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.If):  # version/try guards bind inside a block
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
        elif isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
    return names


@pytest.mark.parametrize(("fn", "mod", "name"), CASES,
                         ids=[f"{f}->{m}.{n}" for f, m, n in CASES])
def test_a_lazy_wrapper_imports_something_that_exists(fn, mod, name):
    target = _module_file(mod)
    if target is None:
        pytest.skip(f"{mod} is not a file in this package (namespace or generated)")
    names = _top_level_names(target)
    # A module with a PEP 562 __getattr__ can supply anything; do not claim a miss there.
    if "__getattr__" in names:
        pytest.skip(f"{mod} defines __getattr__; any name may be lazily supplied")
    assert name in names, (
        f"kernels.{fn}() does `from {mod} import {name}`, and "
        f"{target.relative_to(PKG.parent)} binds no {name!r}. The wrapper is callable, so the "
        f"frozen-surface test passes -- calling it cannot work."
    )
