"""Every driver and checker's lazy imports have to name something that exists.

A driver reaches its kernel through an import inside the function body -- `from
miniworld_engine.kernels.<family>.triton.<file> import <launcher>` -- so that importing the driver
module does not drag in every kernel in the repo. The cost of that is that a rename or a file split
does not break anything until the driver is CALLED, and a driver is only called on a GPU, inside a
build. `cond_transition_expand_swiglu_saveact` and `cond_transition_squeeze_gate_saveact` spent a
whole GPU run raising `ModuleNotFoundError: ...triton.train_fused` -- that file had been split into
`fwd_saveact.py`, registry.csv had been updated, and the driver had not. Their two ops built
nothing, and the build reported it as two bad units rather than as a broken reference.

This reads the imports statically -- no CUDA, no kernel compiled -- so a split like that fails in
the CPU suite, in a second, at the line that is wrong.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path

from paths import REGISTRY as REG


def _driver_specs() -> list[tuple[str, str, str]]:
    """(kernel, module, function) for every registry entry point -- driver AND check.

    Both columns name `module:function` and both reach their kernel through an import in the
    function body, so both rot the same way and neither is exercised without a GPU. The
    train_fused split broke one of each for the same two kernels: the driver built no cache
    entries, and the checker failed its numerics test with the same ModuleNotFoundError.
    """
    with REG.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        for column in ("driver", "check"):
            spec = (r.get(column) or "").strip()
            if spec:
                mod, _, fn = spec.partition(":")
                out.append((f"{r['kernel']} ({column})", mod, fn))
    return out


def _module_file(dotted: str) -> Path | None:
    spec = importlib.util.find_spec(dotted)
    return Path(spec.origin) if spec and spec.origin else None


def test_every_driver_function_exists() -> None:
    missing = []
    for kernel, mod, fn in _driver_specs():
        f = _module_file(mod)
        if f is None:
            missing.append(f"{kernel}: no module {mod}")
            continue
        tree = ast.parse(f.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if fn not in names:
            missing.append(f"{kernel}: {mod} has no {fn}()")
    assert not missing, "registry driver entries pointing at nothing:\n  " + "\n  ".join(missing)


def test_every_lazy_import_inside_a_driver_resolves() -> None:
    """The import is in the function BODY, so only a call would otherwise find it."""
    bad = []
    seen: dict[str, ast.Module] = {}
    for kernel, mod, fn in _driver_specs():
        f = _module_file(mod)
        if f is None:
            continue          # reported by the test above
        tree = seen.setdefault(mod, ast.parse(f.read_text()))
        body = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == fn), None)
        if body is None:
            continue          # reported by the test above
        for node in ast.walk(body):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            target = _module_file(node.module)
            if target is None:
                bad.append(f"{kernel} -> {mod}:{fn}() imports missing module {node.module}")
                continue
            ttree = ast.parse(target.read_text())
            defined = {n.name for n in ast.walk(ttree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            defined |= {t.id for n in ast.walk(ttree) if isinstance(n, ast.Assign)
                        for t in n.targets if isinstance(t, ast.Name)}
            defined |= {a.asname or a.name for n in ast.walk(ttree)
                        if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
            # A module-level `__getattr__` (PEP 562) makes its names at first access -- that is how
            # kernels/layernorm/cuda hands out `layer_norm_cuda`, an extension it compiles on
            # demand. Nothing static can enumerate those, so such a module vouches for all of them.
            if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
                   for n in ttree.body):
                continue
            for a in node.names:
                # `from pkg import name` is also how a SUBMODULE is imported (and how a compiled
                # extension is: kernels.layernorm.cuda.layer_norm_cuda is a .so), so a name the
                # target does not define is still fine when it resolves as a module of its own.
                if a.name in defined or _module_file(f"{node.module}.{a.name}") is not None:
                    continue
                bad.append(f"{kernel} -> {mod}:{fn}() imports {a.name} "
                           f"which {node.module} neither defines nor contains")
    assert not bad, "driver imports that only fail when the driver runs on a GPU:\n  " + \
                    "\n  ".join(bad)
