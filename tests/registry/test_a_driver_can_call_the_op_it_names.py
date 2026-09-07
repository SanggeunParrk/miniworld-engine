"""Every kernel call a registry driver makes must bind against the callee's real signature.

`trimul_outproj_layernorm_gemm_gate_triton`'s driver called `trimul_back_fused` without `eps`,
which is positional and required. Every unit of that op therefore died with a torch.library schema
error before a single config was timed -- on every card, for as long as the op had existed -- and
the kernel looked exactly like one nobody had built a cache for.

Nothing in the repository could see it:

  * the type gate excludes `kernels/**/triton/**` (vendored bodies), so no call INTO a kernel is
    arity-checked anywhere;
  * `@opaque` returns a `torch.library` `CustomOpDef` whose `__call__` is `(*args, **kwargs)`, so
    even `inspect.signature` saw nothing to check -- until `_compile.opaque` started setting
    `__wrapped__`, which is what this test relies on;
  * the build's exit rule was "fail only if nothing succeeded", so 12 identical deaths inside a
    195-unit success exited 0.

The third is fixed in the build and the second in the decorator. This is the first: the mistake is
visible in the SOURCE, with no GPU, before anything is scheduled.

What it does not do is call the drivers -- that needs a card. It binds the arguments each call site
passes against the parameters the callee declares, which is exactly the class of defect that got
through.
"""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "src" / "miniworld_engine" / "kernels" / "registry.csv"


def _entry_refs() -> list[tuple[str, str, str]]:
    """(kernel, module, function) for every driver AND checker the registry names.

    Both columns, because both call the same kernels with the same hand-written argument lists and
    a checker that cannot call its kernel is the same defect wearing a different hat -- it reports
    a kernel as unverified rather than as untuned.
    """
    out = []
    with REGISTRY.open(newline="") as fh:
        for row in csv.DictReader(fh):
            for col in ("driver", "check"):
                ref = (row.get(col) or "").strip()
                if ":" not in ref:
                    continue
                mod, fn = ref.split(":", 1)
                if not mod.startswith("miniworld_engine"):
                    mod = f"miniworld_engine.kernels.{mod}"
                out.append((row["kernel"], mod, fn))
    return out


def _local_imports(fn_node: ast.FunctionDef) -> dict[str, str]:
    """name -> module, for the `from x import y` statements INSIDE a driver body.

    Drivers import their kernel inside the function on purpose: importing at module scope would
    pull every backend into every build child. So the binding this test needs is local, and
    reading it off the module namespace would find nothing.
    """
    found = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                found[alias.asname or alias.name] = node.module
    return found


def _resolve(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name, None)
    except Exception:
        return None


def _calls(fn_node: ast.FunctionDef, imports: dict[str, str]):
    """(callee, n_positional, keyword names) for each call to a locally imported kernel entry."""
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        module = imports.get(node.func.id)
        if not module or not module.startswith("miniworld_engine"):
            continue
        # `*args` / `**kwargs` at a call site makes the count unknowable from the source. No
        # driver uses one today; if one starts, it is skipped rather than guessed at.
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if any(k.arg is None for k in node.keywords):
            continue
        target = _resolve(module, node.func.id)
        if target is None:
            continue
        yield node.func.id, len(node.args), [k.arg for k in node.keywords]


@pytest.mark.parametrize(("kernel", "module", "fname"), _entry_refs(),
                         ids=[f"{k}:{f}" for k, _, f in _entry_refs()])
def test_a_driver_binds_every_kernel_it_calls(kernel: str, module: str, fname: str) -> None:
    mod = importlib.import_module(module)
    src = Path(inspect.getsourcefile(mod) or "").read_text()
    tree = ast.parse(src)
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == fname), None)
    assert node is not None, f"{module}:{fname} is named by registry.csv and is not in the source"

    imports = _local_imports(node)
    checked = 0
    for callee_name, n_pos, kwnames in _calls(node, imports):
        callee = _resolve(imports[callee_name], callee_name)
        try:
            sig = inspect.signature(callee)
        except (TypeError, ValueError):
            continue                      # nothing declared to bind against
        if all(p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD
               for p in sig.parameters.values()):
            continue                      # `(*args, **kwargs)`: the signature was erased
        args = [object()] * n_pos
        kwargs = dict.fromkeys(kwnames, object())
        try:
            sig.bind(*args, **kwargs)
        except TypeError as exc:
            msg = (f"{module}:{fname} calls {callee_name}{sig} with {n_pos} positional "
                   f"argument(s) and {sorted(kwnames)} -- {exc}. {kernel} will die at every "
                   f"shape before a config is timed, and will look like a kernel nobody built "
                   f"a cache for.")
            raise AssertionError(msg) from exc
        checked += 1
    # Not asserted to be non-zero: a few drivers reach their kernel through a module attribute or
    # an nn.Module rather than a plain imported name, and this test is about the calls it CAN see.


def test_opaque_keeps_the_signature_it_wraps() -> None:
    """The whole test above rests on this: without `__wrapped__` there is nothing to bind."""
    from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton

    params = inspect.signature(trimul_back_triton).parameters
    assert "eps" in params, (
        "`opaque` stopped exposing the wrapped signature; every call into a torch.library op is "
        "now unverifiable again, which is how a driver shipped without a required argument")
    assert params["eps"].default is inspect.Parameter.empty, (
        "eps gained a default -- if that is deliberate, this test's premise is gone and the "
        "driver bug it guards could not have happened")
