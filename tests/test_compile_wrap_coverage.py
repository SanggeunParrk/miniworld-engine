"""``compile_wrap="custom_op"`` has to be a real mode, not a switch that raises on import.

It shipped as a documented ``settings`` field with two "interchangeable" values. It was not:
47 of the 58 kernel entry points put ``@opaque()`` on ``autograd.Function.forward``/``backward``,
which ``torch.library.custom_op`` cannot wrap (``ctx`` is not a schema type), so simply SELECTING
the mode raised ``ValueError`` before a single kernel loaded. A configuration that cannot be
selected is not a configuration.

These tests are static -- no GPU, no launches. They check the two things that let the mode rot:

1. every ``opaque`` site can register, i.e. it either carries a ``fake`` or is the ``disable``
   default (:func:`test_custom_op_mode_registers_every_site`);
2. every site that stays a graph break says WHY at the site
   (:func:`test_declared_graph_breaks_state_a_reason`), so "still breaking" is a fact someone
   wrote down rather than the accident of a missing fake.

The second matters because two shapes genuinely cannot be ops, and both fail SILENTLY if wrapped:
an ``nn.Module`` method (``self`` is not a schema type -- that one at least raises), and a wrapper
whose body calls ``Function.apply``. An op is opaque to AUTOGRAD as well as to Dynamo, so wrapping
an ``.apply`` returns a tensor with no ``grad_fn``: the forward numbers stay right and training
quietly stops learning through it.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine"


def _decorator_sites(tree: ast.AST, name: str) -> list[ast.expr]:
    """Every decorator expression in ``tree`` whose callee is ``name``."""
    out = []
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []) or []:
            call = dec if isinstance(dec, ast.Call) else None
            func = call.func if call else dec
            if isinstance(func, ast.Name) and func.id == name:
                out.append(dec)
    return out


def _python_files() -> list[Path]:
    return sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py"))


def test_no_bare_opaque_site_remains() -> None:
    """``@opaque()`` with no ``fake`` is exactly the thing that made the mode unselectable."""
    bare: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for dec in _decorator_sites(tree, "opaque"):
            if isinstance(dec, ast.Call) and not dec.args and not dec.keywords:
                bare.append(f"{path.relative_to(SRC)}:{dec.lineno}")
    assert not bare, (
        "these @opaque() sites have no fake, so compile_wrap='custom_op' cannot register them:\n  "
        + "\n  ".join(bare))


def _is_launch(node: ast.AST) -> bool:
    """Is this ``kernel[grid](...)``, a Triton launch?

    Narrower than "a call whose func is a Subscript": ``candidates[idx][1]()`` in the cute
    dispatcher is that too, and it is a thunk, not a launch. A launch subscripts a plain NAME (the
    kernel) and is called WITH arguments.
    """
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, (ast.Name, ast.Attribute))
            and bool(node.args or node.keywords))


def _launchers_and_ops(paths):
    """(launchers, ops, calls) -- a launcher is a function that launches a Triton kernel."""
    launchers, ops, calls = {}, set(), {}
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = []
            for dec in node.decorator_list:
                f = dec.func if isinstance(dec, ast.Call) else dec
                names.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
            key = f"{path.name}::{node.name}"
            is_op = bool({"opaque", "custom_op", "triton_op"} & set(names))
            if is_op:
                ops.add(node.name)
            if any(_is_launch(n) for n in ast.walk(node)):
                launchers[key] = (node.name, is_op, node.lineno, path)
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    f = n.func
                    nm = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if nm:
                        calls.setdefault(nm, set()).add((node.name, is_op))
    return launchers, ops, calls


def test_every_launcher_is_opaque_or_only_reached_through_one() -> None:
    """No traced path may reach a raw kernel launch.

    A launcher does NOT have to be an op if every one of its callers already is: it can only be
    entered from inside an opaque region, so Dynamo never sees it, and wrapping it would just add
    a second dispatch on a hot inner call. Everything else needs its own fake.

    ``checks_*`` / ``drivers_*`` are excluded: that is the autotune-capture harness, which calls
    kernels directly and is never inside a compiled model.
    """
    paths = [p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")
             if not p.name.startswith(("checks_", "drivers_"))]
    launchers, ops, calls = _launchers_and_ops(paths)

    # "Covered" is TRANSITIVE: `_ln_bwd_atomic`'s only caller is `ln_bwd_mmajor`, which is not an
    # op itself but is only ever called from one. Iterate to a fixed point rather than looking one
    # level up, which is what a first version of this test did -- and it flagged three launchers
    # that were already unreachable.
    covered = set(ops)
    for _ in range(len(calls) + 1):
        grew = False
        for name, callers in calls.items():
            if name in covered or not callers:
                continue
            if all(c in covered or is_op for c, is_op in callers):
                covered.add(name)
                grew = True
        if not grew:
            break

    bad = [f"{path.relative_to(SRC)}:{lineno} {name}"
           for _key, (name, is_op, lineno, path) in sorted(launchers.items())
           if not is_op and name not in covered]
    assert not bad, (
        "these launch a kernel on a path Dynamo can reach, but are not opaque ops:\n  "
        + "\n  ".join(bad))


@pytest.mark.parametrize("wrap", ["disable", "custom_op"])
def test_custom_op_mode_registers_every_site(wrap: str) -> None:
    """Import the whole kernel+module tree under each mode. Registration happens at import.

    A subprocess per mode is not fussiness: ``kernels._compile`` reads ``compile_wrap`` when the
    decorator RUNS, so a single interpreter can only ever hold one of the two.
    """
    script = f"""
import sys
from miniworld_engine import settings
settings.configure(compile_wrap={wrap!r})
import importlib, pathlib
root = pathlib.Path({str(SRC)!r})
mods = sorted({{
    "miniworld_engine." + ".".join(f.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
    for d in ("kernels", "modules") for f in (root / d).rglob("*.py")
}})
missing = []
for m in mods:
    if "cuda.setup" in m:
        continue
    try:
        importlib.import_module(m)
    except ValueError as e:
        if "needs a fake implementation" in str(e):
            missing.append(m + " :: " + str(e))
    except Exception:
        pass          # CUDA/CuTeDSL absent on a CPU runner -- not what this test is about
print("MISSING=" + str(len(missing)))
for m in missing:
    print("  " + m)
sys.exit(1 if missing else 0)
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=1800, check=False)
    assert r.returncode == 0, f"compile_wrap={wrap!r} could not register every site:\n{r.stdout}"
