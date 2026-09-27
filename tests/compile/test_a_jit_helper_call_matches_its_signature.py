"""A ``@triton.jit`` helper call is checked by nothing until a GPU compiles it.

Python does not check it: the body of a jit function is never executed as Python, so a call with the
wrong arity raises at COMPILE time, inside `ast_to_ttir`, on a card. The CPU suite passes, ruff
passes, and the error arrives as a `CompilationError` in a Slurm log twenty minutes later.

That is not hypothetical. `_row_gemm` gained two parameters -- a column mask and `BLOCK_N` -- and one
of its five callers was not updated. Every check in this repository was green; the card said
`TypeError("_row_gemm() missing 2 required positional arguments")`, and because the six checkers of
that family share one launcher, it reported as six failures rather than one.

The check is arity only. Types inside a Triton body are not Python types and a static reading of
them would be a guess; a missing or extra argument is neither.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src"


def _jit_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    """Module-level functions carrying a bare ``@triton.jit``."""
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
            and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in n.decorator_list)}


def _accepts(fn: ast.FunctionDef) -> tuple[set[str], bool]:
    """(every parameter name, whether it takes *args or **kwargs).

    Arity is checked against the WHOLE set, not against the required half: a jit helper in this
    tree declares no defaults -- a `tl.constexpr` with one is a value the tuner cannot vary -- so
    "declared" and "required" are the same set, and counting the other one would let a call omit
    a parameter that has no default to fall back on.
    """
    a = fn.args
    return {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}, bool(a.vararg or a.kwarg)


def test_every_jit_helper_call_passes_what_the_helper_declares() -> None:
    bad: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "notes" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        helpers = _jit_functions(tree)
        if not helpers:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            helper = helpers.get(node.func.id)
            if helper is None:
                continue
            # A starred call hands over a sequence this cannot count, so it is not judged.
            if any(isinstance(x, ast.Starred) for x in node.args) or any(
                    k.arg is None for k in node.keywords):
                continue
            names, flexible = _accepts(helper)
            if flexible:
                continue
            given = [k.arg for k in node.keywords]
            unknown = sorted(set(given) - names)
            where = f"{path.relative_to(SRC)}:{node.lineno}: {node.func.id}()"
            if unknown:
                bad.append(f"{where} passes {unknown}, which it does not declare")
            elif len(node.args) + len(given) != len(names):
                bad.append(f"{where} passes {len(node.args) + len(given)} of {len(names)} "
                           f"parameters; a jit body is only checked when a card compiles it")
    assert not bad, ("a @triton.jit helper called with the wrong arguments:\n  " + "\n  ".join(bad))
