"""Every ``fake`` must describe the function it is attached to.

A ``fake`` is what the compiler believes about a kernel it cannot see: the shapes, dtypes and
arity of the outputs. If it disagrees with the real body, eager stays correct -- the fake never
runs there -- and only the COMPILED path goes wrong, on whichever shape or card reaches the
disagreeing branch. Numerics parity cannot see it (both sides run eager) and ``opcheck`` only
covers ops the test machine can actually launch, which excludes every sm90/sm100 CuTeDSL path.

So this reads instead of running, and it is not hypothetical: it found ``layernorm_bwd``'s
hand-CUDA fast path still returning the pre-split 5-tuple ``(dx, dw, db, None, None)`` against a
3-value schema. That branch is only taken when a row-scale is present (the AF pair-mask), which no
module test passed, so every GPU run had been green.

Two things are checked, both of which a human reading 105 sites will miss at least once:

* the fake's parameter list matches the op's -- a fake with a parameter the op lacks (or a
  different order) misbinds silently under a keyword call;
* the return ARITY agrees, where it can be known statically. A body whose returns are all plain
  tuples is knowable; one that returns a call or a conditional expression is not, and is skipped
  rather than guessed at.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "miniworld_engine"


def _own_return_arities(node: ast.FunctionDef | ast.Lambda) -> set[int] | None:
    """Arities of the returns in this function's OWN scope, or None if not statically knowable.

    Nested scopes are skipped: nearly every launcher defines ``grid = lambda meta: (...)``, and
    counting that lambda's tuple made 62 of 105 sites look broken in a first version of this.
    """
    if isinstance(node, ast.Lambda):
        return {len(node.body.elts)} if isinstance(node.body, ast.Tuple) else None

    arities: set[int] = set()
    knowable = True

    def walk(parent: ast.AST) -> None:
        nonlocal knowable
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                value = child.value
                if isinstance(value, (ast.Tuple, ast.List)):
                    arities.add(len(value.elts))
                else:
                    # `return other(...)` or `return a if c else b`: the arity is whatever that
                    # expression yields, which is not visible here.
                    knowable = False
            walk(child)

    walk(node)
    return arities if knowable and arities else None


def _signature(node: ast.FunctionDef | ast.Lambda) -> list[str]:
    args = node.args
    return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]


def _opaque_sites():
    """(op name, decorated function, fake function, file) for every ``@opaque(fake=...)``."""
    for path in sorted(p for d in ("kernels", "modules") for p in (SRC / d).rglob("*.py")):
        tree = ast.parse(path.read_text())
        named = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "opaque"):
                    continue
                fake = next((k.value for k in dec.keywords if k.arg == "fake"), None)
                if fake is None:
                    continue
                name = next((k.value.value for k in dec.keywords
                             if k.arg == "name" and isinstance(k.value, ast.Constant)), node.name)
                resolved = named.get(fake.id) if isinstance(fake, ast.Name) else fake
                yield str(name), node, resolved, path


def test_every_opaque_site_has_a_resolvable_fake() -> None:
    missing = [f"{n}  ({p.relative_to(SRC)})" for n, _f, fake, p in _opaque_sites() if fake is None]
    assert not missing, "fake= names a function this file does not define:\n  " + "\n  ".join(missing)


def test_fake_signature_matches_the_op() -> None:
    bad = []
    for name, fn, fake, path in _opaque_sites():
        if fake is None:
            continue
        op_params, fake_params = _signature(fn), _signature(fake)
        if op_params != fake_params:
            bad.append(f"{name}  ({path.relative_to(SRC)})\n"
                       f"        op   {op_params}\n        fake {fake_params}")
    assert not bad, "fake parameters differ from the op's:\n  " + "\n  ".join(bad)


def test_fake_return_arity_matches_the_body() -> None:
    bad = []
    for name, fn, fake, path in _opaque_sites():
        if fake is None:
            continue
        body, shadow = _own_return_arities(fn), _own_return_arities(fake)
        if body is None or shadow is None:
            continue                       # not statically knowable; opcheck covers what it can
        if len(body) > 1:
            bad.append(f"{name}  ({path.relative_to(SRC)}): body returns "
                       f"{sorted(body)} values depending on the branch -- a schema has one arity")
        elif body != shadow:
            bad.append(f"{name}  ({path.relative_to(SRC)}): body returns {sorted(body)}, "
                       f"fake returns {sorted(shadow)}")
    assert not bad, "fake/body return arity disagrees:\n  " + "\n  ".join(bad)
