"""The `compiled` column must say what RAN, not what was asked for.

Four of the eight module benches guard `model.compile()` with `and conf.cudagraph == "disabled"`
-- the CUDA-graph capture takes the eager module -- and four do not. So `compile=true
cudagraph=manual` runs eager for half the matrix and compiled for the other half, while the CSV
recorded `conf.compile` for all of them.

330 of the 350 committed result tables say `compiled=True, cudagraph=manual`. For
triangle_multiplication, triangle_attention and bias_only_attention those are measurements of
eager code labelled compiled. `actual_compiled_flag` knew about `transition` alone, by name, which
is how the other three went unnoticed.

The suite cannot import bench.py without a GPU (it raises at import), so this reads the source.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "runners" / "bench.py"
SRC = BENCH.read_text()
TREE = ast.parse(SRC)
LINES = SRC.splitlines()


def _fn_bodies() -> dict[str, str]:
    out = {}
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name.startswith("bench_"):
            out[n.name] = "\n".join(LINES[n.lineno - 1:n.end_lineno])
    return out


BODIES = _fn_bodies()
GATED = {n for n, b in BODIES.items()
         if ".compile()" in b and 'conf.cudagraph == "disabled"' in b}
UNGATED = {n for n, b in BODIES.items()
           if ".compile()" in b and 'conf.cudagraph == "disabled"' not in b}


def test_both_kinds_of_bench_exist():
    """If they did not, the rest of this file would be vacuous."""
    assert GATED, "no bench guards its compile on cudagraph"
    assert UNGATED, "no bench compiles unconditionally"


def test_the_flag_is_not_a_hand_kept_list_of_target_names():
    """It was `if conf.kernel == "transition"`, and three other targets behave the same way."""
    fn = next(n for n in ast.walk(TREE)
              if isinstance(n, ast.FunctionDef) and n.name == "actual_compiled_flag")
    # The docstring explains the old rule; only the CODE must be free of it.
    code = [st for st in fn.body if not (isinstance(st, ast.Expr)
                                         and isinstance(st.value, ast.Constant))]
    body = "\n".join(ast.unparse(st) for st in code)
    assert not any(spelling in body
                   for spelling in ("kernel == 'transition'", 'kernel == "transition"')), (
        "actual_compiled_flag names one target; the rule has to be read from the benches, or it "
        f"drifts again -- {len(GATED)} of them gate their compile today")


def test_the_deriver_agrees_with_every_bench():
    """`_skips_compile_under_cudagraph` must classify each bench the way its source behaves."""
    start = SRC.index("def _skips_compile_under_cudagraph")
    body = SRC[start:start + 900]
    for probe in ('conf.cudagraph == "disabled"', '".compile()"'):
        assert probe in body, (
            f"the deriver no longer looks for {probe}, the gate it is meant to detect")


@pytest.mark.parametrize("name", sorted(GATED))
def test_a_gated_bench_really_skips_compile_under_a_cudagraph(name):
    body = BODIES[name]
    i = body.index(".compile()")
    guard = body.rfind("if ", 0, i)
    assert 'conf.cudagraph == "disabled"' in body[guard:i], (
        f"{name}: the .compile() nearest the guard is not the guarded one")
