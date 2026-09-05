"""The `compiled` column must say what RAN -- and it now does so trivially.

Several module benches used to guard `model.compile()` with `and conf.cudagraph == "disabled"`:
the CUDA-graph capture then took the eager module, so `compile=true cudagraph=manual` ran eager for
half the matrix while the CSV recorded `conf.compile` for all of it -- eager code labelled
compiled (triangle_multiplication, triangle_attention). Under the default
`compile_wrap="custom_op"` there are no graph breaks, so compile+capture is the real deployment
regime; the gates are removed and every bench compiles unconditionally. The recorded flag is then
exactly `conf.compile`, and `actual_compiled_flag` is that and nothing else.

This file guards the two ways that used to drift: a gate creeping back into a bench, and
`actual_compiled_flag` growing a by-hand target rule. The suite cannot import bench.py without a
GPU (it raises at import), so this reads the source.
"""
from __future__ import annotations

import ast
from pathlib import Path

BENCH = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "benchmarks" / "runners" / "bench.py"
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
COMPILING = {n for n, b in BODIES.items() if ".compile()" in b}


def test_no_bench_gates_its_compile_on_the_cudagraph_regime():
    """A gate is what made `compiled=True, cudagraph=manual` a measurement of eager code. It is
    removed; this fails if one returns, so the flag cannot silently start lying again."""
    assert not GATED, (
        f"these benches gate model.compile() on cudagraph again -- compile+capture is the real "
        f"regime under custom_op, so the gate makes the recorded compiled flag lie: {sorted(GATED)}")
    assert COMPILING, "no bench calls .compile() at all -- the source reader is matching nothing"


def test_actual_compiled_flag_is_exactly_conf_compile():
    """No gate to detect means the recorded flag is `conf.compile`, with no target named by hand
    (`if conf.kernel == "transition"` was the old rule, and three other targets behaved the same)."""
    fn = next(n for n in ast.walk(TREE)
              if isinstance(n, ast.FunctionDef) and n.name == "actual_compiled_flag")
    code = [st for st in fn.body
            if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))]
    body = "\n".join(ast.unparse(st) for st in code)
    assert body.strip() == "return conf.compile", (
        f"actual_compiled_flag is no longer just `return conf.compile`; the gates are gone, so it "
        f"has nothing to branch on:\n{body}")
