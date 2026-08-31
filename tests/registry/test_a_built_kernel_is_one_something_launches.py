"""A kernel `build all` tunes must be one the library can actually reach.

`developed=yes` costs real GPU hours -- the sweep compiles and benches every config of it at every
shape -- and ships a cache entry. That is only worth paying for a kernel some path leads to. Five
were not: superseded front variants in `trimul_inproj`, left behind when the front moved to cute,
still tuned in every build and checked in every numerics run because nothing said otherwise.

Nothing could say otherwise, because nothing looked. `undeveloped.csv` records the judgement "this
one is not worth tuning", and it is hand-maintained on purpose -- bias_only_attention is held out
for a reason no rule could derive. But "nothing calls it at all" is not a judgement, it is a fact
about the source, and a fact about the source is exactly what a test can hold.

HOW IT LOOKS. From the module layer, the public `ops` package and the top-level `__init__` -- the
three ways in from outside the kernel package -- follow every name a function references, including
the ones a lazy `from ... import` brings in (every arch branch is that shape:
`if major >= 10: from ...sm100 import bidir_forward_sm100`). A kernel is reached when any function
that launches it is reached. Deliberately over-approximate: a bare reference counts as a call, so a
function handed to `dispatch.pick` as a value is reached. It says "unreachable" only when nothing
anywhere mentions the launcher.

WHAT IT CANNOT SAY. Reachable is not the same as reached at runtime -- a branch may exist for a
card nobody has. That is a different question, and the one `bench`'s coverage report answers by
running.
"""
from __future__ import annotations

import ast
import collections
import csv
from pathlib import Path

import pytest
from paths import ROOT

PKG = ROOT / "src" / "miniworld_engine"
from paths import REGISTRY as REG

#: The ways in from outside the kernel package. A path under any of these is a seed.
ENTRY = ("/modules/", "/ops/")


def _sources() -> dict[Path, ast.Module]:
    """Every module except the drivers and checkers -- those exist to reach kernels the library
    does not, which is the whole point of them and would make every kernel look reached."""
    out = {}
    for p in sorted(PKG.rglob("*.py")):
        s = str(p)
        if "/drivers/" in s or "/checks/" in s:
            continue
        try:
            out[p] = ast.parse(p.read_text())
        except SyntaxError:
            continue
    return out


def _referenced(node: ast.AST) -> set[str]:
    got: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            got.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            got.add(sub.attr)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            got |= {a.asname or a.name.split(".")[-1] for a in sub.names}
    return got


def _reachable(trees: dict[Path, ast.Module]) -> set[str]:
    refs: dict[str, set[str]] = collections.defaultdict(set)
    seeds: set[str] = set()
    for p, t in trees.items():
        key = f"<mod:{p}>"
        refs[key] |= _referenced(t)
        funcs = [n for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for fn in funcs:
            refs[fn.name] |= _referenced(fn)
        if any(e in str(p) for e in ENTRY) or p == PKG / "__init__.py":
            seeds.add(key)
            seeds |= {fn.name for fn in funcs}
    reach, stack = set(seeds), list(seeds)
    while stack:
        for nxt in refs.get(stack.pop(), ()):
            if nxt not in reach:
                reach.add(nxt)
                stack.append(nxt)
    return reach


def _launchers(trees: dict[Path, ast.Module]) -> dict[str, set[str]]:
    """triton kernel symbol -> the functions that launch it, from ANY file.

    Any file: a kernel is regularly defined in one module and launched from another
    (`_sigmul_fwd` lives in gated_projection and is launched by conditioned_transition and
    bias_only_attention). Searching only the defining file reports four false orphans.
    """
    out: dict[str, set[str]] = collections.defaultdict(set)
    for t in trees.values():
        for fn in [n for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                    out[sub.value.id].add(fn.name)
    return out


def _built() -> list[dict]:
    with REG.open(newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if r["backend"] == "triton" and (r.get("developed") or "yes").strip() != "no"]


@pytest.fixture(scope="module")
def analysis():
    trees = _sources()
    assert len(trees) > 100, f"only {len(trees)} modules parsed; the sweep below would be vacuous"
    return _reachable(trees), _launchers(trees)


def test_every_built_kernel_has_a_launcher(analysis) -> None:
    """Before asking whether anything reaches it: something has to launch it at all."""
    _, launch = analysis
    orphan = sorted(r["kernel"] for r in _built() if not launch.get(r["symbol"]))
    assert not orphan, (
        f"no `symbol[grid](...)` anywhere for: {orphan}. Either the registry's `symbol` column is "
        f"wrong, or the kernel is defined and never launched.")


def test_every_built_kernel_is_reachable(analysis) -> None:
    reach, launch = analysis
    unreachable = sorted(
        r["kernel"] for r in _built()
        if not any(fn in reach for fn in launch.get(r["symbol"], ())))
    assert not unreachable, (
        "these are tuned by every `build all` and checked by every numerics run, and no path from "
        "the module layer, the `ops` package or the top-level __init__ reaches them:\n  "
        + "\n  ".join(unreachable)
        + "\nIf a kernel is kept deliberately -- a negative result held as reference, a variant "
          "waiting for its path back -- mark it `developed=no` and write the reason in "
          "kernels/undeveloped.csv. Keeping the code and building it are different decisions.")


def test_the_check_is_not_vacuous(analysis) -> None:
    """Guard the guard. If the seeding or the launcher search broke, everything would look
    reachable and this file would pass on anything."""
    reach, launch = analysis
    built = _built()
    covered = [r for r in built if any(fn in reach for fn in launch.get(r["symbol"], ()))]
    assert len(covered) > 0.8 * len(built), (
        f"only {len(covered)} of {len(built)} built kernels look reachable — the analysis is "
        f"broken, not the code")
