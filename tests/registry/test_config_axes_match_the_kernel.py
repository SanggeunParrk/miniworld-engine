"""A config CSV may only name axes the kernel actually takes.

The two halves are declared apart: `autotune/configs/<set>/<op>.csv` lists the axes, and the
kernel's signature lists the `tl.constexpr` parameters. Triton passes each axis as a keyword, so a
mismatch is not caught by anything until the kernel LAUNCHES -- an axis the kernel does not take
raises `got an unexpected keyword argument`, and a constexpr no set declares raises
`missing required positional arguments`, both on a GPU, inside a build or a numerics run.

Adding GROUP_M to five kernels' CSVs while converting only four of them cost exactly that: the
fifth failed its numerics test with the config half done. This reads both sides statically.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "src/miniworld_engine/kernels/registry.csv"
CFG = ROOT / "src/miniworld_engine/autotune/configs"

#: Set by `triton.Config`, never by the kernel signature.
META = {"num_warps", "num_stages", "maxnreg"}


def _kernel_constexprs(file: str, symbol: str) -> set[str] | None:
    path = ROOT / "src" / file
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == symbol), None)
    if fn is None:
        return None
    return {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
            if a.annotation and "constexpr" in ast.unparse(a.annotation)}


def _declared_axes(path: Path) -> set[str]:
    with path.open(newline="") as fh:
        first = fh.readline().strip()
        if first.startswith("axis,"):
            return {line.split(",", 1)[0].strip() for line in fh
                    if line.strip() and not line.startswith("slice,")}
        return {c.strip() for c in first.split(",")} - META


def test_every_config_axis_is_a_kernel_constexpr() -> None:
    with REG.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["backend"] == "triton"]
    bad = []
    for r in rows:
        ce = _kernel_constexprs(r["file"], r["symbol"])
        if ce is None:
            continue
        for d in sorted(x for x in CFG.iterdir() if x.is_dir()):
            f = d / f"{r['kernel']}.csv"
            if not f.is_file():
                continue
            axes = _declared_axes(f)
            extra = axes - ce - META
            if extra:
                bad.append(f"{d.name}/{r['kernel']}.csv declares {sorted(extra)}, "
                           f"which {r['symbol']} does not take")
            # The other direction is NOT an invariant: a constexpr may come from the launcher
            # rather than the config -- a shape (`K`, `N`, `NC`) or a dispatch switch
            # (`ADD_RESIDUAL`) is passed explicitly at the call site and belongs to no config set.
    assert not bad, ("config and kernel disagree -- each of these is a launch failure on a GPU:"
                     "\n  " + "\n  ".join(bad))
