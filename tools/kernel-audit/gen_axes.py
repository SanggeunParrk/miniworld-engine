"""Regenerate autotune/axes.csv from the code, with only facts that can be checked.

The old file described 88 of its 89 ops with axis names that do not exist in the kernels -- it
kept the normalised names from the rename that was reverted -- so it could not be used to find a
kernel's tile knob. Rather than re-deriving the prose columns (meaning / extent / notes), which
would be guessing, every column here is read off the code or the config CSVs:

  op, kernel, file          the @triton.autotune(configs_for(op)) site
  axis                      the config CSV header -- exactly what triton injects
  <set>                     the value the axis takes in each shipped config set
  grid_axis                 1 if a grid lambda launching this kernel reads this axis
  loop_step                 1 if a `for _ in range(0, EXTENT, axis)` loop steps by it
  loop_extent               that EXTENT expression
  extent_bounded            for a loop_step axis: does the loop body ever bound EXTENT?
                            0 means the tail tile reads past the end (see tiling-audit.md)

The prose is not lost: the previous file is kept verbatim as docs/kernels/axes-legacy.csv.
"""
from __future__ import annotations
import ast
import csv
from pathlib import Path

ROOT = Path("src/miniworld_engine")
META = {"num_warps", "num_stages", "maxnreg"}
SETS = sorted(d.name for d in Path("configs").iterdir() if d.is_dir() and d.name != "devices")

vals: dict[str, dict[str, dict[str, int]]] = {}
for s in SETS:
    for p in sorted((Path("configs") / s).glob("*.csv")):
        rows = list(csv.DictReader(p.open(newline="")))
        if rows:
            for k, v in rows[0].items():
                if k and k not in META and v:
                    vals.setdefault(p.stem, {}).setdefault(k, {})[s] = int(v)


def fname(c: ast.Call) -> str:
    return getattr(c.func, "attr", None) or getattr(c.func, "id", "") or ""


def names(n) -> set[str]:
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}


info: dict[str, dict] = {}
grid_axes: dict[str, set[str]] = {}
for path in sorted(ROOT.rglob("*.py")):
    if "autotune/data" in str(path):
        continue
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue
    kern_op = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for dec in fn.decorator_list:
            for n in ast.walk(dec):
                if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "configs_for"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    kern_op[fn.name] = n.args[0].value
        if fn.name in kern_op:
            op = kern_op[fn.name]
            rec = info.setdefault(op, dict(kernel=fn.name,
                                           file=str(path.relative_to(ROOT.parent)),
                                           line=fn.lineno, loops={}))
            for loop in [n for n in ast.walk(fn) if isinstance(n, ast.For)]:
                it = loop.iter
                # same matcher as .bench/mask_audit.py: tl.static_range counts, and the step is
                # not always named BLOCK_* (HEAD_DIM stepped by HEAD_DIM_PAD)
                if not (isinstance(it, ast.Call) and fname(it) in ("range", "static_range")
                        and len(it.args) == 3):
                    continue
                step = ast.unparse(it.args[2])
                if step.isupper() and step in {"G", "NP"} or not step[:1].isalpha():
                    continue
                extent = ast.unparse(it.args[1])
                ext_names = names(it.args[1]) or {extent}
                body = ast.Module(body=loop.body, type_ignores=[])
                by_cmp = any(ext_names & names(c)
                             for c in ast.walk(body) if isinstance(c, ast.Compare))
                bc = set()
                for n in ast.walk(body):
                    if isinstance(n, ast.Call) and fname(n) == "load":
                        for k in n.keywords:
                            if k.arg == "boundary_check":
                                bc.add(ast.unparse(k.value))
                by_bc = bool(bc) and all(sum(ch.isdigit() for ch in a) >= 2 for a in bc)
                rec["loops"][step] = (extent, 1 if (by_cmp or by_bc) else 0)
    # grid lambdas
    for host in ast.walk(tree):
        if not isinstance(host, ast.FunctionDef):
            continue
        assigns = {}
        for n in ast.walk(host):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                assigns.setdefault(n.targets[0].id, []).append((n.lineno, n.value))
        for n in ast.walk(host):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)):
                continue
            nm = getattr(n.func.value, "id", None) or getattr(n.func.value, "attr", None)
            if nm not in kern_op:
                continue
            g = n.func.slice
            expr = None
            if isinstance(g, ast.Name):
                before = [a for a in assigns.get(g.id, []) if a[0] <= n.lineno]
                if before:
                    expr = max(before, key=lambda a: a[0])[1]
            else:
                expr = g
            if expr is None:
                continue
            keys = {x.slice.value for x in ast.walk(expr)
                    if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant)
                    and isinstance(x.slice.value, str)}
            grid_axes.setdefault(kern_op[nm], set()).update(keys)

out = Path("src/miniworld_engine/autotune/axes.csv")
legacy = Path("docs/kernels/axes-legacy.csv")
if out.is_file() and not legacy.is_file():
    legacy.write_text(out.read_text())

fields = ["op", "kernel", "file", "line", "axis", *SETS,
          "grid_axis", "loop_step", "loop_extent", "extent_bounded"]
rows = []
for op in sorted(vals):
    rec = info.get(op, {})
    for axis in sorted(vals[op]):
        extent, bounded = rec.get("loops", {}).get(axis, ("", ""))
        rows.append({
            "op": op, "kernel": rec.get("kernel", ""), "file": rec.get("file", ""),
            "line": rec.get("line", ""), "axis": axis,
            **{s: vals[op][axis].get(s, "") for s in SETS},
            "grid_axis": 1 if axis in grid_axes.get(op, set()) else 0,
            "loop_step": 1 if extent else 0,
            "loop_extent": extent,
            "extent_bounded": bounded,
        })
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} : {len(rows)} axis rows over {len(vals)} ops")
print(f"legacy prose preserved at {legacy}")
unb = [r for r in rows if r["extent_bounded"] == 0]
print(f"axes stepping an UNBOUNDED loop: {len(unb)}")
for r in unb:
    print(f"   {r['op']:<46} {r['axis']:<12} range(0, {r['loop_extent']}, {r['axis']})  {r['file']}")
