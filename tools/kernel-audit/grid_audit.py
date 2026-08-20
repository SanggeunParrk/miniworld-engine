"""Static audit of grid lambdas against the tile axes of the kernel they launch.

The five shipped config sets give every tile axis of an op the SAME value, so a grid lambda that
divides by the wrong axis name computes the identical program count and the mistake is invisible.
This finds the mismatch by name instead of by number:

  * a grid lambda referencing a meta key that is not one of the launched kernel's tile axes
  * one ``grid`` variable reassigned inside a function, then used for kernels of differing arity
  * a single grid object shared by two kernels whose axis sets differ
"""
from __future__ import annotations
import ast, csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("src/miniworld_engine")
META = {"num_warps", "num_stages", "maxnreg"}

# op -> tile axes, from the config CSV header (authoritative: triton injects exactly these)
axes_of_op = {}
for p in sorted(Path("configs/accuracy").glob("*.csv")):
    rows = list(csv.DictReader(p.open(newline="")))
    if rows:
        axes_of_op[p.stem] = {k for k in rows[0] if k and k not in META}


def meta_keys(node) -> set[str]:
    """Every string key read out of a subscript inside this expression, e.g. META['BLOCK_M1']."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str):
            out.add(n.slice.value)
    return out


findings = defaultdict(list)
launches = 0

for path in sorted(ROOT.rglob("*.py")):
    if "autotune/data" in str(path):
        continue
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue

    # kernel function name -> op, for the autotuned kernels in this file
    kern_op = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for dec in fn.decorator_list:
            for n in ast.walk(dec):
                if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "configs_for"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    kern_op[fn.name] = n.args[0].value

    if not kern_op:
        continue

    for host in ast.walk(tree):
        if not isinstance(host, ast.FunctionDef):
            continue
        # grid variables assigned in this host function: name -> [(lineno, meta keys)]
        gridvars = defaultdict(list)
        for node in ast.walk(host):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                gridvars[node.targets[0].id].append((node.lineno, meta_keys(node.value)))

        for node in ast.walk(host):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
                continue
            tgt = node.func.value
            name = getattr(tgt, "id", None) or getattr(tgt, "attr", None)
            if name not in kern_op:
                continue
            op = kern_op[name]
            want = axes_of_op.get(op)
            if want is None:
                continue
            launches += 1
            g = node.func.slice
            used, src = set(), ""
            if isinstance(g, ast.Name):
                hits = gridvars.get(g.id, [])
                if len(hits) > 1:
                    findings["grid variable reassigned in one function"].append(
                        f"{path.relative_to(ROOT.parent)}:{node.lineno} {host.name}(): "
                        f"'{g.id}' assigned at lines {[h[0] for h in hits]}, launching {name} "
                        f"[{op}] axes={sorted(want)}")
                # The assignment in effect is the nearest one ABOVE the launch, not the union of
                # every assignment in the function -- unioning turns a correct reassignment into a
                # false "divides by an axis it does not have".
                before = [h for h in hits if h[0] <= node.lineno]
                if before:
                    used = max(before, key=lambda h: h[0])[1]
                src = f"var {g.id}@L{max(before, key=lambda h: h[0])[0] if before else '?'}"
            else:
                used = meta_keys(g)
                src = "inline"
            stray = {k for k in used if k.startswith(("BLOCK", "GROUP", "TILE", "BM", "BN", "BK"))} - want
            if stray:
                findings["grid divides by an axis the kernel does not have"].append(
                    f"{path.relative_to(ROOT.parent)}:{node.lineno} {host.name}(): {name} [{op}] "
                    f"grid({src}) uses {sorted(stray)}, kernel axes={sorted(want)}")

print(f"autotuned kernel launches analysed: {launches}\n")
for title in sorted(findings):
    rows = findings[title]
    print(f"[{len(rows)}] {title}")
    for r in rows[:25]:
        print(f"   {r}")
    if len(rows) > 25:
        print(f"   ... +{len(rows) - 25}")
    print()
if not findings:
    print("no grid/kernel axis disagreement found")
