"""Static tiling audit. Three declarations, and only some disagreements are launch errors.

The autotuner injects EXACTLY the config CSV header columns, so:
  * header - kernel_constexprs  -> launch error ("unexpected keyword")
  * header must be identical across all 5 sets, or switching set changes kernel arity
  * axes.csv names must match the code, or the doc points at a knob that does not exist
"""
from __future__ import annotations
import ast, csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("src/miniworld_engine")
META = {"num_warps", "num_stages", "maxnreg"}

axes = defaultdict(dict)
with (ROOT / "autotune/axes.csv").open(newline="") as fh:
    for r in csv.DictReader(fh):
        if r.get("op"):
            axes[r["op"]][r["axis"]] = r

sets = sorted(p.name for p in Path("configs").iterdir() if p.is_dir() and p.name != "devices")
cfg = defaultdict(dict)
for s in sets:
    for p in sorted((Path("configs") / s).glob("*.csv")):
        rows = list(csv.DictReader(p.open(newline="")))
        if rows:
            cfg[p.stem][s] = {k: int(v) for k, v in rows[0].items() if k and k not in META and v}

src: dict[str, set[str]] = {}
where: dict[str, str] = {}
for path in sorted(ROOT.rglob("*.py")):
    if "autotune/data" in str(path):
        continue
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        continue
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        op = None
        for dec in fn.decorator_list:
            for node in ast.walk(dec):
                if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "configs_for"
                        and node.args and isinstance(node.args[0], ast.Constant)):
                    op = node.args[0].value
        if op is None:
            continue
        src[op] = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
        where[op] = f"{path.relative_to(ROOT.parent)}:{fn.lineno}"

print(f"ops: axes.csv {len(axes)}   config {len(cfg)}   source {len(src)}\n")

launch_err, arity, doc_name, doc_missing = [], [], [], []
for op in sorted(cfg):
    heads = {s: set(c) for s, c in cfg[op].items()}
    if len({frozenset(h) for h in heads.values()}) > 1:
        arity.append(f"{op}: " + "; ".join(f"{s}={sorted(h)}" for s, h in heads.items()))
    head = set().union(*heads.values())
    if op in src:
        bad = head - src[op]
        if bad:
            launch_err.append(f"{op} ({where[op]}): config names {sorted(bad)}, kernel takes none")
    else:
        launch_err.append(f"{op}: no @autotune(configs_for(...)) found in source")
    if op in axes:
        declared = set(axes[op])
        if declared - head:
            doc_name.append(f"{op:<44} axes.csv says {sorted(declared - head)}  code takes {sorted(head - declared) or 'same'}")
        elif head - declared:
            doc_missing.append(f"{op:<44} undocumented: {sorted(head - declared)}")

for title, rows in (
    ("LAUNCH ERROR - config column the kernel does not accept", launch_err),
    ("ARITY - config header differs between sets", arity),
    ("DOC - axes.csv axis name does not exist in the code", doc_name),
    ("DOC - tile axis missing from axes.csv", doc_missing),
):
    print(f"[{len(rows)}] {title}")
    for r in rows[:14]:
        print(f"   {r}")
    if len(rows) > 14:
        print(f"   ... +{len(rows) - 14} more")
    print()

print(f"[{len(set(cfg) - set(axes))}] ops with a config CSV but no axes.csv row")
for op in sorted(set(cfg) - set(axes)):
    print(f"   {op}")
