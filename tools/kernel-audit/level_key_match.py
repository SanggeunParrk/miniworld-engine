"""Does each launch key its kernel with the function that kernel's LEVEL requires?

Three ways a shape key can be wrong, and each needed its own check:

  1. not keyed at all          -> a constexpr invisible to the cache      (key_gaps.py)
  2. the wrong SPACE           -> an edge index where a bucket value goes  (fixed; 12 sites)
  3. the wrong LEVEL           -> this one

token_key, atom_key and both_key are all bucket-value functions, so a space check cannot separate
them -- but they clamp to different sets. token_key tops out at 512, so a level=both kernel keyed
through it can never record 1024..8192: those buckets are unreachable, and if another caller uses
both_key the same kernel is written and read in two spaces. Found exactly that on
gated_projection_{,bwd_}gate_flat_triton, launched from bias_only_attention/triton/gate_out.py
through a token_key helper while conditioned_transition launched them with both_key.

Attribution is by the registry's (file, symbol) pair and by following the IMPORT that brought the
name into the calling file -- never by bare symbol. A bare-name version of this same check
reported 36 findings where there are 3, because `_attn_bwd_preprocess` names four different
kernels. That mistake has now been made three times in this repo; see launch_bind.py.

KNOWN LIMIT: imports are collected per FILE, not per scope. A file with several function-local
imports of one name from different modules (checks_attn.py does this four times) attributes them
all to whichever import comes last, which produces a false positive. Such a finding is reported
with a marker rather than silently dropped.
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launch_bind import SRC, _module_of, _resolve_import          # noqa: E402

FN_LEVEL = {"token_key": "token", "atom_key": "atom", "both_key": "both"}
REG = SRC / "miniworld_engine/kernels/registry.csv"


def audit() -> tuple[list, list, int]:
    reg = {r["kernel"]: r for r in csv.DictReader(REG.open())}
    defop = {}
    for k, r in reg.items():
        if r["backend"] != "triton":
            continue
        mod = str(Path(r["file"]).with_suffix("")).replace("/", ".")
        defop[(mod, r["symbol"].split(".")[-1])] = k

    trees, helper = [], {}
    for p in sorted(SRC.rglob("*.py")):
        try:
            t = ast.parse(p.read_text())
        except SyntaxError:
            continue
        trees.append((p, _module_of(p), t))
        for fn in ast.walk(t):
            if not isinstance(fn, ast.FunctionDef):
                continue
            lv = {getattr(n.func, "id", getattr(n.func, "attr", ""))
                  for n in ast.walk(fn) if isinstance(n, ast.Call)} & set(FN_LEVEL)
            if len(lv) == 1:                       # a helper that wraps exactly one level fn
                helper[(p.name, fn.name)] = FN_LEVEL[lv.pop()]

    bad, ambiguous, checked = set(), set(), 0
    for p, mod, tree in trees:
        origin, dupes = {}, set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                origin[fn.name] = (mod, fn.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            tgt = _resolve_import(node, mod)
            if not tgt:
                continue
            for al in node.names:
                nm = al.asname or al.name
                if nm in origin and origin[nm] != (tgt, al.name):
                    dupes.add(nm)                  # same name, two modules, one file
                origin[nm] = (tgt, al.name)
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Subscript):
                continue
            local = getattr(call.func.value, "id", getattr(call.func.value, "attr", None))
            op = defop.get(origin.get(local, (None, None)))
            if op is None:
                continue
            for kw in call.keywords:
                if kw.arg != "shape_key":
                    continue
                fns = {getattr(n.func, "id", getattr(n.func, "attr", ""))
                       for n in ast.walk(kw.value) if isinstance(n, ast.Call)}
                got = ({FN_LEVEL[f] for f in fns if f in FN_LEVEL}
                       | {helper[(p.name, f)] for f in fns if (p.name, f) in helper})
                if not got:
                    continue
                checked += 1
                if reg[op]["level"] not in got:
                    row = (op, reg[op]["level"], ",".join(sorted(got)),
                           f"{p.relative_to(SRC)}:{call.lineno}")
                    (ambiguous if local in dupes else bad).add(row)
    return sorted(bad), sorted(ambiguous), checked


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    bad, ambiguous, checked = audit()
    print(f"launches resolved and level-checked: {checked}")
    print(f"  keyed with the WRONG level: {len(bad)}")
    for op, lvl, got, where in bad:
        print(f"    {op:42s} level={lvl:5s} keyed-as={got:6s} {where}")
    if ambiguous:
        print(f"  undecidable (file re-imports the name from >1 module): {len(ambiguous)}")
        for op, lvl, got, where in ambiguous:
            print(f"    {op:42s} level={lvl:5s} keyed-as={got:6s} {where}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
