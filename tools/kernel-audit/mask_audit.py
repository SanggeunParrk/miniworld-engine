"""Exhaustive static audit: is the tiled axis bounded inside the loop that walks it?

A Triton GEMM/reduce loop walks one axis in BLOCK steps::

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(w_ptrs), acc)

The row mask does not bound the contraction axis, so when ``K % BLOCK_K != 0`` the last trip reads
past the end of the row and multiplies the garbage into the accumulator. Checking for the presence
of ``mask=`` therefore proves nothing -- an earlier version of this audit passed the kernel above.

The test used here: the loop declares its own extent (``range(0, K, BLOCK_K)`` -> ``K``). A
correctly bounded loop has to compare *something* against that extent, in the loop body, on every
trip -- ``mask=offs_k[None, :] < K - k`` or ``boundary_check`` on the advanced axis. A loop whose
body never mentions its extent at all cannot be masking the axis it walks.

``boundary_check`` on a block pointer counts as a bound for the axes it lists; a 2-D block pointer
with ``boundary_check=(0, 1)`` is fully bounded and is not reported.
"""
from __future__ import annotations
import ast
import csv
import re
from pathlib import Path

import os
ROOT = Path(os.environ.get("MASK_AUDIT_ROOT", "src/miniworld_engine"))
META = {"num_warps", "num_stages", "maxnreg"}

axes_of_op = {}
for p in sorted(Path("configs/accuracy").glob("*.csv")):
    rows = list(csv.DictReader(p.open(newline="")))
    if rows:
        axes_of_op[p.stem] = {k for k in rows[0] if k and k not in META}


def fname(call: ast.Call) -> str:
    return getattr(call.func, "attr", None) or getattr(call.func, "id", "") or ""


def names(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


results = []
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
            for n in ast.walk(dec):
                if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "configs_for"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    op = n.args[0].value
        if op is None:
            continue
        for loop in [n for n in ast.walk(fn) if isinstance(n, ast.For)]:
            it = loop.iter
            # tl.static_range is a tiled loop too -- an earlier version matched only "range" and
            # missed back.py's two static_range(0, N, BLOCK_N) loops. And the step is not always
            # named BLOCK_*: the attention preprocess kernels step HEAD_DIM by HEAD_DIM_PAD, a
            # caller-supplied constexpr, so filtering on the name "BLOCK" dropped those as well.
            if not (isinstance(it, ast.Call) and fname(it) in ("range", "static_range")
                    and len(it.args) == 3):
                continue
            step = ast.unparse(it.args[2])
            # a stride-by-program-count persistent loop (range(pid, num_tiles, NUM_PROGRAMS))
            # walks tile INDICES, not an extent, and the range itself bounds it
            if step.isupper() and step in {"G", "NP"} or not step[:1].isalpha():
                continue
            extent = ast.unparse(it.args[1])
            ext_names = names(it.args[1]) or {extent}

            body = ast.Module(body=loop.body, type_ignores=[])
            # does any comparison in the body mention the extent?
            bounded_by_cmp = any(
                ext_names & names(c)
                for c in ast.walk(body) if isinstance(c, ast.Compare))

            # Divisibility guards, same rule as the store pass below: a load under
            # `if EVEN_K:` needs no bound on K, because that branch only compiles when the
            # tile divides the extent. Without this the pass reports the *fixed* form of
            # baseline_dtv1.py -- which branches on EVEN_K and checks both axes in the else --
            # as still unbounded.
            even = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                    if a.annotation is not None and "constexpr" in ast.unparse(a.annotation)
                    and a.arg.startswith("EVEN")}
            for a in ast.walk(fn):
                if (isinstance(a, ast.Assign) and len(a.targets) == 1
                        and isinstance(a.targets[0], ast.Name)
                        and any(isinstance(o, ast.Mod) for o in ast.walk(a.value))):
                    even.add(a.targets[0].id)
            for node in ast.walk(body):
                if not isinstance(node, ast.If):
                    continue
                for sub in node.body + node.orelse:
                    for n in ast.walk(sub):
                        if isinstance(n, ast.Call) and fname(n) == "load":
                            n._guard = getattr(n, "_guard", set()) | names(node.test)

            loads, bc_axes, unmasked, all_ok = 0, set(), 0, True
            for n in ast.walk(body):
                if isinstance(n, ast.Call) and fname(n) == "load":
                    loads += 1
                    kw = {k.arg: k.value for k in n.keywords if k.arg}
                    full_bc = ("boundary_check" in kw
                               and len(re.findall(r"\d", ast.unparse(kw["boundary_check"]))) >= 2)
                    if "boundary_check" in kw:
                        bc_axes |= {ast.unparse(kw["boundary_check"])}
                    elif "mask" not in kw:
                        unmasked += 1
                    if not (full_bc or (getattr(n, "_guard", set()) & even)):
                        all_ok = False
            # a block pointer listing every axis it advances is bounded by triton itself
            bounded_by_bc = (bool(bc_axes) and all(
                len(re.findall(r"\d", a)) >= 2 for a in bc_axes)) or (loads > 0 and all_ok)
            results.append(dict(
                op=op, kernel=fn.name, file=str(path.relative_to(ROOT.parent)),
                line=loop.lineno, extent=extent, step=step, loads=loads,
                unmasked=unmasked, bc=sorted(bc_axes),
                verdict=("bounded" if (bounded_by_cmp or bounded_by_bc) else "UNBOUNDED")))

bad = [r for r in results if r["verdict"] == "UNBOUNDED"]
ops_bad = sorted({r["op"] for r in bad})
print(f"BLOCK-stepped loops examined: {len(results)}   in {len({r['op'] for r in results})} ops")
print(f"loops whose body never bounds the axis they walk: {len(bad)}   in {len(ops_bad)} ops\n")
for r in bad:
    print(f"  {r['op']}")
    print(f"     {r['file']}:{r['line']} {r['kernel']}  for _ in range(0, {r['extent']}, {r['step']})")
    print(f"     loads in loop {r['loads']}  of which no mask at all {r['unmasked']}"
          + (f"  boundary_check={r['bc']}" if r["bc"] else ""))

# ── stores ───────────────────────────────────────────────────────────────────────────────────
# A tl.store writing BLOCK_N columns into a row narrower than BLOCK_N spills into the NEXT row.
# The load audit above cannot see this: the defect is on the write side and does not need to be
# inside a tiled loop.
#
# Two things have to be right or this pass is noise. `mask` is tl.store's third POSITIONAL
# argument, so a keyword-only check misses `tl.store(p, v, m)`. And a store inside
# `if EVEN_N and EVEN_D:` needs no mask at all -- the kernel dispatches on compile-time flags and
# masks in the sibling branches, so treating those as defects reported 5 false positives in
# triangle_attention/triton/atomic.py, whose ragged run measured clean.
store_bad, store_guarded = [], 0
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
            for n in ast.walk(dec):
                if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "configs_for"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    op = n.args[0].value
        if op is None:
            continue
        # An evenness guard is one whose condition is about divisibility, not just any
        # compile-time flag. Excusing a store because *some* constexpr guards it wrongly excused
        # baseline_dtv1.py:350 (guarded by a save-activations flag). Two ways a name qualifies:
        # it is assigned in this function from an expression containing `%`
        # (EVEN_N = (N_CTX % BLOCK_M1 == 0) & ...), or it is a constexpr parameter spelled EVEN_*,
        # which is how the caller passes such a flag in.
        even = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                if a.annotation is not None and "constexpr" in ast.unparse(a.annotation)
                and a.arg.startswith("EVEN")}
        for a in ast.walk(fn):
            if (isinstance(a, ast.Assign) and len(a.targets) == 1
                    and isinstance(a.targets[0], ast.Name)
                    and any(isinstance(o, ast.Mod) for o in ast.walk(a.value))):
                even.add(a.targets[0].id)
        # every store, with the chain of `if` tests enclosing it
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            for sub in node.body + node.orelse:
                for n in ast.walk(sub):
                    if isinstance(n, ast.Call) and fname(n) == "store":
                        n._guard = getattr(n, "_guard", set()) | names(node.test)
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and fname(n) == "store"):
                continue
            if len(n.args) >= 3:                      # tl.store(ptr, value, mask) positional
                continue
            kw = {k.arg: k.value for k in n.keywords if k.arg}
            if "mask" in kw:
                continue
            if getattr(n, "_guard", set()) & even:
                store_guarded += 1
                continue
            if "boundary_check" in kw:
                axes = ast.unparse(kw["boundary_check"])
                if sum(ch.isdigit() for ch in axes) >= 2:
                    continue
                why = f"boundary_check={axes} on a multi-axis tile"
            else:
                why = "no mask, no boundary_check"
            store_bad.append((op, f"{path.relative_to(ROOT.parent)}:{n.lineno}", fn.name, why))

print(f"\nstores guarded by a divisibility condition (correct, not reported): {store_guarded}")
print(f"[{len(store_bad)}] store that does not bound every axis it writes")
for op, where, kern, why in store_bad:
    print(f"   {op}")
    print(f"      {where} {kern}  --  {why}")
