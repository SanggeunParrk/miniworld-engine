"""Every kernel launch in the package, checked against the kernel's real signature.

The failure this exists for is a launch that omits a parameter the kernel has no default for.
Triton reports it at launch as ``dynamic_func() missing 1 required positional argument: 'X'``,
naming the parameter but not the kernel or the call site, and a build logs it as a one-line
"skip" that reads like an unsupported shape. One such omission (shape_key on the canonical
ln-bwd path) killed the trimul backward at every L >= 548.

RESOLUTION, not name matching. A launch is attributed to a kernel by following the import that
brought the name into the calling file:

    same file            -> the definition there
    from .mod import K   -> mod's definition
    from .mod import K as J, then J[grid](...)  -> still mod's K

Matching on the bare name instead is what let the original bug through twice: four different
kernels are named ``_attn_fwd``, so a name-keyed table cross-contaminates their signatures, and
skipping aliased names (the safe-looking alternative) skips exactly the form the real bug took.

WHAT CANNOT BE DECIDED HERE, and is reported rather than hidden:
  starred   ``*q.stride()`` is one AST node that expands to N positionals at runtime, so
            positional coverage is unknown. Counting it as 1 reports every parameter after the
            star as missing.
  kwargs    a ``**kw`` forward can supply anything.
  computed  a callee that is a locally assigned variable, unless the assignment is a plain name.
These need a real launch to check; ``bucket_count.py`` over the build matrix and ``run_all`` over
the drivers do that. This tool prints how many launches fall in each bucket so the gap is a
number, not an assumption.
"""
from __future__ import annotations

import argparse
import ast
import collections
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PKG = "miniworld_engine"
#: The autotuner injects these; a caller never passes them.
INJECTED = ("BLOCK", "GROUP")


def _is_kernel(fn: ast.FunctionDef) -> bool:
    return any("jit" in ast.dump(d) or "autotune" in ast.dump(d) for d in fn.decorator_list)


def _signature(fn: ast.FunctionDef) -> tuple[list[str], set[str]]:
    a = fn.args
    ordered = [p.arg for p in a.posonlyargs + a.args]
    ndef = len(a.defaults)
    required = set(ordered[:-ndef] if ndef else ordered)
    required |= {k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=False) if d is None}
    return ordered, {r for r in required if not r.startswith(INJECTED)}


def _module_of(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import(node: ast.ImportFrom, this_module: str) -> str | None:
    """Absolute module name a ``from ... import`` refers to, or None if unresolvable."""
    if node.level:                       # relative
        base = this_module.split(".")
        up = node.level - 1
        base = base[:-1] if not up else base[:-1 - up]
        if up and len(base) < 0:
            return None
        return ".".join([*base, node.module]) if node.module else ".".join(base)
    return node.module


def collect() -> tuple[dict, list]:
    """(module, kernel name) -> (ordered params, required params), and every parsed tree."""
    kernels: dict[tuple[str, str], tuple[list[str], set[str]]] = {}
    trees = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        mod = _module_of(path)
        trees.append((path, mod, tree))
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and _is_kernel(fn):
                kernels[(mod, fn.name)] = _signature(fn)
    return kernels, trees


def audit() -> tuple[list, collections.Counter, int, list[tuple[str, int, str, str]]]:
    kernels, trees = collect()
    by_name = collections.defaultdict(list)
    for (mod, name) in kernels:
        by_name[name].append(mod)

    findings, skipped, checked = [], collections.Counter(), 0
    undecided: list[tuple[str, int, str, str]] = []
    for path, mod, tree in trees:
        # local name -> (defining module, original name)
        origin: dict[str, tuple[str, str]] = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and _is_kernel(fn):
                origin[fn.name] = (mod, fn.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = _resolve_import(node, mod)
                if target is None:
                    continue
                for al in node.names:
                    origin[al.asname or al.name] = (target, al.name)
        assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)
                    if not isinstance(n.value, ast.Name)}

        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Subscript):
                continue
            local = getattr(call.func.value, "id", getattr(call.func.value, "attr", None))
            if local is None:
                continue
            if local in assigned and local not in origin:
                skipped["not a kernel (dict/cache subscript)"] += 1
                continue
            sig = kernels.get(origin.get(local, (None, None)))
            if sig is None:
                mods = by_name.get(local, [])
                if not mods:
                    # `X[k](...)` is also how a dict of callables is used -- _CACHE[key](...),
                    # _BUILDERS[name](...). Not every subscript-then-call is a kernel launch, and
                    # counting them as unverified launches overstates the gap.
                    skipped["not a kernel (dict/cache subscript)"] += 1
                    continue
                if len(mods) != 1:
                    skipped["callee name ambiguous"] += 1
                    undecided.append((str(path.relative_to(SRC)), call.lineno, local,
                                      "callee name ambiguous"))
                    continue
                sig = kernels[(mods[0], local)]
            if any(k.arg is None for k in call.keywords):
                skipped["**kwargs forward"] += 1
                undecided.append((str(path.relative_to(SRC)), call.lineno, local, "**kwargs forward"))
                continue
            if any(isinstance(a, ast.Starred) for a in call.args):
                skipped["starred positional (*x.stride())"] += 1
                undecided.append((str(path.relative_to(SRC)), call.lineno, local, "starred positional (*x.stride())"))
                continue
            ordered, required = sig
            covered = set(ordered[:len(call.args)]) | {k.arg for k in call.keywords}
            checked += 1
            missing = sorted(required - covered)
            if missing:
                findings.append((str(path.relative_to(SRC)), call.lineno, local, missing))
    return findings, skipped, checked, undecided


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    findings, skipped, checked, undecided = audit()
    kernels, _ = collect()
    print(f"kernels defined: {len(kernels)}   launches statically checked: {checked}")
    print("launches NOT statically decidable (need a real launch):")
    for k, v in skipped.most_common():
        print(f"  {v:4d}  {k}")
    for p_, ln, n, why in sorted(undecided):
        print(f"        {p_}:{ln}  {n}  ({why})")
    print(f"\nlaunches missing a required parameter: {len(findings)}")
    for p, ln, n, miss in findings:
        print(f"  {p}:{ln}  {n}  omits {miss}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
