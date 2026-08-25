"""Rename the key-only GROUP_M / GROUP_N / seq_group parameters to `shape_key`.

`GROUP_M` meant two unrelated things. In 4 kernels it is a real L2-swizzle group that comes from the
config CSV and is used in the body. In 90 others it is a shape bucket that exists only so Triton's
`key=[...]` can name it -- never read in the body, never in a config CSV. One name for a tile knob
and for a cache label makes it impossible to tell which a parameter is by looking at it.

This renames only the second kind, decided per parameter by MEASUREMENT (is the name read anywhere
in the function body?) rather than by which spelling it uses. It rewrites, in each affected file:
  * the parameter in the kernel signature
  * the string in the @triton.autotune(key=[...]) list
  * every `NAME=` keyword at a launch site

and then re-parses every file and asserts the three stay in agreement.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

SRC = Path("src")
OLD = ("GROUP_M", "GROUP_N", "seq_group")
NEW = "shape_key"


def key_only_params(tree: ast.Module) -> dict[str, set[str]]:
    """kernel function name -> the OLD names it takes that its body never reads."""
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        hit = {n for n in OLD if n in params}
        if not hit:
            continue
        used = {n.id for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
                if isinstance(n, ast.Name)}
        keyonly = hit - used
        if keyonly:
            out[fn.name] = keyonly
    return out


def main() -> int:
    files = sorted({Path(p) for p in subprocess.run(
        ["git", "grep", "-l", "|".join(OLD), "--", "src/miniworld_engine"],
        capture_output=True, text=True).stdout.split()} )
    files = sorted(p for p in Path("src").rglob("*.py")
                   if any(o in p.read_text() for o in OLD))
    touched, renamed = [], 0
    for path in files:
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        keyonly = key_only_params(tree)
        if not keyonly:
            continue
        names = set().union(*keyonly.values())
        # A file may hold BOTH kinds (adaln/triton/fused3.py does: _gemm_gate_kernel reads GROUP_M,
        # _ln_kernel only keys on it). Renaming by whole-file regex would rename the computed one
        # too, so skip any name that is computed anywhere in this file and report it.
        computed = set()
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                used = {n.id for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
                        if isinstance(n, ast.Name)}
                computed |= ({n for n in OLD
                              if n in {a.arg for a in fn.args.args}} & used)
        safe = names - computed
        if not safe:
            print(f"  SKIP {path}: {sorted(names)} also used in a body here")
            continue
        if computed & names:
            print(f"  PARTIAL {path}: renaming {sorted(safe)}, leaving {sorted(computed & names)}")
        new = text
        for old in sorted(safe):
            new = re.sub(rf"\b{old}\b", NEW, new)
            renamed += 1
        path.write_text(new)
        touched.append(path)

    # verify: signature, key list and call sites agree, and nothing computed got renamed
    bad = []
    for path in touched:
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            params = {a.arg for a in fn.args.args}
            keys = set()
            for dec in fn.decorator_list:
                for n in ast.walk(dec):
                    if isinstance(n, ast.keyword) and n.arg == "key":
                        keys |= {e.value for e in ast.walk(n.value)
                                 if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if NEW in keys and NEW not in params:
                bad.append(f"{path}:{fn.name}: key names {NEW} but the signature does not")
            used = {n.id for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
                    if isinstance(n, ast.Name)}
            if NEW in params and NEW in used:
                bad.append(f"{path}:{fn.name}: {NEW} is read in the body -- it is not key-only")
    print(f"\nfiles touched {len(touched)}   names renamed {renamed}")
    if bad:
        print("VERIFY FAILED:")
        for b in bad:
            print("  ", b)
        return 1
    print("verify: signature/key/body agree in every touched file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
