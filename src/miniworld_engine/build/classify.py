"""Classify each registered kernel as gemm / reduce / elem, by following the calls it inlines.

Reading only the kernel's own body is wrong and quietly so: a flash-attention `_attn_fwd` keeps its
`tl.dot`s in an inlined `@triton.jit` helper, and `layer_norm_fwd_kernel` does its reduction inside
a `__device__` helper -- both come back with NO signal at all, i.e. "elementwise", which is the
opposite of true. So the classifier walks the transitive closure of same-file callees (Triton
`@triton.jit` / `@triton.autotune` functions, CUDA `__device__` / `__global__` functions) and
classifies the union.

Categories, in priority order:
  gemm    -- a matrix multiply: tl.dot, a cuBLAS gemm call, or an MMA/WGMMA/tcgen05 atom
  reduce  -- no matmul, but a cross-lane or cross-block reduction: tl.sum/max/min, a warp shuffle,
             or an atomic accumulate
  elem    -- neither: per-element work, transposes, casts, gathers

`reduce` is reported separately rather than folded into `elem` because the two behave differently
under everything this audit measures: a reduction has an accumulation ORDER (so it is where tile
size changes the numbers, and where a missing barrier or a bad tail mask corrupts a whole row),
while genuinely elementwise work does not.
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

SRC = Path("src")
REG = SRC / "miniworld_engine/kernels/registry.csv"

GEMM = ((r"\btl\.dot\b", "tl.dot"), (r"cublas\w*Gemm", "cublasGemm"),
        (r"\btcgen05\b|\bwgmma\b|\bmma_atom\b|\bMmaOp\b|\bmake_mma\b|\bSM90_\d+x\d+x\d+_", "mma"))
REDUCE = ((r"\btl\.(sum|max|min)\b", "tl.reduce"), (r"__shfl\w*", "shfl"),
          (r"\batomic_add\b|\batomicAdd\b|\btl\.atomic_\w+", "atomic"))


def _py_bodies(text: str) -> dict[str, tuple[str, set[str]]]:
    """name -> (source, names it calls), for every function in a python file.

    Methods are recorded under BOTH ``method`` and ``Class.method``: six registry symbols are the
    dotted form (the cute kernels are classes whose entry point is ``.kernel``), and a bare-name
    lookup silently missed them -- one came back "elementwise" for a fused LayerNorm+GEMM.
    Attribute calls (``self.helper(...)``, ``cute.gemm(...)``) are followed by their attribute name
    too, since an inlined helper is usually reached that way.
    """
    out: dict[str, tuple[str, set[str]]] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    def record(fn, prefix=""):
        called = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    called.add(f.id)
                elif isinstance(f, ast.Attribute):
                    called.add(f.attr)
        src = ast.unparse(fn)
        out[fn.name] = (src, called)
        if prefix:
            out[f"{prefix}.{fn.name}"] = (src, called)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            record(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    record(sub, node.name)
    for fn in ast.walk(tree):          # nested defs the two passes above did not reach
        if isinstance(fn, ast.FunctionDef) and fn.name not in out:
            record(fn)
    return out


def _cu_bodies(text: str) -> dict[str, tuple[str, set[str]]]:
    """Same for a .cu/.cuh: split on function definitions found by a brace scan."""
    # This attribute has parentheses but is not the function's name/signature.
    text = re.sub(r"__launch_bounds__\s*\([^)]*\)", "", text)
    out: dict[str, tuple[str, set[str]]] = {}
    for m in re.finditer(r"(?:__global__|__device__|__inline__|\bvoid\b|\bstd::vector<[^>]+>|"
                         r"\btorch::Tensor\b)[^;{]*?\b(\w+)\s*\(", text):
        name = m.group(1)
        i = text.find("{", m.end())
        if i < 0:
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[m.start():j + 1]
        called = set(re.findall(r"\b(\w+)\s*(?:<[^;()]*>)?\s*\(", body))
        prev = out.get(name)
        if prev is None or len(body) > len(prev[0]):
            out[name] = (body, called)
    return out


def closure(bodies: dict[str, tuple[str, set[str]]], root: str) -> str:
    """Concatenated source of `root` and everything it transitively calls in this file."""
    if root not in bodies:
        return ""
    seen, stack, parts = {root}, [root], []
    while stack:
        name = stack.pop()
        src, called = bodies[name]
        parts.append(src)
        for c in called:
            if c in bodies and c not in seen:
                seen.add(c)
                stack.append(c)
    return "\n".join(parts)


def source_closure(path: Path, symbol: str, seen: set | None = None) -> str:
    """Follow local calls and explicit imported JIT-body aliases without importing code."""
    seen = set() if seen is None else seen
    identity = (path.resolve(), symbol)
    if identity in seen or not path.is_file():
        return ""
    seen.add(identity)
    text = path.read_text()
    if path.suffix != ".py":
        return closure(_cu_bodies(text), symbol)
    bodies = _py_bodies(text)
    source = closure(bodies, symbol)
    if not source:
        return ""
    tree = ast.parse(text)
    imports = {}
    package_root = next((p.parent for p in path.parents if p.name == "miniworld_engine"), None)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            else:
                if package_root is None:
                    continue
                base = package_root
            module = base / (node.module or "").replace(".", "/")
            for alias in node.names:
                imports[alias.asname or alias.name] = module / (alias.name + ".py")
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not re.search(rf"\b{re.escape(name)}\s*\(", source):
            continue
        value = node.value
        # Triton Autotuner.fn is the underlying shared JIT function.
        while isinstance(value, ast.Attribute) and value.attr == "fn":
            value = value.value
        if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                and value.value.id in imports):
            dependency = source_closure(imports[value.value.id], value.attr, seen)
            source += "\n" + dependency
    return source


def classify(path: Path, symbol: str) -> tuple[str, str, str]:
    """(kind, signals, how) for one kernel."""
    if not path.is_file():
        return "?", "", "file missing"
    text = path.read_text()
    src = source_closure(path, symbol)
    how = "closure"
    if not src:
        src, how = text, "WHOLE FILE (symbol not found)"
    sig = [n for p, n in GEMM if re.search(p, src)] + [n for p, n in REDUCE if re.search(p, src)]
    gemm = {"tl.dot", "cublasGemm", "mma"} & set(sig)
    red = {"tl.reduce", "shfl", "atomic"} & set(sig)
    kind = "gemm" if gemm else ("reduce" if red else "elem")
    return kind, ",".join(sig), how


def main() -> int:
    rows = list(csv.DictReader(REG.open()))
    import collections
    counts, hows, out = collections.Counter(), collections.Counter(), {}
    for r in rows:
        kind, sig, how = classify(SRC / r["file"], r["symbol"])
        out[r["kernel"]] = (kind, sig, how)
        counts[kind] += 1
        hows[how] += 1
    print(f"{len(rows)} kernels   {dict(counts)}")
    print(f"resolution: {dict(hows)}")
    bad = [k for k, v in out.items() if v[2] != "closure"]
    if bad:
        print(f"\nsymbol not resolved, classified on the whole file ({len(bad)}):")
        for k in bad:
            print(f"   {k:<48} {out[k][0]:<7} [{out[k][1]}]")
    if "--list" in sys.argv:
        for kind in ("gemm", "reduce", "elem", "?"):
            ks = sorted(k for k, v in out.items() if v[0] == kind)
            if ks:
                print(f"\n=== {kind} ({len(ks)})")
                for k in ks:
                    print(f"   {k:<48} [{out[k][1]}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
