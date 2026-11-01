"""The rest of the code graph: Triton kernels, native ops and the derivation, with the files each one's evidence hangs on.

``kernels/_codegraph.py`` keys what nvcc compiles. Three more things are invalidated by an edit to a source file:

* a **Triton kernel**: its tuned cache entries (``autotune/data/<op>/<gpu>.json``) are stale when the ``@triton.jit`` body, its
  ``key=[...]`` or a JIT helper it calls changes (``cache.op_identity`` / ``implementation_identity``). The graph of a kernel is
  its registry file, the files of the JIT helpers it imports (followed through ``import`` / ``from ... import``, only into files
  that contain ``triton.jit``) and its registry driver, which decides what gets built;
* a **native op** (``autotune.native.op_identity``): the closure of the op's registry row (``autotune.identity``) and the safety
  set -- its ``*_cuda`` cache is restamped when, and only when, a file of that closure moves;
* the **derivation** (``registry_kernel.units.json``, schema 2): per unit, the files that ran while it was recorded -- ``dev derive``
  re-runs exactly the units a changed file reaches.

What an edit does to each is read from the committed evidence (``autotune.cache_status.scan``, ``plan.status``), so
``dev code-graph --affected FILE`` answers "which Triton caches go stale, which native op is owed a restamp, how many derivation
units re-run", CPU only.
"""

from __future__ import annotations

import ast
import csv
import functools
import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG.parent
REGISTRY = PKG / "kernels" / "registry" / "registry.csv"
TRITON, NATIVE, DERIVATION = "triton", "native-identity", "derivation"


def py_digest(path: Path) -> str:
    """A Python file's digest through its syntax tree: a comment or formatting edit keeps it, any statement edit moves it."""
    try:
        return hashlib.sha256(ast.dump(ast.parse(path.read_text(encoding="utf-8"))).encode()).hexdigest()
    except (OSError, SyntaxError):
        return raw_digest(path)


def raw_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def node_digest(path: Path, kind: str) -> str:
    """The digest ``kind`` records its nodes with: a Triton kernel's files by syntax tree; a native op's files the way its identity
    hashes them (``.py`` by bytes, ``.cu`` normalized); the derivation's by the token its units store."""
    from miniworld_engine.autotune import identity

    if kind == TRITON:
        return py_digest(path)
    try:
        rel = identity.rel(path)
    except ValueError:
        return raw_digest(path)
    if kind == NATIVE:
        return identity._native_token(rel)
    return identity.file_token(rel)

def label(path: Path) -> str | None:
    try:
        return f"pkg:{Path(path).resolve().relative_to(PKG).as_posix()}"
    except ValueError:
        return None


def _module_file(dotted: str) -> Path | None:
    parts = dotted.split(".")
    if parts[0] != "miniworld_engine":
        return None
    base = PKG.joinpath(*parts[1:])
    candidates = [base.with_name(base.name + ".py"), base / "__init__.py"] if parts[1:] else [PKG / "__init__.py"]
    return next((c for c in candidates if c.is_file()), None)


@functools.cache
def _imports(path: Path) -> tuple[Path, ...]:
    """The package files ``path`` imports, anywhere in the file (a function-level import is a dependency too)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ()
    here = ["miniworld_engine", *path.relative_to(PKG).parent.parts]
    found: list[Path] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - (node.level - 1)]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            modules = [module, *(f"{module}.{a.name}" for a in node.names)]
        else:
            continue
        found.extend(p for p in map(_module_file, modules) if p is not None)
    return tuple(dict.fromkeys(found))


@functools.cache
def _is_jit_file(path: Path) -> bool:
    try:
        return "triton.jit" in path.read_text(encoding="utf-8")
    except OSError:
        return False


def kernel_files(kernel_file: Path, driver: str = "") -> list[Path]:
    """A Triton kernel's graph: its file, the JIT-helper files it reaches, and its registry driver's file."""
    seen, queue = [kernel_file], [kernel_file]
    while queue:
        for target in _imports(queue.pop(0)):
            if target not in seen and _is_jit_file(target):
                seen.append(target)
                queue.append(target)
    module = _module_file(driver.split(":")[0]) if driver else None
    if module is not None and module not in seen:
        seen.append(module)
    return seen


def _artifact(name: str, kind: str, files: list[Path], key: str | None = None) -> dict:
    nodes = [[label(p), node_digest(p, kind)] for p in files if p.is_file()]
    key = key or hashlib.sha256("".join(d for _, d in nodes).encode()).hexdigest()[:32]
    return {"name": name, "key": key, "where": "checkout", "material": {"kind": kind, "nodes": nodes}}


def derivation_files() -> list[Path]:
    """The files the committed derivation depends on: the union of every unit's recorded file set (schema 2)."""
    from miniworld_engine.autotune import plan

    try:
        data = json.loads(plan.EVIDENCE.read_text())
    except (OSError, ValueError):
        return []
    return [PKG / p for p in data.get("file_index", [])]


@functools.lru_cache(maxsize=1)
def artifacts() -> tuple[dict, ...]:
    """One artifact per registry Triton kernel and per native op, plus the derivation. Computed from this checkout, no GPU, no
    import of a kernel."""
    from miniworld_engine.autotune import identity, native

    out = []
    with REGISTRY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            file = SRC / row["file"] if row["file"].endswith(".py") else None
            if row["backend"] == "triton" and file is not None and file.is_file():
                out.append(_artifact(f"triton:{row['kernel']}", TRITON, kernel_files(file, row.get("driver", ""))))
    out.extend(_artifact(f"native:{op}", NATIVE, [PKG / p for p in identity.native_files(op)], key=identity.native_identity(op)[:32])
               for op in sorted(native.BUILD_OPS))
    out.append(_artifact("autotune:derivation", DERIVATION, derivation_files()))
    return tuple(out)


def _state(artifact: dict, rows) -> str:
    kind = artifact["material"]["kind"]
    if kind == TRITON:
        op = artifact["name"].removeprefix("triton:")
        stale = [r for r in rows if r.op == op and r.stale]
        if stale:
            return f"cache STALE now: {stale[0].reason} ({', '.join(sorted({r.gpu for r in stale}))})"
        return "tuned cache current" if any(r.op == op for r in rows) else "no committed cache"
    if kind == NATIVE:
        op = artifact["name"].removeprefix("native:")
        stale = [r for r in rows if r.op == op and r.stale and r.reason.startswith("native implementation changed")]
        return f"restamp {op}: restamp tool" if stale else "native cache current"
    from miniworld_engine.autotune import plan

    status = plan.status()
    n = len(status["stale"])
    return f"dev derive owed: {n} of {status['units']} units re-run (the others are reused)" if n else "derivation current"


def affected(path: Path) -> list[tuple[dict, str]]:
    """``(artifact, what the edit does to it)`` for every Triton kernel / native op / the derivation whose graph holds ``path``."""
    where = label(path)
    hits = [a for a in artifacts() if where and any(n[0] == where for n in a["material"]["nodes"])]
    if not hits:
        return []
    from miniworld_engine.autotune import cache_status

    rows = cache_status.scan()
    return [(a, _state(a, rows)) for a in hits]
