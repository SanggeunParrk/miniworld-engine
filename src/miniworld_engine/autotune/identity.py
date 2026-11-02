"""Content identities read off the code graph: what an artifact depends on, and a digest of exactly that.

One rule replaces the hand-made global hashes (``native.source_identity`` over every ``kernels/**`` file, ``plan.source_identity``
over ``modules/``, ``kernels/``, ``baselines/``, ``utils/``): an artifact stores the digest of the part of the graph it depends on, and is
stale iff that digest differs from the current one. Nothing else is judged stale, and nothing that did not change is redone.

Two artifacts use it:

* a **native cache** (``autotune/data/<op>_cuda/<gpu>.json``): :func:`native_identity` -- the op's registry row, the closure of its
  files (kernel file, driver, every python file naming the kernel file, then python imports -- lazy ones too, ``#include`` edges, and
  the paths / dotted module names a source spells out) and the safety set;
* a **derivation unit** (``registry_kernel.units.json`` schema 2): :func:`unit_files` -- the files whose code ran while the unit was
  recorded (``sys.monitoring``, every thread), the data files it opened, the files of every kernel it launched, and the safety set.

The graph must never under-approximate, so wherever a dependency class cannot be proven covered it is included (the safety set; the
python files of a package ``__init__`` chain; every file a source names). It may over-approximate: that costs a redo, never a wrong
answer. The residual under-approximations (string dispatch the closure cannot see, toolchain versions) are listed in
docs/records/graph_identities_20261101.md.

Digests: a ``.py`` is hashed through its syntax tree *and* its bytes (:func:`file_token`): bytes equal means unchanged on any Python,
the tree equal means a comment / formatting edit under the same Python. A ``.cu`` / ``.cuh`` is hashed after the compiler-level
normalization of ``kernels/_codegraph`` (comments, whitespace, ``#if 0`` removed). The native identity hashes a ``.py`` by bytes (one
string must be equal on every Python a wheel runs on), a ``.cu`` by that normalization.
"""

from __future__ import annotations

import ast
import csv
import functools
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

PKG = Path(__file__).resolve().parents[1]
SCHEME = 1                                  # bumped when what an identity covers changes shape: every stored identity then differs
REGISTRY = "kernels/registry/registry.csv"
CODE = {".py", ".cu", ".cuh", ".inc", ".h", ".cpp"}
CCODE = CODE - {".py"}
_SKIP_DIRS = {"notes", "gen", "__pycache__", "plans", "data"}      # not shipped / outputs, never an input (autotune/data is measured FROM the code)

#: Python files every derivation unit (and every native op) depends on: dispatch inputs, the cache / key / policy layer, the
#: builder and the derivation itself, and everything ``native.source_identity`` hashed beside the kernels. Hashed whole.
SAFETY_PY = (
    "settings.py", "modules/dispatch.py",
    "autotune/cache.py", "autotune/shape_key.py", "autotune/policy.py", "autotune/builder.py", "autotune/derive.py",
    "autotune/checkpoint_cases.py", "autotune/module_registry.py", "autotune/native.py", "autotune/fused_config.py",
    "autotune/hopper_cuda_config.py", "autotune/native_history.py",
)
#: The measurement path of a native op: how a candidate is built, timed and judged. Outside ``kernels/``, hashed into every NATIVE
#: identity (an edit to how a candidate is timed can change what the cache claims) but not into a derivation unit (which launches
#: nothing and times nothing).
MEASURE_PY = ("autotune/capture.py", "autotune/native_compile.py", "autotune/configs.py", "autotune/buckets.py")
#: Files outside ``kernels/`` a native closure never follows into: the tools that READ identities (an edit to the rule is a
#: ``SCHEME`` bump, not a kernel edit), the CLI, the build / viz tooling. ``modules/`` files are members when a closure file
#: names them, but not followed (the module layer is a consumer of the kernels, not a part of them).
TOOLING = frozenset({"cli.py", "autotune/identity.py", "autotune/plan.py", "autotune/run_all.py", "autotune/preflight.py",
                     "autotune/cache_backfill.py", "autotune/space_backfill.py", "autotune/cache_status.py"})
SAFETY_DATA = ("kernels/registry/registry.csv", "kernels/registry/registry_module.csv")
SAFETY_GLOBS = ("build/gpu_to_kernels/*.csv",)

_PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|cu|cuh|inc|h|cpp|csv))\b")
_DOTTED_RE = re.compile(r"\bminiworld_engine(?:\.\w+)+")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]', re.MULTILINE)


def rel(path: Path | str) -> str:
    return PurePosixPath(Path(path).resolve().relative_to(PKG)).as_posix()


# ----------------------------------------------------------------------------------------------------------------- digests
@functools.cache
def _raw(path: str, mtime: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.cache
def _norm(path: str, mtime: int, size: int) -> str:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix
    if suffix == ".py":
        try:
            return hashlib.sha256(ast.dump(ast.parse(data)).encode()).hexdigest()
        except (SyntaxError, ValueError):
            pass
    elif suffix in CCODE:
        from miniworld_engine.kernels import _codegraph as cg

        return cg.parse_text(data.decode("utf-8", errors="replace")).digest
    return _raw(path, mtime, size)


def _stat(relpath: str):
    try:
        st = (PKG / relpath).stat()
    except OSError:
        return None
    return str(PKG / relpath), st.st_mtime_ns, st.st_size


DIR = "dir:"


def _listing(relpath: str) -> str:
    """Digest of the NAMES in a package directory (``dir:<rel>``): what a ``listdir`` / ``glob`` / ``scandir`` saw. ``""`` when it
    is gone. Byte-code caches are not part of it."""
    try:
        names = sorted(n for n in os.listdir(PKG / relpath.removeprefix(DIR)) if n != "__pycache__" and not n.endswith((".pyc", ".pyo")))
    except OSError:
        return ""
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def digests(relpath: str) -> tuple[str, str]:
    """``(bytes digest, normalized digest)`` of a package file; ``("", "")`` when it does not exist. A ``dir:<rel>`` entry is the
    listing of a directory (a data directory a unit globbed: a new file in it changes the unit's inputs)."""
    if relpath.startswith(DIR):
        listing = _listing(relpath)
        return listing, listing
    key = _stat(relpath)
    return ("", "") if key is None else (_raw(*key), _norm(*key))


def norm_scheme() -> str:
    """What the normalized half of a :func:`file_token` means: the AST of this Python, the include normalization of ``_codegraph``."""
    from miniworld_engine.kernels import _codegraph as cg

    return f"ast-py{sys.version_info[0]}.{sys.version_info[1]}-cg{cg.KEY_SCHEME}-v{SCHEME}"


def file_token(relpath: str) -> str:
    raw, norm = digests(relpath)
    return f"{raw[:20]}:{norm[:20]}" if raw else ""


def token_matches(relpath: str, stored: str, stored_scheme: str) -> bool:
    """Bytes equal -> unchanged on any Python; else the normalized halves, only where both were made by the same scheme."""
    if relpath.startswith(DIR):
        return bool(stored) and _listing(relpath)[:20] == stored.partition(":")[0]
    key = _stat(relpath)
    if key is None or not stored:
        return False
    s_raw, _, s_norm = stored.partition(":")
    if _raw(*key)[:20] == s_raw:
        return True
    return stored_scheme == norm_scheme() and _norm(*key)[:20] == s_norm


def tokens(relpaths: Iterable[str]) -> dict[str, str]:
    return {p: file_token(p) for p in sorted(set(relpaths))}


# ----------------------------------------------------------------------------------------------------------- the file index
@functools.lru_cache(maxsize=1)
def _index() -> tuple[frozenset[str], dict[str, tuple[str, ...]]]:
    paths = []
    for p in PKG.rglob("*"):
        if p.suffix in CODE | {".csv"} and p.is_file():
            r = p.relative_to(PKG)
            if not _SKIP_DIRS & set(r.parts):
                paths.append(PurePosixPath(r).as_posix())
    by_base: dict[str, list[str]] = {}
    for p in paths:
        by_base.setdefault(PurePosixPath(p).name, []).append(p)
    return frozenset(paths), {k: tuple(v) for k, v in by_base.items()}


def refresh() -> None:
    """Forget the file index and every cached closure (a test that writes files, a long-lived process after a checkout)."""
    _index.cache_clear()
    for cached in (_deps, _imports, _dynamic_split, outside_files, env_names, _launchers, _op_files, _registry_text, _raw, _norm, tree_digest, _is_jit, launch_files):
        cached.cache_clear()
    native_identity.cache_clear()


def module_file(dotted: str) -> str | None:
    """``miniworld_engine.a.b.c`` -> the package file of the longest prefix that is a module or package."""
    parts = dotted.split(".")
    paths = _index()[0]
    for n in range(len(parts), 1, -1):
        base = "/".join(parts[1:n])
        for cand in (f"{base}.py", f"{base}/__init__.py"):
            if cand in paths:
                return cand
    return "__init__.py" if "__init__.py" in paths and len(parts) == 1 else None


def _package_chain(relpath: str) -> list[str]:
    """The ``__init__.py`` files that run when ``relpath`` is imported (every parent package)."""
    paths = _index()[0]
    parts = PurePosixPath(relpath).parts[:-1]
    out = [f"{'/'.join(parts[:n])}/__init__.py" if n else "__init__.py" for n in range(len(parts) + 1)]
    return [p for p in out if p in paths and p != relpath]
@functools.cache
def _imports(relpath: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(strict, lazy)``: the package files a python file imports at module level (they run when it is imported), and the ones only
    a function body imports (they run when the function is called). Both include the ``__init__`` chain of each module."""
    path = PKG / relpath
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return frozenset(), frozenset()
    pkg_here = ["miniworld_engine", *PurePosixPath(relpath).parent.parts]
    strict: set[str] = set()
    lazy: set[str] = set()
    stack = [(tree, False)]
    while stack:
        node, inside = stack.pop()
        inner = inside or isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        stack.extend((child, inner) for child in ast.iter_child_nodes(node))
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_here[: len(pkg_here) - (node.level - 1)]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            names = [module, *(f"{module}.{a.name}" for a in node.names)]
        else:
            continue
        for name in names:
            hit = module_file(name) if name.startswith("miniworld_engine") else None
            if hit:
                (lazy if inside else strict).update([hit, *_package_chain(hit)])
    strict.discard(relpath)
    lazy.discard(relpath)
    return frozenset(strict), frozenset(lazy - strict)


_IMPORT_CALLS = {"import_module", "__import__", "find_spec"}
_REL_STRING = re.compile(r"^\.+(?:\w+(?:\.\w+)*)?$")


def _leading_text(node: ast.AST) -> tuple[str, bool]:
    """``(text, complete)``: the leading constant text of a string expression and whether it is the whole value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.JoinedStr):
        out = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out += part.value
            else:
                return out, False
        return out, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mod):
        return _leading_text(node.left)[0], False
    return "", False


def dynamic_import_specs(tree: ast.AST, *, is_package_init: bool = False) -> list[tuple[str, str]]:
    """The imports a source performs by NAME at run time, as ``(kind, text)``; the one place that knows how dispatch is spelled.

    * ``("exact", "a.b")`` / ``("exact", ".rel.name")`` -- ``import_module("a.b")``, ``__import__``, ``find_spec``;
    * ``("prefix", "a.b.")`` -- ``import_module(f"a.b.{x}")``: any module whose dotted name starts so;
    * ``("opaque", "")`` -- ``import_module(name)``: no text to go by, so every module under the file's own package;
    * ``("relative", ".x.y")`` -- a string literal that is a relative module name (a lazy-export table: ``".adaln.interface"``);
    * ``("children", "")`` -- a package ``__init__`` that defines ``__getattr__`` (PEP 562): every module of the package.

    Everything it cannot read stays conservative: a wider set, never a narrower one."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in _IMPORT_CALLS and node.args:
                text, complete = _leading_text(node.args[0])
                out.append(("exact" if complete and text else "prefix" if text else "opaque", text))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in (".", "..") and _REL_STRING.match(node.value):
            out.append(("relative", node.value))
        elif is_package_init and isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            out.append(("children", ""))
    return out


def _dotted_of(relpath: str) -> str:
    parts = list(PurePosixPath(relpath).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["miniworld_engine", *parts])


def _exact_module(dotted: str) -> list[str]:
    """The package file of exactly ``dotted`` (no shortening to a parent: a string that merely looks like a module name must be one)."""
    parts = dotted.split(".")
    base = "/".join(parts[1:])
    paths = _index()[0]
    return [c for c in (f"{base}.py", f"{base}/__init__.py") if base and c in paths][:1]


@functools.cache
def _dynamic_split(relpath: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(precise, wide)``: the package files a source can import by a name only the running program knows. ``precise`` -- the
    names the source spells out (an ``import_module("a.b")``, a prefix, a table of relative module names); ``wide`` -- what stands
    in when it spells none (``import_module(name)`` of a parameter, a ``__getattr__`` package): the whole package tree / its modules."""
    try:
        tree = ast.parse((PKG / relpath).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return frozenset(), frozenset()
    paths = _index()[0]
    here = PurePosixPath(relpath).parent
    package = ".".join(["miniworld_engine", *here.parts])
    precise: set[str] = set()
    wide: set[str] = set()

    def absolute(text: str) -> str:
        if not text.startswith("."):
            return text
        level = len(text) - len(text.lstrip("."))
        base = package.split(".")
        base = base[: len(base) - (level - 1)] if level > 1 else base
        return ".".join([*base, *([text.lstrip(".")] if text.lstrip(".") else [])])

    for kind, text in dynamic_import_specs(tree, is_package_init=relpath.endswith("__init__.py")):
        if kind == "exact":
            precise.update(module_files(absolute(text)))
        elif kind == "relative":
            precise.update(_exact_module(absolute(text)))
        elif kind == "prefix":
            prefix = absolute(text)
            precise.update(p for p in paths if p.endswith(".py") and _dotted_of(p).startswith(prefix))
        elif kind == "opaque":
            wide.update(p for p in paths if p.endswith(".py") and (here == PurePosixPath(".") or PurePosixPath(p).is_relative_to(here)))
        elif kind == "children":
            wide.update(p for p in paths if p.endswith(".py") and (PurePosixPath(p).parent == here or (
                PurePosixPath(p).name == "__init__.py" and PurePosixPath(p).parent.parent == here)))
    if precise:
        wide = set()               # a dispatch table that names its modules is the answer; the fallback only stands in for no names
    for group in (precise, wide):
        for p in list(group):
            group.update(_package_chain(p))
        group.discard(relpath)
    return frozenset(precise), frozenset(wide)


def _dynamic(relpath: str) -> frozenset[str]:
    return _dynamic_split(relpath)[0] | _dynamic_split(relpath)[1]


def module_files(dotted: str) -> list[str]:
    """The package files that run when ``dotted`` is imported: its module and every parent package ``__init__``."""
    hit = module_file(dotted)
    return [hit, *_package_chain(hit)] if hit else []


def _string_constants(text: str) -> str:
    """The string literals of a python source that can influence behaviour, joined: no comment (not in the tree) and no docstring
    (a docstring that names another family's file is documentation, not a dependency). A source that does not parse is taken whole."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                skip.add(id(body[0].value))
    return "\n".join(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip)


@functools.cache
def _deps(relpath: str) -> frozenset[str]:
    """The package files ``relpath`` reads or builds from, anywhere in the file (a function-level import is a dependency too)."""
    path = PKG / relpath
    paths, by_base = _index()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    found: set[str] = set(_package_chain(relpath))
    here = PurePosixPath(relpath).parent

    def named(token: str) -> None:
        for cand in (token, str(here / token), token.removeprefix("miniworld_engine/")):
            if cand in paths:
                found.add(cand)
                return
        if "/" not in token:
            same = by_base.get(token, ())
            found.update(same if len(same) <= 2 else ())          # a bare name that many files share names nothing in particular

    def includes(source: str) -> None:
        for spelling in _INCLUDE_RE.findall(source):
            near = str(here / spelling)
            if near in paths:
                found.add(near)
                continue
            suffix = "/" + spelling.lstrip("./")
            found.update(p for p in by_base.get(PurePosixPath(spelling).name, ()) if ("/" + p).endswith(suffix))

    if path.suffix == ".py":
        strict, lazy = _imports(relpath)
        found.update(strict, lazy, _dynamic(relpath))        # a name only the running program knows: every module it can mean
        strings = _string_constants(text)                    # names written as code, not in comments or docstrings
        for name in _DOTTED_RE.findall(strings):
            found.update(module_files(name))
        for token in _PATH_RE.findall(strings):
            named(token)
        includes(strings)                                    # CUDA written inline in a python source
    elif path.suffix in CCODE:
        includes(text)
    found.discard(relpath)
    return frozenset(found)


def strict_closure(seeds: Iterable[str]) -> set[str]:
    """The seeds and every module-level import they run when imported, transitively (no function-level import, no named path):
    what a constant imported at module level can come from. Does not widen through a hub that only imports inside functions."""
    seen: set[str] = set()
    queue = [s for s in seeds if (PKG / s).is_file()]
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.endswith(".py"):
            queue.extend(d for d in _imports(node)[0] if d not in seen)
            queue.extend(d for d in _package_chain(node) if d not in seen)
    return seen


def closure(seeds: Iterable[str], *, within: tuple[str, ...] = ("kernels/",)) -> set[str]:
    """Every package file reachable from ``seeds`` through :func:`_deps` (python imports incl. lazy, includes, named paths), staying
    inside ``within`` (the kernel side). A file outside it (the builder, the cache layer, the module layer) is not followed: the ones
    a native op depends on are the safety set, hashed whole. The seeds themselves are always kept."""
    seeds = [s for s in seeds if (PKG / s).is_file()]
    seen: set[str] = set()
    queue = list(seeds)
    while queue:
        node = queue.pop()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(d for d in _deps(node) if d not in seen and d.startswith(within))
    return seen


# ---------------------------------------------------------------------------------------------------------------- safety set
def safety_files(*, data: bool = True) -> list[str]:
    out = [p for p in SAFETY_PY if (PKG / p).is_file()]
    if data:
        out += [p for p in SAFETY_DATA if (PKG / p).is_file()]
        for pattern in SAFETY_GLOBS:
            out += sorted(PurePosixPath(p.relative_to(PKG)).as_posix() for p in PKG.glob(pattern))
    return sorted(set(out))


# --------------------------------------------------------------------------------------------------------------- registry rows
@functools.cache
def _registry_text(mtime: int) -> dict[str, dict]:
    with (PKG / REGISTRY).open(encoding="utf-8", newline="") as handle:
        return {row["kernel"]: row for row in csv.DictReader(handle)}


def registry_row(kernel: str) -> dict | None:
    try:
        return _registry_text((PKG / REGISTRY).stat().st_mtime_ns).get(kernel)
    except OSError:
        return None


def kernel_seed_files(row: dict) -> list[str]:
    """A registry row's files: the kernel's file and its driver (which decides what gets built and launched)."""
    seeds = []
    file = (row.get("file") or "").removeprefix("miniworld_engine/")
    if file:
        seeds.append(file)
    driver = (row.get("driver") or "").split(":")[0]
    hit = module_file(driver) if driver else None
    if hit:
        seeds.append(hit)
    return [s for s in seeds if (PKG / s).is_file()]


def _naming(relpath: str) -> list[str]:
    """Every python file that spells the basename of ``relpath``: the launcher of a ``.cu`` is not reachable FROM the ``.cu``."""
    name = PurePosixPath(relpath).name
    out = []
    for p in _index()[0]:
        if p.endswith(".py"):
            try:
                if name in (PKG / p).read_text(encoding="utf-8", errors="replace"):
                    out.append(p)
            except OSError:
                pass
    return out


@functools.cache
def _launchers(name: str) -> tuple[str, ...]:
    return tuple(_naming(name))


@functools.cache
def _op_files(kernel: str) -> tuple[str, ...]:
    """The files one registry kernel depends on (no safety set): its row's files, whoever names its kernel file, and their closure."""
    row = registry_row(kernel)
    if row is None:
        return ()
    seeds = kernel_seed_files(row)
    file = (row.get("file") or "").removeprefix("miniworld_engine/")
    if file.endswith(tuple(CCODE)):
        seeds += list(_launchers(file))
    return tuple(sorted(closure(seeds)))


def op_files(kernel: str) -> list[str]:
    return list(_op_files(kernel))


# ---------------------------------------------------------------------------------------------------------- native identity
def _native_token(relpath: str) -> str:
    key = _stat(relpath)
    if key is None:
        return ""
    return _norm(*key) if PurePosixPath(relpath).suffix in CCODE else _raw(*key)


def _py_deps(relpath: str) -> frozenset[str]:
    """What a python file names as code: its imports (function-level too), the modules its strings and dispatch tables name."""
    strict, lazy = _imports(relpath)
    found = set(strict) | set(lazy) | set(_dynamic(relpath))
    try:
        strings = _string_constants((PKG / relpath).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        strings = ""
    for name in _DOTTED_RE.findall(strings):
        found.update(module_files(name))
    found.discard(relpath)
    return frozenset(found)


@functools.cache
def outside_files(op: str) -> tuple[str, ...]:
    """The files OUTSIDE ``kernels/`` a native op uses: what its closure files import or name (the capture / compile / config
    helpers, the integrations a kernel module reaches back into) and the measurement path (:data:`MEASURE_PY`), followed through their
    imports. Not the tooling (:data:`TOOLING`), and not into ``modules/`` (a member when named, never followed)."""
    base = [p for p in op_files(op) if p.endswith(".py")]
    queue = [d for f in [*base, *MEASURE_PY] for d in _py_deps(f) if not d.startswith("kernels/")] + [m for m in MEASURE_PY if (PKG / m).is_file()]
    seen: set[str] = set()
    while queue:
        node = queue.pop()
        if node in seen or node in TOOLING or node.startswith(("build/", "viz/")):
            continue
        seen.add(node)
        if node.endswith(".py") and not node.startswith("modules/"):
            queue.extend(d for d in _py_deps(node) if d not in seen and not d.startswith("kernels/"))
    return tuple(sorted(seen))


def native_files(op: str) -> list[str]:
    """Every file the native cache of ``op`` depends on: its closure, the files outside ``kernels/`` it uses and the safety set
    (the registry CSVs enter as the op's own row)."""
    row_only = set(SAFETY_DATA) | {p for p in op_files(op) if p.startswith("build/gpu_to_kernels/")}
    return sorted((set(op_files(op)) - row_only) | set(outside_files(op)) | set(safety_files(data=False)))


def _mathdx() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("nvidia-mathdx")
    except PackageNotFoundError:
        return "absent"


@functools.cache
def native_identity(op: str) -> str:
    """Digest of ``op``'s part of the graph: its registry row, its closure, the safety set. An edit to a file outside it keeps it.

    ``registry.csv`` enters as the op's own row (not the whole file: a row for another kernel names nothing this cache measured), and
    ``registry_module.csv`` / ``gpu_to_kernels`` not at all: they pick which units get built, never what a measured candidate is."""
    row = registry_row(op)
    if row is None:
        raise KeyError(f"{op}: no registry.csv row, no graph to hash")
    h = hashlib.sha256(f"native-identity-v{SCHEME} nvidia-mathdx={_mathdx()}".encode())
    h.update(json.dumps(row, sort_keys=True).encode())
    for path in native_files(op):
        h.update(f"\0{path}\0{_native_token(path)}".encode())
    return h.hexdigest()


def native_aggregate(ops: Iterable[str]) -> str:
    """One digest over several ops' identities (a work generation: moves exactly when one of the ops' graphs moves)."""
    h = hashlib.sha256()
    for op in sorted(ops):
        h.update(f"{op}={native_identity(op)}\n".encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------ a derivation unit
def lazy_reach(executed: Iterable[str]) -> set[str]:
    """What code that ran can still reach without that reach having been observed: the function-level imports and the name-dispatched
    modules of every executed python file (with what they import at module level), and the data files such a file names. A constant
    read through a hub's lazy import in a unit that ran no function of the hub is covered here, not by luck."""
    out: set[str] = set()
    for relpath in set(executed):
        if not relpath.endswith(".py"):
            continue
        for dep in (*_imports(relpath)[1], *_dynamic_split(relpath)[0]):       # the names a unit performed are recorded at run time
            out |= strict_closure([dep])
        out.update(d for d in _deps(relpath) if not d.endswith(".py"))
    return out


def unit_files(executed: Iterable[str], modules: Iterable[str], data: Iterable[str], ops: Iterable[str], listed: Iterable[str] = ()) -> list[str]:
    """The files one derivation unit depends on.

    ``executed`` -- package files whose code ran while the unit was recorded; ``modules`` -- dotted names its import statements
    named (executed lazy imports included); ``data`` -- package files it opened; ``listed`` -- package directories it listed
    (``listdir`` / ``scandir`` / ``glob``), kept as ``dir:<rel>`` entries whose token is the listing. Plus what a constant imported at
    module level can come from (:func:`strict_closure` of the first two), what the executed code can import lazily or by name
    (:func:`lazy_reach`), the files of every kernel it launched (:func:`launch_files`: row file, JIT helpers, driver) and the safety set."""
    executed = set(executed)
    seeds = set(executed)
    for name in modules:
        seeds.update(module_files(name))
    files = strict_closure(seeds) | lazy_reach(seeds) | set(data) | set(safety_files())
    files |= {f"{DIR}{PurePosixPath(d).as_posix()}" for d in listed}
    for op in set(ops):
        files.update(launch_files(op))
    return sorted(files)


# ----------------------------------------------------------------------------------------------------------- the toolchain
#: Distributions whose version changes what a kernel compiles to or which bucket key a launch makes.
TOOLCHAIN_DISTS = ("torch", "triton", "transformer-engine", "cuequivariance", "cuequivariance-ops-torch-cu12", "flash-attn-4",
                   "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12", "nvidia-cublas-cu12", "nvidia-mathdx", "nvidia-cutlass-dsl", "cuda-python")
#: Environment variables the engine reads that cannot change a derivation / a measurement (where it runs, not what it computes).
ENV_OPERATIONAL = frozenset({"MINIWORLD_COMPILE_WRAP", "CUDA_VISIBLE_DEVICES", "HOME", "USER", "PATH", "PWD", "TMPDIR", "XDG_CACHE_HOME",
                             "LD_LIBRARY_PATH", "PYTHONPATH", "CONDA_PREFIX", "CUTLASS_PATH", "TRITON_CACHE_DIR", "QUACK_CACHE_ENABLED"})
ENV_PREFIXES = ("MINIWORLD_", "TRITON_", "TORCH_", "TORCHINDUCTOR_", "CUDA_", "NVIDIA_", "NVCC_", "CUTE_", "QUACK_", "CUEQ_", "NVTE_")
_ENV_RE = re.compile(r"""(?:environ(?:\.get|\.setdefault|\.pop)?\s*[(\[]|getenv\s*\(|\bin\s+(?:os\.)?environ\b)[^"'\n]{0,40}["']([A-Z][A-Z0-9_]+)["']|["']([A-Z][A-Z0-9_]+)["']\s+in\s+(?:os\.)?environ""")


@functools.cache
def env_names() -> tuple[str, ...]:
    """Every environment variable a package source reads by name (a literal passed to ``environ`` / ``getenv``), limited to the
    prefixes that belong to the engine and its toolchain and minus the operational ones."""
    names: set[str] = set()
    for relpath in _index()[0]:
        if relpath.endswith(".py"):
            try:
                text = (PKG / relpath).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            names.update(a or b for a, b in _ENV_RE.findall(text))
    return tuple(sorted(n for n in names if n.startswith(ENV_PREFIXES) and n not in ENV_OPERATIONAL))


def toolchain() -> dict:
    """What a derivation / a native cache was made under and is only valid for: the Python minor, the version of every toolchain
    distribution (``torch`` carries its CUDA build in its local version, ``+cu129``), and the value of every engine environment
    variable that is set. A change of any of them marks the evidence stale: the graph's inputs are not only files."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for dist in TOOLCHAIN_DISTS:
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            versions[dist] = "absent"
    return {"python": f"{sys.version_info[0]}.{sys.version_info[1]}", "versions": versions,
            "env": {n: os.environ[n] for n in env_names() if n in os.environ}}


def toolchain_diff(stored: dict | None) -> list[str]:
    """How ``stored`` (a recorded :func:`toolchain`) differs from now; ``[]`` when equal."""
    now = toolchain()
    out = []
    if (stored or {}).get("python") != now["python"]:
        out.append(f"python {(stored or {}).get('python')} -> {now['python']}")
    for dist in sorted(set(now["versions"]) | set((stored or {}).get("versions", {}))):
        was, is_ = (stored or {}).get("versions", {}).get(dist), now["versions"].get(dist)
        if was != is_:
            out.append(f"{dist} {was} -> {is_}")
    for name in sorted(set(now["env"]) | set((stored or {}).get("env", {}))):
        was, is_ = (stored or {}).get("env", {}).get(name), now["env"].get(name)
        if was != is_:
            out.append(f"env {name} {was!r} -> {is_!r}")
    return out


def files_digest(files: Iterable[str]) -> str:
    """A short digest of the tokens of ``files`` (a work generation: moves exactly when one of them moves)."""
    return hashlib.sha256(repr(tokens(files)).encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def tree_digest() -> str:
    """The coarse fallback: every package source under ``kernels/``, ``modules/``, ``baselines/``, ``utils/`` plus the safety set."""
    top = ("kernels/", "modules/", "baselines/", "utils/")
    return files_digest([p for p in _index()[0] if p.startswith(top) and not p.endswith(".csv")] + safety_files())


@functools.cache
def _is_jit(relpath: str) -> bool:
    try:
        return "triton.jit" in (PKG / relpath).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


@functools.cache
def launch_files(op: str) -> tuple[str, ...]:
    """What a unit that launched ``op`` depends on: the kernel's registry file, the JIT-helper files it imports (followed only into
    files that hold ``triton.jit``) and its driver. NOT the whole closure of :func:`native_files`: a ``.cu`` source is never executed
    while a unit is recorded, so it cannot change which keys the unit launches."""
    row = registry_row(op)
    if row is None:
        return ()
    seeds = kernel_seed_files(row)
    out = list(seeds)
    queue = [s for s in seeds if s.endswith(".py")]
    while queue:
        for dep in sorted(set().union(*_imports(queue.pop(0)))):
            if dep not in out and dep.endswith(".py") and _is_jit(dep):
                out.append(dep)
                queue.append(dep)
    return tuple(sorted(out))


def snapshot() -> dict[str, str]:
    """The bytes digest of every indexed package file and of the safety set, taken before units run: :func:`changed_since` then
    says whether a source was edited while a derivation was running."""
    out = {}
    for relpath in sorted(set(_index()[0]) | set(safety_files())):
        key = _stat(relpath)
        if key is not None:
            out[relpath] = _raw(*key)[:20]
    return out


def changed_since(before: dict[str, str], files: Iterable[str]) -> list[str]:
    """The ``files`` whose bytes differ from the :func:`snapshot` (a file absent from it is not judged)."""
    moved = []
    for relpath in files:
        key = _stat(relpath)
        if relpath in before and (key is None or _raw(*key)[:20] != before[relpath]):
            moved.append(relpath)
    return moved
