"""Symbol-level view of a Python file, for the test selection of the repository graph (:mod:`repo_graph`).

The file-level graph says "``a.py`` imports ``b.py``". That is exact for a CUDA include, but a Python file imports *names*: ``from b import f``
depends on ``f``'s definition and what ``f`` calls, not on every other symbol of ``b``. This module reads one file and says, per top-level
*scope*:

* the **units**: top-level ``def`` / ``class`` and ``NAME = ...`` assignments, each with the names it loads (``uses``), the string
  constants it holds (``strings``, docstrings excluded), and a digest of its syntax tree (a comment / formatting edit keeps it);
* the **module scope** (``"*"``): every other top-level statement (``if`` / ``try`` / calls / ``__all__`` / imports ...). It runs on import, so
  a change to it, or a use of a changed name by it, is a change to the *whole file* (``"*"``);
* the **bindings**: every import, with what it binds (``from b import f`` binds ``f`` to ``(b, f)``; ``from pkg import mod`` and
  ``import b`` bind a module object, of which the attributes used (``mod.f``) are tracked, ``None`` = the object is used bare or ``import b``
  is plain, which stays module-level conservative) and the scope it sits in (a function-level import belongs to that function).

``closure`` closes a set of changed names inside the file (a unit that uses a changed name is changed too; module-level use escalates to the
whole file); ``changed_symbols`` maps an edit (old text -> new text) to the names it changed and falls back to the whole file whenever the
mapping is not certain: a module-level statement, a decorator, an import, a syntax error, a ``globals()`` file.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field

ALL = "*"                      # the whole file / the module scope
MAIN = "<main>"                # the ``if __name__ == "__main__":`` block: never run by an import
_EXT_CALLS = {"load_extension", "load_inline", "load"}
_DYN_IMPORT = {"import_module", "__import__", "spec_from_file_location", "run_path", "run_module"}


@dataclass
class Unit:
    name: str
    kind: str                                   # def | class | assign | main
    uses: set[str] = field(default_factory=set)
    strings: set[str] = field(default_factory=set)
    digest: str = ""
    deco: str = ""
    fixture: str = ""                           # "" | "fixture" | "autouse" (a pytest fixture, in a conftest)


@dataclass
class Bind:
    local: str
    target: str | None                          # the repo file the import resolves to
    orig: str | None                            # the imported name (None: a module object / star)
    plain: bool                                 # ``import x`` / ``import x as y`` / a star import: attributes are not tracked
    scope: str                                  # the scope the import statement sits in (a unit name, or ALL)
    lazy: bool                                  # inside a function body
    exec_only: bool = False                     # runs the package ``__init__`` only (``from pkg import submodule``)


@dataclass
class FileSyms:
    ok: bool = True
    units: dict[str, Unit] = field(default_factory=dict)
    module: Unit = field(default_factory=lambda: Unit(ALL, "module"))
    binds: list[Bind] = field(default_factory=list)
    names: dict[str, set[str]] = field(default_factory=dict)                   # scope -> names loaded
    bare: dict[str, set[str]] = field(default_factory=dict)                    # scope -> names loaded other than as ``name.attr``
    attrs: dict[str, dict[str, set[str]]] = field(default_factory=dict)        # scope -> name -> attributes read
    dynamic_all: bool = False                                                  # globals() / __getattr__: any name may be reached
    dyn_import: set[str] = field(default_factory=set)                          # scopes that import a module by a computed name
    ext_names: dict[str, set[str]] = field(default_factory=dict)               # scope -> extension names it builds
    literal_imports: list[tuple[str, str]] = field(default_factory=list)       # (scope, dotted) of import_module("literal")

    def scope_names(self) -> list[str]:
        return [ALL, *self.units]


def _is_doc(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_main_block(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.If) or not isinstance(stmt.test, ast.Compare):
        return False
    t = stmt.test
    return isinstance(t.left, ast.Name) and t.left.id == "__name__" and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant) \
        and t.comparators[0].value == "__main__" and not stmt.orelse


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _strings(node: ast.AST) -> set[str]:
    out: set[str] = set()
    skip: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and n.body and _is_doc(n.body[0]):
            skip.add(id(n.body[0].value))
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip and len(n.value) >= 3:
            out.add(n.value)
    return out


def _usage(node: ast.AST) -> tuple[set[str], set[str], dict[str, set[str]]]:
    """``(names loaded, names loaded bare, {name: attributes read})`` anywhere under ``node`` (nested functions included)."""
    loaded: set[str] = set()
    attr: dict[str, set[str]] = {}
    covered: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and isinstance(n.ctx, ast.Load):
            attr.setdefault(n.value.id, set()).add(n.attr)
            covered.add(id(n.value))
    bare: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Load, ast.Del)):
            loaded.add(n.id)
            if id(n) not in covered:
                bare.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and not isinstance(n.ctx, ast.Load):
            loaded.add(n.value.id)                      # ``mod.x = ...`` writes through the module object
            bare.add(n.value.id)
        elif isinstance(n, ast.arg):
            loaded.add(n.arg)                           # a pytest fixture is requested by the name of a parameter
            bare.add(n.arg)
        elif isinstance(n, ast.Global):
            loaded.update(n.names)
            bare.update(n.names)
    return loaded, bare, attr


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _scan_calls(node: ast.AST, scope: str, fs: FileSyms) -> None:
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        name = _call_name(n)
        if name in ("globals", "vars", "eval", "exec"):
            if scope == ALL:
                fs.dynamic_all = True
            fs.units[scope].uses.add(ALL) if scope in fs.units else None
        if name in _DYN_IMPORT:
            arg = n.args[0] if n.args else None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                fs.literal_imports.append((scope, arg.value))
            else:
                fs.dyn_import.add(scope)
        if name in _EXT_CALLS:
            cand = list(n.args[:1]) + [k.value for k in n.keywords if k.arg == "name"]
            for a in cand:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    fs.ext_names.setdefault(scope, set()).add(a.value)


def _fixture_kind(stmt: ast.stmt) -> str:
    for d in getattr(stmt, "decorator_list", ()):
        call = d if isinstance(d, ast.Call) else None
        f = call.func if call else d
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
        if name == "fixture":
            auto = call is not None and any(k.arg == "autouse" and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in call.keywords)
            return "autouse" if auto else "fixture"
    return ""


def _targets(stmt: ast.stmt) -> list[str] | None:
    if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
        names = [t.id for t in stmt.targets]
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        names = [stmt.target.id]
    else:
        return None
    return None if any(n.startswith("__") and n.endswith("__") for n in names) else names


def analyze(path: str, text: str, resolve) -> FileSyms:
    """Read ``text`` (the file ``path``) into its units, module scope and bindings. ``resolve`` is the repo's ``_Resolver`` (``dotted`` / ``exact``)."""
    fs = FileSyms()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        fs.ok = False
        return fs
    from pathlib import PurePosixPath

    here = PurePosixPath(path).parent.parts

    def absolute(node: ast.ImportFrom) -> str:
        if node.level:
            base = list(here[: len(here) - (node.level - 1)])
            if base and base[0] == "src":
                base = base[1:]
            return ".".join([*base, *([node.module] if node.module else [])])
        return node.module or ""

    def bind_imports(stmt: ast.AST, scope: str) -> None:
        stack = [(stmt, False)]
        while stack:
            node, lazy = stack.pop()
            if isinstance(node, ast.Import):
                for a in node.names:
                    target = resolve.mod(a.name)
                    if target:
                        fs.binds.append(Bind(a.asname or a.name.split(".")[0], target, None, True, scope, lazy))
            elif isinstance(node, ast.ImportFrom):
                module = absolute(node)
                tfile = resolve.mod(module) if module else None
                if True:
                    for a in node.names:
                        local = a.asname or a.name
                        if a.name == "*":
                            fs.binds.append(Bind("*", tfile, None, True, scope, lazy))
                            continue
                        sub = resolve.exact(f"{module}.{a.name}") if module else None
                        if sub:
                            fs.binds.append(Bind(local, sub, None, False, scope, lazy))
                            if tfile and tfile != sub:
                                fs.binds.append(Bind("", tfile, None, True, scope, lazy, exec_only=True))
                        elif tfile:
                            fs.binds.append(Bind(local, tfile, a.name, False, scope, lazy))
            inside = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            stack.extend((c, inside) for c in ast.iter_child_nodes(node))

    mod = fs.module
    mod_dumps: list[str] = []

    def add_usage(scope: str, node: ast.AST) -> None:
        loaded, bare, attr = _usage(node)
        fs.names.setdefault(scope, set()).update(loaded)
        fs.bare.setdefault(scope, set()).update(bare)
        for k, v in attr.items():
            fs.attrs.setdefault(scope, {}).setdefault(k, set()).update(v)

    def add_unit(name: str, kind: str, node: ast.AST, deco: str = "") -> None:
        loaded, _, _ = _usage(node)
        u = fs.units.get(name)
        if u is None:
            u = fs.units[name] = Unit(name, kind)
        u.uses |= loaded
        u.strings |= _strings(node)
        u.digest = hashlib.sha256((u.digest + _dump(node)).encode()).hexdigest()
        u.deco = u.deco + deco
        add_usage(name, node)
        bind_imports(node, name)
        _scan_calls(node, name, fs)

    for i, stmt in enumerate(tree.body):
        if i == 0 and _is_doc(stmt):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(stmt, ast.ClassDef) else "def"
            add_unit(stmt.name, kind, stmt, deco="".join(_dump(d) for d in stmt.decorator_list))
            fixture = _fixture_kind(stmt)
            if fixture:
                fs.units[stmt.name].fixture = fixture
            continue
        names = _targets(stmt)
        if names:
            for n in names:
                add_unit(n, "assign", stmt)
            continue
        if _is_main_block(stmt):
            add_unit(MAIN, "main", stmt)
            continue
        # module scope: runs on import
        mod_dumps.append(_dump(stmt))
        loaded, _, _ = _usage(stmt)
        mod.uses |= loaded
        mod.strings |= _strings(stmt)
        add_usage(ALL, stmt)
        bind_imports(stmt, ALL)

    mod.digest = hashlib.sha256("".join(mod_dumps).encode()).hexdigest()
    # module-level code that is dynamic reaches everything
    for stmt in tree.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _targets(stmt) is None and not _is_main_block(stmt):
            _scan_calls(stmt, ALL, fs)
    if "__getattr__" in fs.units:
        fs.dynamic_all = True
    return fs


def closure(fs: FileSyms, names: set[str]) -> set[str]:
    """``names`` plus every unit that uses one of them (to a fixed point); ``{ALL}`` when the module scope uses one, or the file is dynamic."""
    if ALL in names or not fs.ok or fs.dynamic_all:
        return {ALL}
    out = set(names)
    while True:
        if fs.module.uses & out:
            return {ALL}
        grown = {u.name for u in fs.units.values() if u.name not in out and (u.uses & out or ALL in u.uses)}
        if not grown:
            return out
        out |= grown


def changed_symbols(old: FileSyms, new: FileSyms) -> set[str]:
    """The names an edit from ``old`` to ``new`` changed; ``{ALL}`` whenever it cannot say (module scope, decorators, imports, parse failure)."""
    if not (old.ok and new.ok) or old.dynamic_all or new.dynamic_all:
        return {ALL}
    if old.module.digest != new.module.digest:
        return {ALL}
    if _bindings(old) != _bindings(new):
        return {ALL}
    out: set[str] = set()
    for name in old.units.keys() | new.units.keys():
        a, b = old.units.get(name), new.units.get(name)
        if a is None or b is None:
            out.add(name)
        elif a.digest != b.digest or a.kind != b.kind:
            if a.deco != b.deco:
                return {ALL}
            out.add(name)
    return out


def _bindings(fs: FileSyms) -> list[tuple]:
    return sorted((b.local, b.target or "", b.orig or "", b.plain) for b in fs.binds if b.scope == ALL)


def scope_nodes(text: str) -> dict[str, list[ast.stmt]]:
    """The top-level statements of each scope (the same split as :func:`analyze`): ``{unit name or ALL: [statements]}``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out: dict[str, list[ast.stmt]] = {}
    for i, stmt in enumerate(tree.body):
        if i == 0 and _is_doc(stmt):
            continue
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(stmt.name, []).append(stmt)
        elif (names := _targets(stmt)):
            for n in names:
                out.setdefault(n, []).append(stmt)
        elif _is_main_block(stmt):
            out.setdefault(MAIN, []).append(stmt)
        else:
            out.setdefault(ALL, []).append(stmt)
    return out
