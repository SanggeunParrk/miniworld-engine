"""Which repository paths a python file computes, read off its syntax tree: ``ROOT / "docs" / "x.md"``, ``PKG / "kernels" / name``,
``Path(__file__).parents[2] / "src"``, ``os.path.join(...)``, ``p.joinpath(...)``, f-strings, names imported from another file of the
repo (``from tests.paths import PKG``).

A literal ``"a/b.py"`` is found by a regular expression (``repo_graph._PATH_RE``); a path BUILT from segments is not, and the tests
and loaders of this repo build nearly all of theirs that way. This module evaluates the expression:

* a fully known path names one file or directory -- an edge to it;
* a path whose last part is not known (``DATA / f"{op}.json"``, ``ROOT / name``) names a PREFIX: it depends on every file below it;
* a file that lists directories (``glob`` / ``rglob`` / ``iterdir`` / ``listdir`` / ``scandir`` / ``walk``) depends on every file
  below each directory it names; one that lists something it cannot resolve (a parameter) is an *opaque scanner*, and so is every
  file that imports one: they depend on the directories they name too, whether or not they list them themselves.

Everything unknown widens the answer (the prefix, then the whole directory), never narrows it: a wider set costs a test run that was
not needed, never a missed one.
"""

from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass, field
from pathlib import PurePosixPath

PATH_CTORS = {"Path", "PurePath", "PosixPath", "PurePosixPath"}
ENUMERATORS = {"glob", "rglob", "iterdir", "listdir", "scandir", "walk", "fwalk", "iglob"}
MAX_VALUES = 12

# A value is (kind, text, open). kind "p": a path relative to the repo root ("" is the root); "s": a string; "x": outside the repo /
# not a path. ``open``: the text is only a prefix of the value.
Val = tuple[str, str, bool]
UNKNOWN: Val = ("s", "", True)
OUTSIDE: Val = ("x", "", True)


@dataclass
class PathUse:
    exact: set[str] = field(default_factory=set)         # repo-relative paths named completely
    prefixes: set[str] = field(default_factory=set)      # repo-relative prefixes: everything below / starting with it
    enumerates: bool = False                             # lists a directory it can resolve
    opaque_enumeration: bool = False                     # lists something it cannot resolve (a parameter): a scanner
    values: set[Val] = field(default_factory=set)        # every path value that occurs (for the scanner rule)


def _norm(rel: str) -> str | None:
    parts: list[str] = []
    for part in rel.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _join(left: Val, right: Val) -> Val:
    if left[0] == "x" or right[0] == "x":
        return OUTSIDE
    if left[2]:
        return left                                       # an open prefix stays a prefix
    text = right[1]
    if right[0] == "p" and right[1] == "":
        return left
    if text.startswith("/"):
        return OUTSIDE if right[0] == "s" else ("p", text.lstrip("/"), right[2])
    rel = _norm(f"{left[1]}/{text}") if left[1] or text else ""
    return OUTSIDE if rel is None else ("p", rel, right[2])


def _parent(v: Val, n: int = 1) -> Val:
    if v[0] != "p":
        return OUTSIDE
    rel = v[1] if not v[2] else posixpath.dirname(v[1])
    for _ in range(n):
        rel = posixpath.dirname(rel)
    return ("p", rel, False)


class Analyzer:
    """Evaluates the path expressions of the files of a repository. ``read(path)`` returns a file's text; ``paths`` is the set of
    repo files; ``resolve_module(module, here, level)`` maps an import to a repo file (or None)."""

    def __init__(self, paths, read, resolve_module) -> None:
        self.paths, self.read, self.resolve_module = paths, read, resolve_module
        self._env: dict[str, dict[str, list[Val]]] = {}
        self._busy: set[str] = set()
        self._tree: dict[str, ast.AST | None] = {}

    def tree(self, path: str):
        if path not in self._tree:
            try:
                self._tree[path] = ast.parse(self.read(path))
            except (SyntaxError, ValueError, OSError):
                self._tree[path] = None
        return self._tree[path]

    # ---------------------------------------------------------------------------------------------- names of a file
    def env(self, path: str) -> dict[str, list[Val]]:
        """``{name: [values]}`` for every simple assignment in the file (any scope; a name assigned twice keeps both) and every
        name imported from another repo file."""
        if path in self._env:
            return self._env[path]
        if path in self._busy:
            return {}
        self._busy.add(path)
        env: dict[str, list[Val]] = {}
        tree = self.tree(path)
        if tree is not None:
            env["__file__"] = [("p", path, False)]
            nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign | ast.AnnAssign | ast.ImportFrom)]
            nodes.sort(key=lambda n: (n.lineno, n.col_offset))
            for _ in range(2):                                                   # a second pass: a name used before its textual assignment
                for node in nodes:
                    if isinstance(node, ast.ImportFrom):
                        target = self.resolve_module(node.module or "", path, node.level)
                        if target and target != path:
                            other = self.env(target)
                            for alias in node.names:
                                if alias.name in other:
                                    env[alias.asname or alias.name] = other[alias.name]
                        continue
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    if value is None:
                        continue
                    for t in targets:
                        if isinstance(t, ast.Name):
                            vals = self.eval(value, env, path)
                            if any(v[0] != "s" or v[1] for v in vals):            # keep path-like and informative strings only
                                have = env.setdefault(t.id, [])
                                for v in vals:
                                    if v not in have and len(have) < MAX_VALUES:
                                        have.append(v)
        self._busy.discard(path)
        self._env[path] = env
        return env

    # ----------------------------------------------------------------------------------------------- expressions
    def eval(self, node: ast.AST, env: dict, path: str) -> list[Val]:
        out = self._eval(node, env, path)
        return out[:MAX_VALUES] or [UNKNOWN]

    def _eval(self, node: ast.AST, env: dict, path: str) -> list[Val]:
        if isinstance(node, ast.Constant):
            return [("s", node.value, False)] if isinstance(node.value, str) else [UNKNOWN]
        if isinstance(node, ast.Name):
            return list(env.get(node.id, [UNKNOWN]))
        if isinstance(node, ast.JoinedStr):
            text, is_open = "", False
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str) and not is_open:
                    text += part.value
                else:
                    is_open = True
                    break
            return [("s", text, is_open)]
        if isinstance(node, ast.Attribute):
            if node.attr == "parent":
                return [_parent(v) for v in self.eval(node.value, env, path)]
            if node.attr in ("name", "stem", "suffix", "parts", "parents"):
                return [UNKNOWN]
            return [UNKNOWN]
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "parents" and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, int):
                return [_parent(v, node.slice.value + 1) for v in self.eval(node.value.value, env, path)]
            return [UNKNOWN]
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                lefts, rights = self.eval(node.left, env, path), self.eval(node.right, env, path)
                return [self._as_path(_join(self._as_path(a), b)) for a in lefts for b in rights][:MAX_VALUES]
            if isinstance(node.op, ast.Add):
                return [self._concat(a, b) for a in self.eval(node.left, env, path) for b in self.eval(node.right, env, path)][:MAX_VALUES]
            if isinstance(node.op, ast.Mod):
                return [(a[0], a[1].split("%", 1)[0], True) if a[0] == "s" else a for a in self.eval(node.left, env, path)]
            return [UNKNOWN]
        if isinstance(node, ast.Call):
            return self._call(node, env, path)
        if isinstance(node, ast.IfExp):
            return self.eval(node.body, env, path) + self.eval(node.orelse, env, path)
        return [UNKNOWN]

    @staticmethod
    def _as_path(v: Val) -> Val:
        return ("p", v[1], v[2]) if v[0] == "s" else v

    @staticmethod
    def _concat(a: Val, b: Val) -> Val:
        if a[0] == "x" or b[0] == "x":
            return OUTSIDE
        if a[2]:
            return a
        return (a[0] if a[0] == "p" else b[0], a[1] + b[1], b[2])

    def _call(self, node: ast.Call, env: dict, path: str) -> list[Val]:
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        receiver = fn.value if isinstance(fn, ast.Attribute) else None
        if name in PATH_CTORS or (name == "join" and isinstance(receiver, ast.Attribute) and receiver.attr == "path"):
            if not node.args:
                return [("p", "", False)] if name in PATH_CTORS else [UNKNOWN]
            acc = self.eval(node.args[0], env, path)
            acc = [self._as_path(a) for a in acc]
            for arg in node.args[1:]:
                acc = [_join(a, b) for a in acc for b in self.eval(arg, env, path)][:MAX_VALUES]
            return acc
        if name == "joinpath" and receiver is not None:
            acc = [self._as_path(a) for a in self.eval(receiver, env, path)]
            for arg in node.args:
                acc = [_join(a, b) for a in acc for b in self.eval(arg, env, path)][:MAX_VALUES]
            return acc
        if name in ("resolve", "absolute", "expanduser", "as_posix", "samefile", "realpath", "abspath", "normpath", "fspath", "str",
                    "Path", "relative_to_root") and (receiver is not None or node.args):
            src = receiver if receiver is not None and name not in ("realpath", "abspath", "normpath", "fspath") else (node.args[0] if node.args else receiver)
            return self.eval(src, env, path) if src is not None else [UNKNOWN]
        if name in ("dirname",) and node.args:
            return [_parent(self._as_path(v)) for v in self.eval(node.args[0], env, path)]
        if name in ("with_name", "with_suffix", "with_stem") and receiver is not None:
            return [("p", _parent(self._as_path(v))[1], True) if v[0] != "x" else v for v in self.eval(receiver, env, path)]
        if name in ("cwd", "getcwd"):
            return [("p", "", False)]
        if name in ("home", "gettempdir", "mkdtemp", "TemporaryDirectory", "NamedTemporaryFile", "tmp_path_factory"):
            return [OUTSIDE]
        return [UNKNOWN]

    # --------------------------------------------------------------------------------------------------- a file
    def use(self, path: str) -> PathUse:
        """Every path ``path`` computes, split into complete paths and prefixes, and whether it lists directories."""
        out = PathUse()
        tree = self.tree(path)
        if tree is None:
            return out
        env = self.env(path)
        for name, vals in env.items():
            if name != "__file__":
                out.values.update(v for v in vals if v[0] == "p")
        children = {id(c) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                    for c in (n.left,) if isinstance(c, ast.BinOp | ast.Call | ast.Attribute)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if name in ENUMERATORS:
                    receiver = fn.value if isinstance(fn, ast.Attribute) else None
                    roots = self.eval(receiver, env, path) if receiver is not None and not (
                        isinstance(receiver, ast.Name) and receiver.id in ("os", "glob", "fnmatch")) else []
                    if name in ("listdir", "scandir", "walk", "fwalk", "iglob", "glob") and node.args and (
                            receiver is None or (isinstance(receiver, ast.Name) and receiver.id in ("os", "glob"))):
                        roots = self.eval(node.args[0], env, path)
                    resolved = [self._as_path(v) for v in roots if v[0] == "p" or (v[0] == "s" and v[1])]
                    if resolved:
                        out.enumerates = True
                    if not resolved or all(v[1] == "" and v[2] for v in resolved):
                        out.opaque_enumeration = True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and id(node) not in children \
                    or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath" \
                    or isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PATH_CTORS:
                for v in self.eval(node, env, path):
                    if v[0] == "p":
                        out.values.add(v)
        for v in list(out.values):
            if v[2]:
                out.prefixes.add(v[1])
            else:
                out.exact.add(v[1])
        return out
