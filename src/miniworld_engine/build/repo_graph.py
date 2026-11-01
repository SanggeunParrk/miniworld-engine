"""The whole repository as one graph, and what a change does to it.

Every tracked file (and every untracked, unignored one) is a node. An edge ``A -> B`` means "A is built from / reads / names B":
a change to B can change A. Edges come from what the files themselves say, not from a hand-kept list:

* ``import``  -- Python ``import`` / ``from ... import`` anywhere in a file (a function-level import is a dependency too);
* ``include`` -- ``#include "..."`` in ``.cu`` / ``.cuh`` / ``.inc`` / ``.h`` / ``.cpp``;
* ``ref``     -- a repo path or a ``miniworld_engine.x.y`` dotted name written in a source, doc, YAML or registry file
  (registry rows name each kernel's file, driver and check; a doc names the code it describes);
* ``cache``   -- ``autotune/data/<op>/<gpu>.json`` is measured from ``<op>``'s kernel file, the JIT helpers that file imports, its
  driver, its config grid ``configs/grid/<op>.csv`` and ``registry.csv`` (``build_rev``), and every cache is keyed by ``cache.py``;
* ``identity`` -- the artifacts that are judged by the digest of their own part of the graph (``autotune/identity.py``) are nodes of their
  own: ``identity:native:<op>`` (``native.op_identity``) feeds that op's ``*_cuda`` cache, ``identity:derivation`` feeds
  ``registry_kernel.units.json`` / ``registry_kernel.csv`` / the packaged plans and through them the sweep page. Their edges go to the
  op's closure (and the safety set), and to the union of the files the derivation's units recorded: an edit outside them reaches
  nothing here.

``changes(base)`` takes the files that differ from ``base`` (``git diff``, plus untracked), drops the ones whose normalized content
is unchanged (comments, whitespace) -- a native identity hashes a ``.py`` by bytes, so those still seed it; a derivation token and a
``.cu`` do not -- and walks the edges backwards.
The result is every node the edit reaches, with the kind of node it is: tests (and which of them need a GPU), docs, tuned caches,
identities, derived pages, CUDA sources. ``actions`` turns that into the list of things the edit owes.
"""

from __future__ import annotations

import ast
import collections
import csv
import functools
import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parents[1]
SRC = "src/miniworld_engine"
NATIVE_PREFIX, DERIVATION = "identity:native:", "identity:derivation"

CCODE = {".cu", ".cuh", ".inc", ".h", ".cpp"}
#: files whose text is scanned for references. Measured data, benchmark tables, figures and lock files name nothing.
SCANNED = {".py", ".md", ".yaml", ".yml"} | CCODE      # not .toml: pyproject names the CLI entry point, it is not built from it
SCANNED_CSV = {f"{SRC}/kernels/registry/registry.csv", f"{SRC}/kernels/registry/registry_module.csv"}
NOT_CODE = {".toml", ".html", ".svg", ".png", ".txt"}
_PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|cu|cuh|inc|h|cpp|csv|json|md|yaml|yml|svg|png|html|toml))\b")
_DOTTED_RE = re.compile(r"\bminiworld_engine(?:\.\w+)+")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def files() -> list[str]:
    """Every tracked file and every untracked, unignored one, repo-relative."""
    out = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted({p for p in out.split("\0") if p and (ROOT / p).is_file()})


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def kind_of(path: str) -> str:
    p = PurePosixPath(path)
    if p.suffix == ".md":
        return "doc"
    if p.parts[0] == "tests":
        return "test-gpu" if "gpu" in p.parts[:3] else "test"
    if p.parts[0] == "benchmarks":
        return "benchmark"
    if p.parts[0] == "docs":
        return "doc"
    if path.startswith(f"{SRC}/autotune/data/"):
        return "cache"
    if p.suffix in CCODE:
        return "cuda-source"
    if p.suffix == ".py":
        return "python"
    return "other"


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}
        self.deps: dict[str, dict[str, str]] = collections.defaultdict(dict)       # node -> {dependency: edge kind}
        self.users: dict[str, dict[str, str]] = collections.defaultdict(dict)      # dependency -> {node: edge kind}

    def edge(self, node: str, dep: str, kind: str) -> None:
        if kind == "ref" and self.nodes.get(dep) == "doc" and self.nodes.get(node) != "doc":
            return              # a comment or docstring pointing at a doc ("see docs/x.md") is not built from it
        if kind == "ref" and PurePosixPath(dep).suffix in NOT_CODE and PurePosixPath(node).suffix in (".py", *CCODE) and not node.startswith("tests/"):
            return              # a comment naming pyproject.toml or a figure; only a test that reads such a file depends on it
        if node != dep and dep in self.nodes and node in self.nodes and (dep not in self.deps[node] or self.deps[node][dep] == "lazy"):
            self.deps[node][dep] = kind                      # a module-level import outranks a lazy one of the same file
            self.users[dep][node] = kind

    def edge_count(self) -> collections.Counter:
        return collections.Counter(k for d in self.deps.values() for k in d.values())


class _Resolver:
    def __init__(self, paths: list[str]) -> None:
        self.paths = set(paths)
        self.by_base: dict[str, list[str]] = collections.defaultdict(list)
        for p in paths:
            self.by_base[PurePosixPath(p).name].append(p)

    def dotted(self, name: str) -> str | None:
        """``miniworld_engine.a.b.c`` -> the file of the longest prefix that is a module or package."""
        parts = name.split(".")
        for n in range(len(parts), 0, -1):
            base = "src/" + "/".join(parts[:n])
            for cand in (f"{base}.py", f"{base}/__init__.py"):
                if cand in self.paths:
                    return cand
        return None

    def path(self, token: str, here: str) -> str | None:
        norm = PurePosixPath(token)
        if ".." in norm.parts:
            return None
        for cand in (token, str(PurePosixPath(here).parent / token), f"{SRC}/{token}", f"src/{token}", f"docs/{token}"):
            if cand in self.paths:
                return cand
        if "/" not in token:
            same = self.by_base.get(token, [])
            if len(same) == 1:
                return same[0]
        return None

    def include(self, spelling: str, here: str) -> list[str]:
        near = str(PurePosixPath(here).parent / spelling)
        if near in self.paths:
            return [near]
        suffix = "/" + spelling.lstrip("./")
        return [p for p in self.by_base.get(PurePosixPath(spelling).name, []) if ("/" + p).endswith(suffix)]


def _imports_with_depth(tree: ast.AST):
    """Every import statement and whether it sits inside a function body (a lazy import: it runs when the function is called)."""
    stack = [(tree, False)]
    while stack:
        node, lazy = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, lazy
        inside = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        stack.extend((child, inside) for child in ast.iter_child_nodes(node))


def _python_imports(path: str, text: str, resolve: _Resolver) -> dict[str, bool]:
    """``{file: lazy}`` of the package files ``path`` imports; ``lazy`` is True when every import of it is inside a function."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    here = PurePosixPath(path).parent.parts                       # ('src', 'miniworld_engine', 'kernels', ...)
    found: dict[str, bool] = {}
    for node, lazy in _imports_with_depth(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(here[: len(here) - (node.level - 1)])
                module = ".".join([*base[1:], *([node.module] if node.module else [])])        # drop the leading 'src'
            else:
                module = node.module or ""
            names = [module, *(f"{module}.{a.name}" for a in node.names)]
        for name in names:
            if name.startswith("miniworld_engine"):
                hit = resolve.dotted(name)
                if hit:
                    found[hit] = found.get(hit, True) and lazy
    return found


def _native_ops() -> set[str]:
    """The names in ``native.BUILD_OPS`` (read from the source: no import, no GPU)."""
    tree = ast.parse(_text(f"{SRC}/autotune/native.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BUILD_OPS" for t in node.targets):
            return {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return set()


@functools.cache
def build() -> Graph:
    paths = files()
    g = Graph()
    g.nodes.update({p: kind_of(p) for p in paths})
    g.nodes.update({f"{NATIVE_PREFIX}{op}": "identity" for op in sorted(_native_ops())})
    g.nodes[DERIVATION] = "identity"
    resolve = _Resolver(paths)
    for p in paths:
        suffix = PurePosixPath(p).suffix
        if suffix not in SCANNED and p not in SCANNED_CSV:
            continue
        text = _text(p)
        if suffix == ".py":
            for dep, lazy in _python_imports(p, text, resolve).items():
                g.edge(p, dep, "lazy" if lazy else "import")
        if suffix in CCODE:
            for spelling in _INCLUDE_RE.findall(text):
                for dep in resolve.include(spelling, p):
                    g.edge(p, dep, "include")
        for token in _PATH_RE.findall(text):
            dep = resolve.path(token, p)
            if dep:
                g.edge(p, dep, "ref")
        for name in _DOTTED_RE.findall(text):
            dep = resolve.dotted(name)
            if dep:
                g.edge(p, dep, "ref")
    _caches(g, resolve)
    _identities(g)
    return g


def _caches(g: Graph, resolve: _Resolver) -> None:
    from miniworld_engine.build import code_graph_extra as extra

    registry = {}
    with (ROOT / SRC / "kernels/registry/registry.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            registry[row["kernel"]] = row
    for kernel in registry:
        g.nodes[f"registry-row:{kernel}"] = "registry-row"      # one node per registry row: an edit to one row reaches that kernel's caches only
    native = _native_ops()
    grid_dir, data_dir = f"{SRC}/autotune/configs/grid", f"{SRC}/autotune/data"
    for p in list(g.nodes):
        if not p.startswith(data_dir + "/"):
            continue
        op = PurePosixPath(p).parent.name
        for common in (f"{SRC}/autotune/cache.py", f"{SRC}/autotune/shape_key.py"):
            g.edge(p, common, "cache")
        if op in native:
            g.edge(p, f"{NATIVE_PREFIX}{op}", "identity")
            continue
        row = registry.get(op)
        g.edge(p, f"{grid_dir}/{op}.csv", "cache")
        g.edge(p, f"{SRC}/autotune/configs/default/{op}.csv", "cache")
        g.edge(p, f"registry-row:{op}", "cache")
        if row and row["file"].endswith(".py") and f"src/{row['file']}" in g.nodes:
            kernel_file = ROOT / "src" / row["file"]
            for f in extra.kernel_files(kernel_file, row.get("driver", "")):
                g.edge(p, str(f.relative_to(ROOT)), "cache")


def is_identity(node: str) -> bool:
    return node.startswith("identity:")


def _identities(g: Graph) -> None:
    from miniworld_engine.autotune import identity
    from miniworld_engine.build import code_graph_extra as extra

    for op in sorted(_native_ops()):
        for f in identity.native_files(op):                 # the op's own closure + the safety set: not "every kernels/ file"
            g.edge(f"{NATIVE_PREFIX}{op}", f"{SRC}/{f}", "identity")
    for f in extra.derivation_files():                      # the union of the files every unit of the committed derivation recorded
        g.edge(DERIVATION, str(f.relative_to(ROOT)), "identity")
    derived = ["kernels/registry/registry_kernel.units.json", "kernels/registry/registry_kernel.csv"]
    derived += [str(PurePosixPath(p).relative_to(SRC)) for p in g.nodes if p.startswith(f"{SRC}/autotune/plans/")]
    for path in derived:
        g.edge(f"{SRC}/{path}", DERIVATION, "identity")
    for page in (p for p in g.nodes if p.startswith("docs/kernels/") and p.endswith("sweep-grid.html")):
        for dep in (f"{SRC}/kernels/registry/registry_kernel.csv", f"{SRC}/viz/sweep_page.py"):
            g.edge(page, dep, "ref")


# --------------------------------------------------------------------------------------------------------- changes
def _normalized(path: str, text: str) -> str:
    suffix = PurePosixPath(path).suffix
    if suffix == ".py":
        try:
            return hashlib.sha256(ast.dump(ast.parse(text)).encode()).hexdigest()
        except SyntaxError:
            pass
    if suffix in CCODE:
        from miniworld_engine.kernels import _codegraph as cg

        return cg.parse_text(text).digest
    return hashlib.sha256(text.encode()).hexdigest()


def diff(base: str) -> tuple[list[str], list[str], list[str]]:
    """``(changed, cosmetic, deleted)``: files that differ from ``base`` in content, in bytes only (comments / whitespace), or are gone."""
    names = set(_git("diff", "--name-only", "-z", base).split("\0")) | set(
        _git("ls-files", "-z", "--others", "--exclude-standard").split("\0"))
    changed, cosmetic, deleted = [], [], []
    for p in sorted(n for n in names if n):
        if not (ROOT / p).is_file():
            deleted.append(p)
            continue
        try:
            old = _git("show", f"{base}:{p}")
        except subprocess.CalledProcessError:
            changed.append(p)                                   # new file
            continue
        except UnicodeDecodeError:
            changed.append(p)                                   # a binary file that differs
            continue
        new = _text(p)
        if old == new:
            continue
        (changed if _normalized(p, old) != _normalized(p, new) else cosmetic).append(p)
    return changed, cosmetic, deleted


def reach(g: Graph, seeds: list[str], lazy: bool = False) -> dict[str, str]:
    """Every node a change to ``seeds`` reaches, with the node it was reached from (the first one found). A ``lazy`` edge (an import
    inside a function) is walked only when ``lazy`` is True: the strong reach is what loading the code runs, the full reach what any
    call path could.

    The walk ends at an identity: what an edit owes the files behind it (the caches a restamp rewrites, the derivation a ``dev derive``
    regenerates) is read off the identity (``derived``), not folded into what the edit itself touches -- or every edit under
    ``kernels/`` would reach the same ~1,400 files."""
    parent = dict.fromkeys(seeds, "")
    queue = collections.deque(seeds)
    while queue:
        node = queue.popleft()
        if is_identity(node):
            continue
        for user, kind in g.users.get(node, {}).items():
            if user not in parent and kind != "identity" and (lazy or kind != "lazy"):
                parent[user] = node
                queue.append(user)
    return parent


REGISTRY_CSV = f"{SRC}/kernels/registry/registry.csv"


def _rows(text: str) -> dict[str, dict]:
    return {r["kernel"]: r for r in csv.DictReader(text.splitlines())}


def registry_rows_changed(base: str) -> list[str]:
    """The kernels whose ``registry.csv`` row was added, removed or edited since ``base``."""
    try:
        old = _rows(_git("show", f"{base}:{REGISTRY_CSV}"))
    except subprocess.CalledProcessError:
        return []
    new = _rows(_text(REGISTRY_CSV))
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


def changes(base: str) -> dict:
    g = build()
    changed, cosmetic, deleted = diff(base)
    seeds = [p for p in changed if p in g.nodes]
    rows = registry_rows_changed(base) if REGISTRY_CSV in changed else []
    seeds += [f"registry-row:{k}" for k in rows if f"registry-row:{k}" in g.nodes]
    # A native identity hashes a .py by bytes (a comment edit moves it) and a .cu normalized; the derivation's file tokens are
    # normalized for both. A changed registry row moves the identity of that kernel only.
    moved: dict[str, list[str]] = {}
    for ident in (i for i in g.nodes if is_identity(i)):
        native = ident.startswith(NATIVE_PREFIX)
        files_ = [p for p in changed if p in g.deps[ident]]
        if native:
            files_ += [p for p in cosmetic if p in g.deps[ident] and p.endswith(".py")]
            files_ += [f"registry-row:{k}" for k in rows if ident == f"{NATIVE_PREFIX}{k}"]
        if files_:
            moved[ident] = files_
            seeds.append(ident)
    return {"graph": g, "changed": changed, "cosmetic": cosmetic, "deleted": deleted, "moved": moved, "rows": rows,
            "reached": reach(g, seeds), "reached_lazy": reach(g, seeds, lazy=True)}


def cuda_rebuilds(result: dict, manifests: list[str]) -> list[tuple[str, str]]:
    """``(artifact, changed file)`` for every known CUDA artifact (local JIT cache, installed bundles, ``manifests``) whose recorded graph
    holds a changed ``.cu`` / ``.cuh`` whose normalized content now differs from the recorded one: what a rebuild will compile."""
    from miniworld_engine.build import kernel_bundle as kb
    from miniworld_engine.kernels import _codegraph as cg

    g = result["graph"]
    files_ = [p for p in result["changed"] if g.nodes.get(p) == "cuda-source" and (ROOT / p).is_file()]
    if not files_:
        return []
    known = kb.known_artifacts(manifests)
    out = []
    for p in files_:
        label, digest = kb.file_label(ROOT / p), cg.parse_file(str(ROOT / p)).digest
        for a in known:
            nodes = {n[0]: n[1] for n in (a.get("material") or {}).get("nodes") or []}
            if label in nodes and nodes[label] != digest:
                out.append((a["name"], p))
    return sorted(set(out))


def live_evidence(result: dict, new_only: bool = False) -> dict:
    """What the committed evidence says NOW, against the working tree (CPU only): the stale tuned caches among the ones reached, the
    native ops whose cache no longer matches their graph, and the derivation units whose recorded files moved.

    ``new_only``: keep only what the diff itself causes -- the native ops whose graph holds a changed file and the units that
    recorded one -- not staleness that was already there at ``base``."""
    from miniworld_engine.autotune import cache_status, identity, plan

    g = result["graph"]
    rows = cache_status.scan()
    reached_ops = {PurePosixPath(n).parent.name for n in result["reached"] if g.nodes[n] == "cache"}
    touched = {p.removeprefix(f"{SRC}/") for p in [*result["changed"], *result["cosmetic"]] if p.startswith(f"{SRC}/")}
    native_stale = sorted({r.op for r in rows if r.stale and r.reason.startswith("native implementation changed")})
    status = plan.status()
    units_stale = status["stale"]
    if new_only:
        rows_changed = set(result.get("rows") or [])
        seeds = {*result["changed"], *(f"registry-row:{k}" for k in rows_changed)}
        direct = {PurePosixPath(n).parent.name for n, kind in g.nodes.items() if kind == "cache" and seeds & set(g.deps[n])}
        native_stale = [op for op in native_stale if op in rows_changed or set(identity.native_files(op)) & touched]
        units_stale = {k: v for k, v in units_stale.items() if set(v) & touched}
        reached_ops &= direct                              # a cache whose own files (kernel, helpers, driver, grid, row) hold a changed file
    return {"stale": [r for r in rows if r.stale and r.op in reached_ops], "native_stale": native_stale,
            "units_stale": units_stale, "units_total": status["units"], "schema": status["schema"], "new_only": new_only}


def _chain(parent: dict[str, str], node: str) -> str:
    out = [node]
    while parent.get(out[-1]):
        out.append(parent[out[-1]])
    return " <- ".join(out[:4]) + (" <- ..." if len(out) > 4 else "")


def actions(result: dict) -> list[str]:
    g, reached, changed = result["graph"], result["reached"], set(result["changed"])
    by_kind: dict[str, list[str]] = collections.defaultdict(list)
    for node in reached:
        by_kind[g.nodes[node]].append(node)
    todo = []
    moved = result.get("moved", {})
    native_moved = {i.removeprefix(NATIVE_PREFIX): f for i, f in moved.items() if i.startswith(NATIVE_PREFIX)}
    if native_moved:
        every = sorted(i.removeprefix(NATIVE_PREFIX) for i in g.nodes if i.startswith(NATIVE_PREFIX))
        todo.append(f"restamp {len(native_moved)} of {len(every)} native ops (dev restamp-native; the other {len(every) - len(native_moved)} stay): "
                    + ", ".join(sorted(native_moved)[:8]) + " -- moved by "
                    + ", ".join(sorted({PurePosixPath(p).name for fs in native_moved.values() for p in fs})[:4]))
    if moved.get(DERIVATION):
        todo.append(f"dev derive --arch sm86: re-runs only the units whose recorded files changed, reuses the rest; then viz.sweep_page "
                    f"-- moved by {len(moved[DERIVATION])} file(s): " + ", ".join(moved[DERIVATION][:4]) + (" ..." if len(moved[DERIVATION]) > 4 else ""))
    if result.get("rows"):
        todo.append(f"registry rows changed ({len(result['rows'])}): " + ", ".join(result["rows"][:8]) + (" ..." if len(result["rows"]) > 8 else "")
                    + " -- their tuned caches, and run_all on each card that holds their manifest rows")
    caches = sorted(n for n in by_kind["cache"] if n not in changed)
    ops = sorted({PurePosixPath(n).parent.name for n in caches})
    if ops:
        todo.append(f"tuned caches reached ({len(ops)} ops, {len(caches)} files): retune / `dev cache-status` -- {', '.join(ops[:12])}"
                    + (" ..." if len(ops) > 12 else ""))
    todo.extend(f"  STALE now: {r.op} [{r.gpu}] {r.reason}" for r in (result.get("evidence") or {}).get("stale", [])[:10])
    rebuilds = result.get("rebuilds") or []
    if rebuilds:
        names = sorted({a for a, _ in rebuilds})
        todo.append(f"CUDA artifacts that rebuild ({len(names)}): " + ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))
    elif by_kind["cuda-source"] or any(g.nodes.get(p) == "cuda-source" for p in changed):
        todo.append(f"CUDA sources reached ({len(by_kind['cuda-source'])}): pass --manifest <bundle> to name the artifacts that rebuild")
    gpu = sorted(by_kind["test-gpu"])
    cpu = sorted(by_kind["test"])
    if cpu:
        todo.append(f"CPU tests reached: {len(cpu)} files")
    if gpu:
        todo.append(f"GPU tests reached: {len(gpu)} files")
    docs = sorted(by_kind["doc"])
    if docs:
        todo.append(f"docs that name what changed ({len(docs)}): " + ", ".join(docs[:8]) + (" ..." if len(docs) > 8 else ""))
    bench = sorted(by_kind["benchmark"])
    if bench:
        todo.append(f"benchmark files reached: {len(bench)}")
    return todo


def pending(result: dict) -> list[str]:
    """What the committed evidence cannot be trusted for, as of now: stale caches among the ones the change reaches, native ops whose
    cache no longer matches their graph, derivation units whose recorded files moved. ``--check`` fails on any."""
    ev = result.get("evidence") or {}
    out = [f"stale cache {r.op} [{r.gpu}]: {r.reason}" for r in ev.get("stale", [])]
    if ev.get("native_stale"):
        out.append(f"native caches stale ({len(ev['native_stale'])}): dev restamp-native -- {', '.join(ev['native_stale'][:6])}")
    if ev.get("units_stale"):
        out.append(f"derivation: {len(ev['units_stale'])} of {ev.get('units_total', '?')} units stale (the rest are reused): dev derive")
    return out


def tests(result: dict, kind: str = "all") -> list[str]:
    """The test files the change reaches (through every import, lazy ones included), one repo path each: ``pytest $(... --tests)``."""
    g = result["graph"]
    want = {"cpu": {"test"}, "gpu": {"test-gpu"}, "all": {"test", "test-gpu"}}[kind]
    return sorted(n for n in result["reached_lazy"] if g.nodes[n] in want and PurePosixPath(n).name.startswith("test_") and n.endswith(".py")
                  and (ROOT / n).is_file())


def report(base: str, verbose: bool = False, manifests: list[str] = (), new_only: bool = False) -> tuple[str, int]:
    result = changes(base)
    result["rebuilds"] = cuda_rebuilds(result, list(manifests))
    result["evidence"] = live_evidence(result, new_only=new_only)
    g, reached = result["graph"], result["reached"]
    lines = [(f"{len(result['changed'])} changed, {len(result['cosmetic'])} cosmetic (comments / whitespace), {len(result['deleted'])} deleted "
              f"vs {base}; the graph: {len(g.nodes)} nodes, {sum(len(d) for d in g.deps.values())} edges {dict(g.edge_count())}")]
    for label, group in (("changed", result["changed"]), ("cosmetic", result["cosmetic"])):
        lines.extend(f"  {label:8s} {p}  ->  {len(reach(g, [p])) - 1 if p in g.nodes else 0} nodes" for p in group)
    lines.extend(f"  deleted  {p}  (not traced: its users still name it)" for p in result["deleted"])
    counts = collections.Counter(g.nodes[n] for n in reached)
    lines.append("reached: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    extra = collections.Counter(g.nodes[n] for n in result["reached_lazy"] if n not in reached)
    if extra:
        lines.append("  plus through lazy imports (an import inside a function): " + ", ".join(f"{k} {v}" for k, v in sorted(extra.items())))
    lines.append(f"tests to run (--tests): {len(tests(result, 'cpu'))} CPU, {len(tests(result, 'gpu'))} GPU")
    lines.append("owed:")
    lines.extend(f"  - {a}" for a in actions(result) or ["nothing"])
    ev = result["evidence"]
    lines.append(f"evidence{' (new-only: what this diff causes)' if new_only else ''}: derivation {len(ev['units_stale'])} of {ev['units_total']} units stale, "
                 f"native {len(ev['native_stale'])} op(s) stale, tuned {len(ev['stale'])} reached cache(s) stale")
    owed = pending(result)
    lines.append(f"check: {len(owed)} pending" + ("" if not owed else " -- " + "; ".join(owed[:4]) + (" ..." if len(owed) > 4 else "")))
    if verbose:
        lines.append("reached nodes:")
        lines.extend(f"  {g.nodes[n]:12s} {_chain(reached, n)}" for n in sorted(reached))
        lines.extend(f"  stale unit  {label}  <- {', '.join(files[:3])}" for label, files in sorted(ev["units_stale"].items())[:200])
    return "\n".join(lines), len(owed)


def diff_tests(base: str, kind: str = "all") -> list[str]:
    return tests(changes(base), kind)


def summary() -> str:
    g = build()
    kinds = collections.Counter(g.nodes.values())
    isolated = sorted(p for p, k in g.nodes.items() if k in ("python", "cuda-source", "test", "test-gpu") and not g.deps.get(p) and not g.users.get(p))
    return (f"{len(g.nodes)} nodes {dict(kinds)}\n{sum(len(d) for d in g.deps.values())} edges {dict(g.edge_count())}\n"
            f"{len(isolated)} code / test files with no edge in or out" + ("".join(f"\n  {p}" for p in isolated[:30])))
