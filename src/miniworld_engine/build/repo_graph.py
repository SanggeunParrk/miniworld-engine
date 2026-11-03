"""The whole repository as one graph, and what a change does to it.

Every tracked file (and every untracked, unignored one) is a node. An edge ``A -> B`` means "A is built from / reads / names B":
a change to B can change A. Edges come from what the files themselves say, not from a hand-kept list:

* ``import``  -- Python ``import`` / ``from ... import`` anywhere in a file (a function-level import is a dependency too);
* ``include`` -- ``#include "..."`` in ``.cu`` / ``.cuh`` / ``.inc`` / ``.h`` / ``.cpp``;
* ``ref``     -- a repo path or a ``miniworld_engine.x.y`` dotted name written in a source, doc, YAML or registry file
  (registry rows name each kernel's file, driver and check; a doc names the code it describes);
* ``ref`` (built paths) -- a path a python file BUILDS from segments (``ROOT / "docs" / name``, ``Path(__file__).parents[2] / "src"``,
  ``glob`` / ``rglob`` / ``iterdir`` of a directory): :mod:`repo_paths` evaluates the expression; a known path is an edge to the file, a prefix
  or an enumerated directory an edge to every file below it (never narrower than what the code can reach);
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

One graph serves every card (:mod:`cards`): each node has a ``cards`` set (``ALL`` for a card-independent node). ``reach_cards`` walks the
same edges but carries, per node, the cards the change reaches it for -- an edge into a card-specific kernel carries that kernel's card, a
card-independent seed carries every card. The report is then one section per card, with the evidence state of that card: computed
(``fresh`` / ``stale``) for the cards this environment can run on (``cards.accessible``) and ``UNVERIFIABLE-HERE (owed on <card>)`` for the
rest. ``--check`` fails only on stale evidence of an accessible card (``--strict-cards``: on unverifiable ones too).
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

from miniworld_engine.build import cards as card_mod
from miniworld_engine.build import repo_paths

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
        self.cards: dict[str, frozenset[str]] = {}          # node -> the cards it depends on (card_mod.ALL: card-independent)
        self.known_cards: frozenset[str] = frozenset()

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
def _module_resolver(resolve: _Resolver):
    """``repo_paths.Analyzer``'s ``resolve_module(module, here, level)``: an import -> the repo file it names (or None)."""

    def find(module: str, here: str, level: int) -> str | None:
        if level == 0:
            target = resolve.dotted(module) if module.startswith("miniworld_engine") else None
            base = module.replace(".", "/")
        else:
            up = PurePosixPath(here).parent
            for _ in range(level - 1):
                up = up.parent
            base = str(up / module.replace(".", "/")) if module else str(up)
            target = None
        for cand in (target, f"{base}.py", f"{base}/__init__.py"):
            if cand and cand in resolve.paths:
                return cand
        return None

    return find


def _built_paths(g: Graph, path: str, use: repo_paths.PathUse, resolve: _Resolver, below: dict[str, list[str]]) -> None:
    """The edges of the paths a python file builds: exact ones to their file, prefixes and enumerated directories to every file below."""
    for exact in sorted(use.exact):
        if exact in resolve.paths and exact != path:
            g.edge(path, exact, "ref")
    wanted = set(use.prefixes)
    if use.enumerates or use.opaque_enumeration:
        wanted |= {v[1] for v in use.values}
    for prefix in sorted(wanted):
        if not prefix:
            continue                                   # the repository root itself: too wide to be an answer, the named parts say more
        for dep in below.get(prefix, ()):
            if dep != path:
                g.edge(path, dep, "ref")


def _files_below(paths: list[str], prefixes: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {p: [] for p in prefixes}
    for f in paths:
        for p in prefixes:
            if f.startswith(p):
                out[p].append(f)
    return out


def build() -> Graph:
    paths = files()
    g = Graph()
    g.nodes.update({p: kind_of(p) for p in paths})
    g.nodes.update({f"{NATIVE_PREFIX}{op}": "identity" for op in sorted(_native_ops())})
    g.nodes[DERIVATION] = "identity"
    resolve = _Resolver(paths)
    analyzer = repo_paths.Analyzer(set(paths), _text, _module_resolver(resolve))
    uses = {p: analyzer.use(p) for p in paths if p.startswith("tests/") and PurePosixPath(p).suffix == ".py"}
    wanted = set()
    for use in uses.values():
        wanted |= {x for x in use.prefixes if x}
        if use.enumerates or use.opaque_enumeration:
            wanted |= {v[1] for v in use.values if v[1]}
    below = _files_below(paths, wanted)
    for p in paths:
        suffix = PurePosixPath(p).suffix
        if suffix not in SCANNED and p not in SCANNED_CSV:
            continue
        text = _text(p)
        if suffix == ".py":
            if p.startswith("tests/"):                    # a test that lists or builds a path depends on it; the sources do not (native.py lists every kernel as a salt, not as a dependency)
                _built_paths(g, p, uses[p], resolve, below)
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
    card_mod.assign(g, paths, ROOT)
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
        for dep in (f"{SRC}/kernels/registry/registry_kernel.csv", f"{SRC}/utils/viz/sweep_page.py"):
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
            "reached": reach(g, seeds), "reached_lazy": reach(g, seeds, lazy=True),
            "card_reached": reach_cards(g, seeds), "card_reached_lazy": reach_cards(g, seeds, lazy=True)}


def cuda_rebuilds(result: dict, manifests: list[str]) -> list[tuple[str, str]]:
    """``(artifact, changed file)`` for every known CUDA artifact (local JIT cache, installed bundles, ``manifests``) whose recorded graph
    holds a changed ``.cu`` / ``.cuh`` whose normalized content now differs from the recorded one: what a rebuild will compile."""
    return sorted({(a, p) for a, p, _ in _rebuild_triples(result, manifests)})


def derivation_plans() -> dict[str, Path]:
    """``{arch: units.json}`` of every committed per-card plan."""
    return {p.parent.name: p for p in sorted((PKG / "autotune" / "plans").glob("*/registry_kernel.units.json"))}


# ------------------------------------------------------------------------------------------------------------------ cards
#: node kinds whose correctness or evidence needs the card's hardware (a doc or a core Python file does not)
HARDWARE_KINDS = {"cuda-source", "cache", "test-gpu", "benchmark", "registry-row", "identity"}


def reach_cards(g: Graph, seeds: list[str], lazy: bool = False) -> dict[str, frozenset[str]]:
    """Like :func:`reach`, but per card: ``{node: the cards the change reaches it for}``.

    A seed carries its own cards (a card-independent one carries every card). The walk intersects: a user is reached for the cards its
    dependency was reached for AND that the user itself depends on, so an edge into an sm100 kernel carries ``sm100`` only and an
    sm100-only test is never reached by an sm90 change. The exception is a ``cache`` edge: a tuned cache is evidence *of its own card* for
    the kernel it was measured from, whatever cards that kernel serves today (an H100 cache of a Triton kernel goes stale when the kernel
    changes), so it is reached for its own card whenever the kernel is reached at all."""
    known = g.known_cards

    def cards_of(node: str) -> frozenset[str]:
        return card_mod.expand(g.cards.get(node, card_mod.ALL), known)

    got: dict[str, set[str]] = {}
    queue = collections.deque()
    for s in seeds:
        c = cards_of(s)
        if c:
            got[s] = set(c)
            queue.append(s)
    while queue:
        node = queue.popleft()
        if is_identity(node):
            continue
        here = got[node]
        for user, kind in g.users.get(node, {}).items():
            if kind == "identity" or (kind == "lazy" and not lazy):
                continue
            mine = cards_of(user)
            new = set(mine) if kind == "cache" else here & mine
            if new - got.get(user, set()):
                got.setdefault(user, set()).update(new)
                queue.append(user)
    return {n: frozenset(c) for n, c in got.items()}


def view(result: dict, card: str, lazy: bool = False) -> set[str]:
    """The nodes the change reaches for ``card``."""
    return {n for n, c in result["card_reached_lazy" if lazy else "card_reached"].items() if card in c}


def reached_cards(result: dict) -> list[str]:
    seen: set[str] = set()
    for c in result["card_reached_lazy"].values():
        seen |= c
    return card_mod.sort_cards(seen)


def hardware_bound(g: Graph, nodes) -> bool:
    """Does what the change reaches for a card need that card's hardware: a CUDA source, a tuned cache, a GPU test, a registry row, an
    identity, a benchmark, a kernel file that is itself card-specific. (A doc or a core Python file reached for every card is not.)"""
    return any(g.nodes[n] in HARDWARE_KINDS or (g.nodes[n] == "python" and g.cards.get(n, card_mod.ALL) != card_mod.ALL) for n in nodes)


def card_evidence(result: dict, card: str, new_only: bool = False, rows: list | None = None) -> dict:
    """What the committed evidence of ``card`` says NOW, against the working tree (CPU only, computable for every card from file digests):
    the stale tuned caches among the ones reached, the native ops whose cache no longer matches their graph, and the derivation units of
    that card's plan (``autotune/plans/<card>/``) whose recorded files moved. A derive on one card re-records that card's plan only, so a
    card's plan is never judged by another's.

    ``new_only``: keep only what the diff itself causes -- the native ops whose graph holds a changed file and the units that recorded
    one -- not staleness that was already there at ``base``."""
    from miniworld_engine.autotune import cache_status, identity, plan

    g = result["graph"]
    rows = cache_status.scan() if rows is None else rows
    mine = [r for r in rows if f"({card})" in r.gpu]
    native = _native_ops()
    reached_ops = {PurePosixPath(n).parent.name for n in view(result, card) if g.nodes[n] == "cache"}
    touched = {p.removeprefix(f"{SRC}/") for p in [*result["changed"], *result["cosmetic"]] if p.startswith(f"{SRC}/")}
    native_stale = sorted({r.op for r in mine if r.stale and r.reason.startswith("native implementation changed")})
    plans = derivation_plans()
    has_plan = card in plans
    status = plan.status(plans[card]) if has_plan else {"units": 0, "stale": {}, "schema": 0}
    units_stale = status["stale"]
    if new_only:
        rows_changed = set(result.get("rows") or [])
        seeds = {*result["changed"], *(f"registry-row:{k}" for k in rows_changed)}
        direct = {PurePosixPath(n).parent.name for n, kind in g.nodes.items() if kind == "cache" and seeds & set(g.deps[n])}
        native_stale = [op for op in native_stale if op in rows_changed or set(identity.native_files(op)) & touched]
        units_stale = {k: v for k, v in units_stale.items() if set(v) & touched}
        reached_ops &= direct                              # a cache whose own files (kernel, helpers, driver, grid, row) hold a changed file
    return {"card": card, "stale": [r for r in mine if r.stale and r.op in reached_ops], "native_stale": native_stale,
            "units_stale": units_stale, "units_total": status["units"], "schema": status["schema"], "new_only": new_only,
            "has_plan": has_plan, "native_rows": sum(1 for r in mine if r.op in native), "tuned_reached": len(reached_ops)}


def _chain(parent: dict[str, str], node: str) -> str:
    out = [node]
    while parent.get(out[-1]):
        out.append(parent[out[-1]])
    return " <- ".join(out[:4]) + (" <- ..." if len(out) > 4 else "")


def actions(result: dict, card: str = "", accessible: bool = True) -> list[str]:
    """What the change owes. ``card`` restricts it to the nodes reached for that card and labels each line: a command is a plain
    ``[sm90] <command>`` on an accessible card and ``[sm100] owed, cannot run here: <command>`` on any other (running it needs that
    card's hardware). Without ``card``: everything reached, unlabelled."""
    g, changed = result["graph"], set(result["changed"])
    reached = set(view(result, card)) if card else set(result["reached"])
    if card:
        changed = {p for p in changed if card in result["card_reached"].get(p, ())}
    by_kind: dict[str, list[str]] = collections.defaultdict(list)
    for node in reached:
        by_kind[g.nodes[node]].append(node)
    tag = f"[{card}] " if card else ""
    run = tag if accessible else f"{tag}owed, cannot run here: "
    todo = []
    moved = {i: f for i, f in result.get("moved", {}).items() if not card or card in result["card_reached"].get(i, ())}
    native_moved = {i.removeprefix(NATIVE_PREFIX): f for i, f in moved.items() if i.startswith(NATIVE_PREFIX)}
    if native_moved:
        every = sorted(i.removeprefix(NATIVE_PREFIX) for i in g.nodes if i.startswith(NATIVE_PREFIX)
                       and (not card or card in card_mod.expand(g.cards.get(i, card_mod.ALL), g.known_cards)))
        todo.append(f"{run}restamp {len(native_moved)} of {len(every)} native ops (dev restamp-native; the other {len(every) - len(native_moved)} stay): "
                    + ", ".join(sorted(native_moved)[:8]) + " -- moved by "
                    + ", ".join(sorted({PurePosixPath(p).name for fs in native_moved.values() for p in fs})[:4]))
    if result.get("moved", {}).get(DERIVATION):
        files_ = result["moved"][DERIVATION]
        for arch in ([card] if card else sorted(derivation_plans())):
            if arch not in derivation_plans():
                continue
            todo.append(f"{run if card else ''}dev derive --arch {arch} --out src/miniworld_engine/autotune/plans/{arch}/registry_kernel.csv: re-runs only "
                        f"the units whose recorded files changed, reuses the rest; then viz.sweep_page "
                        f"-- moved by {len(files_)} file(s): " + ", ".join(files_[:4]) + (" ..." if len(files_) > 4 else ""))
    if result.get("rows"):
        todo.append(f"{tag}registry rows changed ({len(result['rows'])}): " + ", ".join(result["rows"][:8]) + (" ..." if len(result["rows"]) > 8 else "")
                    + " -- their tuned caches, and run_all on each card that holds their manifest rows")
    caches = sorted(n for n in by_kind["cache"] if n not in changed)
    ops = sorted({PurePosixPath(n).parent.name for n in caches})
    if ops:
        todo.append(f"{run}tuned caches reached ({len(ops)} ops, {len(caches)} files): retune / `dev cache-status` -- {', '.join(ops[:12])}"
                    + (" ..." if len(ops) > 12 else ""))
    todo.extend(f"  STALE now: {r.op} [{r.gpu}] {r.reason}" for r in (result.get("evidence") or {}).get("stale", [])[:10])
    rebuilds = (result.get("rebuilds_by_card", {}).get(card, []) if card else result.get("rebuilds")) or []
    cuda_changed = [n for n in by_kind["cuda-source"] if n in changed]
    if rebuilds:
        names = sorted({a for a, _ in rebuilds})
        todo.append(f"{run}CUDA artifacts that rebuild ({len(names)}): " + ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))
    elif by_kind["cuda-source"] or cuda_changed:
        todo.append(f"{run}CUDA sources reached ({len(by_kind['cuda-source'])}): "
                    + ("rebuild the prebuilt bundle of this card" if not accessible else "pass --manifest <bundle> to name the artifacts that rebuild"))
    gpu_tests = sorted(by_kind["test-gpu"])
    cpu_tests = sorted(by_kind["test"])
    if cpu_tests:
        todo.append(f"{tag}CPU tests reached: {len(cpu_tests)} files")
    if gpu_tests:
        todo.append(f"{run}GPU tests reached: {len(gpu_tests)} files")
    docs = sorted(by_kind["doc"])
    if docs:
        todo.append(f"{tag}docs that name what changed ({len(docs)}): " + ", ".join(docs[:8]) + (" ..." if len(docs) > 8 else ""))
    bench = sorted(by_kind["benchmark"])
    if bench:
        todo.append(f"{run}benchmark files reached: {len(bench)}")
    return todo


def pending(result: dict) -> list[str]:
    """What the committed evidence of a card cannot be trusted for, as of now: stale caches among the ones the change reaches, native ops
    whose cache no longer matches their graph, derivation units whose recorded files moved. ``result["evidence"]`` is one card's
    :func:`card_evidence`."""
    ev = result.get("evidence") or {}
    tag = f"[{ev['card']}] " if ev.get("card") else ""
    out = [f"{tag}stale cache {r.op} [{r.gpu}]: {r.reason}" for r in ev.get("stale", [])]
    if ev.get("native_stale"):
        out.append(f"{tag}native caches stale ({len(ev['native_stale'])}): dev restamp-native -- {', '.join(ev['native_stale'][:6])}")
    if ev.get("units_stale"):
        out.append(f"{tag}derivation: {len(ev['units_stale'])} of {ev.get('units_total', '?')} units stale (the rest are reused): dev derive")
    return out


def tests(result: dict, kind: str = "all") -> list[str]:
    """The test files the change reaches (through every import, lazy ones included), one repo path each: ``pytest $(... --tests)``."""
    g = result["graph"]
    want = {"cpu": {"test"}, "gpu": {"test-gpu"}, "all": {"test", "test-gpu"}}[kind]
    return sorted(n for n in result["reached_lazy"] if g.nodes[n] in want and PurePosixPath(n).name.startswith("test_") and n.endswith(".py")
                  and (ROOT / n).is_file())


def plan_tests(result: dict, kind: str = "all", accessible=frozenset()) -> tuple[list[str], dict[str, list[str]]]:
    """``(runnable here, {card: tests that need that card})``. A CPU test is card-independent and runs here. A GPU test reached for an
    accessible card runs here; one reached only for another card (or bound to it: ``tests/gpu/sm100/``) is listed under that card and is
    not claimed verified."""
    g = result["graph"]
    want = {"cpu": {"test"}, "gpu": {"test-gpu"}, "all": {"test", "test-gpu"}}[kind]
    runnable: set[str] = set()
    needs: dict[str, set[str]] = {}
    for card in reached_cards(result):
        for n in view(result, card, lazy=True):
            if g.nodes[n] not in want or not PurePosixPath(n).name.startswith("test_") or not n.endswith(".py") or not (ROOT / n).is_file():
                continue
            if g.nodes[n] == "test" or card in accessible:
                runnable.add(n)
            else:
                needs.setdefault(card, set()).add(n)
    return sorted(runnable), {c: sorted(v) for c, v in sorted(needs.items(), key=lambda kv: card_mod._order(kv[0]))}


# ------------------------------------------------------------------------------------------------------------------ report
class Report:
    """The per-card report. ``blocking``: pending items of ACCESSIBLE cards (``--check`` fails on them); ``unverifiable``: the cards this
    environment cannot judge that the change reaches hardware-bound nodes of (``--strict-cards`` fails on them)."""

    def __init__(self, text: str, blocking: list[str], unverifiable: list[str]) -> None:
        self.text, self.blocking, self.unverifiable = text, blocking, unverifiable

    def code(self, check: bool = True, strict: bool = False) -> int:
        if not check:
            return 0
        if self.blocking:
            return 2
        return 3 if strict and self.unverifiable else 0


def _rebuild_triples(result: dict, manifests: list[str]) -> list[tuple[str, str, str]]:
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
                out.append((a["name"], p, a.get("where", "")))
    return sorted(set(out))


def _bundle_card(where: str) -> str:
    m = re.search(r"(sm\d+)-cu\d+", where)
    return m.group(1) if m else ""


def _evidence_lines(card: str, ev: dict, accessible: bool, bound: bool, new_only: bool) -> list[str]:
    """The state per evidence kind. Staleness of committed files is computed from digests for every card; whatever needs the card's
    hardware (running the tests, building CUDA, retuning, benchmarking) is ``UNVERIFIABLE-HERE`` on a card this environment cannot run."""
    scope = " (new-only: what this diff causes)" if new_only else ""
    out = [f"evidence [{card}]{scope}:"]
    if ev["has_plan"]:
        n = len(ev["units_stale"])
        out.append(f"    derivation plan   : " + (f"stale {n} of {ev['units_total']} units (computed from file digests)" if n
                                                  else f"fresh, 0 of {ev['units_total']} units stale (computed from file digests)"))
    else:
        out.append("    derivation plan   : no committed plan for this card")
    if ev["native_rows"]:
        n = len(ev["native_stale"])
        out.append("    native caches     : " + (f"stale {n} op(s) (computed from identity digests)" if n
                                                 else f"fresh, {ev['native_rows']} committed (computed from identity digests)"))
    else:
        out.append("    native caches     : none committed for this card")
    n = len(ev["stale"])
    out.append("    tuned caches      : " + (("none caused by this diff" if new_only else "none reached") if not ev["tuned_reached"] else
                                              f"stale {n} of {ev['tuned_reached']} reached op(s)" if n else f"fresh, {ev['tuned_reached']} reached op(s)"))
    if accessible:
        out.append("    hardware evidence : run the owed commands and the tests listed (this card is accessible here)")
    elif bound:
        out.append(f"    hardware evidence : UNVERIFIABLE-HERE (owed on {card}): tests, CUDA build, retune, bench")
    else:
        out.append("    hardware evidence : nothing reached needs this card's hardware")
    return out


def evaluate(base: str, verbose: bool = False, manifests: list[str] = (), new_only: bool = False, gpu: str = "",
             accessible: str | list[str] = "") -> Report:
    from miniworld_engine.autotune import cache_status

    result = changes(base)
    g = result["graph"]
    triples = _rebuild_triples(result, list(manifests))
    result["rebuilds"] = sorted({(a, p) for a, p, _ in triples})
    here, source = card_mod.accessible(gpu or accessible)
    known = set(g.known_cards) | set(here)
    cards = reached_cards(result)
    by_card: dict[str, list] = {}
    for a, p, where in triples:
        for c in ([_bundle_card(where)] if _bundle_card(where) else card_mod.sort_cards(result["card_reached"].get(p, ()))):
            by_card.setdefault(c, []).append((a, p))
    result["rebuilds_by_card"] = by_card
    rows = cache_status.scan()
    reached = result["reached"]
    lines = [(f"{len(result['changed'])} changed, {len(result['cosmetic'])} cosmetic (comments / whitespace), {len(result['deleted'])} deleted "
              f"vs {base}; the graph: {len(g.nodes)} nodes, {sum(len(d) for d in g.deps.values())} edges {dict(g.edge_count())}")]
    for label, group in (("changed", result["changed"]), ("cosmetic", result["cosmetic"])):
        lines.extend(f"  {label:8s} {p}  ->  {len(reach(g, [p])) - 1 if p in g.nodes else 0} nodes  [{card_mod.label(g.cards.get(p, card_mod.ALL))}]"
                     for p in group)
    lines.extend(f"  deleted  {p}  (not traced: its users still name it)" for p in result["deleted"])
    lines.append(f"cards: accessible here {', '.join(card_mod.sort_cards(here))} (from {source}); known {', '.join(card_mod.sort_cards(known))}; "
                 f"reached {', '.join(cards) or 'none'}" + (f"; not reached {', '.join(card_mod.sort_cards(known - set(cards)))}" if known - set(cards) else ""))
    counts = collections.Counter(g.nodes[n] for n in reached)
    lines.append("reached (all cards): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    extra = collections.Counter(g.nodes[n] for n in result["reached_lazy"] if n not in reached)
    if extra:
        lines.append("  plus through lazy imports (an import inside a function): " + ", ".join(f"{k} {v}" for k, v in sorted(extra.items())))
    runnable_all, needs_all = plan_tests(result, "all", frozenset(here))
    cpu_here = [t for t in runnable_all if g.nodes[t] == "test"]
    gpu_here = [t for t in runnable_all if g.nodes[t] == "test-gpu"]
    lines.append(f"tests to run here (--tests): {len(cpu_here)} CPU, {len(gpu_here)} GPU; needing another card: "
                 + (", ".join(f"{c} {len(v)}" for c, v in needs_all.items()) or "none"))
    blocking: list[str] = []
    unverifiable: list[str] = []
    for card in cards:
        nodes = view(result, card)
        nodes_lazy = view(result, card, lazy=True)
        ok = card in here
        bound = hardware_bound(g, nodes_lazy)
        state = "accessible here" if ok else ("UNVERIFIABLE-HERE (owed on " + card + ")" if bound else "not hardware-bound")
        lines.append(f"== [{card}] {state} ==")
        kinds = collections.Counter(g.nodes[n] for n in nodes)
        lines.append("  reached: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        card_extra = collections.Counter(g.nodes[n] for n in nodes_lazy if n not in nodes)
        if card_extra:
            lines.append("    plus through lazy imports: " + ", ".join(f"{k} {v}" for k, v in sorted(card_extra.items())))
        run, needs = plan_tests({**result, "card_reached_lazy": {n: frozenset({card}) for n in nodes_lazy}}, "all", frozenset(here))
        if ok:
            lines.append(f"  tests to run: {len([t for t in run if g.nodes[t] == 'test'])} CPU, {len([t for t in run if g.nodes[t] == 'test-gpu'])} GPU")
        else:
            gpu_needed = needs.get(card, [])
            lines.append(f"  tests needing {card} (not run, not claimed verified): {len(gpu_needed)} GPU"
                         + (f"; {len([t for t in run if g.nodes[t] == 'test'])} CPU tests run anywhere (counted above)" if run else ""))
        ev = card_evidence(result, card, new_only=new_only, rows=rows)
        result["evidence"] = ev
        todo = actions(result, card, accessible=ok) if (ok or bound) else []
        lines.append("  owed:")
        lines.extend(f"    - {a}" for a in todo or ["nothing"])
        if ok or bound:
            lines.extend("  " + ln for ln in _evidence_lines(card, ev, ok, bound, new_only))
        if ok:
            blocking.extend(pending(result))
        elif bound:
            unverifiable.append(card)
        if verbose:
            parents = result["reached_lazy"]
            lines.append(f"  reached nodes [{card}]:")
            lines.extend(f"    {g.nodes[n]:12s} {_chain(parents, n)}" for n in sorted(nodes))
            lines.extend(f"    stale unit  {label}  <- {', '.join(files[:3])}" for label, files in sorted(ev["units_stale"].items())[:200])
    lines.append(f"check: {len(blocking)} pending on accessible cards ({', '.join(card_mod.sort_cards(here))})"
                 + ("" if not blocking else " -- " + "; ".join(blocking[:4]) + (" ..." if len(blocking) > 4 else "")))
    lines.append(f"unverifiable here: {len(unverifiable)} card(s)" + ("" if not unverifiable else
                 f" ({', '.join(unverifiable)}): owed on those cards, not judged, never counted as fresh; --strict-cards makes them fail --check"))
    result["evidence"] = {}
    return Report("\n".join(lines), blocking, unverifiable)


def report(base: str, verbose: bool = False, manifests: list[str] = (), new_only: bool = False, gpu: str = "",
           accessible: str | list[str] = "") -> tuple[str, int]:
    r = evaluate(base, verbose=verbose, manifests=manifests, new_only=new_only, gpu=gpu, accessible=accessible)
    return r.text, len(r.blocking)


def diff_tests(base: str, kind: str = "all", accessible: str | list[str] = "", gpu: str = "") -> tuple[list[str], dict[str, list[str]]]:
    """``(tests that run here, {card: tests that need that card})`` for the files that differ from ``base``."""
    here, _ = card_mod.accessible(gpu or accessible)
    return plan_tests(changes(base), kind, frozenset(here))


def summary() -> str:
    g = build()
    kinds = collections.Counter(g.nodes.values())
    isolated = sorted(p for p, k in g.nodes.items() if k in ("python", "cuda-source", "test", "test-gpu") and not g.deps.get(p) and not g.users.get(p))
    per_card = collections.Counter(card_mod.label(c) for c in g.cards.values())
    return (f"{len(g.nodes)} nodes {dict(kinds)}\n{sum(len(d) for d in g.deps.values())} edges {dict(g.edge_count())}\n"
            f"nodes per card set: {dict(sorted(per_card.items()))}\n"
            f"{len(isolated)} code / test files with no edge in or out" + ("".join(f"\n  {p}" for p in isolated[:30])))
