"""Which GPU cards a node of the repository graph depends on (``Graph.cards``), and which cards this environment can judge.

One graph serves every card. A node is either card-independent (``ALL``: core Python, references, utils, modules, docs, tests that
run anywhere) or depends on a set of cards. The sets are read from the repository, not kept in a list:

* a measured file names its card: ``autotune/data/<op>/<gpu name> (smNN).json`` / ``..._smNN_triton<ver>.json`` (tuned and native caches),
  ``autotune/plans/<card>/`` (derivation plans), ``autotune/manifests/<gpu> (smNN).csv``;
* ``docs/gpus/<dir>/`` (and the per-card benchmark results) take the card the ``docs/gpus/README.md`` table gives that directory;
* a hand-CUDA kernel takes its ``registry.csv`` row's ``arch`` (the registry is the authority), else the arch the path spells
  (``sm90``, ``sm100`` ... or a card name: ``h100``, ``b200``, ``a100``), else the arch its text needs (``wgmma`` is sm90, ``tcgen05`` is sm100),
  else the arch of the files that ``#include`` it, else the cards that run hand-written CUDA (the *restricted* cards);
* a Triton kernel (a ``triton/`` directory or a ``backend=triton`` registry row) belongs to the *fallback* cards: the cards that are not
  restricted. ``modules.dispatch.restricted_card`` is the authority (H100 and B200 have no portable-kernel fallback, on purpose:
  ``docs/gpus/h100/dispatch.md``); its capability majors are read from the source, not copied;
* registry rows, native identities: the row's / op's arch; the derivation identity is card-independent (its plans are per card).

A module that imports a card-specific kernel stays ``ALL``; what carries the card is the edge, which :func:`repo_graph.reach_cards` walks.

``accessible()`` says which cards this environment can run on. Everything about another card is ``UNVERIFIABLE-HERE``.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

#: Node kinds whose evidence a card does NOT keep, by policy: an H100 keeps no Triton tuned caches (it runs hand-written CUDA or the PyTorch
#: reference; the Triton fallback exists for the other cards, docs/gpus/h100/dispatch.md). On such a card a node of an exempt kind is neither
#: pending (``--check`` does not fail on it) nor dropped: it is counted and shown as exempt, and shown under the fallback cards (the cards
#: whose evidence a Triton cache is) as UNVERIFIABLE-HERE. Only sm90 is audited so far.
EXEMPT_KINDS: dict[str, frozenset[str]] = {"sm90": frozenset({"triton-cache"})}

#: card-independent. A frozenset (not a string) so ``cards & other`` works; use :func:`expand` before intersecting.
ALL: frozenset[str] = frozenset({"*"})

SRC = "src/miniworld_engine"
#: where this repo is developed: an H100 (docs/gpus/h100/h100.md). Used when no card is named and no GPU is visible (a login node, a CPU job).
DEFAULT_ACCESSIBLE = ("sm90",)
ENV_ACCESSIBLE = "MINIWORLD_ACCESSIBLE_CARDS"

_SM_RE = re.compile(r"(?<![a-z0-9])sm_?(\d{2,3})[a-z]?(?![0-9])")
_NAME_RE = re.compile(r"(?<![a-z0-9])(h100|b200|a100|a5000|a6000)(?![0-9])")
_NAME_CARD = {"h100": "sm90", "b200": "sm100", "a100": "sm80", "a5000": "sm86", "a6000": "sm86"}
_NEEDS_SM100 = re.compile(r"tcgen05|\bsm_?100|__CUDA_ARCH__\s*>=\s*1000|UMMA")
_NEEDS_SM90 = re.compile(r"\bwgmma\b|wgmma\.|\bsm_?90|__CUDA_ARCH__\s*(?:>=|==)\s*900")
_CCODE = {".cu", ".cuh", ".inc", ".h", ".cpp"}
_FALLBACK_MAJORS = (9, 10)           # used only when dispatch.restricted_card cannot be parsed


def exempt_kinds(card: str) -> frozenset[str]:
    return EXEMPT_KINDS.get(card, frozenset())


def card_name(number: str | int) -> str:
    return f"sm{int(number)}"


def expand(cards: frozenset[str], known: frozenset[str]) -> frozenset[str]:
    if cards == ALL:
        return known
    return cards & known if known else cards


def label(cards: frozenset[str]) -> str:
    return "ALL" if cards == ALL else "+".join(sorted(cards, key=_order))


def _order(card: str) -> int:
    return int(card[2:]) if card[2:].isdigit() else 0


def sort_cards(cards) -> list[str]:
    return sorted(cards, key=_order)


# ------------------------------------------------------------------------------------------------------------ the card set
def _read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def gpu_dirs(root: Path) -> dict[str, str]:
    """``{docs/gpus/<dir>: card}`` from the table of ``docs/gpus/README.md`` (the arch column, the directory of the page links)."""
    out: dict[str, str] = {}
    for line in _read(root, "docs/gpus/README.md").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 4 and re.fullmatch(r"sm\d+", cells[2]):
            for d in re.findall(r"\]\(([\w-]+)/", line):
                out[d] = cells[2]
    return out


def _file_card(path: str) -> str | None:
    m = _SM_RE.search(PurePosixPath(path).name.lower())
    return card_name(m.group(1)) if m else None


def known_cards(paths, root: Path) -> frozenset[str]:
    """Every card the repository has evidence or a page for: the README table, the plan directories, the cache / manifest file names."""
    found = set(gpu_dirs(root).values())
    for p in paths:
        if p.startswith(f"{SRC}/autotune/plans/"):
            found.add(PurePosixPath(p).parts[4])
        elif p.startswith((f"{SRC}/autotune/data/", f"{SRC}/autotune/manifests/")):
            c = _file_card(p)
            if c:
                found.add(c)
    return frozenset(c for c in found if re.fullmatch(r"sm\d+", c))


def restricted_majors(root: Path) -> tuple[int, ...]:
    """The capability majors ``modules.dispatch.restricted_card`` restricts (read from its source: ``... [0] in (9, 10)``)."""
    try:
        tree = ast.parse(_read(root, f"{SRC}/modules/dispatch.py"))
    except SyntaxError:
        return _FALLBACK_MAJORS
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "restricted_card":
            for node in ast.walk(fn):
                if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In) and isinstance(node.comparators[0], ast.Tuple):
                    vals = tuple(c.value for c in node.comparators[0].elts if isinstance(c, ast.Constant) and isinstance(c.value, int))
                    if vals:
                        return vals
    return _FALLBACK_MAJORS


def _major(card: str) -> int:
    n = int(card[2:])
    return n // 10


def restricted_cards(known: frozenset[str], root: Path) -> frozenset[str]:
    """H100 / B200: hand-written CUDA, no portable-kernel fallback."""
    majors = restricted_majors(root)
    return frozenset(c for c in known if _major(c) in majors)


def fallback_cards(known: frozenset[str], root: Path) -> frozenset[str]:
    """The cards the Triton kernels run on: the known cards that are not restricted."""
    return known - restricted_cards(known, root)


# -------------------------------------------------------------------------------------------------------------- assignment
def path_cards(path: str) -> frozenset[str] | None:
    """The card a repo path spells: the outermost directory that names one, else the file name's (a comparison names several)."""
    p = PurePosixPath(path)
    for part in p.parts[:-1]:
        low = part.lower()
        m = _SM_RE.search(low)
        if m:
            return frozenset({card_name(m.group(1))})
        n = _NAME_RE.search(low)
        if n:
            return frozenset({_NAME_CARD[n.group(1)]})
    name = p.name.lower()
    found = {card_name(m) for m in _SM_RE.findall(name)} | {_NAME_CARD[n] for n in _NAME_RE.findall(name)}
    return frozenset(found) if found else None


def _registry(root: Path) -> list[dict]:
    try:
        with (root / SRC / "kernels/registry/registry.csv").open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


class Assigner:
    def __init__(self, paths, root: Path) -> None:
        self.root = root
        self.known = known_cards(paths, root) or frozenset(DEFAULT_ACCESSIBLE)
        self.restricted = restricted_cards(self.known, root)
        self.fallback = fallback_cards(self.known, root)
        self.dirs = gpu_dirs(root)
        self.rows = _registry(root)
        self.row_cards: dict[str, frozenset[str]] = {}
        self.file_cards: dict[str, frozenset[str]] = {}
        for r in self.rows:
            cards = self._row_cards(r)
            self.row_cards[r["kernel"]] = cards
            if r["file"].endswith((".py", *_CCODE)):
                self.file_cards.setdefault(f"src/{r['file']}", cards)

    def _row_cards(self, row: dict) -> frozenset[str]:
        if row.get("backend") == "triton":
            return self.fallback
        arch = row.get("arch") or ""
        return frozenset({arch}) if re.fullmatch(r"sm\d+", arch) else self.restricted

    def op_cards(self, op: str) -> frozenset[str]:
        """A tuned / native op: its registry row's cards, else what its name spells, else the restricted cards (a native op is hand-CUDA)."""
        if op in self.row_cards:
            return self.row_cards[op]
        return path_cards(op) or self.restricted

    def explicit(self, path: str, kind: str) -> frozenset[str] | None:
        p = PurePosixPath(path)
        if path.startswith(f"{SRC}/autotune/data/"):
            c = _file_card(path)
            return frozenset({c}) if c else None
        if path.startswith(f"{SRC}/autotune/plans/"):
            return frozenset({p.parts[4]})
        if path.startswith(f"{SRC}/autotune/manifests/"):
            c = _file_card(path)
            return frozenset({c}) if c else None
        if path.startswith("docs/gpus/") and len(p.parts) > 3:
            c = self.dirs.get(p.parts[2])
            return frozenset({c}) if c else None
        if kind == "test":
            return None                            # a CPU test runs anywhere
        if path in self.file_cards:
            return self.file_cards[path]
        under_kernels = path.startswith(f"{SRC}/kernels/")
        if under_kernels and "triton" in p.parts[:-1]:
            return self.fallback
        if under_kernels or path.startswith(("tests/", "benchmarks/", "docs/")):
            return path_cards(path)
        return None

    def content_cards(self, path: str) -> frozenset[str] | None:
        text = _read(self.root, path)
        need = {c for c, rx in (("sm90", _NEEDS_SM90), ("sm100", _NEEDS_SM100)) if rx.search(text)}
        return (frozenset(need) & self.known) or None


_GENCODE_RE = re.compile(r"(?:-arch=sm_|compute_|sm_)(\d{2,3})a?")


def _arches_in(node: ast.AST) -> set[str]:
    """The cards a build call asks nvcc for: ``gencodes("90a", ...)`` arguments and ``-arch=sm_90a`` / ``compute_100`` flag strings under ``node``."""
    found: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")) == "gencodes":
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and re.fullmatch(r"\d{2,3}a?", a.value):
                    found.add(card_name(a.value.rstrip("a")))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("-"):      # an nvcc flag, not prose
            found |= {card_name(m) for m in _GENCODE_RE.findall(n.value)}
    return found


def builder_cards(root: Path, pyfile: str, cu: str) -> frozenset[str]:
    """The cards the python file ``pyfile`` builds ``cu`` for: the arches of the build call(s) that name the file's basename (a
    ``load_extension(... sources=[... "x.cu"] ..., *gencodes("90a"))``), else every arch the file asks for. Empty: it says nothing."""
    try:
        tree = ast.parse(_read(root, pyfile))
    except SyntaxError:
        return frozenset()
    name = PurePosixPath(cu).name
    exact: set[str] = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        names = {c.value for c in ast.walk(call) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        if any(PurePosixPath(s).name == name for s in names):
            exact |= _arches_in(call)
    return frozenset(exact or _arches_in(tree))


def assign(g, paths, root: Path) -> Assigner:
    """Fill ``g.cards`` (node -> frozenset of cards or ``ALL``). ``g`` is a ``repo_graph.Graph``."""
    a = Assigner(paths, root)
    g.known_cards = a.known
    g.restricted_cards = a.restricted
    g.fallback_cards = a.fallback
    cards: dict[str, frozenset[str]] = {}
    undecided = []
    for node, kind in g.nodes.items():
        if node.startswith("registry-row:"):
            cards[node] = a.row_cards.get(node.split(":", 1)[1], ALL)
        elif node.startswith("identity:native:"):
            cards[node] = a.op_cards(node.split(":", 2)[2])
        elif node.startswith("identity:"):
            cards[node] = ALL                      # the derivation: judged per card through that card's plan
        else:
            c = a.explicit(node, kind)
            if c is None and PurePosixPath(node).suffix in _CCODE and node.startswith(f"{SRC}/kernels/"):
                undecided.append(node)             # hand-written CUDA with no card in its name or registry row
            else:
                cards[node] = c if c is not None else ALL
    for node in undecided:
        content = a.content_cards(node)
        if content:
            cards[node] = content
    for node in undecided:                         # a source its builder names: the arches of the nvcc call that compiles it
        if node in cards or PurePosixPath(node).suffix in (".h", ".cuh", ".inc"):      # a header follows the files that include it
            continue
        built: set[str] = set()
        for u, k in g.users.get(node, {}).items():
            if k == "ref" and u.endswith(".py") and u.startswith(f"{SRC}/") and not u.startswith(f"{SRC}/build/"):
                built |= builder_cards(root, u, node)
        built &= a.restricted
        if built:
            cards[node] = frozenset(built)
    for _ in range(4):                             # a header takes the cards of the files that include it
        for node in undecided:
            if node in cards:
                continue
            via = [cards[u] for u, k in g.users.get(node, {}).items() if k == "include" and u in cards and cards[u] != ALL]
            if via:
                cards[node] = frozenset().union(*via)
    for node in undecided:
        cards.setdefault(node, a.restricted)
    g.cards = cards
    return a


# ----------------------------------------------------------------------------------------------------------- accessibility
def parse_cards(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    items = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for item in items:
        for tok in re.split(r"[,\s]+", item.strip()):
            if not tok:
                continue
            m = re.fullmatch(r"(?:sm_?)?(\d{2,3})[a-z]?", tok.lower())
            if not m:
                raise ValueError(f"not a card: {tok!r} (expected e.g. sm90, sm100)")
            out.append(card_name(m.group(1)))
    return out


def visible_card() -> list[str]:
    """The card(s) of the GPU(s) visible to this process (``nvidia-smi``); empty on a node without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({card_name(c.strip().replace(".", "")) for c in out.splitlines() if re.fullmatch(r"\d+\.\d+", c.strip())}, key=_order)


def accessible(explicit: str | list[str] = "") -> tuple[frozenset[str], str]:
    """``(cards, source)``: the cards this environment can run on. ``--accessible`` > ``MINIWORLD_ACCESSIBLE_CARDS`` > the visible GPU >
    the repository default (sm90: its development environment is an H100; a login node or a CPU job shows no GPU)."""
    if explicit:
        return frozenset(parse_cards(explicit)), "--accessible"
    env = os.environ.get(ENV_ACCESSIBLE, "")
    if env.strip():
        return frozenset(parse_cards(env)), ENV_ACCESSIBLE
    seen = visible_card()
    if seen:
        return frozenset(seen), "visible device"
    return frozenset(DEFAULT_ACCESSIBLE), "default (no GPU visible; this repo's H100 environment)"
