"""Content keys of compiled kernel artifacts, from the code each artifact actually compiles.

An artifact (a torch extension ``.so`` or a driver-API ``.cubin``) is identified by a key that is computed the same way
on every machine and at every point in its life -- when the bundle builder records it, when a user process looks it up
in an installed kernel bundle, and when the local JIT cache decides whether it has to be rebuilt:

* the **code graph**: the entry source(s) (or the generated text of ``load_inline`` / TriMul's ``compile_text``), then
  the transitive ``#include`` closure resolved the way the compiler resolves it -- a quoted include first next to the
  including file, then the ``-I`` directories in order; an angle include through the ``-I`` directories only. Only
  files that are actually reached are nodes. Each node is hashed after normalization (comments, whitespace, line
  breaks and ``#if 0`` blocks removed), so a comment or formatting edit, or an edit to an unrelated header in the same
  directory, keeps the key; an edit that reaches the compiler changes it.
* headers outside the engine (torch, CUDA, CUTLASS / CuTe, cuBLASDx) are not read: they are represented by their
  version -- torch and ``torch.version.cuda`` for everything, ``cutlass@<version>`` / ``mathdx`` for an include
  directory that holds those libraries;
* the compile flags as they reach the compiler, normalized: machine paths become ``pkg:<path in the package>``
  (``-I`` into the engine), ``gen`` (a generated directory under the JIT root, whose files are in the graph),
  ``cuda:`` / ``prefix:`` (the toolkit / environment), host-compiler choices (``-ccbin``) dropped;
* the target arch (``sm90`` / ``sm100``), torch and CUDA versions, and the Python ABI for torch extensions.

Per-file parses are cached by (path, mtime, size), so a key costs a few stat calls once its files have been read.
Generated headers (TriMul's transformed upstream headers) live in an in-memory overlay (:func:`publish`) and are only
written to disk when a build needs them, so loading prebuilt artifacts writes nothing.

See docs/kernels/prebuilt-kernels.md.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import re
import sys
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Bumped when the key material changes shape, so old bundles / caches stop matching instead of matching wrongly.
KEY_SCHEME = 1

#: The CUTLASS release the engine's CUTLASS-based kernels are written against. A CUTLASS include directory that is not
#: present on this machine is keyed with this version (a prebuilt artifact needs no headers); one that is present is
#: keyed with the version its ``cutlass/version.h`` states.
CUTLASS_PINNED = "4.2.0"

PKG_ROOT = Path(__file__).resolve().parents[1]
#: Stands in for CUDA_HOME in an include path when no toolkit is installed (a process that only loads prebuilt
#: artifacts needs none); keyed like the real one (``cuda:<path>``).
CUDA_HOME_STANDIN = "/nonexistent/cuda-home"

_LOCK = threading.RLock()


# --------------------------------------------------------------------------------------------------- normalization
_LITERALS = re.compile(
    r"""(?P<line>//[^\n]*)
      | (?P<block>/\*.*?\*/)
      | (?P<raw>(?<![A-Za-z0-9_])(?:u8|u|U|L)?R"(?P<delim>[^()\\\s"]{0,16})\(.*?\)(?P=delim)")
      | (?P<str>(?<![A-Za-z0-9_])(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*")
      | (?P<chr>(?<![A-Za-z0-9_])'(?:\\.|[^'\\\n])+')""",
    re.DOTALL | re.VERBOSE,
)
_WS = re.compile(r"[ \t\f\v]+")
# A space next to punctuation carries no meaning ("a + b" == "a+b"); one between two identifiers ("int x") or between
# two punctuation characters ("- -" is not "--") does, and stays.
_PUNCT_SPACE = re.compile(r"(?<=\w) (?=[^\w\s])|(?<=[^\w\s]) (?=\w)")
_DIRECTIVE = re.compile(r"#\s*(\w+)\s*(.*)", re.DOTALL)
_DEFINE = re.compile(r"(\w+)(\()?(.*)", re.DOTALL)
_INCLUDE = re.compile(r'#\s*include(?:_next)?\s*([<"])([^>"]+)[>"]')
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")


def _squeeze(text: str) -> str:
    return _PUNCT_SPACE.sub("", _WS.sub(" ", text).strip())


def _directive(line: str) -> tuple[str, str]:
    m = _DIRECTIVE.match(line)
    if not m:
        return "", ""
    return m.group(1), m.group(2).strip()


def normalize(text: str) -> str:
    """The text with everything the compiler ignores removed: comments, line continuations, blank lines, runs of
    whitespace, spaces next to punctuation, line breaks inside code, and ``#if 0`` blocks (``#if 0 / #else`` keeps the
    ``#else`` side). String and character literals are kept byte for byte. Not a preprocessor: macros are not expanded
    and other conditionals are kept as they are."""
    text = text.replace("\r\n", "\n").replace("\\\n", "")
    protected: list[str] = []

    def protect(m: re.Match) -> str:
        if m.group("line") is not None:
            return ""
        if m.group("block") is not None:
            return " "
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = _LITERALS.sub(protect, text)
    out: list[str] = []
    code: list[str] = []
    stack: list[bool] = []  # per open conditional: drop its #endif (an `#if 0 ... #else` whose else side was kept)
    skip = 0  # nesting depth inside an `#if 0` block

    def flush() -> None:
        if code:
            out.append(_squeeze(" ".join(code)))
            code.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("#"):
            if not skip:
                code.append(line)
            continue
        word, rest = _directive(line)
        if skip:
            if word in ("if", "ifdef", "ifndef"):
                skip += 1
            elif word == "endif":
                skip -= 1
            elif skip == 1 and word == "else":
                skip = 0
                stack.append(True)
            elif skip == 1 and word == "elif":
                skip = 0
                stack.append(False)
                flush()
                out.append("#if " + _squeeze(rest))
            continue
        flush()
        if word == "if" and _squeeze(rest) == "0":
            skip = 1
            continue
        if word in ("if", "ifdef", "ifndef"):
            stack.append(False)
        elif word == "endif" and stack and stack.pop():
            continue
        if word == "define":
            m = _DEFINE.match(rest)
            if m:  # "#define F(x)" (function-like) and "#define F (x)" (object-like) differ: keep that one space
                out.append(f"#define {m.group(1)}{'(' if m.group(2) else ' '}{_squeeze(m.group(3))}".rstrip())
                continue
        out.append(_squeeze(f"#{word} {rest}"))
    flush()
    return _PLACEHOLDER.sub(lambda m: protected[int(m.group(1))], "\n".join(out))


def includes_of(normalized: str) -> tuple[tuple[str, bool], ...]:
    """``(spelling, quoted)`` of every ``#include`` left in normalized text, in order."""
    found = []
    for line in normalized.split("\n"):
        if line.startswith("#include"):
            m = _INCLUDE.match(line)
            if m:
                found.append((m.group(2).strip(), m.group(1) == '"'))
    return tuple(found)


@dataclasses.dataclass(frozen=True)
class Parsed:
    digest: str
    includes: tuple[tuple[str, bool], ...]


@functools.lru_cache(maxsize=4096)
def parse_text(text: str) -> Parsed:
    norm = normalize(text)
    return Parsed(hashlib.sha256(norm.encode()).hexdigest(), includes_of(norm))


_FILE_CACHE: dict[str, tuple[int, int, Parsed]] = {}
_OVERLAY: dict[str, str] = {}


def publish(path: Path | str, text: str) -> None:
    """Register a generated file (a transformed header) without writing it. Key computation reads it from here; a build
    that needs it writes it first (:func:`materialize`)."""
    with _LOCK:
        _OVERLAY[os.path.abspath(path)] = text


def overlay_text(path: Path | str) -> str | None:
    return _OVERLAY.get(os.path.abspath(path))


def read_text(path: Path | str) -> str:
    """A generated file's registered text, else the file on disk."""
    text = overlay_text(path)
    return text if text is not None else Path(path).read_text()


def _exists(path: str) -> bool:
    return path in _OVERLAY or os.path.isfile(path)


def parse_file(path: str) -> Parsed:
    text = _OVERLAY.get(path)
    if text is not None:
        return parse_text(text)
    st = os.stat(path)
    hit = _FILE_CACHE.get(path)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        parsed = parse_text(fh.read())
    _FILE_CACHE[path] = (st.st_mtime_ns, st.st_size, parsed)
    return parsed


def materialize(paths: Iterable[str]) -> None:
    """Write the generated files among ``paths`` (atomically; unchanged files are left alone)."""
    import tempfile

    for p in paths:
        text = _OVERLAY.get(p)
        if text is None:
            continue
        target = Path(p)
        try:
            if target.read_text() == text:
                continue
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as fh:
            fh.write(text)
        Path(fh.name).replace(target)


# --------------------------------------------------------------------------------------------------- directories
def _jit_root() -> Path | None:
    try:
        from miniworld_engine import settings
    except ImportError:
        return None
    return settings.engine_jit_root()


def _under(path: str, root: Path | str | None) -> str | None:
    """``path`` relative to ``root`` (posix), or None when it is not inside."""
    if root is None:
        return None
    root_s = os.path.abspath(root)
    if path == root_s:
        return "."
    if path.startswith(root_s.rstrip("/") + "/"):
        return path[len(root_s.rstrip("/")) + 1:]
    return None


@functools.lru_cache(maxsize=256)
def _cutlass_version(directory: str) -> str:
    try:
        text = Path(directory, "cutlass", "version.h").read_text()
    except OSError:
        return CUTLASS_PINNED
    parts = [re.search(rf"#define\s+CUTLASS_{k}\s+(\d+)", text) for k in ("MAJOR", "MINOR", "PATCH")]
    if not all(parts):
        return CUTLASS_PINNED
    return ".".join(p.group(1) for p in parts)  # type: ignore[union-attr]


def _cuda_home() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return os.path.abspath(home) if home else None


@dataclasses.dataclass(frozen=True)
class Context:
    """Where the machine-specific roots are. Tests pass their own; everything else uses :func:`context`."""

    package: str
    jit: str | None
    prefix: str
    cuda: str | None


def context() -> Context:
    jit = _jit_root()
    return Context(str(PKG_ROOT), os.path.abspath(jit) if jit else None, os.path.abspath(sys.prefix), _cuda_home())


def classify_dir(directory: str, ctx: Context) -> tuple[str, bool]:
    """``(label, external)`` of an include / library directory. External directories are represented by the label
    (a version or a toolkit-relative path), never read."""
    d = os.path.abspath(directory)
    rel = _under(d, ctx.package)
    if rel is not None:
        return f"pkg:{rel}", False
    if _under(d, ctx.jit) is not None:
        return "gen", False
    low = d.lower()
    if (os.path.isfile(os.path.join(d, "cutlass", "cutlass.h")) or os.path.isfile(os.path.join(d, "cute", "tensor.hpp"))
            or ("cutlass" in low and "mathdx" not in low)):
        return f"cutlass@{_cutlass_version(d)}", True
    if "mathdx" in low or os.path.isfile(os.path.join(d, "cublasdx.hpp")):
        return "mathdx", True
    rel = _under(d, ctx.cuda)
    if rel is not None:
        return f"cuda:{rel}", True
    rel = _under(d, CUDA_HOME_STANDIN)
    if rel is not None:
        return f"cuda:{rel}", True
    rel = _under(d, ctx.prefix)
    if rel is not None:
        return f"prefix:{rel}", True
    return f"dir:{os.path.basename(d.rstrip('/'))}", False


_PATH_FLAGS = ("-I", "-L", "-isystem", "-iquote", "--include-path=", "--library-path=", "-include")


def normalize_flags(flags: Sequence[str], ctx: Context) -> list[str]:
    """Compile flags without machine paths (see the module docstring). Order is kept: it is what the compiler sees."""
    out: list[str] = []
    it = iter(flags)
    for flag in it:
        flag = str(flag)
        if flag == "-ccbin" or flag == "--compiler-bindir":
            next(it, None)
            continue
        if flag.startswith(("-ccbin=", "--compiler-bindir=")) or flag == HOST_MARKER:
            continue
        for prefix in _PATH_FLAGS:
            if flag == prefix.rstrip("="):
                value = next(it, "")
                out.append(f"{prefix.rstrip('=')}{_path_label(value, ctx)}")
                break
            if flag.startswith(prefix) and len(flag) > len(prefix):
                out.append(f"{prefix}{_path_label(flag[len(prefix):], ctx)}")
                break
        else:
            out.append(_scrub(flag, ctx))
    return out


def _path_label(value: str, ctx: Context) -> str:
    return "<" + classify_dir(value, ctx)[0] + ">"


def _scrub(flag: str, ctx: Context) -> str:
    """Any machine path left inside another flag (``-Xcompiler=-I/x``, ``-DROOT=/x``)."""
    for root, label in ((ctx.package, "<pkg>"), (ctx.jit, "<gen>"), (ctx.cuda, "<cuda>"), (ctx.prefix, "<prefix>")):
        if root:
            flag = flag.replace(root, label)
    home = os.path.expanduser("~")
    if home and home != "/":
        flag = flag.replace(home, "<home>")
    return flag


#: Placeholders ``_nvcc`` puts in flag lists instead of machine-dependent choices, expanded only when a build runs:
#: the host compiler (``-ccbin``, found by compiling a probe) and the ``-gencode`` list (filtered by what nvcc knows,
#: and by the bundle's arch when a bundle is being built). Keys see the request, not the machine's answer.
HOST_MARKER = "--miniworld-host-compiler"
GENCODE_MARKER = "--miniworld-gencode="
PTX_MARKER = "--miniworld-ptx="


# --------------------------------------------------------------------------------------------------- graph
@dataclasses.dataclass(frozen=True)
class Node:
    label: str
    digest: str
    path: str | None  # where it was read (informational; not part of the key)


@dataclasses.dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    external: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]  # (from label, spelling, to label or "ext:<spelling>")

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(n.path for n in self.nodes if n.path)

    def material(self) -> dict:
        return {"nodes": sorted([n.label, n.digest] for n in self.nodes), "external": sorted(self.external)}


@dataclasses.dataclass(frozen=True)
class TextRoot:
    """A source that exists only as text (``load_inline``, ``compile_text``): ``directory`` is where it would be
    written, which is where its quoted includes are looked up first."""

    label: str
    text: str
    directory: str


def include_dirs_of(flags: Sequence[str]) -> list[str]:
    """The ``-I`` / ``-isystem`` directories in ``flags``, in order."""
    dirs: list[str] = []
    it = iter(flags)
    for flag in it:
        flag = str(flag)
        for prefix in ("-I", "-isystem", "--include-path="):
            if flag == prefix.rstrip("="):
                dirs.append(next(it, ""))
                break
            if flag.startswith(prefix) and len(flag) > len(prefix):
                dirs.append(flag[len(prefix):])
                break
    return [os.path.abspath(d) for d in dirs if d]


def _label(path: str, include_dirs: Sequence[str], ctx: Context) -> str:
    rel = _under(path, ctx.package)
    if rel is not None:
        return f"pkg:{rel}"
    best = None
    for d in include_dirs:
        r = _under(path, d)
        if r is not None and (best is None or len(r) < len(best)):
            best = r
    return f"inc:{best}" if best is not None else f"file:{os.path.basename(path)}"


def closure(roots: Sequence[str | Path | TextRoot], include_dirs: Sequence[str | Path] = (),
            ctx: Context | None = None) -> Graph:
    """The include graph of ``roots`` (see the module docstring)."""
    ctx = ctx or context()
    inc = [os.path.abspath(d) for d in include_dirs]
    external_dirs = {d for d in inc if classify_dir(d, ctx)[1]}
    nodes: dict[str, Node] = {}
    edges: list[tuple[str, str, str]] = []
    external: set[str] = set()
    todo: list[tuple[str, str, Parsed, str]] = []  # (label, directory, parsed, path)

    for root in roots:
        if isinstance(root, TextRoot):
            parsed = parse_text(root.text)
            nodes.setdefault(root.label, Node(root.label, parsed.digest, None))
            todo.append((root.label, os.path.abspath(root.directory), parsed, ""))
        else:
            p = os.path.abspath(root)
            label = _label(p, inc, ctx)
            if label in nodes:
                continue
            parsed = parse_file(p)
            nodes[label] = Node(label, parsed.digest, p)
            todo.append((label, os.path.dirname(p), parsed, p))

    seen_paths = {n.path for n in nodes.values() if n.path}
    while todo:
        label, directory, parsed, _ = todo.pop()
        for spelling, quoted in parsed.includes:
            found = None
            via_external = False
            if quoted:
                cand = os.path.normpath(os.path.join(directory, spelling))
                if _exists(cand):
                    found = cand
            if found is None:
                for d in inc:
                    cand = os.path.normpath(os.path.join(d, spelling))
                    if _exists(cand):
                        found, via_external = cand, d in external_dirs
                        break
            if found is None or via_external:
                external.add(spelling)
                edges.append((label, spelling, f"ext:{spelling}"))
                continue
            child = _label(found, inc, ctx)
            edges.append((label, spelling, child))
            if found in seen_paths:
                continue
            seen_paths.add(found)
            child_parsed = parse_file(found)
            nodes.setdefault(child, Node(child, child_parsed.digest, found))
            todo.append((child, os.path.dirname(found), child_parsed, found))
    return Graph(tuple(sorted(nodes.values(), key=lambda n: n.label)), tuple(sorted(external)), tuple(edges))


# --------------------------------------------------------------------------------------------------- key
def _versions() -> dict[str, str]:
    try:
        import torch
    except ImportError:
        return {"torch": "-", "cuda": "-"}
    return {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)}


def python_abi() -> str:
    return sys.implementation.cache_tag or f"py{sys.version_info[0]}{sys.version_info[1]}"


def target_arch() -> str:
    """``sm90`` / ``sm100`` / ...: the current CUDA device's arch (what a bundle is built for), ``cpu`` without one."""
    forced = os.environ.get("MINIWORLD_ENGINE_TARGET_ARCH")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception:
        pass
    return "cpu"


def key_material(kind: str, name: str, graph: Graph, flags: dict | Sequence[str], *, arch: str | None = None,
                 python: bool = False, ctx: Context | None = None, versions: dict | None = None) -> dict:
    """Everything the key hashes, as a JSON-able dict (``dev code-graph`` prints it)."""
    ctx = ctx or context()
    if isinstance(flags, dict):
        norm = {k: normalize_flags(v, ctx) if isinstance(v, (list, tuple)) else v for k, v in sorted(flags.items())}
    else:
        norm = normalize_flags(flags, ctx)
    return {
        "scheme": KEY_SCHEME,
        "kind": kind,
        "name": name,
        **graph.material(),
        "flags": norm,
        "arch": arch or target_arch(),
        **(versions or _versions()),
        "python": python_abi() if python else "-",
    }


def key_of(material: dict) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
