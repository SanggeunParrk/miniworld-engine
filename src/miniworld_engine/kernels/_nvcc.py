"""Point the JIT extension build at an nvcc that matches torch, and ask it what it supports.

Two failures this fixes, both observed rather than guessed:

  * ``PATH`` here resolves nvcc to a CUDA 11.7 install that ships with anaconda, while torch is
    built against 12.8. torch's ``cpp_extension.load`` follows ``CUDA_HOME``/``PATH``, so the
    build ran the 11.7 compiler and died on ``nvcc fatal: Unsupported gpu architecture
    'compute_90'``.
  * The ``-gencode`` list was hard-coded to sm_80/90/100. Hard-coding an arch the local compiler
    has never heard of turns a working kernel into a build error, so the list is filtered against
    ``nvcc --list-gpu-arch`` instead of assumed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple


@functools.lru_cache(maxsize=1)
def nvcc_path() -> str | None:
    """An nvcc whose release matches ``torch.version.cuda``, preferring this environment's own."""
    import torch

    want = (torch.version.cuda or "").split(".")[:2]
    candidates = [Path(sys.prefix) / "bin" / "nvcc"]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    from shutil import which

    found = which("nvcc")
    if found:
        candidates.append(Path(found))
    candidates += sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True)

    fallback = None
    for c in candidates:
        if not c.is_file():
            continue
        try:
            out = subprocess.run([str(c), "--version"], capture_output=True, text=True, timeout=20, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        fallback = fallback or str(c)
        if want and f"release {want[0]}.{want[1]}" in out:
            return str(c)
    return fallback


def ensure_cuda_home() -> str | None:
    """Point ``cpp_extension`` at the matching toolkit. Idempotent.

    Setting the environment variable alone is not enough: ``torch.utils.cpp_extension`` resolves
    ``CUDA_HOME`` once, at its own import time, into a module-level constant. If it is already
    imported -- and importing ``load`` from it means it is -- the constant has to be replaced too,
    otherwise the build keeps shelling out to whichever nvcc was on PATH first.
    """
    nvcc = nvcc_path()
    if not nvcc:
        return None
    home = str(Path(nvcc).parent.parent)
    os.environ["CUDA_HOME"] = home
    os.environ["CUDA_PATH"] = home
    mod = sys.modules.get("torch.utils.cpp_extension")
    if mod is not None:
        # torch.utils.cpp_extension caches CUDA_HOME at ITS import time, so setting the env
        # var above is not enough once it is already in sys.modules. setattr, not attribute
        # syntax: the name is not declared on ModuleType.
        setattr(mod, "CUDA_HOME", home)  # noqa: B010 -- see above: attribute syntax is wrong here
    return home


@functools.lru_cache(maxsize=1)
def supported_arches() -> frozenset[str]:
    """What this nvcc will actually accept, from ``--list-gpu-arch``."""
    nvcc = nvcc_path()
    if not nvcc:
        return frozenset()
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(line.strip() for line in out.split("\n") if line.strip())


def gencodes(*arches: str, ptx: tuple[str, ...] = ()) -> list[str]:
    """Placeholders for the ``-gencode`` flags of the requested arches (expanded by :func:`expand_flags` when a build
    runs, so computing a key or loading a prebuilt artifact never asks nvcc anything).

    ``arches`` are like ``"90"`` or ``"90a"``; ``ptx`` names arches to also emit PTX for, so a
    newer card can JIT from it.
    """
    from miniworld_engine.kernels._codegraph import GENCODE_MARKER, PTX_MARKER

    return [f"{GENCODE_MARKER}{a}" for a in arches] + [f"{PTX_MARKER}{a}" for a in ptx]


def real_gencodes(*arches: str, ptx: tuple[str, ...] = ()) -> list[str]:
    """``-gencode`` flags for the requested arches, dropping any this nvcc does not know."""
    ensure_cuda_home()
    have = supported_arches()

    def supported(a):
        # nvcc 12.8's listing omits architecture-specific variants even though
        # it accepts compute_90a/sm_90a. Dropping that flag silently enabled
        # torch's GPU autodetection, which fails on a CPU-only compiler worker.
        return (
            not have
            or f"compute_{a}" in have
            or (a.endswith("a") and a[:-1].isdigit() and int(a[:-1]) >= 90 and f"compute_{a[:-1]}" in have)
        )

    out = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in arches if supported(a)]
    out += [f"-gencode=arch=compute_{a},code=compute_{a}" for a in ptx if supported(a)]
    return out


# The probe must pull the header depth that actually breaks. A bare <type_traits> compiled fine
# under GCC 14 and so reported the default host compiler as usable, while the real build died in
# libstdc++'s c++config.h on `__decltype(0.0bf16)` -- a literal suffix nvcc's frontend rejects.
_PROBE = """
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <string>
#include <vector>
#include <limits>
__global__ void k(float* p) { p[0] = 1.0f; }
"""


#: A release build sets this to the one host compiler every artifact is built with (and CXX to the same, which
#: torch uses to compile and link the C++ side): the libstdc++ it links decides which systems can load the result.
HOST_CXX_ENV = "MINIWORLD_ENGINE_HOST_CXX"

#: g++ installs to try, best first. The environment's own compiler comes first so a working
#: setup needs no override; the cluster's module tree is next, because this cluster ships GCC 12.4
#: and 9.4 under /opt/ohpc/pub (mounted on the compute nodes) while the pixi environment carries
#: only GCC 14.3, which this nvcc cannot parse.
_HOST_CANDIDATES = (
    "{prefix}/bin/g++",
    "/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++",
    "/opt/ohpc/pub/compiler/gcc/9.4.0/bin/g++",
    "/usr/bin/g++",
)


@functools.lru_cache(maxsize=1)
def host_compiler() -> tuple[str | None, bool]:
    """``(compiler_or_None, found_working_one)`` -- decided by compiling, not by version number.

    ``host_config.h`` nominally allows GCC 14, but GCC 14's libstdc++ uses builtins
    (``__is_array``, ``__is_member_object_pointer``) and a ``0.0bf16`` literal that nvcc's
    frontend does not implement, so even ``#include <cuda_bf16.h>`` fails to parse. Version checks
    say the pair is fine; compiling says it is not. Compile.

    The second element matters: returning a bare ``None`` for both "the default works, no override
    needed" and "nothing here works" made a total failure look like a clean pass, and the build
    kept dying with no flag ever added.
    """
    import tempfile
    from shutil import which

    nvcc = nvcc_path()
    if not nvcc:
        return None, False
    cands = [c.format(prefix=sys.prefix) for c in _HOST_CANDIDATES]
    cands += [p for p in (which(f"g++-{v}") for v in (13, 12, 11, 10, 9)) if p]
    default = cands[0]
    forced = os.environ.get(HOST_CXX_ENV)
    if forced:  # a release build pins its toolchain (and sets CXX to the same compiler for torch's link step)
        cands, default = [forced], ""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(_PROBE)
        for cc in cands:
            if not Path(cc).is_file():
                continue
            cmd = [nvcc, "-std=c++17", "-c", str(src), "-o", str(Path(tmp) / "probe.o")]
            if cc != default:
                cmd += ["-ccbin", cc]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                return (None if cc == default else cc), True
    return None, False


def host_flags() -> list[str]:
    """A placeholder for :func:`real_host_flags`, expanded only when a build runs (:func:`expand_flags`): finding the
    host compiler compiles a probe, which a process that loads prebuilt or cached artifacts must not do."""
    from miniworld_engine.kernels._codegraph import HOST_MARKER

    return [HOST_MARKER]


def expand_flags(flags: list[str] | tuple[str, ...]) -> list[str]:
    """``flags`` with the :func:`host_flags` / :func:`gencodes` placeholders replaced by real nvcc flags (every requested
    arch: an artifact is what its flags ask for, whichever GPU the building machine has, or none)."""
    from miniworld_engine.kernels._codegraph import (
        GENCODE_MARKER,
        HOST_MARKER,
        PTX_MARKER,
    )

    out: list[str] = []
    arches: list[str] = []
    ptx: list[str] = []
    slot = None
    for f in flags:
        f = str(f)
        if f == HOST_MARKER:
            out += real_host_flags()
        elif f.startswith((GENCODE_MARKER, PTX_MARKER)):
            if slot is None:
                slot = len(out)
            (arches if f.startswith(GENCODE_MARKER) else ptx).append(f.split("=", 1)[1])
        else:
            out.append(f)
    if slot is not None:
        out[slot:slot] = real_gencodes(*arches, ptx=tuple(ptx))
    return out


def real_host_flags() -> list[str]:
    """``-ccbin`` for a host compiler nvcc can drive.

    Raises if none of the candidates compiles a probe: a JIT build that is going to fail should
    say why here, where the reason is one line, rather than in a page of libstdc++ diagnostics.
    """
    cc, ok = host_compiler()
    if not ok:
        raise RuntimeError(
            "no host compiler on this machine can be driven by "
            f"{nvcc_path()}. Tried: {', '.join(c.format(prefix=sys.prefix) for c in _HOST_CANDIDATES)}. "
            "nvcc's frontend cannot parse GCC 14's libstdc++ (__is_array, 0.0bf16), so a "
            "GCC <= 13 install is required."
        )
    return ["-ccbin", cc] if cc else []


# ---------------------------------------------------------------------------------------------
# mathdx (cuBLASDx) headers. Three transition extensions include them; their include paths were
# written in as `-I/<a developer home>/mathdx_dl/...`, unbuildable for anyone else and, by the
# time it was found, for the author too -- the directory was gone and nothing noticed, because
# those extensions were absent from the old registry.csv and their build is lazy. Resolve instead, and on
# failure name the variable to set rather than emitting a missing-header error.
# ---------------------------------------------------------------------------------------------

MATHDX_ENV_VARS = ("MINIWORLD_MATHDX_HOME", "MATHDX_HOME", "NVIDIA_MATHDX_HOME")

#: Present in every mathdx distribution; used to tell a real root from a plausible directory.
_MATHDX_SENTINEL = Path("include") / "cublasdx.hpp"


def _mathdx_candidates() -> list[Path]:
    """Roots to try, most explicit first. Existence is checked by the caller, not here."""
    roots = [Path(os.environ[v]) for v in MATHDX_ENV_VARS if os.environ.get(v)]
    # A pip/conda install of `nvidia-mathdx` lands under the `nvidia` namespace package.
    # Imported via importlib, not a static `import nvidia`: the package is optional and absent
    # from the lean CPU install CI runs, where a static import is an unresolved-import error at
    # type-check time even though the runtime path is guarded.
    try:
        nvidia = importlib.import_module("nvidia")
    except ImportError:
        pass
    else:
        roots += [Path(p) / "mathdx" for p in getattr(nvidia, "__path__", [])]
    # The header cache (kernels/_headers.py: the pinned nvidia-mathdx wheel, unpacked on first need).
    from miniworld_engine.kernels import _headers

    roots.append(_headers.MATHDX.cache_dir())
    # A conda/pixi package installs the headers into the environment prefix directly.
    roots.append(Path(sys.prefix))
    return roots


def mathdx_home() -> Path | None:
    """First candidate root that actually holds the headers, or None."""
    for root in _mathdx_candidates():
        if (root / _MATHDX_SENTINEL).is_file():
            return root
    return None


def mathdx_includes() -> list[str]:
    """`-I` flags for cuBLASDx and the CUTLASS it vendors.

    Raises rather than returning empty: an empty list would hand nvcc a missing-header error
    naming a file the user has never heard of, which is how this was diagnosed the slow way.
    """
    from miniworld_engine.kernels import _headers

    root = mathdx_home()
    if root is None:
        with contextlib.suppress(_headers.HeadersUnavailable):
            root = _headers.fetch(_headers.MATHDX)
    if root is None:
        tried = ", ".join(MATHDX_ENV_VARS)
        raise RuntimeError(
            "cuBLASDx headers not found, so the CUDA transition extensions cannot build. "
            f"Set one of {tried} to a mathdx root -- the directory containing "
            f"`{_MATHDX_SENTINEL.as_posix()}` -- or `pip install --no-deps nvidia-mathdx==25.6.0`, or run "
            "`python -m miniworld_engine.kernels._headers fetch mathdx` where the network is reachable. "
            f"Looked in: {', '.join(str(c) for c in _mathdx_candidates())}"
        )
    includes = [root / "include", root / "external" / "cutlass" / "include"]
    return [f"-I{p}" for p in includes if p.is_dir()]


# ---------------------------------------------------------------------------------------------
# CUTLASS / CuTe headers. Only a BUILD needs them: an artifact found prebuilt or cached loads without, so a call site
# takes the include directory from here (CUTLASS_PATH, else the pinned release installed or in the header cache -- see
# kernels/_headers.py --, else a stand-in path keyed as the pinned release); a build that meets the stand-in fetches the pinned
# headers into the header cache then, and raises CutlassMissing only when that is impossible.
# ---------------------------------------------------------------------------------------------

#: Stand-in for an absent CUTLASS checkout: never exists, classified as CUTLASS (``_codegraph.CUTLASS_PINNED``).
CUTLASS_STANDIN = Path("/nonexistent/cutlass-headers/include")


class CutlassMissing(RuntimeError):
    """An artifact that compiles CUTLASS / CuTe headers is neither prebuilt nor cached, and the headers are absent."""


def cutlass_include() -> Path:
    """``$CUTLASS_PATH/include`` when it holds the CuTe headers, else the pinned CUTLASS when it is on this machine
    (``_headers.local``: never downloads), else :data:`CUTLASS_STANDIN`."""
    from miniworld_engine.kernels import _headers

    root = os.environ.get("CUTLASS_PATH", "")
    inc = Path(root) / "include"
    if root and (inc / "cute" / "tensor.hpp").is_file():
        return inc
    return _headers.local(_headers.CUTLASS) or CUTLASS_STANDIN


# ---------------------------------------------------------------------------------------------
# Build progress. Every JIT build is silent (verbose=False everywhere), so a first call that spends
# minutes in nvcc looked exactly like a hang. One line when a compile starts, one when it ends,
# nothing on a cache hit -- so a run that prints nothing reused every artifact.
# ---------------------------------------------------------------------------------------------

#: Prefix of every line this module prints.
LOG_PREFIX = "[miniworld-engine build]"

#: How often a waiter on another process's build says it is still waiting.
WAIT_LOG_SECONDS = 60.0

#: A waiter says nothing for this long: a cached load also holds the lock, for about a second.
WAIT_QUIET_SECONDS = 5.0


class Built(NamedTuple):
    """One compile this process ran."""

    name: str
    kind: str
    seconds: float


_BUILT: list[Built] = []
_BUILT_GUARD = threading.Lock()


def engine_jit_root() -> Path:
    """Directory the JIT-built kernels are cached in: $XDG_CACHE_HOME (else ~/.cache) / miniworld_engine_jit."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "miniworld_engine_jit"


#: Prebuilt artifacts shipped inside the package: the same layout as :func:`engine_jit_root` (a copy of the prebuilt part
#: of a cache, see ``kernels/_prebuild.py``), read-only, looked up first.
PACKAGED_ROOT = Path(__file__).resolve().parents[1] / "_prebuilt"


def artifact_roots() -> list[Path]:
    """Where an artifact is looked up, in order: the packaged prebuilt kernels, then the local cache (where builds go)."""
    return [PACKAGED_ROOT, engine_jit_root()]


def build_log(message: str) -> None:
    """Print one ``[miniworld-engine build]`` line to stderr."""
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


def built_this_process() -> tuple[Built, ...]:
    """What this process compiled, in order (cache hits are not listed)."""
    with _BUILT_GUARD:
        return tuple(_BUILT)


def _record(name: str, kind: str, seconds: float) -> None:
    with _BUILT_GUARD:
        _BUILT.append(Built(name, kind, seconds))
        count = len(_BUILT)
    build_log(f"built {name} in {seconds:.1f} s; {count} built in this process")


@contextlib.contextmanager
def building(name: str, kind: str) -> Iterator[None]:
    """Wrap a compile that is known to run (a cache miss): logs its start and its end, and counts it."""
    build_log(f"compiling {name} ({kind}) ...")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        build_log(f"failed to build {name} after {time.perf_counter() - started:.1f} s")
        raise
    _record(name, kind, time.perf_counter() - started)


def flock_wait(handle: Any, name: str) -> None:
    """``fcntl.flock(handle, LOCK_EX)``, saying once a minute that another process holds it.

    No upper bound and no stale check: the kernel drops a flock when its holder dies, so a killed build never leaves a lock
    that blocks the next one (torch's own ``FileBaton`` lock lives inside the private temporary build directory).
    """
    import fcntl

    started = time.monotonic()
    next_report = started + WAIT_QUIET_SECONDS
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            pass
        now = time.monotonic()
        if now >= next_report:
            build_log(f"waiting for {name}, built by another process ({now - started:.0f} s)")
            next_report = now + WAIT_LOG_SECONDS
        time.sleep(0.2)


def _tmp_name(final: Path) -> Path:
    """A private sibling of ``final`` to build into; renamed onto ``final`` only when complete."""
    import socket

    return final.with_name(f".{final.name}.tmp-{socket.gethostname()}-{os.getpid()}")


# ---------------------------------------------------------------------------------------------
# Artifact resolution: packaged prebuilt -> local cache -> build. Every compiled artifact (a torch extension or a cubin) is
# named by a content key from its code graph (kernels/_codegraph.py), the same on every machine; its path relative to a root
# is ``torch_extensions/<name>-<key16>/`` or ``<family>/<stem>-<key16>.cubin``. A build writes into a private temporary
# path and renames it into place when complete, so a killed build leaves nothing a later process mistakes for an artifact.
#
# Planning (``planning()``, used by kernels/_prebuild.py): a load records what it would build and returns a stand-in instead
# of looking anything up, so a build list comes from calling the very functions the runtime calls.
# ---------------------------------------------------------------------------------------------

#: key of every artifact this process resolved, by path (derived artifacts key off their base).
_KEY_OF: dict[str, str] = {}
_LOADED: dict[str, Any] = {}


def key_of_artifact(path: str | Path) -> str | None:
    """The content key of an artifact this process resolved, by its path."""
    return _KEY_OF.get(str(path))


@dataclasses.dataclass
class Spec:
    """One artifact a load asked for while planning: enough to build it in another process (``call`` replays the load)."""

    kind: str  # "ext" | "cubin" | "derived"
    name: str
    key: str
    rel: str  # path relative to an artifact root
    call: dict
    site: str  # "<file relative to kernels/>::<function>" of the call
    overlay: dict  # generated headers (path -> text) the build writes first
    files: list  # package files in its code graph (sources and headers)


_PLAN: list[Spec] | None = None
#: True in a prebuild worker: build a torch extension without failing when this machine cannot import it (no GPU driver).
_BUILD_ONLY = False


class Planned(RuntimeError):
    """A planned (not built) artifact was used."""


class KernelBuildError(RuntimeError):
    """A kernel the dispatch decided to use failed to build or load (nvcc / ptxas / the torch extension build / the import or the
    cubin load). An error, never a reason to take another path: a fallback here silently measured and served the slower path
    for days (the A100 fused Transition, 2026-11). The original exception is the ``__cause__``."""


def build_failed(what: str, exc: BaseException) -> None:
    """Raise :class:`KernelBuildError` naming ``what`` (the extension / kernel family), chained to ``exc``.

    The one exception is planning (``planning()``: the prebuilder's ``--plan`` / ``--list``): there a load returns a stand-in that
    raises :class:`Planned` when used, so a caller probing its build sees an exception that is not a failure; this returns and the
    caller keeps its old answer, so a plan is exactly what it was. Support decisions (shape / dtype / arch / a missing toolkit
    decided before any build) are not build failures and never come here."""
    if _PLAN is not None:
        return
    if isinstance(exc, KernelBuildError):
        raise exc
    raise KernelBuildError(f"{what}: kernel build / load failed: {type(exc).__name__}: {exc}") from exc


class _Stub:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        raise Planned(f"{self._name}.{attr}: only planned")


@contextlib.contextmanager
def planning() -> Iterator[list[Spec]]:
    """Within the block, every load records a :class:`Spec` and returns a stand-in (an extension stub, a cubin's would-be
    path) instead of looking up or building anything. Yields the list the specs are appended to."""
    global _PLAN
    previous, _PLAN = _PLAN, []
    try:
        yield _PLAN
    finally:
        _PLAN = previous


#: modules whose caches never hold a stand-in (the code graph's parses, this module, the header resolution): kept by plan_session
_INFRA_MODULES = ("miniworld_engine.kernels._codegraph", "miniworld_engine.kernels._nvcc", "miniworld_engine.kernels._headers")


def _family_caches() -> dict[Any, int]:
    """Every ``functools.lru_cache`` of the package's kernel code (module-level, methods and closures such as TriMul's
    ``device_cache``) with its current size."""
    import gc

    out: dict[Any, int] = {}
    for o in gc.get_objects():
        if type(o) is functools._lru_cache_wrapper:  # (type(), not isinstance: no __class__ lookup on lazy proxies)
            module = getattr(o, "__module__", None) or ""
            if module.startswith("miniworld_engine.") and module not in _INFRA_MODULES:
                out[o] = o.cache_info().currsize
    return out


@contextlib.contextmanager
def plan_session() -> Iterator[None]:
    """A whole plan (``kernels/_prebuild.plan``, ``kernels/_prebuild_cuda.plan``): on exit, every loader cache that gained entries
    while it ran is cleared. A loader's ``lru_cache`` keeps the stand-in :func:`planning` made it return, so without this a second
    plan in the same process recorded nothing for that loader, and a real load after the plan got the stand-in back."""
    before = _family_caches()
    try:
        yield
    finally:
        for cache, size in _family_caches().items():
            if size > before.get(cache, 0):
                cache.cache_clear()


def _site() -> str:
    """``file::function`` of the first caller outside this module and TriMul's thin compile wrappers."""
    here = Path(__file__).resolve().parent
    skip = {Path(__file__).resolve(), here / "trimul_inproj" / "cuda" / "_h100_runtime.py"}
    f = sys._getframe(1)
    while f is not None and Path(f.f_code.co_filename).resolve() in skip:
        f = f.f_back
    if f is None:
        return "?"
    p = Path(f.f_code.co_filename).resolve()
    rel = p.relative_to(here).as_posix() if p.is_relative_to(here) else p.name
    return f"{rel}::{f.f_code.co_name}"


def _plan(kind: str, name: str, key: str, rel: Path, graph: Any, call: dict) -> None:
    from miniworld_engine.kernels import _codegraph as cg

    overlay = {p: t for p in graph.paths if (t := cg.overlay_text(p)) is not None} if graph is not None else {}
    assert _PLAN is not None
    files = [p for p in graph.paths if cg._under(p, cg.PKG_ROOT) is not None] if graph is not None else []
    _PLAN.append(Spec(kind, name, key, rel.as_posix(), call, _site(), overlay, files))


def _find(rel: Path, marker: str | None = None) -> Path | None:
    """The first root holding ``rel`` (a directory holding ``marker``, or a file)."""
    for root in artifact_roots():
        p = root / rel
        if (p / marker).is_file() if marker else p.is_file():
            return p
    return None


def _write_json(path: Path, data: dict) -> None:
    import json
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    Path(fh.name).replace(path)


def _safe(stem: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "artifact"


# -------------------------------------------------------------------------------- toolchains (only on a build)


# -------------------------------------------------------------------------------- toolchains (only on a build)
def _toolchain_torch() -> tuple[str, list[str]]:
    ensure_cuda_home()
    nvcc = nvcc_path()
    if not nvcc:
        raise RuntimeError("no nvcc matching torch.version.cuda")
    return nvcc, real_host_flags()


def _toolchain_sm100() -> tuple[str, list[str]]:
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, _, host = kernel_toolchain()
    return nvcc, list(host)


def _toolchain_path() -> tuple[str, list[str]]:
    """The nvcc on PATH after :func:`ensure_cuda_home`, with its default host compiler (TriMul H100)."""
    from shutil import which

    ensure_cuda_home()
    nvcc = which("nvcc")
    if nvcc is None:
        home = os.environ.get("CUDA_HOME")
        nvcc = str(Path(home) / "bin" / "nvcc") if home else None
    if not nvcc:
        raise RuntimeError("these kernels require the CUDA nvcc compiler")
    return nvcc, []


#: How a cubin is compiled when it has to be: (nvcc, host flags). A name, not a callable, so a build recipe is data.
def _toolchain_torch_nohost() -> tuple[str, list[str]]:
    """The torch-matched nvcc with its default host compiler (the families that never passed ``-ccbin``)."""
    ensure_cuda_home()
    nvcc = nvcc_path()
    if not nvcc:
        raise RuntimeError("no nvcc matching torch.version.cuda")
    return nvcc, []


TOOLCHAINS: dict[str, Callable[[], tuple[str, list[str]]]] = {
    "torch": _toolchain_torch,
    "torch-nohost": _toolchain_torch_nohost,
    # the old name of "torch" for the Hopper atom families, kept so recorded recipes still replay: their "CUDA <= 12.9"
    # requirement had no reason behind it (the sources build under CUDA 13.0, with the same SASS)
    "torch-cuda<=12.9": _toolchain_torch,
    "sm100": _toolchain_sm100,
    "path": _toolchain_path,
}


def cubin(
    name: str,
    flags: list[str] | tuple[str, ...],
    *,
    source: str | Path | None = None,
    text: str | None = None,
    family: str | None = None,
    toolchain: str = "torch",
    ptxas_log: bool = False,
    timeout: float | None = 900,
) -> Path:
    """The cubin of ``source`` (or of the generated ``text``) compiled with ``flags``: from the packaged prebuilt kernels,
    else from the local cache (``engine_jit_root/<family>/<name>-<key>.cubin``), else built with ``toolchain``.

    ``name`` identifies the artifact (``<family>/<what>``, no hash); the key (``_codegraph``) covers the sources and
    headers it reaches, the flags and the versions. ``ptxas_log`` keeps nvcc's output next to the cubin
    (``<cubin stem>.ptxas.log``), for callers that read the register report.
    """
    from miniworld_engine.kernels import _codegraph as cg

    if (source is None) == (text is None):
        raise ValueError("cubin() takes a source path or a text, not both")
    flags = [str(f) for f in flags]
    family = family or name.split("/", 1)[0]
    fam_dir = engine_jit_root() / family
    ctx = cg.context()
    root: str | cg.TextRoot = cg.TextRoot(f"text:{name}", text, str(fam_dir)) if text is not None else str(source)
    graph = cg.closure([root], cg.include_dirs_of(flags), ctx)
    material = cg.key_material("cubin", name, graph, flags, ctx=ctx)
    key = cg.key_of(material)
    rel = Path(family) / f"{_safe(name.split('/', 1)[-1])}-{key[:16]}.cubin"
    if _PLAN is not None:
        _plan(
            "cubin",
            name,
            key,
            rel,
            graph,
            {
                "name": name,
                "flags": flags,
                "source": None if source is None else str(source),
                "text": text,
                "family": family,
                "toolchain": toolchain,
                "ptxas_log": ptxas_log,
                "timeout": timeout,
            },
        )
        out = engine_jit_root() / rel
        _KEY_OF[str(out)] = key
        return out
    hit = _find(rel)
    if hit is not None:
        _KEY_OF[str(hit)] = key
        return hit
    out = engine_jit_root() / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".lock").open("w") as lock:
        flock_wait(lock, name)
        if not out.exists():
            with building(f"{name} [{key[:8]}]", "cubin"):
                nvcc, host = TOOLCHAINS[toolchain]()
                cg.materialize(graph.paths)
                src = Path(source) if source is not None else out.with_suffix(".cu")
                if text is not None and not src.exists():
                    tmp_src = _tmp_name(src)
                    tmp_src.write_text(text)
                    tmp_src.replace(src)
                tmp = _tmp_name(out)
                res = subprocess.run(
                    [nvcc, *host, *expand_flags(flags), str(src), "-o", str(tmp)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if res.returncode != 0:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"nvcc failed on {name}:\n{res.stderr[-4000:]}")
                if ptxas_log:
                    _replace_text(out.with_suffix(".ptxas.log"), res.stdout + res.stderr)
                _write_json(out.with_suffix(".json"), {"name": name, "key": key, "material": material})
                os.replace(tmp, out)  # last: the cubin's presence means its sidecars are complete
    _KEY_OF[str(out)] = key
    return out


def _replace_text(path: Path, text: str) -> None:
    tmp = _tmp_name(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def derived_cubin(
    name: str,
    base: str | Path,
    params: Any,
    produce: Callable[[Path], bool],
    family: str | None = None,
    replay: tuple[str, list] | None = None,
) -> Path | None:
    """A cubin made from another one by ``produce(tmp_path) -> bool`` (a post-link patch, no compiler): keyed by the
    base's key and ``params`` (which should carry a digest of the patch code). None when ``produce`` declines (the
    caller keeps the base). ``replay`` = ("module:function", args): ``function(base_cubin, *args)`` redoes it (a prebuild
    calls it once the base is built)."""
    from miniworld_engine.kernels import _codegraph as cg

    base_key = key_of_artifact(base)
    if base_key is None:
        raise RuntimeError(f"derived_cubin: {base} was not resolved through _nvcc.cubin in this process")
    material = {"scheme": cg.KEY_SCHEME, "kind": "cubin", "name": name, "derived_from": base_key, "params": repr(params)}
    key = cg.key_of(material)
    family = family or name.split("/", 1)[0]
    rel = Path(family) / f"{_safe(name.split('/', 1)[-1])}-{key[:16]}.cubin"
    if _PLAN is not None:
        if replay is None:
            raise RuntimeError(f"derived_cubin {name}: no replay, so a prebuild cannot make it")
        _plan("derived", name, key, rel, None, {"base_key": base_key, "replay": list(replay)})
        return engine_jit_root() / rel
    hit = _find(rel)
    if hit is not None:
        _KEY_OF[str(hit)] = key
        return hit
    out = engine_jit_root() / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(out)
    with building(f"{name} [{key[:8]}]", "derived cubin"):
        ok = produce(tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        return None
    os.replace(tmp, out)
    _KEY_OF[str(out)] = key
    return out


# -------------------------------------------------------------------------------- torch extensions
def _ext_key(name: str, kwargs: dict, sources: list[str] | None, inline: bool) -> tuple[dict, str, Any]:
    from miniworld_engine.kernels import _codegraph as cg

    ctx = cg.context()
    # load_inline writes main.cpp / cuda.cu into the build directory, where nothing else lives: quoted includes resolve
    # through the -I directories, so the directory only has to be one without headers.
    build_dir = Path(ctx.jit or ".") / "torch_extensions"
    cflags = [str(f) for f in kwargs.get("extra_cflags") or []]
    cuflags = [str(f) for f in kwargs.get("extra_cuda_cflags") or []]
    incs = [str(p) for p in kwargs.get("extra_include_paths") or []]
    include_dirs = [os.path.abspath(p) for p in incs] + cg.include_dirs_of(cuflags) + cg.include_dirs_of(cflags)
    if inline:
        where = str(build_dir or ".")
        roots: list = []
        for label, part in (("inline:cpp", "cpp_sources"), ("inline:cuda", "cuda_sources")):
            value = kwargs.get(part)
            if value:
                roots.append(cg.TextRoot(label, value if isinstance(value, str) else "\n".join(value), where))
    else:
        roots = [str(s) for s in sources or []]
    graph = cg.closure(roots, include_dirs, ctx)
    flags = {
        "cflags": cflags,
        "cuda_cflags": cuflags,
        "ldflags": [str(f) for f in kwargs.get("extra_ldflags") or []],
        "include": [f"-I{p}" for p in incs],
        "with_cuda": kwargs.get("with_cuda"),
        "python_module": kwargs.get("is_python_module", True),
        "standalone": kwargs.get("is_standalone", False),
        "functions": list(kwargs["functions"]) if isinstance(kwargs.get("functions"), (list, tuple)) else kwargs.get("functions"),
    }
    material = cg.key_material("ext", name, graph, flags, python=True, ctx=ctx)
    return material, cg.key_of(material), graph


def _pinned_cutlass(kwargs: dict) -> dict | None:
    """``kwargs`` with every CUTLASS include directory of another release than ``_codegraph.CUTLASS_PINNED`` replaced by
    ``CUTLASS_STANDIN`` (keyed as the pinned release), or None when there is none: the key the prebuilt artifact has. A
    CUTLASS_PATH to another checkout (4.8.0 in FoldForge's runs, 2026-10-30) otherwise re-keyed every CUTLASS extension and
    compiled the prebuilt ones again on the first call."""
    from miniworld_engine.kernels import _codegraph as cg

    incs = [str(p) for p in kwargs.get("extra_include_paths") or []]
    ctx = cg.context()
    pinned = f"cutlass@{cg.CUTLASS_PINNED}"
    swapped = []
    for p in incs:
        label, _ = cg.classify_dir(p, ctx)
        swapped.append(str(CUTLASS_STANDIN) if label.startswith("cutlass@") and label != pinned else p)
    if swapped == incs:
        return None
    return {**kwargs, "extra_include_paths": swapped}


def _import_ext(so: Path, module: str, python_module: bool) -> Any:
    from torch.utils.cpp_extension import _import_module_from_library

    return _import_module_from_library(module, str(so.parent), python_module)


def _ext_rel(name: str, key: str) -> Path:
    return Path("torch_extensions") / f"{name}-{key[:16]}"


def _load_built(directory: Path, key: str, python_module: bool) -> Any:
    import json

    info = json.loads((directory / "artifact.json").read_text())
    so = directory / info["file"]
    ext = _import_ext(so, info["module"], python_module)
    _KEY_OF[str(so)] = key
    return ext


def _load(kind_label: str, name: str, kwargs: dict, sources: list[str] | None = None, inline: bool = False) -> Any:
    # Lists copied: torch's load appends its own link flags to `extra_ldflags` in place, which changed the caller's list
    # after the key was computed.
    kwargs = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    kwargs.pop("build_directory", None)  # content-addressed below: one directory per key, never rebuilt in place
    material, key, graph = _ext_key(name, kwargs, sources, inline)
    rel = _ext_rel(name, key)
    if _PLAN is not None:
        _plan(
            "ext", name, key, rel, graph, {"kind_label": kind_label, "name": name, "kwargs": kwargs, "sources": sources, "inline": inline}
        )
        return _Stub(name)
    if key in _LOADED:
        return _LOADED[key]
    python_module = bool(kwargs.get("is_python_module", True))
    hit = _find(rel, "artifact.json")
    if hit is None and (pinned := _pinned_cutlass(kwargs)) is not None:
        # CUTLASS_PATH names another CUTLASS release than the prebuilt artifacts were built with: the headers only matter to a
        # build, so the pinned release's artifact is the one (a build below still uses this checkout's key)
        _, p_key, _ = _ext_key(name, pinned, sources, inline)
        hit = _find(_ext_rel(name, p_key), "artifact.json")
        if hit is not None:
            key = p_key
    if hit is None:
        hit = _build_ext(kind_label, name, kwargs, sources, inline, key, material, graph)
    if _BUILD_ONLY:
        return hit
    ext = _load_built(hit, key, python_module)
    _LOADED[key] = ext
    return ext


def _real_cutlass(name: str, kwargs: dict) -> dict:
    """``kwargs`` with :data:`CUTLASS_STANDIN` replaced by the pinned CUTLASS headers, fetched into the header cache if need be
    (the key stays: both are keyed ``cutlass@<pinned>``); CutlassMissing when they cannot be had."""
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels import _headers

    incs = [str(p) for p in kwargs.get("extra_include_paths") or []]
    if str(CUTLASS_STANDIN) not in [os.path.abspath(p) for p in incs]:
        return kwargs
    try:
        real = str(_headers.fetch(_headers.CUTLASS))
    except _headers.HeadersUnavailable as e:
        raise CutlassMissing(
            f"{name} compiles CUTLASS {cg.CUTLASS_PINNED} / CuTe headers and is not prebuilt, and the pinned headers could not be "
            f"fetched ({e}). Set CUTLASS_PATH to a CUTLASS v{cg.CUTLASS_PINNED} checkout, `pip install --no-deps "
            f"nvidia-cutlass=={_headers.CUTLASS.version}`, or run `python -m miniworld_engine.kernels._headers fetch cutlass` "
            "where the network is reachable (see docs/install.md)"
        ) from e
    return {**kwargs, "extra_include_paths": [real if os.path.abspath(p) == str(CUTLASS_STANDIN) else p for p in incs]}


def _build_ext(
    kind_label: str, name: str, kwargs: dict, sources: list[str] | None, inline: bool, key: str, material: dict, graph: Any
) -> Path:
    """Build the extension into a private temporary directory and rename it to its key's directory; returns that directory.
    A flock (dropped by the kernel when its holder dies) keeps two processes from building the same key at once."""
    import shutil

    from miniworld_engine.kernels import _codegraph as cg

    kwargs = _real_cutlass(name, kwargs)
    final = engine_jit_root() / _ext_rel(name, key)
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.with_name(f".{final.name}.lock").open("w") as lock:
        flock_wait(lock, name)
        if (final / "artifact.json").is_file():
            return final  # another process built it while this one waited
        ensure_cuda_home()
        cg.materialize(graph.paths)
        tmp = _tmp_name(final)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        call_kwargs: dict[str, Any] = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        call_kwargs["build_directory"] = str(tmp)
        for f in ("extra_cflags", "extra_cuda_cflags"):
            if call_kwargs.get(f):
                call_kwargs[f] = expand_flags(call_kwargs[f])
        forced = os.environ.get(HOST_CXX_ENV)
        cu = call_kwargs.get("extra_cuda_cflags") or []
        if forced and not any(str(f) == "-ccbin" or str(f).startswith(("-ccbin=", "--compiler-bindir")) for f in cu):
            call_kwargs["extra_cuda_cflags"] = [*cu, "-ccbin", forced]  # a pinned release toolchain covers every .cu
        python_module = bool(kwargs.get("is_python_module", True))
        built = lambda: sorted(tmp.glob("*.so"))  # torch names it <name>.so, or <name>_v<n>.so on a rebuild in one process
        try:
            with building(f"{name} [{key[:8]}]", kind_label):
                try:
                    if inline:
                        from torch.utils.cpp_extension import load_inline

                        load_inline(name=name, **call_kwargs)
                    else:
                        from torch.utils.cpp_extension import load

                        load(name=name, sources=sources or [], **call_kwargs)
                except (ImportError, OSError):
                    if not (_BUILD_ONLY and built()):
                        raise  # a prebuild worker may not be able to import a CUDA extension; the build itself succeeded
            if len(built()) != 1:
                raise RuntimeError(f"{name}: the build produced {len(built())} shared libraries")
            so = built()[0]
            for junk in tmp.glob("*.o"):  # objects are only build intermediates
                junk.unlink()
            _write_json(
                tmp / "artifact.json",
                {
                    "name": name,
                    "key": key,
                    "file": so.name,
                    "module": so.name[: -len(".so")],
                    "python_module": python_module,
                    "material": material,
                },
            )
            os.rename(tmp, final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def load_extension(name: str, sources: list[str], **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load`, resolved through the packaged prebuilt kernels and a content-keyed cache.

    The extension's key (its code graph, flags and versions) picks ``<root>/torch_extensions/<name>-<key>/`` in the packaged
    prebuilt kernels or the local cache; only a miss runs torch's build, into a private directory renamed into place when
    complete. An edit that does not reach the extension's sources never rebuilds it, and an edit that does always does (a new
    directory). ``build_directory`` is ignored.
    """
    return _load("torch extension", name, kwargs, sources=[str(s) for s in sources])


def load_inline_extension(name: str, **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load_inline`, resolved like :func:`load_extension` (the inline text is the graph's root)."""
    return _load("inline torch extension", name, kwargs, inline=True)
