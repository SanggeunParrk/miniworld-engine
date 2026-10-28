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
            out = subprocess.run([str(c), "--version"], capture_output=True, text=True,
                                 timeout=20, check=False).stdout
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
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True,
                             timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(line.strip() for line in out.split("\n") if line.strip())


#: Set by ``dev bundle-kernels`` (``sm90`` / ``sm100``): a build keeps only that arch out of a multi-arch ``gencodes``
#: request, so a bundle carries one arch's code. The artifact's key is computed from the request, not from this.
BUNDLE_ARCH_ENV = "MINIWORLD_ENGINE_BUNDLE_ARCH"


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
        return (not have or f"compute_{a}" in have
                or (a.endswith("a") and a[:-1].isdigit() and int(a[:-1]) >= 90
                    and f"compute_{a[:-1]}" in have))
    out = [f"-gencode=arch=compute_{a},code=sm_{a}"
           for a in arches if supported(a)]
    out += [f"-gencode=arch=compute_{a},code=compute_{a}"
            for a in ptx if supported(a)]
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
    """``flags`` with the :func:`host_flags` / :func:`gencodes` placeholders replaced by real nvcc flags. Under
    ``dev bundle-kernels`` (``MINIWORLD_ENGINE_BUNDLE_ARCH``) a multi-arch request keeps only the bundle's arch."""
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
        bundle = os.environ.get(BUNDLE_ARCH_ENV, "").removeprefix("sm").removeprefix("_")
        if bundle:
            keep = [a for a in arches if a.rstrip("a") == bundle.rstrip("a")]
            if keep:
                arches, ptx = keep, [a for a in ptx if a.rstrip("a") == bundle.rstrip("a")]
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
# those extensions are absent from registry.csv and their build is lazy. Resolve instead, and on
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
    root = mathdx_home()
    if root is None:
        tried = ", ".join(MATHDX_ENV_VARS)
        raise RuntimeError(
            "cuBLASDx headers not found, so the CUDA transition extensions cannot build. "
            f"Set one of {tried} to a mathdx root -- the directory containing "
            f"`{_MATHDX_SENTINEL.as_posix()}` -- or `pip install nvidia-mathdx`. "
            f"Looked in: {', '.join(str(c) for c in _mathdx_candidates())}")
    includes = [root / "include", root / "external" / "cutlass" / "include"]
    return [f"-I{p}" for p in includes if p.is_dir()]


# ---------------------------------------------------------------------------------------------
# CUTLASS / CuTe headers. Only a BUILD needs them: an artifact found prebuilt or cached loads without, so a call site
# takes the include directory from here (a stand-in path when CUTLASS_PATH is unset or incomplete, keyed as the pinned
# CUTLASS release) and the requirement is checked when a build is about to run (CutlassMissing).
# ---------------------------------------------------------------------------------------------

#: Stand-in for an absent CUTLASS checkout: never exists, classified as CUTLASS (``_codegraph.CUTLASS_PINNED``).
CUTLASS_STANDIN = Path("/nonexistent/cutlass-headers/include")


class CutlassMissing(RuntimeError):
    """An artifact that compiles CUTLASS / CuTe headers is neither prebuilt nor cached, and the headers are absent."""


def cutlass_include() -> Path:
    """``$CUTLASS_PATH/include`` when it holds the CuTe headers, else :data:`CUTLASS_STANDIN`."""
    root = os.environ.get("CUTLASS_PATH", "")
    inc = Path(root) / "include"
    return inc if root and (inc / "cute" / "tensor.hpp").is_file() else CUTLASS_STANDIN


def cutlass_available() -> bool:
    return cutlass_include() != CUTLASS_STANDIN


# ---------------------------------------------------------------------------------------------
# Build progress. Every JIT build is silent (verbose=False everywhere), so a first call that spends
# minutes in nvcc looked exactly like a hang. One line when a compile starts, one when it ends,
# nothing on a cache hit -- so a run that prints nothing reused every artifact.
# ---------------------------------------------------------------------------------------------

#: Prefix of every line this module prints.
LOG_PREFIX = "[miniworld-engine build]"

#: A torch extension load whose ``.so`` cannot be located counts as a build when it took longer
#: than this; a no-op ninja check plus the import is far below it.
LOAD_BUILD_SECONDS = 2.0

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


def _log_enabled() -> bool:
    try:
        from miniworld_engine import settings
    except ImportError:
        return True
    return bool(getattr(settings.current(), "jit_build_log", True))


def build_log(message: str) -> None:
    """Print one ``[miniworld-engine build]`` line to stderr (unless ``settings.jit_build_log`` is off)."""
    if _log_enabled():
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


def cache_roots() -> list[Path]:
    """The JIT cache directories that exist here: ``settings.engine_jit_root()`` (every cubin and
    every extension built with a ``build_directory``) and torch's extensions root (``$TORCH_EXTENSIONS_DIR``,
    else torch's default, for ``load_extension`` calls without one). See docs/kernels/jit-cache.md."""
    from miniworld_engine import settings

    roots = [settings.engine_jit_root()]
    torch_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not torch_root:
        try:
            from torch.utils.cpp_extension import get_default_build_root
            torch_root = get_default_build_root()
        except Exception:  # no torch, no torch cache
            torch_root = None
    if torch_root:
        roots.append(Path(torch_root))
    out: list[Path] = []
    for r in roots:
        if r.is_dir() and r.resolve() not in {o.resolve() for o in out}:
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------
# JIT extension loading, with the two failure modes that cost five GPU-hours.
#
# `torch.utils.cpp_extension.load` serialises concurrent builds with a `FileBaton` on
# `<build dir>/lock`, and `FileBaton.wait()` polls for that file to disappear FOREVER, with no
# timeout and no message. A build that is killed -- a cancelled Slurm job, a Ctrl-C -- leaves the
# lock behind, and every later run on that machine blocks on it indefinitely.
#
# That happened. A run left a 13-hour-old lock in `layer_norm_cuda/`; three subsequent GPU jobs
# (45 min killed, 4 h timed out, one stalled at test 42 of 100) all sat on it, and all three looked
# exactly like "slow". Deleting the file moved the stalled job forward within 20 seconds.
#
# So: reclaim a lock that is too old to be real, say once a minute that we are waiting on a live
# one, and raise naming the file only when the wait is beyond any real build. The remedy is one `rm`.
# ---------------------------------------------------------------------------------------------

#: A lock older than this cannot belong to a live build -- the longest real build here is a few
#: minutes -- so it is a leftover and gets reclaimed, also while waiting on it. `build --reclaim`
#: makes the same judgement about the builder's own O_EXCL claims.
STALE_LOCK_SECONDS = 1800.0

#: Upper bound on a wait for another process's build. A waiter does not give up on a live build
#: (it waits until the lock is gone, or reclaims it once stale); this only fires when something is
#: clearly wrong, such as a lock that cannot be removed or whose timestamps never age.
LOCK_WAIT_SECONDS = 7200.0


def _build_lock(name: str, build_directory: str | None = None) -> Path | None:
    """The lock file `torch.utils.cpp_extension.load` will use for `name`, if it can be located.

    `load(build_directory=...)` puts the baton there, not under TORCH_EXTENSIONS_DIR. Checking the
    default dir instead left a 3-day-old leftover in ~/.cache/miniworld_engine_jit/miniworld_pwa_ctr
    unguarded, and every rank of a MiniWorld training job sat in torch's unbounded
    FileBaton.wait() (2026-09-28).
    """
    if build_directory:
        return Path(build_directory) / "lock"
    try:
        from torch.utils.cpp_extension import _get_build_directory
    except ImportError:      # torch too old / not installed: let `load` do whatever it does
        return None
    try:
        return Path(_get_build_directory(name, verbose=False)) / "lock"
    except Exception:        # locating the lock must never be what fails a build
        return None


def clear_stale_lock(lock: Path, stale_after: float = STALE_LOCK_SECONDS) -> bool:
    """Delete `lock` if it is too old to belong to a running build. True if it was reclaimed."""
    try:
        st = lock.stat()
    except OSError:
        return False
    # NFSv3 exclusive create (torch's FileBaton uses O_CREAT|O_EXCL) stores the create verifier
    # in atime/mtime, so a LIVE lock on an NFS home can show an mtime in 1981. Trusting it deleted
    # another rank's lock mid-build (2026-09-27): the builder's baton.release() raised
    # FileNotFoundError and the job hung. ctime is set by the server; take the newer of the two,
    # and never reclaim a timestamp from before 2000.
    stamp = max(st.st_mtime, st.st_ctime)
    if stamp < 946684800:
        return False
    age = time.time() - stamp
    if age < stale_after:
        return False
    try:
        lock.unlink()
    except OSError:
        return False
    print(f"{LOG_PREFIX} reclaimed a stale build lock ({age / 60:.0f} min old): {lock}",
          file=sys.stderr, flush=True)
    return True


def wait_for_lock(lock: Path, limit: float = LOCK_WAIT_SECONDS, name: str | None = None,
                  stale_after: float = STALE_LOCK_SECONDS, poll: float = 2.0) -> None:
    """Wait while another process builds behind `lock`; return once it is gone or reclaimed as stale.

    Says so once a minute (after a few quiet seconds: a cached load holds the lock briefly too).
    Raises, naming the file, only past `limit` -- far beyond any real build.
    """
    if not lock.exists():
        return
    what = name or lock.parent.name
    started = time.monotonic()
    next_report = started + min(WAIT_QUIET_SECONDS, limit)
    while lock.exists():
        if clear_stale_lock(lock, stale_after):
            return
        now = time.monotonic()
        if now - started > limit:
            msg = (f"waited {(now - started) / 60:.0f} min for another process's build lock and it "
                   f"is still there:\n    {lock}\n"
                   f"If no build is running, it is a leftover from a killed one -- torch waits on "
                   f"it forever, which is why this raises instead. Remove it:\n"
                   f"    rm {lock}")
            raise TimeoutError(msg)
        if now >= next_report:
            build_log(f"waiting for {what}, built by another process ({now - started:.0f} s)")
            next_report = now + WAIT_LOG_SECONDS
        time.sleep(poll)


def flock_wait(handle: Any, name: str) -> None:
    """``fcntl.flock(handle, LOCK_EX)``, saying once a minute that another process holds it.

    No upper bound and no stale check: the kernel drops a flock when its holder dies.
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


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _will_build(library: Path, before: float | None, sources: list[str] | None) -> bool:
    """Cheaply predict that torch's ninja is about to compile: no library yet, or (when the sources are
    known) a ``build.ninja`` that does not name them -- another checkout built this cache -- or a source
    newer than the library. Included headers are not checked; a rebuild they cause gets only the
    "built" line."""
    if before is None:
        return True
    if not sources:
        return False
    try:
        ninja = (library.parent / "build.ninja").read_text()
    except OSError:
        return True
    for s in sources:
        path = os.path.abspath(s)
        if path not in ninja:
            return True
        mtime = _mtime(Path(path))
        if mtime is not None and mtime > before:
            return True
    return False


def _guarded_load(name: str, kind: str, build_directory: str | None, call: Callable[[], Any],
                  sources: list[str] | None = None) -> Any:
    """Run a torch JIT load behind the lock guard, and log it if it compiled.

    torch decides internally whether to rebuild (ninja), so the build is confirmed afterwards: the
    library ``<build dir>/<name>.so`` is new or changed. When that is predictable up front
    (:func:`_will_build`) the "compiling" line comes first.
    """
    lock = _build_lock(name, build_directory)
    if lock is not None and not clear_stale_lock(lock):
        wait_for_lock(lock, name=name)
    library = lock.parent / f"{name}.so" if lock is not None else None
    before = _mtime(library)
    announced = library is not None and _will_build(library, before, sources)
    if announced:
        build_log(f"compiling {name} ({kind}) ...")
    started = time.perf_counter()
    try:
        ext = call()
    except BaseException:
        # torch does not always release its baton when the build raises, and the leftover is
        # indistinguishable from a live build -- so the NEXT call in this same process waits on a
        # lock nobody holds. Seen for real: one nvcc failure in `transition_cuda_ext_v2` left the
        # lock, and the following checker sat in `wait_for_lock` until its deadline. Guarding only
        # the lock that exists on entry (above) covers another process; this covers our own.
        if lock is not None:
            lock.unlink(missing_ok=True)
        if announced:
            build_log(f"failed to build {name} after {time.perf_counter() - started:.1f} s")
        raise
    seconds = time.perf_counter() - started
    if library is not None:
        after = _mtime(library)
        compiled = after is not None and after != before
    else:
        compiled = seconds > LOAD_BUILD_SECONDS
    if compiled:
        _record(name, kind, seconds)
    elif announced:
        build_log(f"{name} needed no rebuild")
    return ext


# ---------------------------------------------------------------------------------------------
# Artifact resolution: installed bundle -> local cache -> build. Every compiled artifact (a torch extension or a cubin)
# is named by a content key from its code graph (kernels/_codegraph.py), the same on every machine. A bundle hit loads
# the shipped binary and never touches nvcc; a cache hit loads the local one; only a miss compiles, and with
# settings.jit_fallback False a miss raises instead. See docs/kernels/prebuilt-kernels.md.
# ---------------------------------------------------------------------------------------------

#: key of every artifact this process resolved, by path (derived artifacts key off their base).
_KEY_OF: dict[str, str] = {}
_LOADED: dict[str, Any] = {}
_MISS_EXPLAINED = False


def key_of_artifact(path: str | Path) -> str | None:
    """The content key of an artifact this process resolved, by its path."""
    return _KEY_OF.get(str(path))


def _jit_fallback() -> bool:
    try:
        from miniworld_engine import settings
    except ImportError:
        return True
    return bool(getattr(settings.current(), "jit_fallback", True))


def _before_build(kind: str, name: str, key: str) -> None:
    """A miss: say so (once per artifact), or raise when builds are not allowed."""
    global _MISS_EXPLAINED
    from miniworld_engine.kernels import prebuilt

    if not _jit_fallback():
        raise prebuilt.PrebuiltMissing(
            f"{name} ({kind}, key {key[:16]}) is not in the prebuilt kernel bundle ({prebuilt.describe()}) and "
            "settings.jit_fallback is False, so it is not compiled. Rebuild the bundle with this artifact covered "
            "(docs/kernels/prebuilt-kernels.md), or allow JIT builds.")
    if prebuilt.enabled():
        why = "" if _MISS_EXPLAINED else f" ({prebuilt.describe()})"
        _MISS_EXPLAINED = True
        build_log(f"not in the prebuilt bundle: {name}{why}")


def _resolved(kind: str, name: str, key: str, path: Path, origin: str, material: dict, **extra: Any) -> None:
    _KEY_OF[str(path)] = key
    from miniworld_engine.kernels import prebuilt

    if prebuilt.recording():
        sidecars = [s for s in (".ptxas.log",) if path.with_suffix(s).is_file()] if kind == "cubin" else []
        prebuilt.record({"kind": kind, "name": name, "key": key, "path": str(path), "origin": origin,
                         "material": material, "sidecars": sidecars, **extra})


def _write_json(path: Path, data: dict) -> None:
    import json
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    Path(fh.name).replace(path)


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
    # requirement had no reason behind it (the sources build under CUDA 13.0, with the same SASS; docs/kernels/prebuilt-kernels.md)
    "torch-cuda<=12.9": _toolchain_torch,
    "sm100": _toolchain_sm100,
    "path": _toolchain_path,
}


#: Toolchains whose nvcc is not torch's (``torch.version.cuda`` is already in every key): a cubin built by one of them keys on the
#: toolchain and the nvcc release it resolved to, so a 12.9 and a 13.1 build of the same source never share a cache entry or a bundle
#: slot (the B200 families pick the newest nvcc that knows sm_100a, which differs between hosts). The torch-matched toolchains keep
#: their keys (and the H100 wheel's prebuilt artifacts stay valid).
_KEYED_TOOLCHAINS = ("sm100",)


@functools.lru_cache(maxsize=None)
def _nvcc_release(nvcc: str) -> str:
    """``V13.1.80``-style version of ``nvcc`` (its ``--version`` line), cached per path."""
    import re

    res = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=120, check=False)
    found = re.search(r"release [\d.]+, (V[\d.]+)", res.stdout)
    return found.group(1) if found else (res.stdout.strip().splitlines() or ["unknown"])[-1]


def _toolchain_versions(toolchain: str) -> dict[str, str] | None:
    from miniworld_engine.kernels import _codegraph as cg

    if toolchain not in _KEYED_TOOLCHAINS:
        return None
    return {**cg._versions(), "toolchain": toolchain, "nvcc": _nvcc_release(TOOLCHAINS[toolchain]()[0])}


def _safe(stem: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "artifact"


def cubin(name: str, flags: list[str] | tuple[str, ...], *, source: str | Path | None = None, text: str | None = None,
          family: str | None = None, toolchain: str = "torch", ptxas_log: bool = False,
          timeout: float | None = 900) -> Path:
    """The cubin of ``source`` (or of the generated ``text``) compiled with ``flags``: from an installed kernel bundle,
    else from the local cache (``engine_jit_root/<family>/<name>-<key>.cubin``), else built with ``toolchain``.

    ``name`` identifies the artifact (``<family>/<what>``, no hash); the key (``_codegraph``) covers the sources and
    headers it reaches, the flags and the versions. ``ptxas_log`` keeps nvcc's output next to the cubin
    (``<cubin stem>.ptxas.log``), for callers that read the register report.
    """
    from miniworld_engine import settings
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels import prebuilt

    if (source is None) == (text is None):
        raise ValueError("cubin() takes a source path or a text, not both")
    flags = [str(f) for f in flags]
    family = family or name.split("/", 1)[0]
    fam_dir = settings.engine_jit_root() / family
    ctx = cg.context()
    root: str | cg.TextRoot = cg.TextRoot(f"text:{name}", text, str(fam_dir)) if text is not None else str(source)
    graph = cg.closure([root], cg.include_dirs_of(flags), ctx)
    material = cg.key_material("cubin", name, graph, flags, ctx=ctx, versions=_toolchain_versions(toolchain))
    key = cg.key_of(material)
    recipe = {"fn": "cubin", "name": name, "flags": flags, "source": None if source is None else str(source), "text": text,
              "family": family, "toolchain": toolchain, "ptxas_log": ptxas_log, "timeout": timeout, "jit": ctx.jit,
              "overlay": {p: t for p in graph.paths if (t := cg.overlay_text(p)) is not None}}
    hit = prebuilt.find("cubin", name, key)
    if hit is not None:
        _resolved("cubin", name, key, hit.path, "prebuilt", material, recipe=recipe)
        return hit.path
    out = fam_dir / f"{_safe(name.split('/', 1)[-1])}-{key[:16]}.cubin"
    if not out.exists():
        _before_build("cubin", name, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix(".lock").open("w") as lock:
            flock_wait(lock, name)
            if not out.exists():
                with building(f"{name} [{key[:8]}]", "cubin"):
                    nvcc, host = TOOLCHAINS[toolchain]()
                    cg.materialize(graph.paths)
                    src = Path(source) if source is not None else out.with_suffix(".cu")
                    if text is not None and not src.exists():
                        tmp_src = src.with_name(f"{src.name}.{os.getpid()}.tmp")
                        tmp_src.write_text(text)
                        tmp_src.replace(src)
                    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp")
                    res = subprocess.run([nvcc, *host, *expand_flags(flags), str(src), "-o", str(tmp)],
                                         capture_output=True, text=True, timeout=timeout, check=False)
                    if ptxas_log:
                        out.with_suffix(".ptxas.log").write_text(res.stdout + res.stderr)
                    if res.returncode != 0:
                        tmp.unlink(missing_ok=True)
                        raise RuntimeError(f"nvcc failed on {name}:\n{res.stderr[-4000:]}")
                    _write_json(out.with_suffix(".json"), {"name": name, "key": key, "material": material})
                    os.replace(tmp, out)
                _resolved("cubin", name, key, out, "built", material, recipe=recipe)
                return out
    _resolved("cubin", name, key, out, "cache", material, recipe=recipe)
    return out


def derived_cubin(name: str, base: str | Path, params: Any, produce: Callable[[Path], bool],
                  family: str | None = None, replay: tuple[str, list] | None = None) -> Path | None:
    """A cubin made from another one by ``produce(tmp_path) -> bool`` (a post-link patch, no compiler): keyed by the
    base's key and ``params`` (which should carry a digest of the patch code). None when ``produce`` declines (the
    caller keeps the base)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels import prebuilt

    base_key = key_of_artifact(base)
    if base_key is None:
        raise RuntimeError(f"derived_cubin: {base} was not resolved through _nvcc.cubin in this process")
    material = {"scheme": cg.KEY_SCHEME, "kind": "cubin", "name": name, "derived_from": base_key,
                "params": repr(params)}
    key = cg.key_of(material)
    # how to redo it elsewhere: "module:function" called as function(base_cubin, *args)
    recipe = {"fn": "derived", "name": name, "base_key": base_key, "replay": list(replay) if replay else None}
    hit = prebuilt.find("cubin", name, key)
    if hit is not None:
        _resolved("cubin", name, key, hit.path, "prebuilt", material, recipe=recipe)
        return hit.path
    family = family or name.split("/", 1)[0]
    out = settings.engine_jit_root() / family / f"{_safe(name.split('/', 1)[-1])}-{key[:16]}.cubin"
    if not out.exists():
        _before_build("cubin", name, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp")
        with building(f"{name} [{key[:8]}]", "derived cubin"):
            ok = produce(tmp)
        if not ok:
            tmp.unlink(missing_ok=True)
            return None
        os.replace(tmp, out)
        _resolved("cubin", name, key, out, "built", material, recipe=recipe)
        return out
    _resolved("cubin", name, key, out, "cache", material, recipe=recipe)
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
        "functions": list(kwargs["functions"]) if isinstance(kwargs.get("functions"), (list, tuple))
        else kwargs.get("functions"),
    }
    material = cg.key_material("ext", name, graph, flags, python=True, ctx=ctx)
    return material, cg.key_of(material), graph


def _import_ext(so: Path, module: str, python_module: bool) -> Any:
    from torch.utils.cpp_extension import _import_module_from_library

    return _import_module_from_library(module, str(so.parent), python_module)


def _load(kind_label: str, name: str, kwargs: dict, sources: list[str] | None = None, inline: bool = False) -> Any:
    import json

    from miniworld_engine import settings
    from miniworld_engine.kernels import _codegraph as cg
    from miniworld_engine.kernels import prebuilt

    # Lists copied: torch's load appends its own link flags to `extra_ldflags` in place, which changed the caller's list
    # and the recorded recipe after the key was computed (a replay then keyed the artifact differently).
    kwargs = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    kwargs.pop("build_directory", None)  # content-addressed below: one directory per key, never rebuilt in place
    material, key, graph = _ext_key(name, kwargs, sources, inline)
    if key in _LOADED:
        return _LOADED[key]
    recipe = {"fn": "ext", "name": name, "inline": inline, "sources": sources, "jit": str(settings.engine_jit_root()),
              "kwargs": {k: v for k, v in kwargs.items() if k != "verbose"}}
    python_module = bool(kwargs.get("is_python_module", True))
    hit = prebuilt.find("ext", name, key)
    if hit is not None:
        module = hit.entry.get("module", name)
        ext = _import_ext(hit.path, module, python_module)
        _resolved("ext", name, key, hit.path, "prebuilt", material, module=module, python_module=python_module, recipe=recipe)
        _LOADED[key] = ext
        return ext
    build = settings.engine_jit_root() / "torch_extensions" / f"{name}-{key[:16]}"
    marker = build / "artifact.json"

    def cached() -> Any:
        try:
            info = json.loads(marker.read_text())
        except (OSError, ValueError):
            return None
        so = build / info["file"]
        if not so.is_file():
            return None
        ext = _import_ext(so, info["module"], python_module)
        _resolved("ext", name, key, so, "cache", material, module=info["module"], python_module=python_module, recipe=recipe)
        return ext

    ext = cached()
    if ext is None:
        lock = build / "lock"
        if lock.exists() and not clear_stale_lock(lock):
            wait_for_lock(lock, name=name)
            ext = cached()
    if ext is None:
        if str(CUTLASS_STANDIN) in [os.path.abspath(p) for p in kwargs.get("extra_include_paths") or []]:
            raise CutlassMissing(f"{name} compiles CUTLASS 4.2 / CuTe headers and is not in the prebuilt kernels: set "
                                 "CUTLASS_PATH to a CUTLASS 4.2 checkout to build it")
        _before_build("ext", name, key)
        ensure_cuda_home()
        build.mkdir(parents=True, exist_ok=True)
        cg.materialize(graph.paths)
        call_kwargs = {k: list(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        call_kwargs["build_directory"] = str(build)
        for f in ("extra_cflags", "extra_cuda_cflags"):
            if call_kwargs.get(f):
                call_kwargs[f] = expand_flags(call_kwargs[f])
        forced = os.environ.get(HOST_CXX_ENV)
        cu = call_kwargs.get("extra_cuda_cflags") or []
        if forced and not any(str(f) == "-ccbin" or str(f).startswith(("-ccbin=", "--compiler-bindir")) for f in cu):
            call_kwargs["extra_cuda_cflags"] = [*cu, "-ccbin", forced]  # a pinned release toolchain covers every .cu
        if inline:
            from torch.utils.cpp_extension import load_inline

            ext = _guarded_load(name, kind_label, str(build), lambda: load_inline(name=name, **call_kwargs))
        else:
            from torch.utils.cpp_extension import load

            ext = _guarded_load(name, kind_label, str(build),
                                lambda: load(name=name, sources=sources, **call_kwargs), sources=sources)
        if python_module:
            so, module = Path(ext.__file__), ext.__name__
        else:
            so = Path(str(ext))
            module = so.name[: -len(".so")] if so.name.endswith(".so") else so.stem
        _write_json(marker, {"name": name, "key": key, "file": so.name, "module": module, "material": material})
        _resolved("ext", name, key, so, "built", material, module=module, python_module=python_module, recipe=recipe)
    _LOADED[key] = ext
    return ext


def load_extension(name: str, sources: list[str], **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load`, resolved through the prebuilt bundle and a content-keyed cache.

    The extension's key (its code graph, flags and versions) picks a shipped binary when an installed kernel bundle has
    it, else ``engine_jit_root/torch_extensions/<name>-<key>/`` when this machine built it before; only a miss runs
    torch's build, into that directory. An edit that does not reach the extension's sources never rebuilds it, and an
    edit that does always does (a new directory). The build still cannot hang on a leftover lock (reclaimed when too
    old to be real, else waited on, saying so) and says when it compiles. ``build_directory`` is ignored.
    """
    return _load("torch extension", name, kwargs, sources=[str(s) for s in sources])


def load_inline_extension(name: str, **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load_inline`, resolved like :func:`load_extension` (the inline text is the graph's root)."""
    return _load("inline torch extension", name, kwargs, inline=True)
