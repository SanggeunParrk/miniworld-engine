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


def gencodes(*arches: str, ptx: tuple[str, ...] = ()) -> list[str]:
    """``-gencode`` flags for the requested arches, dropping any this nvcc does not know.

    ``arches`` are like ``"90"`` or ``"90a"``; ``ptx`` names arches to also emit PTX for, so a
    newer card can JIT from it.
    """
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


def load_extension(name: str, sources: list[str], **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load`, but it cannot hang on a leftover lock, and says when it compiles.

    Reclaims a lock too old to be real, then waits on a fresh one (saying so). Correctness still
    comes from torch's own baton: this only decides how long to tolerate it.
    """
    from torch.utils.cpp_extension import load

    return _guarded_load(name, "torch extension", kwargs.get("build_directory"),
                         lambda: load(name=name, sources=sources, **kwargs), sources=sources)


def load_inline_extension(name: str, **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load_inline` behind the same guard and log as :func:`load_extension`."""
    from torch.utils.cpp_extension import load_inline

    return _guarded_load(name, "inline torch extension", kwargs.get("build_directory"),
                         lambda: load_inline(name=name, **kwargs))
