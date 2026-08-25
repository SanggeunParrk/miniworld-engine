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

import functools
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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
    out = [f"-gencode=arch=compute_{a},code=sm_{a}"
           for a in arches if not have or f"compute_{a}" in have]
    out += [f"-gencode=arch=compute_{a},code=compute_{a}"
            for a in ptx if not have or f"compute_{a}" in have]
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
# So: reclaim a lock that is too old to be real, and turn an unbounded wait into an error that
# names the file. A message beats a hang, and the remedy is one `rm`.
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# mathdx (cuBLASDx / CommonDx) headers.
#
# Three transition extensions -- transition_b2b_cuda, transition_expand_gate_cuda,
# transition_gatebwd_cuda -- include cuBLASDx. Their include paths used to be written into the
# build as `-I<a developer home>/mathdx_dl/extracted/nvidia/mathdx/...`, six occurrences across three
# functions. That is unbuildable for any other user by construction; it also stopped being
# buildable HERE, because the directory no longer exists and nothing noticed. Nothing noticed
# because these extensions are absent from registry.csv and their build is lazy, so no test, no
# audit, and no CI job ever asks for them.
#
# Resolve instead, and when resolution fails say which variable to set rather than emitting a
# compiler error about a missing header.
# ---------------------------------------------------------------------------------------------

#: Checked in order. The project-prefixed name wins so a user can override a system install.
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



#: A lock older than this cannot belong to a live build -- the longest real build here is a few
#: minutes -- so it is a leftover and gets reclaimed. `build --reclaim` makes the same judgement
#: about the builder's own O_EXCL claims.
STALE_LOCK_SECONDS = 1800.0

#: How long to wait for a FRESH lock (another process really is building) before giving up. Long
#: enough for a genuine four-arch nvcc build, short enough that a job notices inside its time limit.
LOCK_WAIT_SECONDS = 600.0


def _build_lock(name: str) -> Path | None:
    """The lock file `torch.utils.cpp_extension.load` will use for `name`, if it can be located."""
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
    import time

    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    if age < stale_after:
        return False
    try:
        lock.unlink()
    except OSError:
        return False
    print(f"[miniworld.nvcc] reclaimed a stale build lock ({age / 60:.0f} min old): {lock}",
          file=sys.stderr)
    return True


def wait_for_lock(lock: Path, limit: float = LOCK_WAIT_SECONDS) -> None:
    """Wait for a fresh `lock` to clear, or raise naming it. Never block indefinitely."""
    import time

    if not lock.exists():
        return
    print(f"[miniworld.nvcc] another process is building this extension; waiting up to "
          f"{limit / 60:.0f} min for {lock}", file=sys.stderr)
    deadline = time.time() + limit
    while lock.exists():
        if time.time() > deadline:
            msg = (f"waited {limit / 60:.0f} min for another process's build lock and it is still "
                   f"there:\n    {lock}\n"
                   f"If no build is running, it is a leftover from a killed one -- torch waits on "
                   f"it forever, which is why this raises instead. Remove it:\n"
                   f"    rm {lock}")
            raise TimeoutError(msg)
        time.sleep(2.0)


def load_extension(name: str, sources: list[str], **kwargs: Any) -> Any:
    """`torch.utils.cpp_extension.load`, but it cannot hang on a leftover lock.

    Reclaims a lock too old to be real, then bounds the wait on a fresh one. Correctness still
    comes from torch's own baton: this only decides how long to tolerate it.
    """
    from torch.utils.cpp_extension import load

    lock = _build_lock(name)
    if lock is not None and not clear_stale_lock(lock):
        wait_for_lock(lock)
    return load(name=name, sources=sources, **kwargs)
