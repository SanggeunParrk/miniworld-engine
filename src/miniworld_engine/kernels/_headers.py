"""Third-party C++ headers a JIT build compiles against: CUTLASS 4.2 / CuTe and MathDx (cuBLASDx), found without a checkout.

Only a BUILD needs them -- an artifact found prebuilt or cached loads without -- so nothing here runs, and nothing is
downloaded, unless a kernel has to be compiled on this machine. Where a library's headers are taken from, first hit wins:

1. the environment variable(s) (``CUTLASS_PATH`` = a CUTLASS checkout; ``MINIWORLD_MATHDX_HOME`` / ``MATHDX_HOME`` /
   ``NVIDIA_MATHDX_HOME`` = a mathdx root) -- resolved by ``_nvcc``, unchanged;
2. the pinned PyPI distribution, when it is installed in this environment at exactly the pinned version
   (``pip install --no-deps nvidia-cutlass==4.2.0.0`` / ``nvidia-mathdx==25.6.0``);
3. the header cache, ``$MINIWORLD_ENGINE_HEADER_CACHE`` (else ``$XDG_CACHE_HOME`` or ``~/.cache``, ``/miniworld_engine_headers``),
   filled on the first build that needs it by downloading that same wheel from PyPI, checking its SHA-256 and unpacking only
   the header tree (``python -m miniworld_engine.kernels._headers fetch`` does it ahead of time, e.g. on a node with network
   access, or ``--wheel FILE`` from a wheel downloaded elsewhere; ``MINIWORLD_ENGINE_NO_DOWNLOAD=1`` forbids the download).

Why the wheels and not a dependency: ``nvidia-cutlass`` 4.2.0.0 ships the complete ``include/`` tree of the CUTLASS v4.2.0 release
(byte-identical to the v4.2.0 git tag the prebuilt artifacts were built with), but it also depends on ``cuda-python>=11.8``, whose
current release pins ``cuda-bindings~=13.5`` and pulls the CUDA 13 runtime wheels -- an install-time change to the user's CUDA
stack for headers that most installs (prebuilt H100 kernels) never read. The header cache sits outside the JIT cache on purpose:
an include directory there is keyed by its library version (``cutlass@4.2.0`` / ``mathdx``, ``_codegraph.classify_dir``), exactly
like a checkout, so the artifact keys do not depend on where the headers came from.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class HeaderPackage:
    """One pinned header distribution: a PyPI wheel and the part of it that is the header tree."""

    name: str  # cache directory stem and CLI name
    dist: str  # PyPI distribution
    version: str  # exact distribution version
    url: str
    sha256: str
    member: str  # prefix inside the wheel that becomes the root
    sentinel: str  # relative to the root: present in a complete tree
    module: str  # top-level module of the installed distribution
    module_sub: str  # the root, relative to that module's directory

    def cache_dir(self) -> Path:
        return cache_root() / f"{self.name}-{self.version}"


#: CUTLASS v4.2.0 (``_codegraph.CUTLASS_PINNED``); the root is the ``include`` directory itself.
CUTLASS = HeaderPackage(
    name="cutlass",
    dist="nvidia-cutlass",
    version="4.2.0.0",
    url="https://files.pythonhosted.org/packages/21/6b/e86fc2fd536dd4b8bd2209aa31d7002610c501b0802337787eac9c9b328c/"
        "nvidia_cutlass-4.2.0.0-py3-none-any.whl",
    sha256="f9599ada45c7bcb6bf53f7d3f0a7d154183e53451c01bfa59eecd37dd4a693f6",
    member="cutlass_library/source/include/",
    sentinel="cute/tensor.hpp",
    module="cutlass_library",
    module_sub="source/include",
)

#: MathDx 25.6.0 (cuBLASDx + the CUTLASS 3.9.2 it vendors); the root is the mathdx root (``include/``, ``external/``).
MATHDX = HeaderPackage(
    name="mathdx",
    dist="nvidia-mathdx",
    version="25.6.0",
    url="https://files.pythonhosted.org/packages/20/1a/a418b8c1adc58abd87fd69414c19883af5c1b10514e3dbfcc27cde831b13/"
        "nvidia_mathdx-25.6.0-py3-none-any.whl",
    sha256="22e6ad5d0d005f836be5cbd14e836cf2e9ea42c82deb602707246ce8198eaa96",
    member="nvidia/mathdx/",
    sentinel="include/cublasdx.hpp",
    module="nvidia",
    module_sub="mathdx",
)

PACKAGES = {p.name: p for p in (CUTLASS, MATHDX)}

CACHE_ENV = "MINIWORLD_ENGINE_HEADER_CACHE"
NO_DOWNLOAD_ENV = "MINIWORLD_ENGINE_NO_DOWNLOAD"
_COMPLETE = ".complete"


class HeadersUnavailable(RuntimeError):
    """The pinned headers are neither installed, nor cached, nor downloadable here."""


def cache_root() -> Path:
    forced = os.environ.get(CACHE_ENV)
    if forced:
        return Path(forced).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "miniworld_engine_headers"


def _installed(pkg: HeaderPackage) -> Path | None:
    """The root inside an installed copy of ``pkg.dist`` at exactly ``pkg.version`` (found without importing it)."""
    try:
        if importlib.metadata.version(pkg.dist) != pkg.version:
            return None
        spec = importlib.util.find_spec(pkg.module)
    except (importlib.metadata.PackageNotFoundError, ImportError, ValueError):
        return None
    for base in (spec.submodule_search_locations or []) if spec else []:
        root = Path(base) / pkg.module_sub
        if (root / pkg.sentinel).is_file():
            return root
    return None


def _cached(pkg: HeaderPackage) -> Path | None:
    root = pkg.cache_dir()
    return root if (root / _COMPLETE).is_file() and (root / pkg.sentinel).is_file() else None


def local(pkg: HeaderPackage) -> Path | None:
    """The pinned headers when present on this machine (installed distribution, then the cache); never downloads."""
    return _installed(pkg) or _cached(pkg)


def _download(pkg: HeaderPackage, dest: Path) -> None:
    if os.environ.get(NO_DOWNLOAD_ENV, "") not in ("", "0"):
        raise HeadersUnavailable(f"{NO_DOWNLOAD_ENV} is set")
    print(f"[miniworld-engine build] downloading {pkg.dist}=={pkg.version} headers ({pkg.url})", file=sys.stderr, flush=True)
    try:
        with urllib.request.urlopen(pkg.url, timeout=60) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)
    except OSError as e:
        raise HeadersUnavailable(f"download of {pkg.url} failed: {e}") from e


def _unpack(pkg: HeaderPackage, wheel: Path, out: Path) -> None:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != pkg.sha256:
        raise HeadersUnavailable(f"{wheel} is not {pkg.dist}=={pkg.version}: sha256 {digest}, expected {pkg.sha256}")
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(pkg.member):
                continue
            rel = Path(name[len(pkg.member):])
            if rel.is_absolute() or ".." in rel.parts:
                raise HeadersUnavailable(f"unsafe path in {wheel}: {name}")
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    if not (out / pkg.sentinel).is_file():
        raise HeadersUnavailable(f"{wheel} holds no {pkg.member}{pkg.sentinel}")


def fetch(pkg: HeaderPackage, wheel: str | os.PathLike | None = None) -> Path:
    """The pinned headers, downloading (or unpacking ``wheel``) into the cache when they are not present. Concurrent callers
    wait on one lock; the tree is unpacked into a private directory and renamed into place only when complete."""
    import fcntl

    if wheel is None and (hit := local(pkg)) is not None:
        return hit
    final = pkg.cache_dir()
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.with_name(f".{final.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if wheel is None and (hit := _cached(pkg)) is not None:
            return hit  # another process fetched it while this one waited
        tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            src = Path(wheel) if wheel is not None else tmp / "download.whl"
            if wheel is None:
                _download(pkg, src)
            _unpack(pkg, src, tmp / "tree")
            (tmp / "tree" / _COMPLETE).write_text(f"{pkg.dist}=={pkg.version} sha256:{pkg.sha256}\n")
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp / "tree", final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m miniworld_engine.kernels._headers",
                                 description="Show or fetch the pinned CUTLASS / MathDx headers JIT builds compile against.")
    ap.add_argument("action", choices=("show", "fetch"))
    ap.add_argument("packages", nargs="*", metavar="PACKAGE", help=f"any of {', '.join(PACKAGES)} (default: all)")
    ap.add_argument("--wheel", default=None, help="unpack this already-downloaded wheel instead of downloading (one package)")
    args = ap.parse_args(argv)
    names = args.packages or list(PACKAGES)
    if unknown := [n for n in names if n not in PACKAGES]:
        ap.error(f"unknown package(s): {', '.join(unknown)}")
    if args.wheel and len(names) != 1:
        ap.error("--wheel needs exactly one package")
    rc = 0
    for name in names:
        pkg = PACKAGES[name]
        if args.action == "show":
            print(f"{name}: {local(pkg) or 'absent'} (cache {pkg.cache_dir()})")
            continue
        try:
            print(f"{name}: {fetch(pkg, args.wheel)}")
        except HeadersUnavailable as e:
            print(f"{name}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
