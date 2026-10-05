"""Loader of the sm_80 MPNN edge extension (``sm80/ops.cu``: weight packing, the edge-tail chain kernels, the edge-MLP, LayerNorm and dropout kernels).

Built on first use, never at import: ``ext()`` raises when the toolchain cannot build it (the callers in ``integrations/mpnn_edge_sm80.py`` turn that into a one-time warning
and keep the Triton path).  The sources are staged into a directory named by their content hash under the JIT root and built there under a hash-named extension, so a checkout, a
copy of it or a snapshot with the same kernel sources shares one build, and a changed source can never meet a stale object or a leftover lock of an older one.
``MINIWORLD_MPNN_EDGE_PTXAS=1`` prints the ptxas resource report of the build.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

_dir = Path(__file__).parent / "sm80"


def _extra_flags() -> list[str]:
    """``MINIWORLD_MPNN_EDGE_FLAGS``: extra nvcc flags (``-DNAME=value`` knobs of the kernels) for A/B experiments; they are part of the build's name."""
    return os.environ.get("MINIWORLD_MPNN_EDGE_FLAGS", "").split()


def _staged() -> tuple[Path, str]:
    """(directory holding a private copy of the kernel sources, their content hash)."""
    files = sorted(p for p in _dir.iterdir() if p.suffix in {".cu", ".cuh"})
    digest = hashlib.sha1()
    digest.update(" ".join(_extra_flags()).encode())
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    tag = digest.hexdigest()[:12]
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT") or Path.home() / ".cache" / "miniworld-engine-jit")
    dst = root / f"mpnn_edge_sm80_src_{tag}"
    if not dst.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=dst.name + ".", dir=root))
        for path in files:
            shutil.copy2(path, tmp / path.name)
        try:
            tmp.rename(dst)
        except OSError:                      # another process staged the same content first
            shutil.rmtree(tmp, ignore_errors=True)
    return dst, tag


@functools.lru_cache(maxsize=1)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    src, tag = _staged()
    flags = [*host_flags(), "-O3", "-std=c++17", "-lineinfo", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
             "-U__CUDA_NO_BFLOAT162_OPERATORS__", *gencodes("80"), *_extra_flags()]
    verbose = os.environ.get("MINIWORLD_MPNN_EDGE_PTXAS", "0") == "1"
    if verbose:
        flags += ["-Xptxas", "-v"]
    return load_extension(name=f"mpnn_edge_sm80_{tag}", sources=[str(src / "ops.cu")], extra_include_paths=[str(src)], extra_cuda_cflags=flags,
                          extra_cflags=["-std=c++17", "-O3"], verbose=verbose)
