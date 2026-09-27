"""CTA-wide CUDA normalization candidate (width <=16384)."""

from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(None)
def extension():
    from miniworld_engine.kernels._nvcc import (
        ensure_cuda_home,
        host_flags,
        load_extension,
    )

    ensure_cuda_home()
    source = Path(__file__).with_name("wide.cu")
    return load_extension(
        name="mw_norm_wide_" + hashlib.sha256(source.read_bytes()).hexdigest()[:12],
        sources=[str(source)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-lineinfo"],
    )
