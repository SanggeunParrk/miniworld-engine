"""Own CUDA TriAttn output projection + dropout + residual candidate."""

import functools
from pathlib import Path
from . import sm80_arch as _arch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name=_arch.name("triattn_native_out_fused_sm80"),
        sources=[str(d / "native_out_fused.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            *_arch.gencode(),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def run(a, w, res, ds, out, length, transposed=False, cfg=0, grid=0):
    _ext().run(a, w, out, res, ds, length, transposed, 0, cfg, grid)
