"""Own CUDA D256 LN reused and streamed front fragments/bias-plane stage, unconnected experimental path."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="triattn_native_front_stream2_sm80",
        sources=[str(d / "native_front_stream2.cu")],
        extra_include_paths=[str(d)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def run(x, wp, mask, y, bias, length, transposed=False, eps=1e-5, cfg=0, grid=0):
    _ext().run(x, wp, mask, y, bias, length, transposed, eps, cfg, grid)
