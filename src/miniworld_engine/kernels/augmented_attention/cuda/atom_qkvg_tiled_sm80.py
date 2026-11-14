"""Own-CUDA BF16 packed atom Q/K/V/gate projection candidate."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    adaln = Path(__file__).parents[2] / "adaln/cuda/sm80"
    return load_extension(
        name="atom_qkvg_tiled_sm80",
        sources=[str(directory / "atom_qkvg_tiled.cu")],
        extra_include_paths=[str(adaln)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def run(x, w, bias, out, cfg=0):
    _ext().run(x, w, bias, out, cfg)


def result(x, w, bias, cfg=0):
    out = x.new_empty((x.shape[0], 512))
    run(x, w, bias, out, cfg)
    return out
