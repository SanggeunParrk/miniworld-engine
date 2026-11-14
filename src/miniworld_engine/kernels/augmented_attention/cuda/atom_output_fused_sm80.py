"""Separate own-CUDA atom attention output/gate fusion candidate."""

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
        name="atom_output_fused_sm80",
        sources=[str(directory / "atom_output_fused.cu")],
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


def result(og, cond, wo, ws, bias, residual=None, save=False, cfg=1):
    if residual is None:
        residual = og.new_empty((0,))
    return _ext().result(og, cond, wo, ws, bias, residual, save, cfg)
