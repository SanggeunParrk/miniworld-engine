"""L128 two native gradient GEMMs; the forward outer product still uses cuBLAS."""

import functools
from pathlib import Path
import torch

from miniworld_engine import settings

from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="opm_pipeline_native128_sm80",
        sources=[str(directory / "pipeline_native128.cu")],
        extra_include_paths=[str(directory)],
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
        extra_ldflags=[
            "-lcublasLt"
        ],  # the searched explicit GEMM algorithms (opm_lt_gemm.cuh)
        verbose=False,
    )


def forward(m, mask, residual, leaves, eps, norm_first, training, save_o):
    if mask is None:
        mask = m.new_empty((0,), dtype=torch.uint8)
    if residual is None:
        residual = m.new_empty((0,))
    return _ext().forward(
        m,
        mask,
        residual,
        leaves,
        eps,
        norm_first,
        training,
        save_o,
        settings.current().opm_sm80_pl,
        settings.current().opm_sm80_epi,
    )


def backward(m, mask, leaves, saved, dz, norm_first):
    if mask is None:
        mask = m.new_empty((0,), dtype=torch.uint8)
    a, b, stats, o, norm = saved
    return _ext().backward(
        m,
        mask,
        leaves,
        a,
        b,
        stats,
        o,
        norm,
        dz,
        norm_first,
        settings.current().opm_sm80_dg,
        settings.current().opm_sm80_pb,
    )
