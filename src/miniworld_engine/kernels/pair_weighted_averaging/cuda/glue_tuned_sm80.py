"""Separate own-CUDA PWA glue occupancy candidate; original extension/default untouched."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card
from . import sm80


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name=_card.name("pwa_glue_tuned_sm80"),
        sources=[str(directory / "glue_tuned_ops.cu")],
        extra_include_paths=[str(directory)],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ]),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def glue(m, dres, o, lnw, lnb, wg, wo, keep, eps, dscale, raw=False, mode=0):
    depth, length, _ = m.shape
    nbx = sm80.ext().bwd_grids(depth, length, 64, 32)[0]
    do = torch.empty_like(o)
    dgp = torch.empty_like(o)
    part = m.new_empty((nbx, 4, 8192), dtype=torch.float32)
    _ext().glue(
        m,
        dres,
        o,
        lnw,
        lnb,
        wg,
        wo,
        m.new_empty(0) if keep is None else keep,
        do,
        dgp,
        part,
        eps,
        dscale,
        mode,
    )
    total = sm80.reduce_rows(part.view(nbx, -1), torch.float32).view(4, 8192)
    if raw:
        return do, dgp, total
    dwo = total[:, :4096].reshape(4, 64, 64).permute(1, 0, 2).reshape(64, 256)
    return do, dgp, dwo, total[:, 4096:].reshape(256, 64)
