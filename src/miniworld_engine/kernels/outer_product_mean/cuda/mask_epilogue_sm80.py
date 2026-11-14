"""Own-CUDA packed mask count and projection epilogue candidate."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name=_card.name("opm_mask_epilogue_sm80"),
        sources=[str(directory / "mask_epilogue.cu")],
        extra_include_paths=[str(directory)],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ]),
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def result(o, mask, wo, bias, residual, norm_first, save_norm=False, cfg=None, dmin=1.0, doff=0.0):
    """The pair divisor is ``max(n, dmin) + doff``; the saved counts (``save_norm``) are ``max(n, dmin)`` (the dgrad adds ``doff``)."""
    if cfg is None:
        cfg = 2 if mask.shape[0] <= 1024 or mask.shape[1] == 768 else 1
    if residual is None:
        residual = o.new_empty((0,))
    return _ext().result(o, mask, wo, bias, residual, norm_first, save_norm, cfg, float(dmin), float(doff))
