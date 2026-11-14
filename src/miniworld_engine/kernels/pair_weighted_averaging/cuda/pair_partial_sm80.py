"""E110 own-CUDA short DW partials reduced inside pair backward; opt-in only."""

import functools
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name=_card.name("pwa_pair_partial_sm80"),
        sources=[str(d / "pair_partial.cu")],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ]),
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def contract(a, b):
    return _ext().contract(a, b)


def pair(z, w, dw, mask, lnw, lnb, wb, eps):
    return _ext().pair(z, w, dw, mask, lnw, lnb, wb, eps)
