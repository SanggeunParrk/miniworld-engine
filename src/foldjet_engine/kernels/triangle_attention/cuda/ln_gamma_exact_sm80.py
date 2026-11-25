"""Own CUDA exact normalized-input LN gamma-gradient candidate."""

import functools, torch
from pathlib import Path
from . import sm80_arch as _arch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name=_arch.name("triattn_ln_gamma_exact_sm80"),
        sources=[str(d / "ln_gamma_exact.cu")],
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


def run(dd, db, x, stats, wt, wbt, gamma, partial, out, length, transposed=False):
    _ext().run(dd, db, x, stats, wt, wbt, gamma, partial, out, length, transposed)


def gradient(dd, db, x, stats, wt, wbt, gamma, like, transposed=False):
    partial = torch.empty(
        (torch.cuda.get_device_properties(x.device).multi_processor_count * 8, 128),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_like(like)
    run(dd, db, x, stats, wt, wbt, gamma, partial, out, x.shape[1], transposed)
    return out
