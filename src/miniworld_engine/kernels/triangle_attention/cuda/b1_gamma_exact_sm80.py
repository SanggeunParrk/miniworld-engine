"""Own CUDA front input backward + exact FP32 LN scale derivative candidate."""

import functools, torch
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="triattn_b1_gamma_exact_sm80",
        sources=[str(d / "b1_gamma_exact.cu")],
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


def run(
    dd,
    db,
    x,
    stats,
    dout,
    dpair,
    wt,
    wbt,
    gamma,
    partial,
    out,
    length,
    transposed=False,
):
    _ext().run(dd, db, x, stats, dout, dpair, wt, wbt, gamma, partial, out, length, transposed)


def gradients(dd, db, x, stats, dout, wt, wbt, gamma, like, transposed=False):
    partial = torch.empty(
        (torch.cuda.get_device_properties(x.device).multi_processor_count * 8, 128),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_like(like)
    dpair = torch.empty_like(x)
    run(
        dd,
        db,
        x,
        stats,
        dout,
        dpair,
        wt,
        wbt,
        gamma,
        partial,
        out,
        x.shape[1],
        transposed,
    )
    return dpair, out
