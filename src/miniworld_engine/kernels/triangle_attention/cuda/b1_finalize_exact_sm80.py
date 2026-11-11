"""Own CUDA B1 scale derivative reduced inside the weight-gradient finalizer candidate."""

import functools, torch
from pathlib import Path
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    d = Path(__file__).parent / "sm80"
    return load_extension(
        name="triattn_b1_finalize_exact_sm80",
        sources=[str(d / "b1_finalize_exact.cu")],
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


def finish(g, weights, grads, gamma, beta, partial, dgamma, dbeta):
    _ext().finish(g, list(weights), list(grads), gamma, beta, partial, dgamma, dbeta)


def gradients(dd, db, x, stats, dout, wt, wbt, gamma32, xh, weights, gamma, beta, transposed=False):
    partial = torch.empty(
        (torch.cuda.get_device_properties(x.device).multi_processor_count * 8, 128),
        device=x.device,
        dtype=torch.float32,
    )
    dpair = torch.empty_like(x)
    dgamma = torch.empty_like(gamma)
    dbeta = torch.empty_like(beta)
    run(
        dd,
        db,
        x,
        stats,
        dout,
        dpair,
        wt,
        wbt,
        gamma32,
        partial,
        dgamma,
        x.shape[1],
        transposed,
    )
    t = x.numel() // 128
    xh2 = xh.view(t, 136)
    from .sm80 import _front_weight_products

    g = _front_weight_products(dd, db, xh2)
    grads = [torch.empty_like(w) for w in weights]
    finish(g, weights, grads, gamma, beta, partial, dgamma, dbeta)
    return dpair, grads, dgamma, dbeta
