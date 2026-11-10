"""Natural-bias Token DiT CUDA gradient staging candidate, not a cuBLAS replacement."""

import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    directory = Path(__file__).parent / "sm80"
    return load_extension(
        name="token_grad_sm80",
        sources=[str(directory / "token_grad_ops.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def backward(q, k, v, dob, bias, lse, delta, packed, pbq, samples, length, heads):
    rows = 4 if samples % 4 == 0 else 2 if samples % 2 == 0 else 1
    db = torch.empty_like(bias)
    partial = bias.new_empty((samples // rows, heads, length, length))
    bias_t = torch.empty_like(bias)
    _ext().backward(
        q,
        k,
        v,
        dob,
        bias,
        lse,
        delta,
        packed,
        db,
        partial,
        bias_t,
        pbq,
        samples,
        length,
        heads,
        rows,
    )
    return db
