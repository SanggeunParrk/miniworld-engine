"""Accurate BF affine-LN front, training-only exact D64 geometry.

Original E263 CUDA and E314 direct pack/run body, fresh weights every call.
Eligibility is evaluated once before the selected whole forward opaque.
"""

import functools
import pathlib
from pathlib import Path
import torch
from . import sm80_arch as _arch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from . import sm80_wide as wide

P = Path(__file__).parent


def eligible(x, weights, gamma, beta, d, dh, hd, length):
    if (
        (d, dh, hd) != (64, 64, 32)
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or (not x.is_contiguous())
        or (length < 128)
        or length % 128
    ):
        return False
    if (
        gamma.dtype != torch.float32
        or beta.dtype != torch.float32
        or tuple(gamma.shape) != (64,)
        or (tuple(beta.shape) != (64,))
        or (not gamma.is_contiguous())
        or (not beta.is_contiguous())
    ):
        return False
    if len(weights) != 5:
        return False
    return (
        all(
            (
                w.dtype == torch.bfloat16
                and w.is_contiguous()
                and (tuple(w.shape) == ((2, 64) if i == 4 else (64, 64)))
                and (w.device == x.device)
                for i, w in enumerate(weights)
            )
        )
        and gamma.device == x.device
        and (beta.device == x.device)
    )


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name=_arch.name("triattn_affine_training_a100_sm80"),
        sources=[str(P / "affine263.cu")],
        extra_include_paths=[str(pathlib.Path(wide.__file__).parent / "sm80")],
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


def front(
    x,
    weights,
    gamma,
    beta,
    mask,
    y,
    bias,
    stats,
    xh,
    length,
    transposed,
    eps,
    d,
    dh,
    hd,
):
    packed = weights[0].new_empty((264, 64))
    zeros = gamma.new_empty((264,))
    extension = _ext()
    extension.pack(weights, packed, zeros)
    extension.run(
        x,
        packed,
        zeros,
        gamma,
        beta,
        mask,
        y,
        bias,
        stats,
        xh,
        length,
        transposed,
        eps,
        False,
    )
    return y
