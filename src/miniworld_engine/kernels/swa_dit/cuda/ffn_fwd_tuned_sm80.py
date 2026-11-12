"""Own A100 FFN forward with one-stage weights and consecutive sequence tiles."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="swa_ffn_fwd_tuned_sm80",
        sources=[str(Path(__file__).parent / "sm80" / "ffn_fwd_tuned.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def ffn_forward(
    qi: torch.Tensor,
    g: torch.Tensor,
    o: torch.Tensor,
    mod: torch.Tensor,
    wo: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    s: int,
    b: int,
    eps: float,
):
    """Return five fresh BF16 outputs, including the original backward saves."""
    return tuple(_ext().compute(qi, g, o, mod, wo, wu, wd, s, b, eps))
