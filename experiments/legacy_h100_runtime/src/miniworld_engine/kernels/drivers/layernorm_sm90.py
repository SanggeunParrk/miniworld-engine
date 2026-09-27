"""Native TMA build workloads: both layouts and actual requested widths/dtypes."""

import torch
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.layernorm_linear import _D, _M


def inputs(m=_M, n=_D, layout="col"):
    # Pad only the physical leading stride to TMA's 16-byte alignment; logical
    # tails remain present. No copied activation is used by the kernel.
    align = 16 // torch.empty((), dtype=BF16).element_size()
    leading = m if layout == "col" else n
    pitch = (leading + align - 1) // align * align
    strides = (1, pitch) if layout == "col" else (pitch, 1)
    x = torch.empty_strided((m, n), strides, device=dev(), dtype=BF16).normal_()
    dy = torch.empty_strided((m, n), strides, device=dev(), dtype=BF16).normal_()
    w = torch.randn(n, device=dev(), dtype=torch.float32)
    mean = x.float().mean(1)
    rs = (x.float().var(1, unbiased=False) + 1e-5).rsqrt()
    return dy, x, w, mean, rs


def layernorm_bwd_split_sm90_cute():
    from miniworld_engine.kernels.layernorm.cute.tma_backward import backward_impl

    for layout in ("col", "row"):
        dy, x, w, mean, rs = inputs(layout=layout)
        backward_impl(dy, x, w, mean, rs, x.stride())
