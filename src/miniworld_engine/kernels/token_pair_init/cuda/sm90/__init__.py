"""sm_90a (H100) build and launch of the token-pair initialisation kernels (``token_pair_init.cu``): a torch extension built on
first use, launched on torch's current stream (CUDA-graph capturable). Plain CUDA-core kernels -- the op is bandwidth bound.
The backward is bitwise deterministic (no atomics: per-row-block partials summed in fixed order)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels.token_pair_init.cuda.sm100 import P, check_packable, n_rel

_dir = Path(__file__).parent
RB = 8


def fwd_rows(L: int) -> int:
    """Rows of z per forward CTA (the kernel loads the class-row table once per CTA and right[j] once per R rows): 4 up to L 256, else 2 (measured L128-L384, B16 / B48)."""
    return 4 if L <= 256 else 2


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="token_pair_init_sm90", sources=[str(_dir / "token_pair_init.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")], extra_cflags=["-std=c++17"],
                          verbose=False)


def forward(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32 from left / right [B, L, 128], the table [nbin, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    B, L, _ = left.shape
    check_packable(r_max, s_max)
    out = torch.empty(B, L, L, P, device=left.device, dtype=torch.float32)
    _ext().forward(left, right, tbl, ids, bond, out, int(r_max), int(s_max), fwd_rows(L))
    return out


def backward(g, ids, bond, r_max: int, s_max: int):
    """(dleft [B, L, 128], dright [B, L, 128], dtbl [nbin, 128]) of :func:`forward` from dz = g [B, L, L, 128] fp32."""
    B, L = g.shape[:2]
    check_packable(r_max, s_max)
    nbin = n_rel(r_max, s_max) + 2
    nrb = (L + RB - 1) // RB
    dev = g.device
    dleft = torch.empty(B, L, P, device=dev, dtype=torch.float32)
    dright = torch.empty(B, L, P, device=dev, dtype=torch.float32)
    dtbl = torch.empty(nbin, P, device=dev, dtype=torch.float32)
    dpart = torch.empty(B * nrb * L * P, device=dev, dtype=torch.float32)
    wpart = torch.empty(B * nrb * 4 * nbin * 32, device=dev, dtype=torch.float32)
    _ext().backward(g, ids, bond, dleft, dright, dtbl, dpart, wpart, int(r_max), int(s_max))
    return dleft, dright, dtbl
