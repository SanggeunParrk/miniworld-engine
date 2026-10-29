"""sm_100a (B200) build and launch of the backbone pair-embedding kernels (``pair_embed_geom_sm100.cu``): a torch extension built on first use,
launched on torch's current stream (CUDA-graph capturable). The forward / LayerNorm-backward kernels are the sm_90a ones built for sm_100a
(CUDA-core gathers); the table gradient is the tcgen05 kernel of ``pe_tbl_sm100.cuh``. Same calls as ``cuda.sm90``."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent

NBLK = 74          # pair slices of the tcgen05 table gradient (x 2 channel halves = 148 CTAs); pe100::NBLK
NBLK_SM90 = 33     # slices of the A/B kernels (impl 0 / 2)
N_ROWS = {0: 317, 1: 157}


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="pair_embed_geom_sm100", sources=[str(_dir / "pair_embed_geom_sm100.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("100a")], extra_cflags=["-std=c++17"],
                          verbose=False)


def forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=False, ln_eps=None):
    """pair [B, n, n, 256] bf16 (and the binned pair codes [B, n, n, 4] int32 when ``with_codes``); ``ln_eps``: LayerNorm(pair), no affine."""
    B, n = idx.shape
    out = torch.empty(B, n, n, 256, device=tbl.device, dtype=torch.bfloat16)
    codes = torch.empty(B, n, n, 4, device=tbl.device, dtype=torch.int32) if with_codes else torch.empty(0, device=tbl.device, dtype=torch.int32)
    _ext().forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, out, codes, int(ln_eps is not None), float(ln_eps or 0.0))
    return (out, codes) if with_codes else out


def forward_ca(tbl, ca, mask, idx, dlim, with_codes=False, ln_eps=None):
    """Decoder pair [B, n, n, 256] bf16 from the table [157, 256] (sequence separation | CA distance bins)."""
    B, n = idx.shape
    out = torch.empty(B, n, n, 256, device=tbl.device, dtype=torch.bfloat16)
    codes = torch.empty(B, n, n, 4, device=tbl.device, dtype=torch.int32) if with_codes else torch.empty(0, device=tbl.device, dtype=torch.int32)
    _ext().forward_ca(tbl, ca, mask, idx, dlim, out, codes, int(ln_eps is not None), float(ln_eps or 0.0))
    return (out, codes) if with_codes else out


IMPL = int(os.environ.get("MINIWORLD_PAIR_EMBED_BWD_IMPL", "1"))   # 1: tcgen05, 2: the sm_90 mma.sync kernel, 0: the shared-memory kernel (A/B)


def backward(g, codes, mode, impl=None):
    """Table gradient [N_ROWS[mode], 256] bf16 from d pair = g [B, n, n, 256] bf16 (deterministic, fixed-order partial sums)."""
    impl = IMPL if impl is None else impl
    nt = N_ROWS[mode]
    tpart = torch.empty((NBLK if impl == 1 else NBLK_SM90) * nt * 256, device=g.device, dtype=torch.float32)
    dtbl = torch.empty(nt, 256, device=g.device, dtype=torch.bfloat16)
    _ext().backward(g, codes, tpart, dtbl, int(mode), int(impl))
    return dtbl


def ln_backward(tbl, codes, dh, mode, eps):
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) = ``dh`` (the pair is recomputed from the codes and the table; zeros for padding)."""
    dx = torch.empty_like(dh)
    _ext().ln_backward(tbl, codes, dh, dx, int(mode), float(eps))
    return dx
