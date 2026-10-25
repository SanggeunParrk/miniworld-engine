"""sm_90a (H100) build and launch of the pair-embedding + adaptive-LayerNorm kernels (``pair_embed_adaln.cu``): a torch extension
built on first use, launched on torch's current stream (CUDA-graph capturable). Plain CUDA-core kernels -- the op is bandwidth
bound. The backward is bitwise deterministic (no atomics: per-CTA partials summed in fixed order)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

_dir = Path(__file__).parent
C = 256
NBLK = 33
DX_CHUNK_BYTES = 1 << 29     # the backward's dx scratch (bf16 grad of the table sum) is built for at most this many bytes of pairs at a time


@functools.lru_cache(maxsize=None)
def _ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(name="pair_embed_adaln_sm90", sources=[str(_dir / "pair_embed_adaln.cu")],
                          extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")], extra_cflags=["-std=c++17"],
                          verbose=False)


def forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2):
    """(pair [B, n, n, 256] bf16, bins [B, n, n] int32)."""
    B, n = idx.shape
    out = torch.empty(B, n, n, C, device=tbl.device, dtype=torch.bfloat16)
    bins = torch.empty(B, n, n, device=tbl.device, dtype=torch.int32)
    _ext().forward(tbl, lnw, lnb, gamma, beta, xt, sc, int(has_sc), idx, mask, xlim, clim, out, bins, int(nS), int(nX), float(eps1), float(eps2))
    return out, bins


IMPL = int(os.environ.get("MINIWORLD_PAIR_EMBED_BWD_IMPL", "1"))   # 1: table gradient on the tensor cores (wgmma), 0: the shared-memory read-modify-write kernel (A/B)


def backward(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2, impl=None):
    """(dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]) from d pair = dout [B, n, n, 256] bf16."""
    B, n = bins.shape[:2]
    NT = tbl.shape[0]
    dev = tbl.device
    chunk_b = max(1, min(B, DX_CHUNK_BYTES // (n * n * C * 2)))
    nchunk = -(-B // chunk_b)
    nrb = -(-n // 8)
    dx = torch.empty(chunk_b, n, n, C, device=dev, dtype=torch.bfloat16)
    part = torch.empty(B * nrb * 4 * C, device=dev, dtype=torch.float32)
    tpart = torch.empty(nchunk * NBLK * NT * C, device=dev, dtype=torch.float32)
    dtbl = torch.empty(NT, C, device=dev, dtype=torch.float32)
    dgamma = torch.empty(B, C, device=dev, dtype=torch.float32)
    dbeta = torch.empty(B, C, device=dev, dtype=torch.float32)
    dlw = torch.empty(C, device=dev, dtype=torch.float32)
    dlb = torch.empty(C, device=dev, dtype=torch.float32)
    _ext().backward(tbl, lnw, lnb, gamma, bins, dout, dx, part, tpart, dtbl, dgamma, dbeta, dlw, dlb, int(IMPL if impl is None else impl), int(chunk_b), int(nS), int(nX),
                    float(eps1), float(eps2))
    return dtbl, dlw, dlb, dgamma, dbeta
