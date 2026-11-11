"""sm_90a (H100) build and launch of the backbone pair-embedding kernels (``pair_embed_geom.cu``): a torch extension built on first use,
launched on torch's current stream (CUDA-graph capturable). The encoder pair (mode 0, 317 table rows) and the decoder pair (mode 1, 157 rows),
each forward-only or with the codes a backward replays, optionally with the attention-bias LayerNorm folded in, and the deterministic table gradient."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
NBLK = 33
N_ROWS = {0: 317, 1: 157}
GT = 1  # forwards: 1 reads the table through the L1 (8-warp CTAs, high occupancy; the faster), 0 keeps a copy in shared memory (A/B)
BWD_IMPL = (
    1  # table gradient on the tensor cores (wgmma); the .cu keeps 2 (mma.sync) and 0 (shared-memory read-modify-write) as its A/B variants
)


@functools.lru_cache(maxsize=None)
def ext():
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="pair_embed_geom_sm90",
        sources=[str(_dir / "pair_embed_geom.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", *gencodes("90a")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=False, ln_eps=None, jp=0, gt=GT):
    """pair [B, n, n, 256] bf16 (and, for the backward, the binned pair codes [B, n, n, 4] int32 when ``with_codes``). ``ln_eps``: the output is
    LayerNorm(pair) (no affine, that eps) instead of the pair."""
    B, n = idx.shape
    out = torch.empty(B, n, n, 256, device=tbl.device, dtype=torch.bfloat16)
    codes = (
        torch.empty(B, n, n, 4, device=tbl.device, dtype=torch.int32)
        if with_codes
        else torch.empty(0, device=tbl.device, dtype=torch.int32)
    )
    ext().forward(
        tbl, bb, ca, cb, mask, idx, chains, dlim, alim, out, codes, int(ln_eps is not None), float(ln_eps or 0.0), int(jp), int(gt)
    )
    return (out, codes) if with_codes else out


def forward_ca(tbl, ca, mask, idx, dlim, with_codes=False, ln_eps=None, jp=0, gt=GT):
    """Decoder pair [B, n, n, 256] bf16 from the table [157, 256] (sequence separation | CA distance bins)."""
    B, n = idx.shape
    out = torch.empty(B, n, n, 256, device=tbl.device, dtype=torch.bfloat16)
    codes = (
        torch.empty(B, n, n, 4, device=tbl.device, dtype=torch.int32)
        if with_codes
        else torch.empty(0, device=tbl.device, dtype=torch.int32)
    )
    ext().forward_ca(tbl, ca, mask, idx, dlim, out, codes, int(ln_eps is not None), float(ln_eps or 0.0), int(jp), int(gt))
    return (out, codes) if with_codes else out


def backward(g, codes, mode):
    """Table gradient [N_ROWS[mode], 256] bf16 from d pair = g [B, n, n, 256] bf16 (deterministic, fixed-order partial sums)."""
    nt = N_ROWS[mode]
    tpart = torch.empty(NBLK * nt * 256, device=g.device, dtype=torch.float32)
    dtbl = torch.empty(nt, 256, device=g.device, dtype=torch.bfloat16)
    ext().backward(g, codes, tpart, dtbl, int(mode), BWD_IMPL)
    return dtbl


def ln_backward(tbl, codes, dh, mode, eps):
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) = ``dh`` (the pair is recomputed from the codes and the table; zeros for padding)."""
    dx = torch.empty_like(dh)
    ext().ln_backward(tbl, codes, dh, dx, int(mode), float(eps))
    return dx
