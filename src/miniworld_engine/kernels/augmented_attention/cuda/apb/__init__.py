"""CUDA passes of AttentionPairBias (the Pairformer single track) on B200 (``apb_rows.cu``): the pair -> bias projection
with its LayerNorm (mma.sync), its backward, and the row passes of the single track (LayerNorm, sigmoid gate, their
backwards). Its attention cores are this family's sm_100a kernels (``../sm100``). Built on first use (``load_extension``),
never at import; ``integrations/attention_pair_bias_b200.py`` drives them."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
RW = 4                    # rows per block of the row kernels (a warp each)
DP = 128                  # d_pair
# (heads, d_single) -> (real head dim, head width in memory: 16 x 24 is padded to 32)
GEOMETRY = {(8, 384): (48, 48), (12, 384): (32, 32), (16, 384): (24, 32), (24, 384): (16, 16), (16, 512): (32, 32)}


def width(heads: int, d: int = 384) -> int:
    """Row width of q / k / v / g / O and their gradients: d_single, or 512 for 16 x 24 (padded to 32)."""
    return heads * GEOMETRY[(heads, d)][1]


def acc_n(heads: int, d: int = 384) -> int:
    """The backward's fp32 accumulator: dlnw [d] | dlnb [d] | dbq [W] | dWf [heads, 128] | per-head dbias sums [heads]."""
    return 2 * d + width(heads, d) + heads * DP + heads


@functools.lru_cache(maxsize=None)
def ext():
    from ...._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="apb_rows_cuda",
        sources=[str(_dir / "apb_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def available() -> bool:
    ext()
    return True


def pair_bias(pair2d: torch.Tensor, wf: torch.Tensor, mask: torch.Tensor | None, L: int, eps: float, neg: float) -> torch.Tensor:
    """bias [H, L, L] bf16 = wf . LN(pair) (no LN affine: fold its weight into ``wf`` [H, 128] bf16, H = 8, 16 or 24), ``neg`` on the
    keys where ``mask`` [L] is False. pair2d [L L, 128] bf16 contiguous, L a multiple of 16."""
    out = torch.empty(wf.shape[0], L, L, device=pair2d.device, dtype=torch.bfloat16)
    ext().pair_bias_fwd(pair2d, wf, mask, out, int(L), float(eps), float(neg))
    return out


def pair_bias_bwd(pair2d: torch.Tensor, dbias: torch.Tensor, wf: torch.Tensor, L: int, eps: float, acc: torch.Tensor | None = None,
                  d: int = 384):
    """(dpair [L L, 128] bf16, dWf [H, 128] fp32, per-head dbias sums [H]) of ``pair_bias`` (without the mask: masked keys
    carry zero dbias). dWf and the sums are added into ``acc`` (``acc_n(H)`` floats, zeroed by the caller) when given; the
    returned dWf / sums are views of it."""
    heads = wf.shape[0]
    if acc is None:
        acc = torch.zeros(acc_n(heads, d), device=pair2d.device)
    dz = torch.empty_like(pair2d)
    ext().pair_bias_bwd(pair2d, dbias.contiguous(), wf, dz, acc, int(L), float(eps), int(d))
    o = 2 * d + width(heads, d)
    return dz, acc[o:o + heads * DP].view(heads, DP), acc[o + heads * DP:]


__all__ = ["DP", "GEOMETRY", "RW", "acc_n", "available", "ext", "pair_bias", "pair_bias_bwd", "width"]
