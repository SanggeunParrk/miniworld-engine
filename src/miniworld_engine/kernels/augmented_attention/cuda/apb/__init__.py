"""CUDA passes of AttentionPairBias (the Pairformer single track) on B200 (``apb_rows.cu``): the pair -> bias projection
with its LayerNorm (mma.sync), its backward, and the row passes of the single track (LayerNorm, sigmoid gate, their
backwards). Its attention cores are this family's sm_100a kernels (``../sm100``). Built on first use (``load_extension``),
never at import; ``kernels/integrations/attention_pair_bias_b200.py`` drives them."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent
RW = 4                    # rows per block of the row kernels (a warp each)
DP = 128                  # d_pair of the 128-wide pair kernels (256 / 384 / 512: the wide forms, 8 or 16 heads)
# (heads, d_single) -> (real head dim, head width in memory: 16 x 24 is padded to 32)
GEOMETRY = {(8, 384): (48, 48), (12, 384): (32, 32), (16, 384): (24, 32), (24, 384): (16, 16), (16, 512): (32, 32)}
#: d_pair -> the head counts the pair kernels are built for (forward; the wide backward takes the same)
PAIR_WIDTHS = {128: (8, 12, 16, 24), 256: (8, 16), 384: (8, 16), 512: (16,)}


def width(heads: int, d: int = 384) -> int:
    """Row width of q / k / v / g / O and their gradients: d_single, or 512 for 16 x 24 (padded to 32)."""
    return heads * GEOMETRY[(heads, d)][1]


def acc_n(heads: int, d: int = 384, dp: int = DP) -> int:
    """The backward's fp32 accumulator: dlnw [d] | dlnb [d] | dbq [W] | dWf [heads, dp] | per-head dbias sums [heads]."""
    return 2 * d + width(heads, d) + heads * dp + heads


def ext(shadow: bool = False):
    """The row / pair extension. ``shadow`` (H100 only) also builds the fp32 pair-weight oracle (``integrations.attention_pair_bias_b200.EXACT_DWB``), which
    needs CUTLASS 4.2 headers (``CUTLASS_PATH``); the default build has no such dependency."""
    arch = "90a" if torch.cuda.get_device_capability() == (9, 0) else "100"
    return _ext(arch, shadow and arch == "90a")


@functools.lru_cache(maxsize=None)
def _ext(arch: str, shadow: bool = False):
    from ...._nvcc import cutlass_include, ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    includes, shadow_flags = [], []
    if shadow:
        # only a build needs the headers: unset, the stand-in keys as the pinned CUTLASS 4.2 and the prebuilt artifact loads
        # (a build without them raises _nvcc.CutlassMissing)
        includes = [str(cutlass_include())]
        shadow_flags = ["-DAPB_H100_FP32_SHADOW=1"]
    return load_extension(
        name=f"apb_rows_cuda_sm{arch}" + ("_shadow" if shadow else ""),
        sources=[str(_dir / "apb_rows.cu")],
        extra_include_paths=includes,
        extra_ldflags=["-lcublas", "-lcublasLt"],
        extra_cuda_cflags=[*host_flags(), *shadow_flags, "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes(arch, ptx=() if arch == "90a" else ("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def pair_bias(pair2d: torch.Tensor, wf: torch.Tensor, mask: torch.Tensor | None, L: int, eps: float, neg: float,
              saved: bool = False):
    """bias [H, L, L] bf16 = wf . LN(pair) (no LN affine: fold its weight into ``wf`` [H, dp] bf16; dp 128 with H = 8, 12, 16 or 24,
    or a wide pair of ``PAIR_WIDTHS``), ``neg`` on the keys where ``mask`` [L] is False. pair2d [L L, dp] bf16 contiguous, L a
    multiple of 16. ``saved`` (a wide pair's training forward): returns (bias, row stats [L L, 2], unmasked bias [H, L, L] fp32),
    what ``pair_bias_bwd`` reads for a wide pair."""
    out = torch.empty(wf.shape[0], L, L, device=pair2d.device, dtype=torch.bfloat16)
    if not saved:
        ext().pair_bias_fwd(pair2d, wf, mask, out, int(L), float(eps), float(neg))
        return out
    st = torch.empty(L * L, 2, device=pair2d.device, dtype=torch.float32)
    bu = torch.empty(wf.shape[0], L, L, device=pair2d.device, dtype=torch.float32)
    ext().pair_bias_fwd_train(pair2d, wf, mask, out, st, bu, int(L), float(eps), float(neg))
    return out, st, bu


def pair_bias_bwd(pair2d: torch.Tensor, dbias: torch.Tensor, wf: torch.Tensor, L: int, eps: float, acc: torch.Tensor | None = None,
                  d: int = 384, saved: tuple[torch.Tensor, torch.Tensor] | None = None):
    """(dpair [L L, dp] bf16, dWf [H, dp] fp32, per-head dbias sums [H]) of ``pair_bias`` (without the mask: masked keys
    carry zero dbias). dWf and the sums are added into ``acc`` (``acc_n(H, d, dp)`` floats, zeroed by the caller) when given; the
    returned dWf / sums are views of it. A wide pair (dp > 128) takes ``saved`` = (row stats, unmasked bias) of its forward."""
    heads, dp = wf.shape
    if acc is None:
        acc = torch.zeros(acc_n(heads, d, dp), device=pair2d.device)
    dz = torch.empty_like(pair2d)
    if dp == DP:
        ext().pair_bias_bwd(pair2d, dbias.contiguous(), wf, dz, acc, int(L), float(eps), int(d))
    else:
        st, bu = saved
        ext().pair_bias_bwd_wide(pair2d, dbias.contiguous(), wf, st, bu, dz, acc, int(L), int(d))
    if acc.ndim == 2:
        return dz, None, None  # Hopper block partials are reduced by finalize.
    o = 2 * d + width(heads, d)
    return dz, acc[o:o + heads * dp].view(heads, dp), acc[o + heads * dp:]


__all__ = ["DP", "GEOMETRY", "PAIR_WIDTHS", "RW", "acc_n", "ext", "pair_bias", "pair_bias_bwd", "width"]
