"""A100 (sm_80) hand-CUDA OuterProductMean, forward and backward (bf16, d_hidden 32, d_msa 64 / 128, d_pair 128 / 256 / 384).

``OuterProductMean`` is LN(msa) -> two projections -> the outer product of the projected rows averaged over the MSA depth -> the 32 x 32 -> d_pair
projection. The outer product is one GEMM in the *grouped* layout, O[(i, c), (j, e)] = sum_s a[s, i, c] b[s, j, e] (M = N = 32 L, K = S): cuBLAS runs
it (and the two gradient GEMMs), and the hand kernels (``sm80/``) are everything around it:

  prologue       LN + both projections + mask -> A, B [S, 32 L] (s-major operands: O = A^T B runs on cuBLAS's fastest NT-class kernel), and the LN statistics
  epilogue       O -> the [i, j, (c, e)] permute, / n_ij, the 1024 -> d_pair projection, bias, pair residual
  dgrad          dz -> dzn = dz / n, dO in the grouped layout (no permuted copy), the bias-gradient partials
  dwo            dWo partials off the kept O (split over rows of i)
  prologue_bwd   dA | dB -> dm, the dWl / dWr / dgamma / dbeta partials (persistent CTAs)
  reduce_rows    fixed-order sums of the partial rows (no atomics: every gradient is bit-reproducible)

``integrations/opm_sm80.py`` drives these from ``OuterProductMean.forward``; the extension is built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

CH = 32                                    # d_hidden
D_MSA = (64, 128)
D_PAIR = (128, 256, 384)
SPLITS = {128: 24, 256: 12, 384: 8}        # dWo: splits over rows of i (CTAs ~ 192 per card)


@functools.lru_cache(maxsize=1)
def ext():
    """The extension (built on first use into ``TORCH_EXTENSIONS_DIR``)."""
    ensure_cuda_home()
    return load_extension(
        name="opm_sm80", sources=[str(_dir / "ops.cu")], extra_include_paths=[str(_dir)],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-lineinfo", *gencodes("80"), "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False)


def supported(d_msa: int, d_hidden: int, d_pair: int) -> bool:
    return d_hidden == CH and d_msa in D_MSA and d_pair in D_PAIR


def _none(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.new_empty((0,), dtype=dtype)


def prologue(m: torch.Tensor, mask: torch.Tensor | None, lnw: torch.Tensor, lnb: torch.Tensor, wl: torch.Tensor, wr: torch.Tensor, eps: float,
             save_stats: bool, variant: int | None = None):
    """m [S, L, CM] bf16, mask [S, L] uint8 (or None), lnw / lnb fp32 [CM], wl / wr bf16 [32, CM] -> (A, B [S, 32 L] bf16, stats [S, L, 2] fp32 or an empty tensor)."""
    S, L, _ = m.shape
    a = m.new_empty((S, L * CH))
    b = m.new_empty((S, L * CH))
    stats = m.new_empty((S, L, 2), dtype=torch.float32) if save_stats else _none(m, torch.float32)
    if variant is None:
        variant = int(os.environ.get("MINIWORLD_OPM_SM80_PL", "0"))       # schedule experiments (tile buffers); 0 is the shipped one
    ext().prologue(m, _none(m, torch.uint8) if mask is None else mask, lnw, lnb, wl, wr, a, b, stats, eps, variant)
    return a, b, stats


def epilogue(o: torch.Tensor, norm: torch.Tensor | None, norm_const: float, wo: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor | None,
             norm_first: bool, variant: int | None = None) -> torch.Tensor:
    """O [32 L, 32 L] bf16, norm fp32 [L, L] (None: the constant ``norm_const``), wo bf16 [CZ, 1024], bias fp32 [CZ], residual bf16 [L, L, CZ] (or None) -> [L, L, CZ] bf16."""
    length = o.shape[0] // CH
    out = o.new_empty((length, length, wo.shape[0]))
    if variant is None:
        variant = int(os.environ.get("MINIWORLD_OPM_SM80_EPI", "0"))      # schedule experiments (ring depth / CTAs per SM); 0 is the shipped one
    ext().epilogue(o, _none(o, torch.float32) if norm is None else norm, norm_const, wo, bias, _none(o, torch.bfloat16) if residual is None else residual, out,
                   int(norm_first), variant)
    return out


def dgrad(dz: torch.Tensor, norm: torch.Tensor | None, norm_const: float, wo: torch.Tensor, norm_first: bool, variant: int | None = None):
    """dz [L, L, CZ] bf16 -> (dO [32 L, 32 L] bf16, dzn [L, L, CZ] bf16, dbo fp32 [CZ])."""
    length, _, cz = dz.shape
    dO = dz.new_empty((length * CH, length * CH))
    dzn = torch.empty_like(dz)
    parts = dz.new_empty((-(-length // 32) * -(-length // 4), cz), dtype=torch.float32)
    if variant is None:
        variant = int(os.environ.get("MINIWORLD_OPM_SM80_DG", "0"))        # schedule experiments (ring depth / CTAs per SM); 0 is the shipped one
    ext().dgrad(dz, _none(dz, torch.float32) if norm is None else norm, norm_const, wo, dO, dzn, parts, int(norm_first), variant)
    return dO, dzn, reduce_rows(parts, torch.float32)


def dwo(dzn: torch.Tensor, o: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """dWo [CZ, 1024] (``out_dtype``) = sum over the pairs of dzn[i, j, z] O[(i, c), (j, e)]."""
    length, _, cz = dzn.shape
    i_per = -(-length // SPLITS[cz])
    splits = -(-length // i_per)
    part = dzn.new_empty((splits, cz, CH * CH), dtype=torch.float32)
    ext().dwo(dzn, o, part, i_per)
    return reduce_rows(part.view(splits, -1), out_dtype).view(cz, CH * CH)


def prologue_bwd(da: torch.Tensor, db: torch.Tensor, m: torch.Tensor, stats: torch.Tensor, mask: torch.Tensor | None, lnw: torch.Tensor, lnb: torch.Tensor,
                 wl: torch.Tensor, wr: torch.Tensor, variant: int | None = None):
    """dA, dB [S, 32 L] bf16 (the cuBLAS products dA = B dO^T, dB = A dO), m [S, L, CM], stats [S, L, 2], ... -> (dm bf16, dWl, dWr [32, CM], dgamma, dbeta [CM]) all fp32 but dm."""
    S, L, cm = m.shape
    if variant is None:
        variant = int(os.environ.get("MINIWORLD_OPM_SM80_PB", "0"))       # schedule experiments (buffers / CTAs per SM); 0 is the shipped one
    ntile = -(-S // 64) * -(-L // 2)
    nblk = min(ntile, torch.cuda.get_device_properties(m.device).multi_processor_count * (1 if variant == 1 or cm != 64 else 2))
    dm = torch.empty_like(m)
    part = m.new_empty((nblk, 66 * cm), dtype=torch.float32)
    ext().prologue_bwd(da, db, m, stats, _none(m, torch.uint8) if mask is None else mask, lnw, lnb, wl, wr, dm, part, variant)
    total = reduce_rows(part, torch.float32)
    return dm, total[:32 * cm].view(32, cm), total[32 * cm:64 * cm].view(32, cm), total[64 * cm:65 * cm], total[65 * cm:]


def reduce_rows(part: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The rows of an fp32 [R, W] partial buffer summed in order (W a multiple of 4) -> [W] in ``dtype`` (fp32 or bf16)."""
    out = part.new_empty((part.shape[1],), dtype=dtype)
    ext().reduce_rows(part, out)
    return out
