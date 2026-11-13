"""The TriMul contraction bounded by the live extents of the pair mask (H100: ``h100_sources/bounded``; A100: ``sm80/contract_bounded.cu``).

A padded sample (tokens ``[0, n)`` live, the tail masked) has a / b planes that are zero past ``n`` in every row and column (K1 masks
them by ``m[i] & m[k]``), so its contraction past ``n`` is exact zeros. These kernels read each plane's live extents on the device (a
CUDA graph re-reads them on every replay: FoldForge refreshes the mask per fold) and run the K loop only to the live ``k``, writing a
tile past the live rows / columns as the zeros the full product holds. With every token live they are the full product.

They replace cuBLAS only on the cells ``CELLS`` lists, where they measured at least as fast as the cuBLAS contraction on UNPADDED
inputs (interleaved medians; the speed ladder times unpadded inputs). Outputs are bitwise those of the cuBLAS contraction on the
measured cells (both accumulate each output in fp32 over k in tile order; checked by tests/gpu/test_trimul_bounded_contraction.py).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from ..._nvcc import cutlass_include, ensure_cuda_home, gencodes, host_flags, load_extension

_H100 = Path(__file__).parent / "h100_sources" / "bounded"
_SM80 = Path(__file__).parent / "sm80"
H100_CONFIGS = 6


@functools.lru_cache(maxsize=None)
def _ext_h100():
    ensure_cuda_home()
    return load_extension(
        name="trimul_contract_bounded_h100_v1",
        sources=[str(_H100 / "binding.cu"), *(str(_H100 / f"cfg{i}.cu") for i in range(H100_CONFIGS))],
        extra_include_paths=[str(_H100), str(cutlass_include())],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[*host_flags(), *gencodes("90a"), "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG"],
    )


@functools.lru_cache(maxsize=None)
def _ext_sm80():
    ensure_cuda_home()
    return load_extension(
        name="trimul_contract_bounded_sm80_v1",
        sources=[str(_SM80 / "contract_bounded.cu")],
        extra_include_paths=[str(cutlass_include())],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[*host_flags(), "-gencode=arch=compute_80,code=sm_80", "-O3", "-std=c++17", "--expt-relaxed-constexpr"],
    )


#: (card, planes, n, mode) -> the measured configuration (cfg, swizzle, raster) (H100) / (cfg, log tile)
#: (A100). Filled from tools/bench/contract_bounded_*.py; a cell absent here keeps cuBLAS (no skip).
CELLS: dict[tuple, tuple] = {
    # A100 (2026-11-13, tools/bench/contract_bounded_sm80.py; unpadded median ratio vs torch.bmm, then the L // 8 and L // 4 padded ratios):
    # only the cells within 1 % of cuBLAS or faster; L <= 1536 measured 0.91-0.99 (cuBLAS's ampere 128x128x32 x5 kernel stays ahead there)
    ("a100", 64, 2048, 0): (4, 3),  # 1.023 unpadded; 1.52x / 2.37x padded
    ("a100", 64, 2048, 1): (2, 1),  # 1.049; 1.55x / 2.40x
    ("a100", 128, 2048, 0): (1, 3),  # 0.995; 1.44x / 2.22x
    ("a100", 128, 2048, 1): (2, 2),  # 1.028; 1.51x / 2.37x
    ("a100", 256, 2048, 1): (2, 2),  # 1.000-1.016; 1.40x / 2.09x
    ("a100", 384, 2048, 1): (2, 0),  # 1.005; 1.40x / 2.09x
    # H100 (2026-11-14, tools/bench/contract_bounded_h100.py; vs _h100_bmm.contract): (config, swizzle, raster 8 + 1 | 2: CUTLASS's tile order when
    # every token is live, live tiles first otherwise). Config 3 = pingpong 128 x 192, cluster 2 x 1. Only cells >= 0.99 of cuBLAS unpadded
    # in two separate sweeps; L 640-1280 and most L 2048 cells measured 0.80-0.98 (cuBLAS's nvjet kernels stay ahead), L 768 padded L // 8 slower.
    ("h100", 64, 1536, 0): (3, 2, 10),  # 1.039 / 1.040 unpadded; 1.12x / 1.49x padded (L // 8, L // 4)
    ("h100", 64, 1536, 1): (3, 2, 10),  # 1.065 / 1.065; 1.14x / 1.53x
    ("h100", 64, 2048, 1): (3, 1, 9),  # 0.990 / 0.992; 1.19x / 1.67x
    ("h100", 128, 1536, 0): (3, 1, 10),  # 1.017 / 1.055; 1.11x / 1.48x
    ("h100", 128, 1536, 1): (3, 1, 10),  # 1.054 / 1.056; 1.12x / 1.53x
    ("h100", 256, 1536, 1): (3, 2, 10),  # 1.040 / 1.054; 1.13x / 1.56x
    ("h100", 384, 1536, 1): (3, 2, 10),  # 1.059 / 1.055; 1.12x / 1.55x
}


def live_bounds(mask: torch.Tensor | None, samples: int) -> torch.Tensor | None:
    """int32 [samples, 3]: (1 + the last live token) of each sample's token mask, three times (rows, columns, k); None without a mask.
    On the device, no host sync (graph-capture safe; a replay recomputes it from the mask it reads)."""
    if mask is None:
        return None
    t = mask.reshape(samples, -1)
    pos = torch.arange(1, t.shape[1] + 1, device=t.device, dtype=torch.int32)
    last = (t.to(torch.int32) * pos).amax(-1, keepdim=True)
    return last.expand(samples, 3).contiguous()


def _card(device: torch.device) -> str | None:
    cap = torch.cuda.get_device_capability(device)
    return "h100" if cap == (9, 0) else "a100" if cap == (8, 0) else None


def cell(planes: int, n: int, mode: int, device: torch.device) -> tuple | None:
    """The measured configuration of this contraction, or None (cuBLAS keeps it)."""
    card = _card(device)
    return None if card is None else CELLS.get((card, planes, n, mode))


def _planes(t: torch.Tensor) -> tuple[torch.Tensor, bool] | None:
    """(the stored [P, n, n] planes, transposed?) of a [P, n, n] bf16 view of contiguous planes, else None."""
    n = t.shape[-1]
    if t.ndim != 3 or t.shape[1] != n or t.dtype != torch.bfloat16 or t.stride(0) != n * n:
        return None
    if t.stride(1) == n and t.stride(2) == 1:
        return t, False
    if t.stride(1) == 1 and t.stride(2) == n:
        return t.transpose(1, 2), True
    return None


def contract(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, bounds: torch.Tensor | None, samples: int, config: tuple) -> None:
    """``out = a @ b`` per plane (a, b as ``_h100_bmm.contract`` takes them: outgoing ``(left, right^T)``, incoming ``(left^T, right)``),
    bounded by ``bounds`` (``live_bounds``; plane l is sample l % samples) on the configuration ``config`` of ``cell``."""
    pa, pb = _planes(a), _planes(b)
    if pa is None or pb is None or pa[1] == pb[1] or not out.is_contiguous():
        raise ValueError("bounded contraction: a / b must be (planes, planes^T) or (planes^T, planes) views of contiguous bf16 planes")
    mode = 1 if pa[1] else 0
    if _card(a.device) == "h100":
        _ext_h100().contract(pa[0], pb[0], out, mode, config[0], bounds, samples, config[1], config[2])
    else:
        _ext_sm80().contract(pa[0], pb[0], out, mode, config[0], bounds, samples, config[1])


def bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, token: torch.Tensor | None) -> torch.Tensor:
    """A100 (one sample): ``torch.bmm(a, b, out=out)``, or with ``token`` (inference: the [L] token mask, uint8 / bool, empty = no mask)
    on a cell of ``CELLS`` the bounded contraction, which stops at the last live token."""
    if token is not None:
        pa = _planes(a)
        cfg = None if pa is None else cell(a.shape[0], a.shape[-1], int(pa[1]), a.device)
        if cfg is not None:
            contract(a, b, out, None if token.numel() == 0 else live_bounds(token, 1), 1, cfg)
            return out
    return torch.bmm(a, b, out=out)
