"""A100: the TriMul inference contraction bounded by the live extents of the pair mask (``sm80/contract_bounded.cu``).

A padded sample (tokens ``[0, n)`` live, the tail masked) has a / b planes that are zero past ``n`` in every row and column (K1 masks
them by ``m[i] & m[k]``), so its contraction past ``n`` is exact zeros. The kernel reads the live extents on the device (a CUDA graph
re-reads them on every replay: FoldForge refreshes the mask per fold) and runs the K loop only to the live ``k`` (rounded up to the K
tile), writing a tile past the live rows / columns as the zeros the full product holds. With every token live it is the full product.

It replaces cuBLAS only on the cells ``CELLS`` lists, where it measured at least as fast as cuBLAS on UNPADDED inputs (interleaved
medians; the speed ladder times unpadded inputs). Its outputs are bitwise those of the cuBLAS contraction (cuBLAS's sm_80 kernel
accumulates each output in fp32 over the same k tiles in the same order; tests/gpu/test_trimul_bounded_contraction.py).

H100: the same skip on a CUTLASS 3 sm_90 GEMM with a bounded tile scheduler measured 1.04-1.06x of cuBLAS at the GEMM level on a few
cells, but 0.96-0.98x at the module level unpadded, so H100 keeps cuBLAS (the work is on branch mask-skip-trimul-h100-wip).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from ..._nvcc import cutlass_include, ensure_cuda_home, host_flags, load_extension

_SM80 = Path(__file__).parent / "sm80"


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


#: (planes, n, mode: 0 outgoing a b^T / 1 incoming a^T b) -> (config, log tile) of ``sm80/contract_bounded.cu``. A100, 2026-11-13 / 14
#: (tools/bench/contract_bounded_sm80.py, two runs; unpadded median ratio vs torch.bmm, then the L // 8 and L // 4 padded ratios). Only
#: the cells within 1 % of cuBLAS or faster: L <= 1536 measured 0.91-0.99 (cuBLAS's ampere 128x128x32 x5 kernel stays ahead there), as
#: did the outgoing d 256 / 384 cells at L 2048 (0.97-0.99). A cell absent here keeps cuBLAS (no skip).
CELLS: dict[tuple[int, int, int], tuple[int, int]] = {
    (64, 2048, 0): (4, 3),  # 1.023 / 0.997 unpadded; 1.46-1.52x / 2.22-2.37x padded
    (64, 2048, 1): (2, 1),  # 1.049 / 1.046; 1.54-1.55x / 2.39-2.40x
    (128, 2048, 0): (1, 3),  # 0.995 / 0.993; 1.44-1.47x / 2.22-2.31x
    (128, 2048, 1): (2, 2),  # 1.028; 1.51x / 2.37x
    (256, 2048, 1): (2, 2),  # 1.016 / 1.010; 1.40-1.44x / 2.09-2.20x
    (384, 2048, 1): (2, 0),  # 1.005; 1.40x / 2.09x
}


def live_bounds(mask: torch.Tensor, samples: int) -> torch.Tensor:
    """int32 [samples, 3]: (1 + the last live token) of each sample's token mask, three times (rows, columns, k). On the device, no
    host sync (graph-capture safe; a replay recomputes it from the mask it reads)."""
    t = mask.reshape(samples, -1)
    pos = torch.arange(1, t.shape[1] + 1, device=t.device, dtype=torch.int32)
    last = (t.to(torch.int32) * pos).amax(-1, keepdim=True)
    return last.expand(samples, 3).contiguous()


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


def _route(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """(a planes, b planes, mode) when this product is a contraction the kernel takes on a CELLS cell, else None."""
    if not a.is_cuda or torch.cuda.get_device_capability(a.device) != (8, 0) or not out.is_contiguous():
        return None
    pa, pb = _planes(a), _planes(b)
    if pa is None or pb is None or pa[1] == pb[1] or tuple(b.shape) != tuple(a.shape) or tuple(out.shape) != tuple(a.shape):
        return None
    mode = 1 if pa[1] else 0
    return (pa[0], pb[0], mode) if (a.shape[0], a.shape[-1], mode) in CELLS else None


def contract(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, bounds: torch.Tensor | None, config: tuple[int, int], mode: int) -> None:
    """``out = a @ b`` per plane for planes ``a`` / ``b`` as stored ([P, n, n]; mode 0: a b^T, mode 1: a^T b), bounded by ``bounds``
    (``live_bounds``, one sample; None: the full product) on ``config`` (a ``CELLS`` value)."""
    _ext_sm80().contract(a, b, out, mode, config[0], bounds, 1, config[1])


def bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, token: torch.Tensor | None) -> torch.Tensor:
    """A100, one sample: ``torch.bmm(a, b, out=out)``, or with ``token`` (inference: the [L] token mask, uint8 / bool; empty = no mask)
    on a cell of ``CELLS`` the bounded contraction, which stops at the last live token."""
    route = None if token is None else _route(a, b, out)
    if route is None:
        return torch.bmm(a, b, out=out)
    pa, pb, mode = route
    contract(pa, pb, out, None if token.numel() == 0 else live_bounds(token, 1), CELLS[(a.shape[0], a.shape[-1], mode)], mode)
    return out
