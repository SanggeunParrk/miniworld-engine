"""Row kernels of the hoisted pair bias (``integrations/bias_only_dit_hoist.py``), bf16 and fp32.

  forward    LN0(pair): the token DiT rows' ``layernorm128_rows`` (one warp per row, no affine; the inference hoist's kernel)
  backward   ``bias_only_dit_hoist_rows.cu`` ln0_bwd_rows: d pair from d LN0 (fp32) and the pair rows, the row statistics
             recomputed with the forward's arithmetic, and the dW' GEMM's operand made from the same fp32 LN0 (bf16: the two-term
             split [hi | lo], 256 columns; fp32: LN0 itself)

Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_rows_cuda",
        sources=[str(_dir / "bias_only_dit_hoist_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                           *gencodes("80", "90", "100", ptx=("100",))],
        extra_cflags=["-std=c++17"], verbose=False)


def ln0_rows(x: torch.Tensor, out: torch.Tensor, eps: float = 1e-5) -> None:
    """out = LN(x) without affine; x [R, 128] (bf16 / fp32 rows), out contiguous [R, 128] (bf16 / fp32)."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    rows._ext().layernorm128_rows(x, out, float(eps), False)


def ln0_bwd_rows(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> None:
    """dx = the LayerNorm-without-affine backward of dy at x: rstd (dy - mean(dy) - y mean(dy y)), and ``y`` = LN0(x) as the dW'
    GEMM's operand: bf16 [R, 256] = [bf16(y) | bf16(y - bf16(y))], or fp32 [R, 128]. x / dx bf16 or fp32, dy fp32 [R, 128]; all
    contiguous."""
    ext().ln0_bwd_rows_cuda(x, dy, dx, y, float(eps))


@functools.lru_cache(maxsize=None)
def bwd_ext():
    """``bo_hoist_bwd.cu`` (sm_90a): the hoisted pair bias's backward in one kernel. Built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_hoist_bwd_sm90a",
        sources=[str(_dir / "bo_hoist_bwd.cu")],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", *gencodes("90a")],
        extra_cflags=["-std=c++17"], verbose=False)


#: Most rows of dbias_all (blocks x heads) the fused backward takes
FUSED_MAX_ROWS = 192


def fused_bwd_served(pair2d: torch.Tensor, wf: torch.Tensor) -> bool:
    """The fused backward serves bf16 pair rows [R, 128] with R a multiple of 64, at most ``FUSED_MAX_ROWS`` weight rows, on sm_90."""
    return (
        pair2d.dtype is torch.bfloat16
        and pair2d.is_cuda
        and torch.cuda.get_device_capability(pair2d.device) == (9, 0)
        and pair2d.shape[0] % 64 == 0
        and wf.shape[0] <= FUSED_MAX_ROWS
    )


def hoist_bwd_fused(pair2d: torch.Tensor, wf: torch.Tensor, dball: torch.Tensor, eps: float = 1e-5):
    """(d pair [R, 128] bf16, dW' [N, 128] fp32) from pair rows [R, 128], W'_all [N, 128] and dbias_all [N, R] (all bf16): d LN0 = dbias_all^T
    W', the LayerNorm backward and dW' = dbias_all LN0 (hi + lo) in one pass over the pair, the forward's LN0 arithmetic. The per-CTA fp32
    partials of dW' are summed here, in a fixed order."""
    R, N = pair2d.shape[0], wf.shape[0]
    npt = (N + 15) // 16
    grid = min(torch.cuda.get_device_properties(pair2d.device).multi_processor_count, R // 64)
    dpair = torch.empty_like(pair2d)
    part = torch.empty(grid, npt * 16, 128, device=pair2d.device, dtype=torch.float32)
    bwd_ext().hoist_bwd(pair2d, wf, dball, dpair, part, float(eps))
    return dpair, part.sum(0)[:N]


def operand_cols(dtype: torch.dtype) -> int:
    """Columns of ``ln0_bwd_rows``'s y for a path's dtype: 256 (bf16, the two-term split) or 128 (fp32)."""
    return 256 if dtype is torch.bfloat16 else 128


def kernel_attrs() -> list[tuple[str, int, int]]:
    """(name, registers, local-memory bytes) of every kernel of ``bias_only_dit_hoist_rows.cu``."""
    return [tuple(a) for a in ext().func_attrs()]


__all__ = ["FUSED_MAX_ROWS", "bwd_ext", "ext", "fused_bwd_served", "hoist_bwd_fused", "kernel_attrs", "ln0_bwd_rows", "ln0_rows", "operand_cols"]
