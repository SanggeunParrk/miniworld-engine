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
    rows._ext().layernorm128_rows(x, out, float(eps))


def ln0_bwd_rows(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> None:
    """dx = the LayerNorm-without-affine backward of dy at x: rstd (dy - mean(dy) - y mean(dy y)), and ``y`` = LN0(x) as the dW'
    GEMM's operand: bf16 [R, 256] = [bf16(y) | bf16(y - bf16(y))], or fp32 [R, 128]. x / dx bf16 or fp32, dy fp32 [R, 128]; all
    contiguous."""
    ext().ln0_bwd_rows_cuda(x, dy, dx, y, float(eps))


def operand_cols(dtype: torch.dtype) -> int:
    """Columns of ``ln0_bwd_rows``'s y for a path's dtype: 256 (bf16, the two-term split) or 128 (fp32)."""
    return 256 if dtype is torch.bfloat16 else 128


def kernel_attrs() -> list[tuple[str, int, int]]:
    """(name, registers, local-memory bytes) of every kernel of ``bias_only_dit_hoist_rows.cu``."""
    return [tuple(a) for a in ext().func_attrs()]


__all__ = ["ext", "kernel_attrs", "ln0_bwd_rows", "ln0_rows", "operand_cols"]
