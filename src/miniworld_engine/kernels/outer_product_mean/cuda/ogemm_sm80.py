"""The OuterProductMean's grouped outer product O = A^T B on A100 with the dead tokens (masked in every MSA row) skipped: ``sm80/ogemm_cutlass.cu``
(a CUTLASS 2.x sm_80 GEMM; a threadblock whose row or column tokens are all dead writes zeros). Exact: those O blocks are 0 in the dense product."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_dir = Path(__file__).parent / "sm80"


@functools.lru_cache(maxsize=1)
def ext():
    from ..._nvcc import cutlass_include, ensure_cuda_home, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="opm_ogemm_sm80",
        sources=[str(_dir / "ogemm_cutlass.cu")],
        extra_include_paths=[str(_dir), str(cutlass_include())],
        extra_cuda_cflags=[*host_flags(), "-O3", "-std=c++17", "-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def live_tokens(mask_u8: torch.Tensor) -> torch.Tensor:
    """[S, L] uint8 mask -> [L] uint8: 1 where some MSA row keeps the token."""
    return mask_u8.amax(0).contiguous()


def outer(a: torch.Tensor, b: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """a, b [S, 32 L] bf16 -> O [32 L, 32 L] bf16 = a^T b (the dead tokens' blocks written as zeros without the product)."""
    m = a.shape[1]
    out = a.new_empty((m, m))
    ext().ogemm_skip(a, b, live, out)
    return out
