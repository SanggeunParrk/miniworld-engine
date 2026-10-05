"""A100 (sm_80) hand-CUDA bucket reduction of the ProteinMPNN relative-position embedding backward.

``grad_table[b] = sum of the gradient rows whose edge falls in bucket b``, ``grad_bias`` = the sum of all rows: one pass over the edges as a one-hot MATMUL on the tensor cores
(``cuda/sm80/relpos_sm80.cuh``; the bias rides in an extra all-ones row), deterministic (per-warp register tables, a fixed-order combine) and an HBM stream (40 B / edge).
Served: a [rows, 16] bf16 or fp32 gradient, an int64 bucket index of ``rows`` entries, a table of at most 79 buckets (the shipped one is 66); everything else keeps the Triton reduction.
Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"
_common = Path(__file__).resolve().parents[2] / "mpnn_message" / "cuda" / "sm80"

WIDTH = 16
MAX_BUCKETS = 79


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="mpnn_relpos_sm80",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", f"-I{_common}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def supported(grad_output: torch.Tensor, bucket: torch.Tensor, buckets: int) -> bool:
    """Whether :func:`bucket_reduce` serves this reduction (the capability of the device is the caller's gate)."""
    return (grad_output.is_cuda and bucket.is_cuda and grad_output.device == bucket.device and bucket.dtype == torch.long
            and grad_output.dtype in (torch.bfloat16, torch.float32) and grad_output.shape[-1] == WIDTH and 1 <= buckets <= MAX_BUCKETS
            and bucket.numel() > 0 and grad_output.numel() == bucket.numel() * WIDTH)


def bucket_reduce(grad_output: torch.Tensor, bucket: torch.Tensor, buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(grad_table [buckets, 16], grad_bias [16])``, both fp32, of the gradient rows ``grad_output`` [..., 16] (bf16 or fp32) and their bucket index ``bucket`` [...] (int64)."""
    g = grad_output.contiguous().reshape(-1, WIDTH)
    b = bucket.contiguous().reshape(-1)
    table, bias = _ext().relpos_reduce(g, b, int(buckets))
    return table, bias
