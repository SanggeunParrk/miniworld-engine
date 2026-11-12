"""H100 TriangleAttention parameter-pack / dropout-mask extension (``h100_sources/pack.cu``).

The attention forward / backward of the H100 TriangleAttention are in ``h100_sm90`` / ``h100_sm90_generic`` / ``h100_dh``.
What remains here is the Philox dropout draw of the H100 training paths: the MSA row keep-mask (``drop_keep``, ``integrations/pwa_train``)
and the pair dropout scale of TriangleMultiplication / TriangleAttention (``drop_scale``, :func:`drop_scale`).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import (
    cutlass_include,
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")
_PARENT = _SRC.parent


@functools.lru_cache(maxsize=None)
def extension(part: str):
    if part != "pack":
        raise ValueError(part)
    ensure_cuda_home()
    # CUTLASS 4.2 headers are needed only to build (prebuilt / cached artifacts load without; _nvcc raises CutlassMissing)
    cutlass_inc = cutlass_include()
    return load_extension(
        name="triattn_h100_b200style_pack_v1",
        sources=[str(_SRC / "pack.cu")],
        extra_include_paths=[str(_PARENT), str(cutlass_inc)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            *host_flags(),
            *gencodes("90a"),
            "-O3",
            "-std=c++17",
            "-DMW_TRIATTN_KV_TILE=64",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
    )


@functools.lru_cache(maxsize=None)
def _drop_operands(p: float) -> tuple[float, float]:
    """``(threshold, inverse)`` of ``(rand > p) / (1 - p)`` on a BF16 tensor as torch evaluates it on CUDA: the comparison casts its scalar
    to the tensor's dtype (BF16(p)); the division multiplies by the FP32 reciprocal of its FP32 scalar (1.f / float(1 - p)), the product
    rounded to BF16."""
    threshold = torch.tensor(p, dtype=torch.bfloat16).item()
    inverse = (torch.ones((), dtype=torch.float32) / torch.tensor(1.0 - p, dtype=torch.float32)).item()
    return threshold, inverse


def _drop_scale_fake(pair: torch.Tensor, threshold: float, inverse: float) -> torch.Tensor:
    return pair.new_empty((pair.shape[0], pair.shape[1], pair.shape[3]))


@opaque(fake=_drop_scale_fake, name="triangle_h100_drop_scale_cuda")
def _drop_scale(pair: torch.Tensor, threshold: float, inverse: float) -> torch.Tensor:
    return extension("pack").drop_scale(pair, threshold, inverse)


def drop_scale_serves(pair: torch.Tensor, p: float) -> bool:
    """The Philox kernel draws this call's scale: a BF16 square pair [B, L, L, C] on an H100 and 0 < p < 1."""
    return (
        pair.is_cuda
        and pair.dtype == torch.bfloat16
        and pair.ndim == 4
        and pair.shape[1] == pair.shape[2]
        and pair.numel() > 0
        and 0.0 < p < 1.0
        and torch.cuda.get_device_capability(pair.device) == (9, 0)
    )


def drop_scale(pair: torch.Tensor, p: float, broadcast_dim: int) -> torch.Tensor:
    """``modules.functional.dropout_scale(pair, p, broadcast_dim)`` (``broadcast_dim`` 1: [B, 1, L, C], 2: [B, L, 1, C]) in one launch
    of ``drop_scale_kernel``: the same default-generator Philox subsequences, counter advance, BF16 uniform rounding, threshold and
    scale as the torch statements (bitwise the same tensor for the same generator state; CUDA-graph capturable). The caller checked
    :func:`drop_scale_serves`."""
    threshold, inverse = _drop_operands(float(p))
    out = _drop_scale(pair.detach(), threshold, inverse)
    return out.unsqueeze(broadcast_dim)
