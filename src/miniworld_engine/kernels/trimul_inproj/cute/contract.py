"""Bidirectional contractions write directly into the downstream packed buffers.

Two forward and four backward GEMMs need no concatenation copies. These opaque
launch sequences choose the same backend in eager, cold compile, and CUDA graphs;
no shape inspection or calibration occurs while Dynamo traces fake tensors.
"""

from functools import lru_cache

import torch

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cute.dispatch import _cute_allowed


@lru_cache(None)
def _measured_gpu(device):
    return (
        torch.cuda.get_device_capability(device) == (9, 0)
        and torch.cuda.get_device_name(device) == "NVIDIA H100 80GB HBM3"
    )


def _use_quack(*operands):
    """Whole-training graph measurements: Quack wins these exact contracts.

    BF16 [256, L, L], L=384/768: all six Quack GEMMs beat packed cuBLAS.
    L=128 and unmeasured layouts/shapes/devices retain cuBLAS. Weight-gradient
    reductions are separate operations and are deliberately not covered here.
    Evidence: MiniWorld docs/trimul-packed-training.md and its paired samples.
    """
    first = operands[0]
    return (
        first.device.type == "cuda"
        and first.dtype == torch.bfloat16
        and tuple(first.shape) in ((256, 384, 384), (256, 768, 768))
        and all(
            t.shape == first.shape and t.dtype == first.dtype
            and t.device == first.device and t.is_contiguous()
            for t in operands
        )
        and _measured_gpu(first.device)
        and _cute_allowed(first.device, first.dtype, "triangle_multiplication_bidirectional")
    )


def _bmm(a, b, out, quack):
    if quack:
        from miniworld_engine.kernels._quack_compat import gemm
        gemm(a, b, out=out)
    else:
        torch.bmm(a, b, out=out)


def _forward_fake(left, right, h):
    return left.new_empty(left.shape)


@opaque(fake=_forward_fake, name="trimul_bidir_contract_fwd")
def packed_forward(left: torch.Tensor, right: torch.Tensor, h: int) -> torch.Tensor:
    quack = _use_quack(left, right) and h == 128
    tri = left.new_empty(left.shape)
    _bmm(left[:h], right[:h].transpose(1, 2), tri[:h], quack)
    _bmm(left[h:].transpose(1, 2), right[h:], tri[h:], quack)
    return tri


def _backward_fake(grad, left, right, h):
    return left.new_empty(left.shape), right.new_empty(right.shape)


@opaque(fake=_backward_fake, name="trimul_bidir_contract_bwd")
def packed_backward(
    grad: torch.Tensor, left: torch.Tensor, right: torch.Tensor, h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    quack = _use_quack(left, right, grad) and h == 128
    dl, dr = left.new_empty(left.shape), right.new_empty(right.shape)
    _bmm(grad[:h], right[:h], dl[:h], quack)
    _bmm(grad[:h].transpose(1, 2), left[:h], dr[:h], quack)
    _bmm(right[h:], grad[h:].transpose(1, 2), dl[h:], quack)
    _bmm(left[h:], grad[h:], dr[h:], quack)
    return dl, dr
