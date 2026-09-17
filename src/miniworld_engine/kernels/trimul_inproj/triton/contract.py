"""Portable packed contractions for the Triton/cuBLAS bidirectional path.

GEMMs write into final buffers, avoiding one forward and two backward copies.
Fresh outputs and fixed launch sequences preserve the custom-op alias contract.
The enclosing autograd.Function owns the backward formulas.
"""

import torch

from miniworld_engine.kernels._compile import opaque


def _forward_fake(left, right, h):
    return left.new_empty(left.shape)


@opaque(fake=_forward_fake, name="trimul_triton_contract_fwd")
def packed_forward(left: torch.Tensor, right: torch.Tensor, h: int) -> torch.Tensor:
    tri = left.new_empty(left.shape)
    torch.bmm(left[:h], right[:h].transpose(1, 2), out=tri[:h])
    torch.bmm(left[h:].transpose(1, 2), right[h:], out=tri[h:])
    return tri


def _backward_fake(grad, left, right, h):
    return left.new_empty(left.shape), right.new_empty(right.shape)


@opaque(fake=_backward_fake, name="trimul_triton_contract_bwd")
def packed_backward(
    grad: torch.Tensor, left: torch.Tensor, right: torch.Tensor, h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dl, dr = left.new_empty(left.shape), right.new_empty(right.shape)
    torch.bmm(grad[:h], right[:h], out=dl[:h])
    torch.bmm(grad[:h].transpose(1, 2), left[:h], out=dr[:h])
    torch.bmm(right[h:], grad[h:].transpose(1, 2), out=dl[h:])
    torch.bmm(left[h:], grad[h:], out=dr[h:])
    return dl, dr
