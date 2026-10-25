"""Portable packed contractions for the Triton/cuBLAS bidirectional path.

GEMMs write into final buffers, avoiding one forward and two backward copies.
Fresh outputs and fixed launch sequences preserve the custom-op alias contract.
The enclosing autograd.Function owns the backward formulas.
"""

import torch

from miniworld_engine.kernels._compile import opaque


def _packed_forward_fake(left, right, h):
    """Allocate outputs with the same shape, dtype and strides as packed_forward."""
    return left.new_empty(left.shape)


@opaque(fake=_packed_forward_fake, name='trimul_triton_contract_fwd')
def packed_forward(left: torch.Tensor, right: torch.Tensor, h: int) -> torch.Tensor:
    """Execute packed forward behind an opaque compiler boundary."""
    tri = left.new_empty(left.shape)
    torch.bmm(left[:h], right[:h].transpose(1, 2), out=tri[:h])
    torch.bmm(left[h:].transpose(1, 2), right[h:], out=tri[h:])
    return tri


def _packed_backward_fake(grad, left, right, h):
    """Allocate outputs with the same shape, dtype and strides as packed_backward."""
    return (left.new_empty(left.shape), right.new_empty(right.shape))


@opaque(fake=_packed_backward_fake, name='trimul_triton_contract_bwd')
def packed_backward(grad: torch.Tensor, left: torch.Tensor, right: torch.Tensor, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute packed backward behind an opaque compiler boundary."""
    (dl, dr) = (left.new_empty(left.shape), right.new_empty(right.shape))
    torch.bmm(grad[:h], right[:h], out=dl[:h])
    torch.bmm(grad[:h].transpose(1, 2), left[:h], out=dr[:h])
    torch.bmm(right[h:], grad[h:].transpose(1, 2), out=dl[h:])
    torch.bmm(left[h:], grad[h:], out=dr[h:])
    return (dl, dr)


#: The widths the portable TriMul front / back serve when ``d_hidden != d_pair`` (equal widths keep their own, wider, contract): both a
#: multiple of 16 (the MMA tile) inside this box. Outside it the module refuses (``KernelRefusedWarning``) and runs the PyTorch reference.
HIDDEN_NE_PAIR_MAX_PAIR = 256
HIDDEN_NE_PAIR_MAX_HIDDEN = 512


def hidden_refusal(d_pair: int, d_hidden: int) -> str | None:
    """None if the portable TriMul serves ``d_hidden != d_pair`` (the per-direction hidden width of a unidirectional or a bidirectional
    module), else why not. ``d_hidden == d_pair`` is the family's own contract and is not judged here."""
    if d_hidden == d_pair:
        return None
    if d_pair % 16 or d_hidden % 16:
        return f"d_hidden ({d_hidden}) and d_pair ({d_pair}) must both be multiples of 16 when they differ"
    if d_pair > HIDDEN_NE_PAIR_MAX_PAIR or d_hidden > HIDDEN_NE_PAIR_MAX_HIDDEN:
        return (f"d_hidden ({d_hidden}) != d_pair ({d_pair}) is served up to d_pair {HIDDEN_NE_PAIR_MAX_PAIR} / "
                f"d_hidden {HIDDEN_NE_PAIR_MAX_HIDDEN}")
    return None
