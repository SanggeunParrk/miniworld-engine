"""Host side of the Hopper TMA path shared by the portable Transition kernels.

``transition_fwd_kernel``, ``_squeeze_residual_kernel`` and ``_squeeze_swiglu_bwd_kernel`` each have two data-movement
modes behind the constexpr ``USE_TMA``: the pointer loads every card runs, and, on a Hopper card, tensor-map
loads/stores in a persistent grid (one program per SM walking the tiles). The tensor-map mode keeps the
copy engine busy through the epilogue (the stores of a tile overlap the loads of the next) and measured
~730 TFLOPS against ~650 for the same tile with pointer loads on the (M=147456, N=768, K=3072) squeeze GEMM,
past cuBLAS on that shape. The arithmetic and the rounding points are identical in both modes.

A tensor map needs a 16-byte aligned base, 16-byte aligned row strides and a unit inner stride; a call that
cannot meet that (a ragged width, a sliced operand) takes the pointer kernel. Only ``sm_90`` takes the tensor
maps: Triton's descriptor path is not validated for the portable family on any other card.
"""
from __future__ import annotations

import functools
import os

import torch
import triton
from torch._subclasses.fake_tensor import FakeTensor


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


@functools.cache
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index)[0] == 9


def _scratch(size: int, alignment: int, stream) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _install_allocator() -> None:
    """In-kernel tensor maps keep their 128 B descriptors in global scratch memory, which Triton asks a process-wide
    allocator for. Install torch's only when nobody has: an application's own allocator is left alone."""
    from triton.runtime._allocation import NullAllocator, _allocator

    if isinstance(_allocator.get(), NullAllocator):
        triton.set_allocator(_scratch)


def num_sms(device: torch.device) -> int:
    return _sm_count(device.index if device.index is not None else torch.cuda.current_device())


def tma_ok(*tensors: torch.Tensor) -> bool:
    """Whether every operand can be addressed by a 2-D tensor map on this card."""
    first = tensors[0]
    if not first.is_cuda or os.environ.get("MINIWORLD_PORTABLE_NO_TMA"):   # the env switch is the A/B and test hook
        return False
    index = first.device.index if first.device.index is not None else torch.cuda.current_device()
    if not _is_hopper(index):
        return False
    for t in tensors:
        if isinstance(t, FakeTensor):     # `dev derive` records launches on fake tensors: no address to check, no kernel to run
            return False
        if t.dim() != 2 or t.stride(1) != 1 or t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        if t.data_ptr() % 16 or (t.stride(0) * t.element_size()) % 16 or t.shape[0] == 0:
            return False
    _install_allocator()      # a ContextVar: the autograd engine runs the backward launches on threads of its own
    return True


