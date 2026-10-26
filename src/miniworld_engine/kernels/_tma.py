"""Persistent, TMA-fed GEMM kernels on Hopper: the plumbing the kernels that have a ``TMA`` config axis share.

A Triton GEMM that loads its operands with pointers and lets one program own one tile runs at 55-80 % of cuBLAS on an H100 and, with an epilogue that
reads and writes whole tiles, its memory traffic is not overlapped with the MMAs of the next tile. The same GEMM with tensor descriptors (TMA loads and
stores) in a persistent loop (``tl.range(pid, n_tiles, NUM_SMS, flatten=True)``) matches cuBLAS (measured on an H100, M = 49152, K = 768 / 1536:
328 vs 323 us and 167 vs 166 us) and pipelines the epilogue of one tile under the MMAs of the next, which is what makes a fused epilogue win.

Descriptors are made inside the kernel; Triton asks for their scratch memory through an allocator that has to be set once per process.
A ``TMA`` config of a kernel is only launched where ``tma_launchable`` holds: sm_90 or newer, 16-bit operands, 16-byte aligned bases and row strides;
``prune_tma`` takes the others off the autotuner's list, so a card without TMA (or a shape that cannot use it) searches the pointer-based configs only.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

def _scratch(size: int, alignment: int, stream) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ensure_allocator() -> None:
    """Triton's scratch allocator for on-device tensor descriptors. It is a ContextVar, so it is set in the calling thread at every launch
    (the autograd engine runs a backward in a thread of its own, which does not see the one the forward set)."""
    triton.set_allocator(_scratch)


@functools.lru_cache(maxsize=16)
def num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def tma_capable(device: torch.device) -> bool:
    if device.type != "cuda" or not hasattr(tl, "make_tensor_descriptor"):
        return False
    return torch.cuda.get_device_capability(device)[0] >= 9


def tma_launchable(*tensors: torch.Tensor) -> bool:
    """16-bit, 16-byte aligned base, unit inner stride and a row stride that is a multiple of 16 bytes, for every operand."""
    for t in tensors:
        if t.dtype not in (torch.bfloat16, torch.float16) or not tma_capable(t.device):
            return False
        if t.dim() != 2 or t.stride(1) != 1 or (t.stride(0) * t.element_size()) % 16 or t.data_ptr() % 16:
            return False
    return True


def prune_tma(operands):
    """``early_config_prune`` for an autotuned kernel with a ``TMA`` axis: ``operands(named_args)`` returns the tensors a TMA launch would read and write.
    The TMA configs stay only when ``tma_launchable`` holds for all of them."""

    def prune(configs, named_args, **kwargs):
        try:
            ok = tma_launchable(*operands(named_args))
        except Exception:  # noqa: BLE001 - a prune never fails a launch
            ok = False
        return [c for c in configs if ok or not c.kwargs.get("TMA", 0)]

    return prune
