"""One-launch dtype cast of a handful of weights (the fp32 master -> bf16 copies a training forward makes at every call).

``torch._foreach_copy_`` casts a list of tensors in one launch too, but its grid is a few CTAs (11 for a 2.5 M-element list on an H100) and it
runs at a third of the HBM rate: 16 - 20 us per call for 8 - 18 MB of traffic, four calls in a bias-only DiT training step. The kernel here gives
every 4096 elements its own program, so the copy is a memory-bound pass (a few us). Used by the portable training forwards only; the CUDA paths
keep their own packing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._compile import opaque

_BLOCK = 4096
MAX_TENSORS = 6


@triton.jit
def _seg(src, dst, n, blk, BLOCK: tl.constexpr):
    off = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(dst + off, tl.load(src + off, mask=m, other=0.0).to(dst.dtype.element_ty), mask=m)


@triton.jit
def _cast_kernel(s0, d0, n0, s1, d1, n1, s2, d2, n2, s3, d3, n3, s4, d4, n4, s5, d5, n5, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    e0 = tl.cdiv(n0, BLOCK)
    e1 = e0 + tl.cdiv(n1, BLOCK)
    e2 = e1 + tl.cdiv(n2, BLOCK)
    e3 = e2 + tl.cdiv(n3, BLOCK)
    e4 = e3 + tl.cdiv(n4, BLOCK)
    if pid < e0:
        _seg(s0, d0, n0, pid, BLOCK)
    elif pid < e1:
        _seg(s1, d1, n1, pid - e0, BLOCK)
    elif pid < e2:
        _seg(s2, d2, n2, pid - e1, BLOCK)
    elif pid < e3:
        _seg(s3, d3, n3, pid - e2, BLOCK)
    elif pid < e4:
        _seg(s4, d4, n4, pid - e3, BLOCK)
    else:
        _seg(s5, d5, n5, pid - e4, BLOCK)


def _launch(srcs, dsts):
    k = len(srcs)
    args = []
    total = 0
    for s, d in zip(srcs, dsts, strict=True):
        args += [s, d, s.numel()]
        total += triton.cdiv(s.numel(), _BLOCK)
    while len(args) < 3 * MAX_TENSORS:           # unused slots: size 0
        args += [srcs[0], dsts[0], 0]
    _cast_kernel[(total,)](*args, BLOCK=_BLOCK, num_warps=4)


def _cast_pack_fake(srcs, n_cat):
    rows = sum(s.shape[0] for s in srcs[:n_cat])
    return [srcs[0].new_empty((rows, *srcs[0].shape[1:]), dtype=torch.bfloat16),
            *(s.new_empty(s.shape, dtype=torch.bfloat16) for s in srcs[n_cat:])]


@opaque(fake=_cast_pack_fake, name="weights_cast_pack")
def cast_pack(srcs: list[torch.Tensor], n_cat: int) -> list[torch.Tensor]:
    """bf16 copies of up to six contiguous weights in ONE launch: ``[cat(srcs[:n_cat], dim=0), *srcs[n_cat:]]`` (the first ``n_cat`` share their
    trailing shape and land as row blocks of one buffer, the layout a GEMM over their concatenation reads)."""
    k = len(srcs)
    if not 1 <= n_cat <= k <= MAX_TENSORS or not all(s.is_contiguous() for s in srcs):
        raise ValueError(f"cast_pack takes 1 <= n_cat <= len(srcs) <= {MAX_TENSORS} contiguous tensors")
    first = srcs[0]
    cat = first.new_empty((sum(s.shape[0] for s in srcs[:n_cat]), *first.shape[1:]), dtype=torch.bfloat16)
    rest = [s.new_empty(s.shape, dtype=torch.bfloat16) for s in srcs[n_cat:]]
    dsts, r = [], 0
    for s in srcs[:n_cat]:
        dsts.append(cat[r:r + s.shape[0]])
        r += s.shape[0]
    _launch(srcs, dsts + rest)
    return [cat, *rest]
