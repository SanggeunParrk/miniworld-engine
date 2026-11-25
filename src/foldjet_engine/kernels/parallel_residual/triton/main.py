"""The fused residual add of parallel branches: ``out = x + sum_i (y_i - x)`` in ONE pass (``modules/residual.py``).

Each branch ``y_i`` is the output of a part that owns its residual (``y_i = x + delta_i``), so the stage's update is the sum of the deltas.
PyTorch spells it as ``2 n`` elementwise kernels (n subtractions, n additions), each reading and writing the whole tensor; this kernel reads
``x`` and the ``n`` branches once and writes the output once (``n + 2`` tensor passes instead of ``~6 n``), with the differences and the sum
in fp32 and one rounding to the output dtype at the end (``y_i - x`` of two bf16 values close to each other is exact in fp32).

Elementwise over the flat contiguous tensors: no shape dependence, so no launch-config table; one block of ``BLOCK`` elements per
program (``BLOCK`` 2048, 8 warps: 8 bf16 / 4 fp32 elements per thread per load, 128-bit accesses).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The most branches one launch reads (Chai-1's pair stage has three; more are refused by the module).
MAX_BRANCHES = 4
_BLOCK = 2048
_WARPS = 8


@triton.jit
def parallel_residual_fwd_kernel(X, Y0, Y1, Y2, Y3, OUT, numel, NB: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    acc = tl.load(Y0 + offs, mask=m, other=0.0).to(tl.float32) - x
    if NB > 1:
        acc += tl.load(Y1 + offs, mask=m, other=0.0).to(tl.float32) - x
    if NB > 2:
        acc += tl.load(Y2 + offs, mask=m, other=0.0).to(tl.float32) - x
    if NB > 3:
        acc += tl.load(Y3 + offs, mask=m, other=0.0).to(tl.float32) - x
    tl.store(OUT + offs, (x + acc).to(OUT.dtype.element_ty), mask=m)


def parallel_residual_fwd(x: torch.Tensor, branches: list[torch.Tensor]) -> torch.Tensor:
    """``x + sum_i (y_i - x)`` for 1 .. ``MAX_BRANCHES`` contiguous tensors of ``x``'s shape, dtype and device (the module checks)."""
    nb = len(branches)
    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return out
    ys = [*branches, *[x] * (MAX_BRANCHES - nb)]  # unused slots: never read (NB is a constexpr)
    with torch.cuda.device(x.device):
        parallel_residual_fwd_kernel[(triton.cdiv(numel, _BLOCK),)](x, *ys, out, numel, NB=nb, BLOCK=_BLOCK, num_warps=_WARPS)
    return out
