"""cuBLAS GEMMs over a long reduction (weight gradients over pair / MSA / atom rows) that keep every row of the reduction.

cuBLAS 12.9 (torch 2.13 cu129) on B200 drops the last ``K % 64`` rows of the reduction in some of its bf16 / fp16 split-K GEMMs
(``nvjet_tss_*_splitK_*`` + ``splitKreduce_kernel``, which its heuristics pick for a long K over a small M x N: the weight gradients
dW = dY^T X over B L^2 pair rows, S N MSA rows or the atoms). Measured 2026-10-29 against fp64 of the same operands: the error matrix
has rank ``K % 64`` and is exactly those rows' outer products (0.1-2.7 % relative error); about 3 % of the K values swept between
2 049 and 1.2 M were hit, at every M x N tried (1 x 128 to 512 x 512), for both the weight-gradient (TN) and the row-major (NN)
layout, bf16 and fp16 alike, with fp32 or bf16 output; never a K that is a multiple of 64, never below K = 2 049, never an fp32 /
TF32 GEMM nor ``bmm``. ``allow_bf16_reduced_precision_reduction``, ``CUBLAS_WORKSPACE_CONFIG`` and the cuBLASLt backend do not
change it. A pair reduction B L^2 is a multiple of 64 whenever L is a multiple of 8, so MiniWorld's bucketed crops never hit it;
FoldForge's crops (any L) and ragged atom / MSA counts do.

``mm`` runs the multiple-of-64 part of the reduction on cuBLAS with an fp32 result and adds the last ``K % 64`` rows as a second
(short, never split) GEMM with an fp32 result: one small GEMM and one add more, only for a reduction that is long and not a multiple
of 64; anything else is the plain call.
"""

from __future__ import annotations

import torch

#: the K granularity the split-K kernels keep (a multiple of it is exact)
TAIL = 64
#: shorter reductions never reach the split-K kernels (first failure seen at K = 2 049); a margin below it
MIN_K = 1024
_LOW = (torch.bfloat16, torch.float16)


def _mm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype | None) -> torch.Tensor:
    if out_dtype is None or out_dtype == a.dtype:
        return torch.mm(a, b)
    try:
        return torch.mm(a, b, out_dtype=out_dtype)
    except (TypeError, RuntimeError):  # a torch without mm(out_dtype=): the result rounds through the operand dtype
        return torch.mm(a, b).to(out_dtype)


def ragged(k: int, dtype: torch.dtype) -> bool:
    """Whether a reduction of length ``k`` in ``dtype`` needs the tail split (a long bf16 / fp16 reduction not a multiple of 64)."""
    return dtype in _LOW and k >= MIN_K and k % TAIL != 0


def mm(a: torch.Tensor, b: torch.Tensor, *, out_dtype: torch.dtype | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` (a [M, K], b [K, N], any strides) on cuBLAS without the split-K tail drop.

    ``out_dtype``: the result dtype (``torch.float32`` for bf16 / fp16 operands: fp32 accumulation, unrounded; default the operands'
    dtype). ``out``: write the result there (its dtype wins). A long bf16 / fp16 reduction that is not a multiple of 64 runs as the
    multiple-of-64 head plus the last ``K % 64`` rows, both on cuBLAS with fp32 results; anything else is the plain call."""
    if out is not None:
        out_dtype = out.dtype
    k = a.shape[1]
    if not ragged(k, a.dtype):
        if out is None:
            return _mm(a, b, out_dtype)
        if out_dtype == a.dtype:
            return torch.mm(a, b, out=out)
        return out.copy_(_mm(a, b, out_dtype))
    h = k - k % TAIL
    y = _mm(a[:, :h], b[:h], torch.float32)
    y += _mm(a[:, h:], b[h:], torch.float32)  # K % 64 < 64 rows: never split, fp32 accumulation of the operands as they are
    if out is not None:
        return out.copy_(y)
    if out_dtype is None:
        out_dtype = a.dtype
    return y if out_dtype == torch.float32 else y.to(out_dtype)
