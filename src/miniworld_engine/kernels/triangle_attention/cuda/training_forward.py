"""Qualified H100 BF16 D32 training forward with FP32 base-2 LSE.

Unsupported inputs retain the existing implementation. No runtime compilation.
"""
from __future__ import annotations
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from miniworld_engine.kernels._compile import opaque

_ROOT=Path(__file__).resolve().parent
_EXT=None
SUPPORTED_LENGTHS=(384,768,1024)


@functools.lru_cache(maxsize=1)
def _artifact_ready():
    try:
        data=json.loads((_ROOT/'fwd_manifest.json').read_text())
        if data['python_abi']!=sys.implementation.cache_tag or data['torch']!=str(torch.__version__):
            return False
        for filename,digest in data['files'].items():
            if hashlib.sha256((_ROOT/filename).read_bytes()).hexdigest()!=digest:
                return False
        dispatcher=data['dispatcher']
        return hashlib.sha256((_ROOT/dispatcher['path']).read_bytes()).hexdigest()==dispatcher['sha256']
    except (OSError,ValueError,KeyError):
        return False


@functools.lru_cache(maxsize=None)
def _device_ready(device):
    return torch.cuda.get_device_capability(device)==(9,0)


def can_use(q,k,v,b):
    if not(q.is_cuda and q.dtype==torch.bfloat16 and q.ndim==5 and q.shape[0]==1
           and q.shape[1]==4 and q.shape[4]==32 and q.shape[2] in SUPPORTED_LENGTHS
           and q.shape[3]==q.shape[2]):
        return False
    L=q.shape[2]
    return (all(t.device==q.device and t.dtype==q.dtype and t.shape==q.shape
                and t.stride(4)==1 and t.stride(1)==32 and t.stride(3)==128
                and t.stride(2)==L*128 and t.data_ptr()%16==0 for t in (q,k,v))
            and b.device==q.device and b.dtype==q.dtype and b.shape==(1,4,L,L)
            and b.is_contiguous() and b.data_ptr()%16==0 and _artifact_ready() and _device_ready(q.device))


def extension():
    global _EXT
    if _EXT is None:
        data=json.loads((_ROOT/'fwd_manifest.json').read_text())
        spec=importlib.util.spec_from_file_location(data['module_name'],_ROOT/data['binary'])
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT


def forward(q,k,v,b):
    return tuple(extension().forward(q,k,v,b))


# --- the caller-side choice between this CUDA forward and the portable Triton one -------------------------------------------------
# The Triton forward (``triton/main.py:_tri_attn_fwd``) is pure Triton; the choice to take the CUDA forward is made HERE and by
# ``triangle_attention/interface.py``, never from inside a Triton file.
def _routed_fwd_fake(q, k, v, bias, shape_key):
    """``(out, m)`` exactly as the Triton forward allocates them (projection-layout strided ``out``, fp32 ``m``)."""
    from miniworld_engine.kernels.triangle_attention.triton.main import _tri_attn_fwd_fake
    return _tri_attn_fwd_fake(q, k, v, bias, shape_key)


@opaque(fake=_routed_fwd_fake, name="triangle_attention_fwd_routed")
def routed_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(out, m)``: this CUDA forward where :func:`can_use`, the Triton forward otherwise (the old ``_tri_attn_fwd`` behaviour)."""
    from miniworld_engine.kernels.triangle_attention.triton import main as core

    if can_use(q, k, v, bias):
        return forward(q, k, v, bias)
    return core._tri_attn_fwd(q, k, v, bias, shape_key)


@functools.lru_cache(maxsize=1)
def _function():
    from einops import reduce
    from miniworld_engine.autotune.shape_key import token_key
    from miniworld_engine.kernels.triangle_attention.triton import main as core

    class TriangleAttentionCudaForwardFunction(torch.autograd.Function):
        """``TritonTriangleAttentionPairBiasFunction`` with the forward routed to this CUDA kernel (same saved tensors, same Triton backward)."""

        @staticmethod
        def forward(ctx, q, k, v, bias):
            bias = bias.contiguous()
            out, m = routed_fwd(q, k, v, bias, token_key(q.shape[2]))
            ctx.save_for_backward(q, k, v, bias, m, out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, bias, m, out = ctx.saved_tensors
            if grad_output.dtype != q.dtype:
                grad_output = grad_output.to(q.dtype)
            L = q.shape[2]
            dq, dk, dv, dbias = core._tri_attn_bwd(q, k, v, bias, m, out, grad_output, token_key(L))
            dbias = reduce(dbias, "B (H L3) L L2 -> B H L L2", "sum", L3=L)
            return dq, dk, dv, dbias

    return TriangleAttentionCudaForwardFunction


def pair_bias_attention(q, k, v, bias):
    """The autograd entry with the CUDA forward; the caller checks :func:`can_use` (and ``force_portable_kernels``) first."""
    return _function().apply(q, k, v, bias)
