"""H100 (sm_90) side of the RMSNorm entry points: the general CUDA RMSNorm on contiguous rows, and the activation-dtype statements of
``rms_norm_modulation`` (an H100 / B200 call launches no Triton). Nothing here launches Triton and nothing here is a new kernel.

``rms_norm_rows`` answers ``None`` when no CUDA kernel serves the call (another card, a dtype / layout outside the kernel's contract, a CPU tensor,
a ``torch.compile`` trace) and the caller runs the PyTorch statements.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels._device import on_h100


def rms_norm_rows(x, weight, eps):
    """RMSNorm over the last axis of a contiguous activation on the general CUDA RMSNorm (``kernels/rmsnorm/cuda``: any width, bf16 / fp16 / fp32,
    fp32 statistics, forward and backward with a bit-reproducible weight gradient), or None. The triangle-attention QK norm (rows of 32 channels)."""
    if not on_h100(x) or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float32):
        return None
    from miniworld_engine.kernels.layernorm.cuda.general import supported
    from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda

    if not supported(x):
        return None
    return rmsnorm_cuda(x, weight, eps)


_ONE_ZERO: dict = {}


def _one_plus_bias(d: int, like: torch.Tensor) -> torch.Tensor:
    """The GEMM bias [1 .. 1 | 0 .. 0 | 0 .. 0] (3 d) of ``rms_norm_modulation``: the projection leaves ``1 + scale``. Cached per (device, dtype, d)."""
    key = (like.device, like.dtype, d)
    got = _ONE_ZERO.get(key)
    if got is None:
        got = _ONE_ZERO[key] = torch.cat([like.new_ones(d), like.new_zeros(2 * d)])
    return got


def rms_norm_modulation(q, c, w_scale, w_shift, w_gate, weight, eps):
    """``(rmsnorm(q) * (1 + c Ws^T) + c Wsh^T, c Wg^T)`` in the activation's dtype with as few launches as the statements allow: one RMSNorm, one
    GEMM over [Ws; Wsh; Wg] whose bias epilogue adds the 1 of ``1 + scale`` (fp32 accumulation), one fused ``addcmul``. Replaces the fp32 statements
    of ``rmsnorm_adamod_reference`` (a dozen launches and fp32 temporaries of the whole activation). The gate comes back as a view of the GEMM result."""
    dt = q.dtype
    d = q.shape[-1]
    w_all = torch.cat([w_scale, w_shift, w_gate]).to(dt)
    proj = F.linear(c.to(dt), w_all, _one_plus_bias(d, q))
    scale1, shift, gate = proj[..., :d], proj[..., d : 2 * d], proj[..., 2 * d :]
    normed = F.rms_norm(q, (d,), None if weight is None else weight.to(dt), eps)
    return torch.addcmul(shift, normed, scale1), gate
