"""Bias-only attention on A100 (sm_80), hand CUDA: ``out[b, h, t, m, :] = sum_n softmax_n(bias[b, h, m, :])[n] v[b, h, t, n, :]`` -- the hand-written twin of ``triton/main.py``.

The logits ARE the pair bias (no query, no key), so one softmax ``P [L, L]`` per (batch, head) plane serves the ``t`` axis, and the product ``P v_t`` is the bias-only token DiT's
attention core: the ``t`` slices are the SAMPLES of ``kernels/bias_only_dit/cuda/sm80`` (``pv_gate``: one CTA = 128 query rows of a plane x a group of samples that share the P tile,
``mma.sync`` bf16 -> fp32, ``cp.async`` ring) with the head PLANES ``[L (t), L (n), D]`` of v / out one after the other instead of the token DiT's head columns.

    forward   P = softmax(bias) (the family's CUDA row softmax, bf16 like the Triton kernel's probabilities);  out = P v, one launch over every plane
    backward  P and P^T (``softmax_t``);  dv = P^T dout (the same core on the transposed P);  D[t, h, m] = sum_d dout out (``delta_planes``);
              dbias[h, m, n] = P (sum_t dout v^T - sum_t D)  (``dpb_planes``: the sum over t is the K loop of one GEMM per plane, so dbias needs no per-t partials)

``serves`` is the gate: A100 (sm_80), bf16, v ``[B, H, L, L, D]`` with D 32 / 48 / 64 and L a multiple of 128 up to 1024, bias ``[B, H, L, L]``; ``MINIWORLD_BIAS_ONLY_ATTN_SM80=0`` turns the
path off (the family's ``bias_only_attention`` door then takes Triton). One autograd Function whose forward and backward are each one opaque op.
"""

from __future__ import annotations

import os

import torch

from miniworld_engine.kernels._compile import opaque

HEAD_DIMS = (32, 48, 64)


def serves(v: torch.Tensor, bias: torch.Tensor) -> bool:
    if os.environ.get("MINIWORLD_BIAS_ONLY_ATTN_SM80", "1") == "0" or not v.is_cuda or v.dtype is not torch.bfloat16 or bias.dtype is not torch.bfloat16:
        return False
    if v.ndim != 5 or bias.ndim != 4 or v.shape[2] != v.shape[3] or tuple(bias.shape) != tuple(v.shape[:4]) or v.shape[-1] not in HEAD_DIMS:
        return False
    length = v.shape[2]
    if length % 128 or length > 1024:
        return False
    if torch.cuda.get_device_capability(v.device) != (8, 0):
        return False
    return _loads()


#: a training call (autograd on) below this L keeps the Triton kernels: at L = 128 the CUDA backward (softmax, two cores, the row term, the bias-gradient core) loses to Triton (0.172 against 0.109 ms, 2026-10-04,
#: ``bench.py target=bias_only_attention``); every other L wins (training 1.7-5.2x Triton from L = 256, inference at every L).  ``MINIWORLD_BIAS_ONLY_ATTN_SM80=all`` takes every call ``serves`` accepts.
MIN_TRAIN_LENGTH = 256


def wanted(v: torch.Tensor, bias: torch.Tensor) -> bool:
    """The door's choice: ``serves`` and not a small training call (see ``MIN_TRAIN_LENGTH``)."""
    if not serves(v, bias):
        return False
    if os.environ.get("MINIWORLD_BIAS_ONLY_ATTN_SM80", "1") == "all":
        return True
    return not (torch.is_grad_enabled() and (v.requires_grad or bias.requires_grad) and v.shape[2] < MIN_TRAIN_LENGTH)


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the extensions; False, with one warning, when the toolchain fails (the Triton path then serves).  A process-level constant for ``torch.compile``."""
    global _FAILED
    if _FAILED:
        return False
    try:
        from miniworld_engine.kernels.bias_only_dit import cuda as rows
        from miniworld_engine.kernels.bias_only_dit.cuda import sm80 as core

        rows._ext()
        core._ext()
    except Exception as exc:  # a toolchain problem keeps the Triton path
        import warnings

        _FAILED = True
        warnings.warn(f"sm_80 bias-only attention kernels unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


_FAILED = False


def _fwd_fake(v, bias):
    """The output of ``_fwd``: a fresh tensor shaped and typed like ``v``."""
    return torch.empty_like(v)


@opaque(fake=_fwd_fake, name="bias_only_attention_sm80_fwd")
def _fwd(v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``out = softmax(bias) v`` over v [B, H, L, L, D] (bf16), bias [B, H, L, L]: a fresh tensor shaped like v."""
    from miniworld_engine.kernels.bias_only_dit import cuda as rows
    from miniworld_engine.kernels.bias_only_dit.cuda import sm80 as core

    b, h, length, _, d = v.shape
    planes = b * h
    p = torch.empty(planes * length, length, device=v.device, dtype=torch.bfloat16)
    rows.softmax_rows(bias.contiguous().view(planes * length, length), p, None)
    out = torch.empty_like(v)
    core.pv_planes(v.contiguous().view(planes * length * length, d), p, out.view(planes * length * length, d), length, planes, d)
    return out


def _bwd_fake(v, bias, out, dout):
    """``[dv, dbias]`` of ``_bwd``: fresh tensors shaped and typed like ``v`` and ``bias``."""
    return [torch.empty_like(v), torch.empty_like(bias)]


@opaque(fake=_bwd_fake, name="bias_only_attention_sm80_bwd")
def _bwd(v: torch.Tensor, bias: torch.Tensor, out: torch.Tensor, dout: torch.Tensor) -> list[torch.Tensor]:
    """``[dv, dbias]`` of ``_fwd`` (bf16, shaped like v and bias)."""
    from miniworld_engine.kernels.bias_only_dit import cuda as rows
    from miniworld_engine.kernels.bias_only_dit.cuda import sm80 as core

    b, h, length, _, d = v.shape
    planes = b * h
    dev = v.device
    p = torch.empty(planes * length, length, device=dev, dtype=torch.bfloat16)
    pt = torch.empty_like(p)
    rows.softmax_t(bias.contiguous().view(planes * length, length), p, pt, None)
    v2, do2, o2 = (t.contiguous().view(planes * length * length, d) for t in (v, dout, out))
    dv = torch.empty_like(v)
    core.pv_planes(do2, pt, dv.view(planes * length * length, d), length, planes, d)           # dv[t, n] = sum_m P[m, n] dout[t, m]
    dd = torch.empty(length, planes, length, device=dev, dtype=torch.float32)
    core.delta_planes(do2, o2, dd, length, length, planes, d)
    dbias = torch.empty(planes * length, length, device=dev, dtype=torch.bfloat16)
    core.dpb_planes(do2, v2, p, dd, dbias, length, planes, d)
    return [dv, dbias.view(b, h, length, length)]


class BiasOnlyAttentionSm80Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, bias):
        out = _fwd(v, bias)
        ctx.save_for_backward(v, bias, out)
        return out

    @staticmethod
    def backward(ctx, dout):
        v, bias, out = ctx.saved_tensors
        dv, dbias = _bwd(v, bias, out, dout.contiguous().to(v.dtype))
        return dv if ctx.needs_input_grad[0] else None, dbias if ctx.needs_input_grad[1] else None


def bias_only_attention_sm80(v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The hand-CUDA bias-only attention (call ``serves`` first); differentiable in v and bias."""
    if torch.is_grad_enabled() and (v.requires_grad or bias.requires_grad):
        return BiasOnlyAttentionSm80Function.apply(v, bias)
    return _fwd(v, bias)


__all__ = ["MIN_TRAIN_LENGTH", "bias_only_attention_sm80", "serves", "wanted"]
