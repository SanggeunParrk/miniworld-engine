"""Public entry point for the rmsnorm family.

RMSNorm is ``x / sqrt(mean(x^2) + eps) * weight`` over the last axis -- LayerNorm with the mean
removed. Two call sites in this repo were reaching for ``F.rms_norm`` directly, which is three
HBM passes and holds the normalized activation for the backward:
``modules/swa_atom_attention`` (q and k, no learnable weight) and
the triangle-attention QK norm (with one). The exported name is the autograd-aware
entry, not the ``torch.autograd.Function`` behind it.
"""

from __future__ import annotations

from miniworld_engine.kernels.rmsnorm.triton.main import (
    triton_rmsnorm,
    triton_rmsnorm_adamod,
)

__all__ = ["triton_rmsnorm", "triton_rmsnorm_adamod"]


def rms_norm(x, weight=None, eps=1e-5):
    """RMSNorm over the last axis on the kernel backend the running card takes (H100 / B200: the general CUDA RMSNorm, the PyTorch statements for a call it does not take; a card without a record: the PyTorch statements; every other card: the portable ``triton_rmsnorm``): the entry point for a caller that
    needs the op alone, without building a module."""
    import torch.nn.functional as F

    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(x):
        from miniworld_engine.kernels.rmsnorm.cuda import h100

        out = h100.rms_norm_rows(x, weight, eps)
        return out if out is not None else F.rms_norm(x, (x.shape[-1],), weight, eps)
    if x.is_cuda and portable_blocked("rmsnorm", x.device, x, *(t for t in (weight,) if t is not None)):
        return F.rms_norm(x, (x.shape[-1],), weight, eps)
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm

    return triton_rmsnorm(x, weight, eps)


def rms_norm_modulation(
    q,
    c,
    w_scale,
    w_shift,
    w_gate,
    weight=None,
    eps: float = 1e-5,
):
    """``(rmsnorm(q) * (1 + c @ w_scale^T) + c @ w_shift^T, c @ w_gate^T)`` with ``c`` the ACTIVATED conditioning, routed per card -- see
    ``kernels.rmsnorm.triton.main.triton_rmsnorm_adamod``. H100 / B200 launch no Triton: the activation-dtype statements with the fewest launches
    (``cuda/h100``). On a card without a record: the PyTorch reference (``kernels.rmsnorm.reference.rmsnorm_adamod_reference``, fp32 statistics and
    projections, the outputs in ``q``'s dtype)."""
    from miniworld_engine.kernels._device import no_triton
    from miniworld_engine.modules.dispatch import portable_blocked

    if no_triton(q):
        from miniworld_engine.kernels.rmsnorm.cuda import h100

        return h100.rms_norm_modulation(q, c, w_scale, w_shift, w_gate, weight, eps)
    if q.is_cuda and portable_blocked(
        "rms_norm_modulation", q.device, q, c, w_scale, w_shift, w_gate, *(t for t in (weight,) if t is not None)
    ):
        from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

        return rmsnorm_adamod_reference(q, c, w_scale, w_shift, w_gate, weight, eps)
    from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm_adamod

    return triton_rmsnorm_adamod(q, c, w_scale, w_shift, w_gate, weight, eps)
