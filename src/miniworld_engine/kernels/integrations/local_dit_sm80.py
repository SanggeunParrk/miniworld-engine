"""A100 AF3 local attention. Surrounding AdaLN/projections/transition keep their measured paths.

settings.local_dit_sm80=False retains the previous module composition for comparisons.
The core and pair bias are own CUDA; the surrounding GEMMs remain separate backend work.
"""

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels._compile import device_constant, opaque


@device_constant
def _ready(index):
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    sm80_atom_local.ready(index)
    return True


def serves(block, single, cond, pair, mask):
    if (
        not settings.current().local_dit_sm80
        or settings.current().engine_backend == "triton"
    ):
        return False
    att = block.attention
    if att.use_qk_norm:
        return False
    if not single.is_cuda or torch.cuda.get_device_capability(single.device) != (8, 0):
        return False
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[-1] != 128
        or single.shape[2] < 1
    ):
        return False
    if (single.dtype, cond.dtype, pair.dtype) != (torch.bfloat16,) * 3:
        return False
    n = single.shape[2]
    if cond.shape != single.shape or pair.shape != (1, (n + 31) // 32, 32, 128, 16):
        return False
    if (
        cond.device != single.device
        or pair.device != single.device
        or not pair.is_contiguous()
    ):
        return False
    if att.n_head != 4 or att.to_query.weight.shape != (128, 128):
        return False
    if att.ln_pair.eps != 1e-5:
        return False
    if any(
        t.dtype not in (torch.bfloat16, torch.float32) or not t.is_contiguous()
        for t in (att.ln_pair.weight, att.to_bias.weight)
    ):
        return False
    if mask is not None and (
        mask.shape != (1, n)
        or mask.dtype != torch.bool
        or mask.device != single.device
        or not mask.is_contiguous()
    ):
        return False
    return _ready(
        single.device.index
        if single.device.index is not None
        else torch.cuda.current_device()
    )


def _attn_fwd_fake(q, k, v, bias, mask):
    return torch.empty_like(q), torch.empty(
        q.shape[0], 4, q.shape[1], device=q.device, dtype=torch.float32
    )


@opaque(fake=_attn_fwd_fake, name="local_attn_sm80_fwd")
def _attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    return sm80_atom_local.attn_fwd(q, k, v, bias, None if mask.numel() == 0 else mask)


def _attn_bwd_fake(q, k, v, dout, bias, out, lse, mask):
    # Native backward writes three views of a 512-column workspace; expose matching strides.
    packed = torch.empty(q.shape[0] * q.shape[1], 512, device=q.device, dtype=q.dtype)
    return packed, torch.empty_like(bias)


@opaque(fake=_attn_bwd_fake, name="local_attn_sm80_bwd")
def _attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    return sm80_atom_local.attn_bwd(
        q,
        k,
        v,
        dout.contiguous(),
        bias,
        out,
        lse,
        None if mask.numel() == 0 else mask,
        packed_output=True,
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask):
        out, lse = _attn_fwd(q, k, v, bias, mask)
        ctx.save_for_backward(q, k, v, bias, out, lse, mask)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, bias, out, lse, mask = ctx.saved_tensors
        packed, dbias = _attn_bwd(q, k, v, dout, bias, out, lse, mask)
        dq, dk, dv = (
            packed[:, offset : offset + 128].view_as(q) for offset in (0, 128, 256)
        )
        return dq, dk, dv, dbias, None


def _bias_fwd_fake(z, gamma, wb):
    return torch.empty(4, z.shape[0], 32, 128, device=z.device, dtype=torch.float32)


@opaque(fake=_bias_fwd_fake, name="local_pair_bias_sm80_fwd")
def _bias_fwd(z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    return sm80_atom_local.pair_bias_fwd(z, gamma, wb)


def _bias_bwd_fake(z, gamma, wb, dbias):
    return torch.empty_like(z), torch.empty_like(gamma), torch.empty_like(wb)


@opaque(fake=_bias_bwd_fake, name="local_pair_bias_sm80_bwd")
def _bias_bwd(
    z: torch.Tensor, gamma: torch.Tensor, wb: torch.Tensor, dbias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    dz, dgamma, dwb = sm80_atom_local.pair_bias_bwd(z, gamma, wb, dbias.contiguous())
    return dz, dgamma.to(gamma.dtype), dwb.to(wb.dtype)


class _PairBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, gamma, wb):
        ctx.save_for_backward(z, gamma, wb)
        return _bias_fwd(z, gamma, wb)

    @staticmethod
    @once_differentiable
    def backward(ctx, dbias):
        return _bias_bwd(*ctx.saved_tensors, dbias)


def _gate_fwd_fake(gate, x):
    return torch.empty_like(x), torch.empty_like(x)


@opaque(fake=_gate_fwd_fake, name="local_gate_sm80_fwd")
def _gate_fwd(gate: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    return sm80_atom_local.gate_fwd(gate, x)


def _gate_bwd_fake(dy, sigmoid, x):
    return torch.empty_like(x), torch.empty_like(x)


@opaque(fake=_gate_bwd_fake, name="local_gate_sm80_bwd")
def _gate_bwd(
    dy: torch.Tensor, sigmoid: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    return sm80_atom_local.gate_bwd(dy.contiguous(), sigmoid, x)


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, x):
        out, sigmoid = _gate_fwd(gate, x)
        ctx.save_for_backward(sigmoid, x)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        return _gate_bwd(dy, *ctx.saved_tensors)


def attention_delta(block, single, cond, pair, mask):
    att = block.attention
    samples, _, n, _ = single.shape
    x = att.ada_ln_in(single, cond)
    q = att.to_query(x).view(samples, n, 128)
    xkv = att.ada_ln_kv(x, cond) if block.cross_attention else x
    k, v = (
        att.to_key(xkv).view(samples, n, 128),
        att.to_value(xkv).view(samples, n, 128),
    )
    bias = _PairBias.apply(pair[0], att.ln_pair.weight, att.to_bias.weight)
    keymask = (
        torch.empty(0, device=single.device, dtype=torch.bool)
        if mask is None
        else mask.view(n)
    )
    out = _Attention.apply(q, k, v, bias, keymask).view_as(single)
    out = att.to_out(_Gate.apply(att.to_gate(x), out))
    return _Gate.apply(att.to_scale(cond), out)


from miniworld_engine.modules.primitives import Linear as _OriginalProjectionLinear


class _ProjectionLinear(_OriginalProjectionLinear):
    """Original one-argument Linear, with explicit typed inference operands.

    The same module and Parameter objects keep their names, hooks, insertion
    order and checkpoint storage. The default call retains F.linear exactly.
    """

    def forward(self, value, weight=None, bias=None):
        import torch.nn.functional as F

        if weight is None:
            return F.linear(value, self.weight, self.bias)
        assert not torch.is_grad_enabled()
        return F.linear(value, weight, bias)


def prepare_projection_modules(attention):
    # Host constructor preparation: no tensor allocations, parameter writes,
    # RNG calls, forward monkeypatch or GPU operation. B200/default Linear
    # calls retain their original one-argument F.linear computation.
    for name in ("to_query", "to_key", "to_value", "to_gate", "to_out", "to_scale"):
        linear = getattr(attention, name)
        assert type(linear) in (_OriginalProjectionLinear, _ProjectionLinear)
        linear.__class__ = _ProjectionLinear


def inference_attention_current(block, single, cond, pair, mask):
    from miniworld_engine.kernels.augmented_attention.cuda.sm80_atom_local_post import (
        _GateResidual,
    )
    from miniworld_engine.kernels.augmented_attention.cuda.sm80_atom_local_projection import (
        projection_parameters,
    )

    att = block.attention
    samples, _, n, _ = single.shape
    wq, bq, wk, wv, wg, wo, ws, bs = projection_parameters(att)
    x = att.ada_ln_in(single, cond)
    q = att.to_query(x, wq, bq).view(samples, n, 128)
    xkv = att.ada_ln_kv(x, cond) if block.cross_attention else x
    k, v = (
        att.to_key(xkv, wk).view(samples, n, 128),
        att.to_value(xkv, wv).view(samples, n, 128),
    )
    bias = _PairBias.apply(pair[0], att.ln_pair.weight, att.to_bias.weight)
    keymask = (
        torch.empty(0, device=single.device, dtype=torch.bool)
        if mask is None
        else mask.view(n)
    )
    out = _Attention.apply(q, k, v, bias, keymask).view_as(single)
    out = att.to_out(_Gate.apply(att.to_gate(x, wg), out), wo)
    return _GateResidual.apply(single, att.to_scale(cond, ws, bs), out, False, 0)


def inference_block(block, single, cond, pair, mask=None):
    assert not torch.is_grad_enabled()
    current = inference_attention_current(block, single, cond, pair, mask)
    return block.transition(current, cond)
