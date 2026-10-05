"""The ``projected_attention`` leaf of the model code on an A100 (sm_80): ``ops.augmented_attention_pair_bias`` at the triangle-attention row geometry, through the generalised
triangle-attention core (``sm80_core2.py``, hand-written CUDA), inference and training.

    out[a, b, h, i] = softmax_j(D^-1/2 q[a, b, h, i] . k[a, b, h, j] + bias[b, h, i, j]   (-inf where mask[a, b, j] is False)) v[a, b, h, j]          q / k / v / out [A, B, H, L, D]

The pair rows ``a`` are the L tokens of one pair stack (``A == L``: "the triangle rows are independent augmentation streams sharing one pair bias"), the head dim is 16 or 32, L a multiple of
128.  The operands are read in place through their element strides (a head-major tensor and a permuted view of a token-major projection output alike: no transposing copies), the key mask is
the op's per-row ``[A, B, L]`` mask (a row's keys as bits, a key tile without a masked key skips the masking arithmetic), the pair bias ``[B, H, L, L]`` is shared by the rows.  The backward is
the core's backward (``sm80_core2.backward``: dq / dk / dv, the pair bias' gradient summed over the rows from bf16 partials in a fixed order) after a row-term kernel (``delta = sum_d o . do``).

Shapes the core does not serve (other row counts, head dims, dtypes, a bias in fp32, ``kernel_type="memory_efficient"``) keep the op's Triton path: ``serves()`` is the whole gate.
``MINIWORLD_TRIATTN_SM80=0`` turns the path off (the switch of the triangle-attention kernels).
"""

import os

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine import settings

from ..._compile import opaque
from . import sm80 as _core
from . import sm80_core2 as _core2

BF = torch.bfloat16
F32 = torch.float32
#: head dims of the core (the registry's 4 x 64 is 16, the others 32); the rows are the L tokens, L a multiple of 128
HEAD_DIMS = (16, 32)
MAX_L = 8192


def enabled() -> bool:
    return os.environ.get("MINIWORLD_TRIATTN_SM80", "1") != "0"


def _needs_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def serves(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None = None) -> bool:
    """The path's whole gate: sm_80 (exactly), bf16 ``[A, B, H, L, D]`` operands with ``A == L`` (a multiple of 128, at most ``MAX_L``) and D = 16 or 32, a bf16 bias ``[B, H, L, L]``, a bool
    ``[A, B, L]`` key mask if any, the engine's backend not forced to Triton, ``MINIWORLD_TRIATTN_SM80`` not 0; a gradient wanted by an operand needs the bias-gradient partials to fit
    ``sm80_core2.MAX_DBP_BYTES``."""
    if not enabled() or settings.current().engine_backend == "triton":
        return False
    if query.ndim != 5 or not query.is_cuda or query.dtype is not BF or key.dtype is not BF or value.dtype is not BF or bias.dtype is not BF:
        return False
    if key.shape != query.shape or value.shape != query.shape or key.device != query.device or value.device != query.device or bias.device != query.device:
        return False
    a, b, h, length, d = query.shape
    if d not in HEAD_DIMS or a != length or length % 128 or not 128 <= length <= MAX_L or b * h > 65535 or tuple(bias.shape) != (b, h, length, length):
        return False
    if mask is not None and (mask.dtype is not torch.bool or tuple(mask.shape) != (a, b, length) or mask.device != query.device):
        return False
    if _needs_grad(query, key, value, bias) and not _core2.dbp_fits(length, h, b):
        return False
    return _core._is_ampere(query.device.index if query.device.index is not None else torch.cuda.current_device()) and _core2.loads()


def _lays(t: torch.Tensor) -> tuple[int, int, int, int]:
    """The core's layout of an ``[A, B, H, L, D]`` tensor: the element strides of (token, pair row, batch, head)."""
    return (t.stride(3), t.stride(0), t.stride(1), t.stride(2))


def _readable(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself when the core can read it where it is (unit channel stride, the other strides multiples of 8 elements: 16-byte granules), else a contiguous copy.  Strides only: the
    storage offset of a view is checked inside the ops (``_aligned``), where the tensors are real."""
    st = t.stride()
    if st[4] == 1 and all(s % 8 == 0 for s in st[:4]):
        return t
    return t.contiguous()


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself when its first element is 16-byte aligned (a storage offset that is a multiple of 8 bf16 elements; an allocation is 512-byte aligned), else a fresh copy."""
    return t if t.storage_offset() % 8 == 0 else t.clone(memory_format=torch.contiguous_format)


def _mask_rows(mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """The ``[A, B, L]`` bool key mask as the core's per-row uint8 ``[B, A, L]`` (reinterpreted, a copy only when B > 1); empty = no mask."""
    if mask is None:
        return like.new_empty((0,), dtype=torch.uint8)
    return mask.transpose(0, 1).contiguous().view(torch.uint8)


def _forward_fake(q, k, v, bias, mask, save):
    """[out in q's shape and strides, lse [B, H, A, L] fp32 when ``save`` (else empty)]: fresh tensors."""
    a, b, h, length, _ = q.shape
    return [torch.empty_like(q), q.new_empty((b, h, a, length), dtype=F32) if save else q.new_empty((0,), dtype=F32)]


@opaque(fake=_forward_fake, name="triangle_attention_sm80p_forward")
def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, save: bool) -> list[torch.Tensor]:
    """The core's forward on ``[A, B, H, L, D]`` operands in place; ``bias`` ``[B, H, L, L]`` bf16 contiguous, ``mask`` ``[B, A, L]`` uint8 or empty.  Returns [out, lse] (lse: the base-2
    log-sum-exp the backward reads, empty unless ``save``)."""
    a, b, h, length, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, h, a, length), dtype=F32, device=q.device) if save else q.new_empty((0,), dtype=F32)
    q, k, v, bias = _aligned(q), _aligned(k), _aligned(v), _aligned(bias)
    _core2.forward(q, k, v, bias, mask, out, lse, (_lays(q), _lays(k), _lays(v), _lays(out)), length, h, b, d, _core2.default_scale(d))
    return [out, lse]


def _backward_fake(q, k, v, bias, mask, out, lse, dout):
    """[dq, dk, dv in q / k / v's shapes and strides, db in the bias' shape]: fresh tensors."""
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), bias.new_empty(bias.shape)]


@opaque(fake=_backward_fake, name="triangle_attention_sm80p_backward")
def _backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, dout: torch.Tensor) -> list[torch.Tensor]:
    """The gradients of ``_forward``: [dq, dk, dv, db] (db ``[B, H, L, L]`` bf16: the sum over the pair rows of dS)."""
    a, b, h, length, d = q.shape
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    q, k, v, bias, dout = _aligned(q), _aligned(k), _aligned(v), _aligned(bias), _aligned(dout)
    delta = torch.empty((b, h, a, length), dtype=F32, device=q.device)
    _core2.delta_rows(out, dout, delta, (_lays(out), _lays(dout)), length, h, b, d)
    lays = (_lays(q), _lays(k), _lays(v), _lays(dout), _lays(dq), _lays(dk), _lays(dv))
    db = _core2.backward(q, k, v, dout, bias, mask, lse, delta, dq, dk, dv, lays, length, h, b, d, _core2.default_scale(d))
    return [dq, dk, dv, db]


class _Leaf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask):
        out, lse = _forward(q, k, v, bias, mask, True)
        ctx.save_for_backward(q, k, v, bias, mask, out, lse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, bias, mask, out, lse = ctx.saved_tensors
        dq, dk, dv, db = _backward(q, k, v, bias, mask, out, lse, _readable(dout))
        return dq, dk, dv, db, None


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``softmax(D^-1/2 q k^T + bias [masked keys -inf]) v`` for ``[A, B, H, L, D]`` operands (``A == L``), the bias ``[B, H, L, L]`` shared by the rows, the key mask ``[A, B, L]`` bool:
    returns ``[A, B, H, L, D]`` (in the strides of ``query``).  Autograd-aware (q, k, v and the bias).  Call ``serves()`` first."""
    q, k, v = _readable(query), _readable(key), _readable(value)
    bias = bias.contiguous()
    mk = _mask_rows(mask, query)
    if _needs_grad(q, k, v, bias):
        return _Leaf.apply(q, k, v, bias, mk)
    return _forward(q, k, v, bias, mk, False)[0]
