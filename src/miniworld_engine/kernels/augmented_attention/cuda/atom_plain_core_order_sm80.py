"""E237 independent original plain backward with CTA-order-only dQ/dKV."""
import math
import torch
from .sm80 import _ext,_penalty,_dq_schedule,_dkv_schedule,HEAD_DIMS,_L2E

def plain_backward_core_order(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dob: torch.Tensor, bias: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, samples: int,
                   length: int, heads: int, head_dim: int, *, db_natural: bool = True, sm_scale: float | None = None, kpen: torch.Tensor | None = None, kps: int = 0,
                   dq: torch.Tensor | None = None, dk: torch.Tensor | None = None, dv: torch.Tensor | None = None, dq_order: int = 0, dkv_order: int = 0):
    """The backward of ``plain_forward``: ``dob`` the output's gradient bf16 ``[A L, H hd]``, ``o`` / ``lse`` the forward's, ``bias`` in raw units as there.  Returns
    ``(dq, dk, dv, db)``: bf16 ``[A L, H hd]`` each and the bias gradient fp32 ``[H, L, L]`` (the sum over the samples; masked keys 0) -- with respect to the
    NATURAL-unit bias (``db_natural``), or with respect to the raw-unit ``bias`` passed in (``db_natural=False``: times sm_scale).  A replay is bit-identical.
    ``kpen`` / ``kps``: as for :func:`plain_forward`.  ``dq`` / ``dk`` / ``dv``: write the gradients into these bf16 ``[A L, >= H hd]`` views (e.g. the first three
    column blocks of the projection's gradient) instead of new tensors."""
    assert head_dim in HEAD_DIMS or (head_dim == 16 and samples == 1 and kpen is None), head_dim
    ext = _ext()
    from .atom_core_order_sm80 import _ext as order_ext
    core = order_ext()
    assert (samples,head_dim,length)==(48,32,length) and length in (4096,8192)
    assert dq_order in (0,1,2,3,4,6) and dkv_order in (0,1,2,3,4,6)
    dev, m, d = q.device, samples * length, heads * head_dim
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    scl = sm_scale * _L2E
    delta = torch.empty(samples, heads, length, device=dev, dtype=torch.float32)
    ext.attn_delta(dob, o, delta, samples, length, heads, head_dim)
    dq, dk, dv = (torch.empty(m, d, device=dev, dtype=torch.bfloat16) if t is None else t for t in (dq, dk, dv))
    pen, kps = _penalty(kpen, kps, length)
    rows, nkv, minb = _dq_schedule(samples)
    groups = samples // rows
    dbp = torch.empty(groups, heads, length, length, device=dev, dtype=torch.bfloat16)
    core.attn_bwd_dq(q, k, v, dob, bias, lse, delta, dq, dbp, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dq.stride(0),
                    scl, _L2E, sm_scale, rows, nkv, minb, 0, pen, kps, dq_order)
    db = torch.empty(heads, length, length, device=dev, dtype=torch.float32)
    ext.db_reduce(dbp, db, groups, 1.0 if db_natural else sm_scale)
    bias_t = torch.empty_like(bias)
    ext.bias_transpose(bias, bias_t, length)
    core.attn_bwd_dkv(q, k, v, dob, bias_t, lse, delta, dk, dv, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dk.stride(0),
                     dv.stride(0), scl, _L2E, sm_scale, *_dkv_schedule(head_dim), 0, pen, kps, dkv_order)
    return dq, dk, dv, db
