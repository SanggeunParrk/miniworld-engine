"""E237 private Dense atom AA backward: unchanged E187 body, only dQ/dKV CTA ownership order."""
from __future__ import annotations
import functools
import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque
from .augattn_sm80 import (_bwd_fake,_sm80,_pad,_pack,_stage,_penalties,_kind,_apb,_generic_kernel,_mm32,_unstage,EPS,BF,F32)

@opaque(fake=_bwd_fake, name="dense_atom_core_order_e237_bwd")
def _bwd(dout: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, kmask: torch.Tensor | None, params: list[torch.Tensor], saved: list[torch.Tensor],
         a: int, batch: int, length: int, heads: int, mask_kind: int) -> list[torch.Tensor]:
    """The backward of ``_fwd`` (``save=True``): [d x, d cond, d pair, *d params], each in its input's dtype (the residual's gradient is dout itself: the caller returns it)."""
    sm80 = _sm80()
    qkvg, o, lse, og, y, g2, bias = saved
    dt = x.dtype
    f32 = torch.float32
    d = x.shape[-1]
    hd = d // heads
    padded = _pad(length)
    m = a * batch * padded
    n = a * padded
    dev = x.device
    dp = pair.shape[-1]
    lnw, wb = params[8], params[9]
    wo, ws = params[5].to(dt), params[6].to(dt)
    pack = _pack(params, heads, dt)
    wqkvg, wf, w32 = pack[0], pack[2], pack[4]
    xs, cs = _stage(x, batch, padded), _stage(cond, batch, padded)
    ds = _stage(dout, batch, padded)
    dy = torch.empty(m, d, device=dev, dtype=dt)
    dg2 = torch.empty(m, d, device=dev, dtype=dt)
    sm80.res_gate_bwd(ds, y, g2, dy, dg2)
    dwo = _mm32(dy.t(), og)
    dob = torch.empty(m, d, device=dev, dtype=dt)
    dqkvg = torch.empty(m, 4 * d, device=dev, dtype=dt)
    assert (a,batch,length,d,heads,dt)==(48,1,length,128,4,BF) and length in (4096,8192)
    from miniworld_engine.kernels.augmented_attention.cuda import atom_post_grad_rolled_sm80
    # E168 cfg3: physical Wo within CTA, DOG first rounds BF, exact original F32 gate arithmetic.
    atom_post_grad_rolled_sm80.run(dy,wo,o,qkvg[:,3*d:],dob,dqkvg[:,3*d:],None,3)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    kind = _kind(dp, heads, dt)
    acc = torch.zeros(_apb().acc_n(heads, 384), device=dev) if kind == "apb" else None
    dzs, dw_sum = [], None
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        from miniworld_engine.kernels.augmented_attention.cuda.atom_plain_core_order_sm80 import plain_backward_core_order
        assert dt is BF
        core_bwd = functools.partial(plain_backward_core_order, db_natural=True, dq_order=settings.current().dense_core_dq_order, dkv_order=settings.current().dense_core_dkv_order)
        _, _, _, db = core_bwd(qkvg[rows, :d], qkvg[rows, d:2 * d], qkvg[rows, 2 * d:3 * d], dob[rows], bias[b], o[rows], lse[b * a:(b + 1) * a], a, padded, heads, hd,
                               kpen=None if kpen is None else kpen[b], dq=dqkvg[rows, :d], dk=dqkvg[rows, d:2 * d], dv=dqkvg[rows, 2 * d:3 * d])
        shared_uniform = settings.current().augattn_sm80_shared_uniform
        if shared_uniform is None:
            shared_uniform = dt is BF and d == 128 and hd == 32
        if mask_kind == 1 and shared_uniform:
            from miniworld_engine.kernels.augmented_attention.cuda import shared_mask_uniform_sm80
            shared_mask_uniform_sm80.backward(kmask[b], dob[rows], dqkvg[rows, :d], dqkvg[rows, d:2 * d],
                                              dqkvg[rows, 2 * d:3 * d], db, length, padded, a, heads)
        mask_b = kmask[b] if mask_kind == 1 else None
        pair_b = pair[b]
        if kind == "atom":
            dz, dw = sm80.pair_bias_backward_w(pair_b.unsqueeze(0), w32, db, None if mask_b is None else mask_b.unsqueeze(0), EPS)
            dzs.append(dz.view(length, length, dp))
            dw_sum = dw if dw_sum is None else dw_sum + dw
        elif kind == "apb":
            dz, _, _ = _apb().pair_bias_bwd(pair_b.reshape(length * length, dp), db, wf, length, EPS, acc, 384)
            dzs.append(dz.view(length, length, dp))
        elif sm80.pair_bias_generic_bwd_supported(dp, heads, pair_b.dtype) and _generic_kernel():   # any supported width: dz and dW' in one pass over the pair
            dz, dw = sm80.pair_bias_generic_backward(pair_b, pack[5], db, mask_b, padded, EPS)
            dzs.append(dz.view(length, length, dp))
            dw_sum = dw if dw_sum is None else dw_sum + dw
        else:
            db_c = db[:, :length, :length]
            if mask_b is not None:
                db_c = db_c.masked_fill(~mask_b.reshape(1, 1, length), 0.0)           # a masked key has no gradient
            dbb = db_c.reshape(heads, length * length).to(dt)
            p2 = pair_b.reshape(length * length, dp)
            zz, mean, rstd = torch.native_layer_norm(p2, (dp,), None, None, EPS)
            dwf_b = _mm32(dbb, zz)                                             # [H, dp]: the gradient of the natural-unit folded weight
            dzh = torch.mm(dbb.t(), wf)                                        # [L L, dp]
            dzs.append(torch.ops.aten.native_layer_norm_backward(dzh, p2, (dp,), mean, rstd, None, None, [True, False, False])[0].view(length, length, dp))
            dw_sum = dwf_b if dw_sum is None else dw_sum + dwf_b
    dpair = dzs[0].view(pair.shape) if batch == 1 else torch.stack(dzs)
    dx = torch.mm(dqkvg, wqkvg)                                                # [M, d]
    dwqkvg = _mm32(dqkvg.t(), xs)                                              # [4 d, d]
    dbq = torch.sum(dqkvg[:, :d], dim=0, dtype=f32)
    dcond = torch.mm(dg2, ws)                                                  # [M, dc]
    dws = _mm32(dg2.t(), cs)                                                   # [d, dc]
    dbs = torch.sum(dg2, dim=0, dtype=f32)
    # the pair bias's parameter gradients: d wf (the folded weight, natural units) -> d to_bias.weight = d wf x ln_pair.weight, d ln_pair.weight = sum_h d wf x to_bias.weight
    if kind == "apb":
        o_ = 2 * 384 + _apb().width(heads, 384)
        dw_sum = acc[o_:o_ + heads * dp].view(heads, dp)
    dlnw = (dw_sum * wb.float()).sum(0)
    dwb = dw_sum * lnw.float()[None]
    grads = [dwqkvg[:d], dbq, dwqkvg[d:2 * d], dwqkvg[2 * d:3 * d], dwqkvg[3 * d:], dwo, dws, dbs, dlnw, dwb]
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [g_.reshape(p.shape) for g_, p in zip(grads, params, strict=True)])
    return [_unstage(dx, a, batch, length, padded, d), _unstage(dcond, a, batch, length, padded, cond.shape[-1]), dpair, *outs]

