"""A100 atom training attention: fixed CUDA projection, Qresident dQ, dKV, delta and pair reductions."""
from __future__ import annotations
import functools
import os
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.augmented_attention.cuda import atom_plain_partials_delta_mixed_sm80,atom_pair_reduce_vec_sm80,atom_pair_uniform_no_db_sm80
from miniworld_engine.kernels.augmented_attention.cuda import atom_sample_uniform_sm80
from .augattn_sm80 import _fwd_fake,_bias_into
from .augattn_sm80 import (_bwd_fake,_sm80,_pad,_pack,_stage,_penalties,_kind,_apb,_generic_kernel,_mm32,_unstage,EPS,BF,F32)

@opaque(fake=_fwd_fake, name="augmented_attention_atom_sm80_fwd")
def _fwd(x: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, res: torch.Tensor | None, kmask: torch.Tensor | None, bias_in: torch.Tensor | None,
         params: list[torch.Tensor], a: int, batch: int, length: int, heads: int, mask_kind: int, save: bool) -> list[torch.Tensor]:
    """The module's update after the AdaLN: x [A, B, L, d] (the AdaLN's output), cond [A, B, L, dc], pair [B, L, L, dp], res [A, B, L, d] or None (the residual), kmask
    a bool [B, L] / [A, B, L] key mask or None (``mask_kind`` 1 / 2 / 0), bias_in a pair bias made earlier (the compute dtype [1, H, Lp, Lp]: ``cache_pair_bias``) or None, params =
    (Wq, bq, Wk, Wv, Wg, Wo, Ws, bs, ln_pair.weight, to_bias.weight). Returns ``[out]``, or with ``save`` also the activations of the backward: qkvg, o, lse, og, y, g2, bias."""
    sm80 = _sm80()
    dt = x.dtype
    f32 = dt is F32
    d = x.shape[-1]
    hd = d // heads
    padded = _pad(length)
    m = a * batch * padded
    n = a * padded                                                             # rows of one batch element
    dev = x.device
    wo, ws, bs = params[5].to(dt), params[6].to(dt), params[7].to(dt)             # no copies for bf16 parameters (the autograd function hands cast ones for fp32)
    pack = _pack(params, heads, dt, save=save)
    wqkvg, bvec = pack[0], pack[1]
    xs, cs = _stage(x, batch, padded), _stage(cond, batch, padded)
    rs = None if res is None else _stage(res, batch, padded)
    qkvg_override = os.environ.get("MINIWORLD_AUGATTN_SM80_ATOM_QKVG")
    qkvg_native = (qkvg_override == "1" if qkvg_override is not None else
                   (save and batch == 1 and a == 48 and length == 4096)
                   or (not save and batch == 1 and a == 5 and length in (2048, 4096)))
    tiled_override = os.environ.get("MINIWORLD_AUGATTN_SM80_ATOM_QKVG_TILED")
    tiled_qkvg = (dt is BF and d == 128 and hd == 32
                  and (tiled_override == "1" if tiled_override is not None else
                       not save and batch == 1 and (a, length) in
                       ((1, 1024), (1, 2048), (1, 4096), (5, 1024), (5, 2048))))
    if tiled_qkvg:
        from miniworld_engine.kernels.augmented_attention.cuda import atom_qkvg_tiled_sm80
        cfg = {1024: 7, 2048: 6, 4096: 10, 5120: 10, 10240: 11, 20480: 11}.get(m, 10)
        qkvg = atom_qkvg_tiled_sm80.result(xs, wqkvg, bvec, cfg)
    elif dt is BF and d == 128 and qkvg_native:
        from miniworld_engine.kernels.augmented_attention.cuda import atom_qkvg_resident_sm80
        cfg = 1 if m == 10240 or m >= 393216 else 3 if m >= 20480 else 0
        qkvg = atom_qkvg_resident_sm80.result(xs, wqkvg, bvec, cfg)
    else:
        qkvg = torch.addmm(bvec, xs, wqkvg.t())                               # [M, 4 d]
    q, k, v, g = (qkvg[:, i * d:(i + 1) * d] for i in range(4))
    if bias_in is not None:
        bias = bias_in
    else:
        bias = torch.empty(batch, heads, padded, padded, device=dev, dtype=dt)
        for b in range(batch):
            _bias_into(bias[b], pair[b], pack, heads, hd, length, padded, kmask[b] if mask_kind == 1 else None)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    if save:
        o = torch.empty(m, d, device=dev, dtype=dt)
        lse = torch.empty(batch * a, heads, padded, device=dev, dtype=torch.float32)
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        kp = None if kpen is None else kpen[b]
        shared_core = (dt is BF and d == 128 and hd == 32 and mask_kind == 1
                       and os.environ.get("MINIWORLD_AUGATTN_SM80_SHARED_CORE", "0") == "1")
        shared_score = (dt is BF and d == 128 and hd == 32 and mask_kind == 1
                        and os.environ.get("MINIWORLD_AUGATTN_SM80_SHARED_SCORE", "0") == "1")
        if shared_score:
            from miniworld_engine.kernels.augmented_attention.cuda import shared_score_core_sm80
            shared_score_core_sm80.forward(q[rows], k[rows], v[rows], None if save else g[rows],
                                          bias[b], kmask[b], length, padded, a, heads, hd, save,
                                          o[rows] if save else q[rows],
                                          lse[b * a:(b + 1) * a] if save else None)
        elif shared_core:
            from miniworld_engine.kernels.augmented_attention.cuda import shared_uniform_core_sm80
            shared_uniform_core_sm80.forward(q[rows], k[rows], v[rows], None if save else g[rows],
                                            bias[b], kmask[b], length, padded, a, heads, hd, save,
                                            o[rows] if save else q[rows],
                                            lse[b * a:(b + 1) * a] if save else None)
        elif f32 and save:
            sm80.tf32_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, kpen=kp, out=o[rows], lse=lse[b * a:(b + 1) * a])
        elif f32:
            sm80.tf32_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, gate=g[rows], out=q[rows], kpen=kp)
        elif save:
            sm80.plain_forward(q[rows], k[rows], v[rows], bias[b], a, padded, heads, hd, save_lse=True, kpen=kp, out=o[rows], lse=lse[b * a:(b + 1) * a])
        else:
            sm80.pgate_forward(q[rows], k[rows], v[rows], g[rows], bias[b], a, padded, heads, hd, kpen=kp, out=q[rows])
        uniform_default = "1" if dt is BF and d == 128 and hd == 32 else "0"
        if mask_kind == 1 and not shared_core and not shared_score and os.environ.get("MINIWORLD_AUGATTN_SM80_SHARED_UNIFORM", uniform_default) == "1":
            from miniworld_engine.kernels.augmented_attention.cuda import shared_mask_uniform_sm80
            shared_mask_uniform_sm80.forward(kmask[b], v[rows], None if save else g[rows],
                                             o[rows] if save else q[rows],
                                             lse[b * a:(b + 1) * a] if save else None,
                                             length, padded, a, heads)
        if mask_kind == 2:
            # A sample whose own keys are all masked follows the measured
            # original Dense finite-min uniform softmax. Other samples untouched.
            atom_sample_uniform_sm80.forward(kmask[:, b], v[rows], None if save else g[rows],
                                             o[rows] if save else q[rows],
                                             lse[b * a:(b + 1) * a] if save else None,
                                             length, padded, a, heads)
    if save:
        og = torch.empty(m, d, device=dev, dtype=dt)
        sm80.gate_rows(o, g, og)
    else:
        og = q                                                                 # sigmoid(g) o, written over q's columns
    output_override = os.environ.get("MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_FUSED")
    output_fused = (output_override == "1" if output_override is not None else
                    batch == 1 and rs is not None
                    and ((save and a == 48 and length == 4096) or
                         (not save and (a, length) in ((1, 4096), (5, 1024), (5, 2048), (5, 4096)))))
    output_tiled_override = os.environ.get("MINIWORLD_AUGATTN_SM80_ATOM_OUTPUT_TILED")
    # The small A1 inference rows qualify own 32-channel output tiles;
    # larger rows retain the existing faster full-output fusion.
    output_tiled = ((output_tiled_override == "1" if output_tiled_override is not None else True)
                    and not save and batch == 1 and hd == 32
                    and (a, length) in ((1, 1024), (1, 2048)))
    if dt is BF and d == 128 and cs.shape[1] == 128 and output_tiled:
        from miniworld_engine.kernels.augmented_attention.cuda import atom_output_tiled_sm80
        cfg = 1 if m == 1024 else 3
        out, y, g2 = atom_output_tiled_sm80.result(og, cs, wo, ws, bs, rs, save, cfg)
    elif dt is BF and d == 128 and cs.shape[1] == 128 and output_fused:
        from miniworld_engine.kernels.augmented_attention.cuda import atom_output_fused_sm80
        cfg = 1 if save else (0 if m <= 5120 else 2 if m <= 10240 else 4)
        out, y, g2 = atom_output_fused_sm80.result(og, cs, wo, ws, bs, rs, save, cfg)
    else:
        y = torch.mm(og, wo.t())
        g2 = torch.addmm(bs, cs, ws.t())
        out = torch.empty(m, d, device=dev, dtype=dt)
        sm80.res_gate(y, g2, rs, out)
    outs = [_unstage(out, a, batch, length, padded, d)]
    if save:
        outs += [qkvg, o, lse, og, y, g2, bias]
    return outs


@opaque(fake=_bwd_fake, name="augmented_attention_atom_sm80_bwd")
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
    from miniworld_engine.kernels.augmented_attention.cuda import atom_post_grad_delta_sm80
    # E274 retains E205/E221 arithmetic and changes only fixed core mapping/Q residency.
    # Original BFdo first, then exact sequential32FMA delta, original12BFpartials
    # enter ascendingG12 F32/shared vector-pairgrad; no standalone delta/DB write.
    delta=torch.empty(a,heads,padded,device=dev,dtype=torch.float32)
    atom_post_grad_delta_sm80.run(dy,wo,o,qkvg[:,3*d:],dob,dqkvg[:,3*d:],delta,padded,None,3)
    kpen = _penalties(kmask, a, batch, length, padded) if mask_kind == 2 else None
    kind = _kind(dp, heads, dt)
    acc = torch.zeros(_apb().acc_n(heads, 384), device=dev) if kind == "apb" else None
    dzs, dw_sum = [], None
    for b in range(batch):
        rows = slice(b * n, (b + 1) * n)
        core_bwd = sm80.tf32_backward if dt is F32 else functools.partial(atom_plain_partials_delta_mixed_sm80.plain_backward_partials_delta_mixed, db_natural=True, delta=delta, dq_order=1, dkv_order=6)
        _, _, _, db = core_bwd(qkvg[rows, :d], qkvg[rows, d:2 * d], qkvg[rows, 2 * d:3 * d], dob[rows], bias[b], o[rows], lse[b * a:(b + 1) * a], a, padded, heads, hd,
                               kpen=None if kpen is None else kpen[b], dq=dqkvg[rows, :d], dk=dqkvg[rows, d:2 * d], dv=dqkvg[rows, 2 * d:3 * d])
        uniform_default = "1" if dt is BF and d == 128 and hd == 32 else "0"
        if mask_kind == 1 and os.environ.get("MINIWORLD_AUGATTN_SM80_SHARED_UNIFORM", uniform_default) == "1":
            atom_pair_uniform_no_db_sm80.backward(kmask[b], dob[rows], dqkvg[rows, :d], dqkvg[rows, d:2 * d],
                                                  dqkvg[rows, 2 * d:3 * d], length, padded, a, heads)
        if mask_kind == 2:
            # Finite corrected LSE makes this sample's score/DB contribution 0.
            # Only dV needs its original uniform mean; do not zero shared DB.
            atom_sample_uniform_sm80.backward(kmask[:, b], dob[rows], dqkvg[rows, :d],
                                              dqkvg[rows, d:2*d], dqkvg[rows, 2*d:3*d],
                                              length, padded, a, heads)
        mask_b = kmask[b] if mask_kind == 1 else None
        pair_b = pair[b]
        if kind == "atom":
            dz=torch.empty_like(pair_b.unsqueeze(0));pw=torch.empty((padded//64)*(padded//32),4,16,device=dev,dtype=torch.float32)
            atom_pair_reduce_vec_sm80.run(pair_b.unsqueeze(0),w32,db,dz,pw,None if mask_b is None else mask_b.contiguous(),EPS)
            dw=pw.sum(0)
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

