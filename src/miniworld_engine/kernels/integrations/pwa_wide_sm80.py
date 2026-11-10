"""A100 (sm_80) MSAPairWeightedAveraging at the BIG width (d_msa 256, d_pair 512, 8 heads x 32: IntelliFold2), inference and training.

The other sm_80 widths are ``pwa_sm80``'s fused kernels; at this width their resident tiles do not fit an A100 SM (the H100 BIG build splits
its kernels for the same reason, ``csrc/pwa_wide.cu``). Here every product is a cuBLAS GEMM on the layout it wants, with the engine's row
LayerNorm (the TriangleAttention glue's ``ln_fwd`` / ``ln_bwd``, its sm_80 build) and four layout / gate kernels
(``pair_weighted_averaging/cuda/sm80/wide_ops.cu``) between them::

    pair     zn = LN(z)                       glue ln_fwd              [L L, 512]         (inference: pair_logits, LN and the 8 dots in one pass)
             logits = zn Wb^T                 GEMM                     [L L, 8] -> masked softmax over the keys -> w [8, L, L] bf16
    msa      y = LN(m)                        glue ln_fwd              [S L, 256]
             vg = y [Wv; Wg]^T                GEMM                     [S L, 512]  (v | g, natural rows (s, j))
             v_hm                             to_hm                    [8, L, S 32] (head-major)
             o_hm = w v_hm                    batched GEMM (M = L, N = S 32)    (inference: head_bmm, per head on the natural layout)
             u = sigmoid(g) o                 gate_fwd                 [S L, 256]
             out = m + drop(u Wo^T)           cublasLt GEMM with the residual as C (D != C); with the row dropout (keep-mask [L, 256] shared
                                              by the MSA rows) a GEMM and one fused multiply-add

    backward dWo = dt^T u, du = dt Wo (dt = drop(dout)) -> gate_bwd (do_hm, dg) -> dv_hm = w^T do_hm (from_hm into [dv | dg]),
             dw = do_hm v_hm^T (fp32) -> dy = [dv | dg] [Wv; Wg], d[Wv; Wg] = [dv | dg]^T y -> glue ln_bwd (+ the residual's gradient)
             pair: the softmax backward (fp32), d logits Wb -> glue ln_bwd, dWb = d logits^T zn

Contract: bf16 msa [1, S, L, 256] and pair [1, L, L, 512], an optional bool key mask [1, L], capability 8.0 (``refusal``).
"""

from __future__ import annotations

import torch

D_MSA, D_PAIR, HEADS, D_HIDDEN = 256, 512, 8, 32
HC = HEADS * D_HIDDEN


def supported(d_msa: int, d_pair: int, n_head: int, d_hidden: int) -> bool:
    return (d_msa, d_pair, n_head, d_hidden) == (D_MSA, D_PAIR, HEADS, D_HIDDEN)


def _glue():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_bidir

    return sm80_bidir.glue()


def _lt():
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import _lt as lt   # host-only cublasLt glue (no device code)

    return lt()


def _ext():
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import wide_sm80

    return wide_sm80.ext()


def _weights(leaves):
    """The kernels' forms of the module's parameters: LayerNorm affines fp32, [Wv; Wg] / Wb / Wo bf16; the pair LayerNorm's offset is zeros (the
    module has none: ``pwa_sm80._ln_pair_bias``). Cached while the parameters are unmodified (``_capture.param_forms``)."""
    from miniworld_engine.kernels import _capture

    lnm_w, lnm_b, wv, wg, lnz_w, _, wb, wo = leaves
    bf, f32 = torch.bfloat16, torch.float32
    params = (lnm_w, lnm_b, wv, wg, lnz_w, wb, wo)
    return _capture.param_forms(
        "pwa_wide_sm80", params,
        lambda: (lnm_w.to(f32).contiguous(), lnm_b.to(f32).contiguous(), torch.cat([wv, wg]).to(bf).contiguous(), lnz_w.to(f32).contiguous(),
                 torch.zeros(lnz_w.shape, dtype=f32, device=lnz_w.device), wb.to(bf).contiguous(), wo.to(bf).contiguous()))


def _forward(msa, pair, mask, leaves, eps_m, eps_z, keep, dscale, save):
    g, e = _glue(), _ext()
    lnm_w, lnm_b, wvg, lnz_w, lnz_b, wb, wo = _weights(leaves)
    _, S, L, _ = msa.shape
    zf = pair[0].reshape(L * L, D_PAIR)
    if save:   # training: LN(z) and its statistics are kept (dWb, the LayerNorm backward)
        zn, zmean, zrstd = g.ln_fwd(zf, lnz_w, lnz_b, eps_z, L, False)
        logits = torch.mm(zn, wb.t())
    else:      # inference: one pass over the pair
        zn = zmean = zrstd = None
        logits = e.pair_logits(zf, lnz_w, lnz_b, wb, eps_z, 0)
    key = logits.new_empty((0,), dtype=torch.uint8) if mask is None else mask[0].contiguous().view(torch.uint8)
    w = e.softmax_keys(logits, key, L)                                             # [8, L, L] bf16: the masked softmax over the keys j
    mf = msa[0].reshape(S * L, D_MSA)
    y, mmean, mrstd = g.ln_fwd(mf, lnm_w, lnm_b, eps_m, L, False)
    vg = _lt().lt_mm(y, wvg, None, 0, False, None, None)                           # [S L, 512] = [v | g] (cublasLt, the timed algorithm)
    u = torch.empty(S * L, HC, device=msa.device, dtype=torch.bfloat16)
    v_hm = torch.empty(HEADS, L, S * D_HIDDEN, device=msa.device, dtype=torch.bfloat16)
    e.to_hm(vg, v_hm, S, L)
    o_hm = torch.bmm(w, v_hm)
    e.gate_fwd(o_hm, vg, HC, u, S, L)
    if keep is None:   # m + u Wo^T in one cublasLt call (D != C: no copy of m into the output first)
        out = _lt().lt_mm(u, wo, mf, 0, False, None, None)
    else:
        out = torch.addcmul(mf.view(S, L, D_MSA), torch.mm(u, wo.t()).view(S, L, D_MSA), keep * dscale)   # keep [L, 256] shared by the MSA rows
    saved = (zn, zmean, zrstd, w, y, mmean, mrstd, vg, v_hm, o_hm, u) if save else None
    return out.view(1, S, L, D_MSA), saved


def _backward(dy, msa, pair, mask, leaves, eps_m, eps_z, keep, dscale, saved):
    g, e = _glue(), _ext()
    lnm_w, lnm_b, wvg, lnz_w, lnz_b, wb, wo = _weights(leaves)
    zn, zmean, zrstd, w, y, mmean, mrstd, vg, v_hm, o_hm, u = saved
    _, S, L, _ = msa.shape
    f32 = torch.float32
    mf = msa[0].reshape(S * L, D_MSA)
    dres = dy[0].reshape(S * L, D_MSA)
    dt = dres if keep is None else (dres.view(S, L, D_MSA) * (keep * dscale)).view(S * L, D_MSA)
    dwo = torch.mm(dt.t(), u, out_dtype=f32)
    du = torch.mm(dt, wo)
    del dt, u
    do_hm = torch.empty_like(o_hm)
    dvg = torch.empty(S * L, 2 * HC, device=msa.device, dtype=torch.bfloat16)      # [dv | dg]
    e.gate_bwd(du, vg, HC, o_hm, do_hm, dvg, S, L)
    del du
    e.from_hm(torch.bmm(w.transpose(1, 2), do_hm), dvg, S, L)
    dw = torch.bmm(do_hm, v_hm.transpose(1, 2), out_dtype=f32)                     # [8, L, L]
    del do_hm
    dyln = torch.mm(dvg, wvg)
    dwvg = torch.mm(dvg.t(), y, out_dtype=f32)
    del dvg
    dm, mpart = g.ln_bwd(dyln, mf, dres, lnm_w, mmean, mrstd, L, False)            # + the residual's gradient
    del dyln
    mgb = mpart.sum(0)
    # pair side: softmax backward, the masked keys' logits get nothing (masked_fill), then the bias projection and the pair LayerNorm
    wf = w.float()
    dl = wf * (dw - (wf * dw).sum(-1, keepdim=True))
    if mask is not None:
        dl = dl.masked_fill(~mask[0][None, None, :], 0.0)
    dlog = dl.permute(1, 2, 0).reshape(L * L, HEADS).to(torch.bfloat16)
    dwb = torch.mm(dlog.t(), zn, out_dtype=f32)
    zf = pair[0].reshape(L * L, D_PAIR)
    dz, zpart = g.ln_bwd(torch.mm(dlog, wb), zf, torch.zeros_like(zf), lnz_w, zmean, zrstd, L, False)
    zgb = zpart.sum(0)
    grads = (mgb[0], mgb[1], dwvg[:HC], dwvg[HC:], zgb[0], zgb[1], dwb, dwo)
    return dm.view_as(msa), dz.view_as(pair), [t.to(p.dtype) for t, p in zip(grads, leaves, strict=True)]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps_m, eps_z, p_drop, msa, pair, mask, *leaves):
        keep, dscale = None, 1.0
        if p_drop > 0:   # Dropout(broadcast_dim=1): one keep decision per (token, channel), shared by the MSA rows (pwa_sm80's draw)
            keep = (torch.rand(msa.shape[2], msa.shape[3], device=msa.device) > p_drop).to(torch.bfloat16)
            dscale = 1.0 / (1.0 - p_drop)
        with torch.autocast("cuda", enabled=False):
            out, saved = _forward(msa, pair, mask, list(leaves), eps_m, eps_z, keep, dscale, True)
        ctx.args, ctx.saved = (eps_m, eps_z, keep, dscale), saved
        ctx.save_for_backward(msa, pair, mask, *leaves)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        msa, pair, mask, *leaves = ctx.saved_tensors
        eps_m, eps_z, keep, dscale = ctx.args
        dm, dz, grads = _backward(dy.contiguous(), msa, pair, mask, leaves, eps_m, eps_z, keep, dscale, ctx.saved)
        ctx.saved = None
        return (None, None, None, dm, dz, None, *grads)


def update_inference(leaves, msa, pair, mask, eps_m, eps_z) -> torch.Tensor:
    with torch.cuda.device(msa.device), torch.autocast("cuda", enabled=False):   # the dtypes are the kernels': bf16 activations, fp32 statistics
        return _forward(msa, pair, mask, leaves, eps_m, eps_z, None, 1.0, False)[0]


def update_train(leaves, msa, pair, mask, eps_m, eps_z, p_drop) -> torch.Tensor:
    return _Training.apply(eps_m, eps_z, p_drop, msa, pair, mask, *leaves)
