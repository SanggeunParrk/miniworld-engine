"""CUDA/cuBLAS compositions for the four A100 families' general API paths.

Use the existing tiled kernels first. These expressions cover the remaining
geometries in BF16/FP32, with native CUDA row operations and explicit autograd.
The numerical kernels use CUDA and cuBLAS. Opaque launches support fullgraph
torch.compile; the compiler still owns surrounding view/gradient bookkeeping.
"""

from __future__ import annotations

import os

from miniworld_engine.kernels import cuda_native as C


def serves(module, x):
    from miniworld_engine.modules.dispatch import KernelBackend
    from miniworld_engine.modules.exceptions import ImplementationType

    if (
        hasattr(module, "use_self_attention")
        and os.environ.get("MINIWORLD_TRIATTN_SM80", "1") == "0"
    ):
        return False
    if (
        hasattr(module, "to_left_gate")
        and os.environ.get("MINIWORLD_TRIMUL_SM80", "1") == "0"
    ):
        return False
    if (
        hasattr(module, "expand_a")
        and os.environ.get("MINIWORLD_CONDTRANS_SM80", "1") == "0"
    ):
        return False
    return (
        getattr(module, "_sm80_cuda", True)
        and C.enabled(x)
        and (
            getattr(module, "_backend", None) == KernelBackend.TRITON
            or getattr(module, "implementation", None) == ImplementationType.MINIWORLD
        )
    )


def _ln(x, module):
    return C.norm(x, module.weight, module.bias, module.eps)


def adaln(module, x, cond):
    cn = _ln(cond, module.ln_cond)
    xn = C.norm(x, None, None, module.ln_in.eps)
    scale = C.linear(cn, module.to_scale.weight, module.to_scale.bias)
    shift = C.linear(cn, module.to_bias.weight, module.to_bias.bias)
    # AF3 AdaLN uses sigmoid(scale), not 1 + scale.
    return C.add(C.gate(scale, xn), shift)


def conditioned_tail(x, cond, wa, wb, ws, wsc, bsc):
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80

    if fused_wide_sm80.available(x, wa, ws):
        y = fused_wide_sm80.swiglu_ffn_sm80(x, wa, wb, ws)
    else:
        y = C.linear(C.swiglu(C.linear(x, wa), C.linear(x, wb)), ws)
    return C.gate(C.linear(cond, wsc, bsc), y)


def conditioned(module, x, cond, residual=True):
    y = conditioned_tail(
        adaln(module.ada_ln_in, x, cond),
        cond,
        module.expand_a.weight,
        module.expand_b.weight,
        module.squeeze.weight,
        module.to_scale.weight,
        module.to_scale.bias,
    )
    return C.add(x, y) if residual else y


def _bias_value(prob, value):
    # One GEMM per batch/head. Broadcasting prob across pair rows would make
    # torch.matmul materialize O(L^3) repeated probability matrices.
    a, b, h, length, d = value.shape
    packed = value.permute(1, 2, 3, 0, 4).reshape(b, h, length, a * d)
    out = C.matmul(prob, packed)
    return out.reshape(b, h, length, a, d).permute(3, 0, 1, 2, 4)


def triangle(
    pair,
    mask,
    *,
    n_head,
    ln_pair_weight,
    ln_pair_bias,
    to_value_weight,
    to_bias_weight,
    to_gate_weight,
    to_out_weight,
    to_query_weight=None,
    to_key_weight=None,
    norm_query_weight=None,
    norm_key_weight=None,
    starting=True,
    eps=1e-5,
    qk_eps=None,
):
    x = pair if starting else pair.transpose(1, 2)
    b, l, _, _ = x.shape
    xn = C.norm(x, ln_pair_weight, ln_pair_bias, eps)
    h = n_head
    d = to_value_weight.shape[0] // h

    def heads(t):
        return t.reshape(b, l, l, h, d).permute(
            1, 0, 3, 2, 4
        )  # pair-row, batch, head, query, channel

    v = heads(C.linear(xn, to_value_weight))
    bias = C.linear(xn, to_bias_weight).permute(0, 3, 1, 2)
    if to_query_weight is not None:
        q, k = heads(C.linear(xn, to_query_weight)), heads(C.linear(xn, to_key_weight))
        if norm_query_weight is not None:
            eq, ek = (eps, eps) if qk_eps is None else qk_eps
            q = C.norm(q, norm_query_weight, eps=eq, rms=True)
            k = C.norm(k, norm_key_weight, eps=ek, rms=True)
        m = None if mask is None else mask.unsqueeze(0).expand(l, -1, -1)
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected

        if sm80_projected.serves(q, k, v, bias, m):
            out = sm80_projected.attention(q, k, v, bias, m)
        else:
            out = C.attention(q, k, v, bias, m)
    else:
        p = C.softmax(bias, None if mask is None else mask[:, None, None, :])
        out = _bias_value(p, v)
    out = out.permute(1, 0, 3, 2, 4).reshape(b, l, l, h * d)
    out = C.linear(C.gate(C.linear(xn, to_gate_weight), out), to_out_weight)
    return out if starting else out.transpose(1, 2)


def triangle_module(module, pair, mask):
    kw = {
        "n_head": module.n_head,
        "ln_pair_weight": module.ln_pair.weight,
        "ln_pair_bias": module.ln_pair.bias,
        "to_value_weight": module.to_value.weight,
        "to_bias_weight": module.to_bias.weight,
        "to_gate_weight": module.to_gate.weight,
        "to_out_weight": module.to_out.weight,
        "starting": module.starting,
        "eps": module.ln_pair.eps,
    }
    if module.use_self_attention:
        kw.update(
            to_query_weight=module.to_query.weight, to_key_weight=module.to_key.weight
        )
    if module.use_qk_norm:
        kw.update(
            norm_query_weight=module.norm_query.weight,
            norm_key_weight=module.norm_key.weight,
            qk_eps=(module.norm_query.eps, module.norm_key.eps),
        )
    out = triangle(pair, mask, **kw)
    if module.training and module.p_drop:
        out = C.mul(out, module._make_drop_scale(pair, module.p_drop))
    return C.add(pair, out)


def triangle_bidir(module, pair, mask):
    # Both directions share normalization. Concatenate before the one output GEMM
    # to preserve the original BF16 rounding boundary.
    x = _ln(pair, module.ln_pair)
    b, l, _, _ = x.shape
    h, dh = module.n_head, module.d_hidden
    vals, gates, biases = (
        C.linear(x, module.to_value.weight),
        C.linear(x, module.to_gate.weight),
        C.linear(x, module.to_bias.weight),
    )
    if module.use_self_attention:
        qs, ks = C.linear(x, module.to_query.weight), C.linear(x, module.to_key.weight)
    outs = []
    for direction in range(2):

        def frame(t, direction=direction):
            return t if direction == 0 else t.transpose(1, 2)

        def heads(t, direction=direction, frame=frame):
            return (
                frame(t[..., direction * dh : (direction + 1) * dh])
                .reshape(b, l, l, h, dh // h)
                .permute(1, 0, 3, 2, 4)
            )

        v = heads(vals)
        bias = frame(biases[..., direction * h : (direction + 1) * h]).permute(
            0, 3, 1, 2
        )
        if module.use_self_attention:
            q, k = heads(qs), heads(ks)
            m = None if mask is None else mask.unsqueeze(0).expand(l, -1, -1)
            from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected

            out = (
                sm80_projected.attention(q, k, v, bias, m)
                if sm80_projected.serves(q, k, v, bias, m)
                else C.attention(q, k, v, bias, m)
            )
        else:
            out = _bias_value(
                C.softmax(bias, None if mask is None else mask[:, None, None, :]), v
            )
        outs.append(frame(out.permute(1, 0, 3, 2, 4).reshape(b, l, l, dh)))
    return C.add(pair, C.linear(C.gate(gates, C.cat(outs, -1)), module.to_out.weight))


def trimul(
    x,
    wl,
    wlg,
    wr,
    wrg,
    wg,
    wo,
    gi,
    bi,
    go,
    bo,
    eps_in,
    eps_out,
    outgoing,
    mask=None,
    bidirectional=False,
    dropscale=None,
):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80, sm80_wide

    dh = wl.shape[0] // (2 if bidirectional else 1)
    # Native tiled paths accept a token mask; the general path also accepts a pair mask.
    native = (
        sm80
        if sm80.available(x, dh, mask)
        else sm80_wide
        if sm80_wide.available(x, dh, mask, hs=wl.shape[0])
        else None
    )
    if native is not None and gi is not None and bi is not None:
        direction = (
            sm80.BIDIR
            if bidirectional
            else sm80.OUTGOING
            if outgoing
            else sm80.INCOMING
        )
        return C.cat(
            [
                native.trimul(
                    x[z : z + 1],
                    wl.to(x.dtype),
                    wlg.to(x.dtype),
                    wr.to(x.dtype),
                    wrg.to(x.dtype),
                    wg.to(x.dtype),
                    wo.to(x.dtype),
                    gi,
                    bi,
                    go,
                    bo,
                    sm80.token_mask(
                        None if mask is None else mask[z : z + 1], x.shape[1], x.device
                    ),
                    x.new_empty(0)
                    if dropscale is None
                    else dropscale[z : z + 1]
                    .reshape(x.shape[1], x.shape[-1])
                    .to(x.dtype)
                    .contiguous(),
                    direction,
                    eps_in,
                    eps_out,
                )
                for z in range(x.shape[0])
            ],
            0,
        )
    xn = C.norm(x, gi, bi, eps_in)
    a = C.gate(C.linear(xn, wlg), C.linear(xn, wl))
    b = C.gate(C.linear(xn, wrg), C.linear(xn, wr))
    if mask is not None:
        pm = mask if mask.ndim == 3 else C.pair_mask(mask)
        a, b = C.mul(a, pm[..., None]), C.mul(b, pm[..., None])

    def contract(aa, bb, out):
        aa, bb = aa.permute(0, 3, 1, 2), bb.permute(0, 3, 1, 2)
        y = (
            C.matmul(aa, bb.transpose(-1, -2))
            if out
            else C.matmul(aa.transpose(-1, -2), bb)
        )
        return y.permute(0, 2, 3, 1)

    y = (
        C.cat(
            [
                contract(a[..., :dh], b[..., :dh], True),
                contract(a[..., dh:], b[..., dh:], False),
            ],
            -1,
        )
        if bidirectional
        else contract(a, b, outgoing)
    )
    y = C.linear(C.norm(y, go, bo, eps_out), wo)
    y = C.gate(C.linear(xn, wg), y)
    return C.add(x, y if dropscale is None else C.mul(y, dropscale))


def trimul_module(module, pair, mask, dropscale=None, bidirectional=False):
    return trimul(
        pair,
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        module.ln_pair.eps,
        module.ln_out.eps,
        getattr(module, "outgoing", True),
        mask,
        bidirectional,
        dropscale,
    )


def token(module, single, cond, pair, mask=None, compute_dtype=None):
    a = module.attention
    if compute_dtype is not None:
        single, cond, pair = (
            single.to(compute_dtype),
            cond.to(compute_dtype),
            pair.to(compute_dtype),
        )
    x = adaln(a.ada_ln_in, single, cond)
    s, b, l, _ = x.shape
    h = a.n_head

    def heads(t):
        return t.reshape(s, b, l, h, -1).permute(0, 1, 3, 2, 4)

    q = heads(C.linear(x, a.to_query.weight, a.to_query.bias))
    k, v = heads(C.linear(x, a.to_key.weight)), heads(C.linear(x, a.to_value.weight))
    if a.use_qk_norm:
        q, k = (
            C.norm(
                q,
                a.norm_query.weight,
                eps=a.norm_query.effective_eps(q.dtype),
                rms=True,
            ),
            C.norm(
                k, a.norm_key.weight, eps=a.norm_key.effective_eps(k.dtype), rms=True
            ),
        )
    bias = C.linear(_ln(pair, a.ln_pair), a.to_bias.weight).permute(0, 3, 1, 2)
    m = None if mask is None else mask.unsqueeze(0).expand(s, -1, -1)
    out = C.attention(q, k, v, bias, m).permute(0, 1, 3, 2, 4).reshape(s, b, l, -1)
    out = C.linear(C.gate(C.linear(x, a.to_gate.weight), out), a.to_out.weight)
    single = C.add(
        single, C.gate(C.linear(cond, a.to_scale.weight, a.to_scale.bias), out)
    )
    return conditioned(module.transition, single, cond)
