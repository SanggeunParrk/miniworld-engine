"""H100 (sm_90a) :class:`~miniworld_engine.modules.triangle.BidirectionalTriangleAttention`, bias-only (``use_self_attention=False``, the
MiniWorld block), bf16 activations, fp32 masters or bf16 parameters; forward (inference and training) and backward.

The TriangleAttention family's width-generic glue (``h100_sm90_generic._glue``: LayerNorm forward / backward, the gate) around cuBLAS
GEMMs over channel planes [R, L L] (row r = one channel over the L x L pair tokens of a sample), the gate's backward in
``h100_bidir`` (``h100_sources/triattn_bidir_sm90.cu``). Bias-only attention weights do not depend on the row, so the softmax is formed
once per (sample, head) and each direction's contraction is a plain GEMM (the flash cores of the q.k family would recompute it per row)::

    y        = LN(pair)                                   glue ``ln_fwd``
    C        = [Wv; Wg; Wb] y^T                           one GEMM -> planes [2 Dh | 2 Dh | 2 H, L L] (the module's channel order)
    P_s[h]   = softmax_k(bias_s[h, j, k])                 starting: the bias plane as it is (masked keys -inf, fp32 softmax, bf16 P)
    P_e[h]   = softmax_k(bias_e[h, k, i])                 ending: the plane transposed (L x L per head: small)
    O_s[h]   = V_s[h] P_s[h]^T                            [(d, i), k] x [k, j]: one GEMM per head (M = d_head L)
    O_e[c]   = P_e[h] V_e[c]                              [i, k] x [k, j]: per head, batched over its channels (P shared)
    U        = sigmoid(G) O                               glue ``gate_u``
    out      = pair + U^T Wo^T                            one GEMM (K = 2 Dh), the residual in its C

Backward: dWo = dy^T U^T, dU = Wo^T dy^T -> dO, dG (``gate_bwd``) -> dV_s = dO_s P_s, dP_s = dO_s^T V_s, dV_e = P_e^T dO_e, dP_e = sum_d
dO_e V_e^T (fp32) -> the softmax gradients -> dW = dC y, dy_ln = dC^T W (fp32) -> glue ``ln_bwd`` (+ the residual's dy).
"""

from __future__ import annotations

import torch

#: pair widths the glue LayerNorm takes (C % 8 == 0, C <= 512)
MAX_C = 512


def refusal(module, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
    """Why the H100 path does not serve this call (None: it does). Shapes / dtypes / flags only."""
    if not pair.is_cuda:
        return "the kernels run on CUDA tensors only"
    if torch.cuda.get_device_capability(pair.device) != (9, 0):
        return "the cuda backend of BidirectionalTriangleAttention is the H100 (sm_90a) path"
    if module.use_self_attention:
        return "self-attention (use_self_attention=True) is not part of the H100 path (the bias-only block is)"
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: the H100 path is bf16 activations only"
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return "mask must be a bool [B, L] key mask"
    C, L = pair.shape[-1], pair.shape[1]
    if C % 8 or C > MAX_C:
        return f"d_pair {C}: the glue LayerNorm takes d_pair % 8 == 0, <= {MAX_C}"
    if L % 8:
        return f"L {L}: the planes' GEMMs and the gate kernels take L % 8 == 0"
    if (module.d_hidden // module.n_head) % 8:
        return f"head width {module.d_hidden // module.n_head}: the planes' GEMMs take heads of a multiple of 8 channels"
    if module.ln_pair.weight.dtype != torch.float32 or module.ln_pair.bias.dtype != torch.float32:
        return "the LayerNorm affine must be fp32 (the module's pin)"
    return None


def _glue():
    from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90_generic import _glue as glue

    return glue()


def _ext():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_bidir

    return h100_bidir.ext()


def _mm_f32(a, b):
    return torch.mm(a, b, out_dtype=torch.float32)


def _probs(bias: torch.Tensor, keep: torch.Tensor | None, H: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The bias planes [2 H, L L] of one sample -> (P_s [H, L, L], P_e [H, L, L]), each a softmax over its last axis k (masked keys -inf)."""
    b = bias.view(2 * H, L, L).float()
    logits = torch.cat([b[:H], b[H:].transpose(1, 2)])          # starting [h, j, k]; ending [h, i, k] = bias_e[h, k, i]
    if keep is not None:
        logits = logits.masked_fill(~keep[None, None, :], float("-inf"))
    p = torch.softmax(logits, -1).to(torch.bfloat16)
    return p[:H], p[H:]


def _forward(pair, mask, lnw, lnb, eps, w, wo, H, Dh, save):
    """-> (out [B, L, L, C], saved or None). w = [Wv; Wg; Wb] [4 Dh + 2 H, C], wo [C, 2 Dh] (bf16)."""
    g = _glue()
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    y, mean, rstd = g.ln_fwd(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, L, False)
    out = torch.empty_like(x)
    planes, probs, os_, us = [], [], [], []
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c = torch.mm(w, y[rows].t())                                 # [4 Dh + 2 H, L L]
        ps, pe = _probs(c[4 * Dh:], None if mask is None else mask[b], H, L)
        o = torch.empty(2 * Dh, LL, device=x.device, dtype=torch.bfloat16)
        torch.bmm(c[:Dh].view(H, dh * L, L), ps.transpose(1, 2), out=o[:Dh].view(H, dh * L, L))
        for h in range(H):
            ch = slice(Dh + h * dh, Dh + (h + 1) * dh)
            torch.bmm(pe[h].expand(dh, L, L), c[ch].view(dh, L, L), out=o[ch].view(dh, L, L))
        u = g.gate_u(c[2 * Dh:4 * Dh], o, 0)
        torch.addmm(x[rows], u.t(), wo.t(), out=out[rows])
        if save:
            planes.append(c)
            probs.append((ps, pe))
            os_.append(o)
            us.append(u)
    saved = (y, mean, rstd, planes, probs, os_, us) if save else None
    return out.view(B, L, L, C), saved


def _backward(dy, pair, lnw, w, wo, saved, H, Dh):
    g, e = _glue(), _ext()
    y, mean, rstd, planes, probs, os_, us = saved
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    dy = dy.contiguous().view(-1, C)
    dW = torch.zeros(w.shape, device=x.device, dtype=torch.float32)
    dWo = torch.zeros(wo.shape, device=x.device, dtype=torch.float32)
    dyln = torch.empty(B * LL, C, device=x.device, dtype=torch.float32)
    dc = torch.empty(w.shape[0], LL, device=x.device, dtype=torch.bfloat16)
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c, (ps, pe), o, u = planes[b], probs[b], os_[b], us[b]
        dWo += _mm_f32(dy[rows].t(), u.t())
        du = torch.mm(wo.t(), dy[rows].t())                          # [2 Dh, L L]
        do = e.gate_bwd(du, c[2 * Dh:4 * Dh], o, dc[2 * Dh:4 * Dh])     # dO; dG into dc
        del du
        # starting: O_s[h] = V_s[h] P_s[h]^T
        dos = do[:Dh].view(H, dh * L, L)
        torch.bmm(dos, ps, out=dc[:Dh].view(H, dh * L, L))
        dps = torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32)   # [H, j, k]
        # ending: O_e[c] = P_e[h] V_e[c]
        dpe = []
        for h in range(H):
            ch = slice(Dh + h * dh, Dh + (h + 1) * dh)
            doe = do[ch].view(dh, L, L)
            torch.bmm(pe[h].t().expand(dh, L, L), doe, out=dc[ch].view(dh, L, L))
            dpe.append(torch.bmm(doe, c[ch].view(dh, L, L).transpose(1, 2), out_dtype=torch.float32).sum(0))   # [i, k]
        dp = torch.cat([dps, torch.stack(dpe)])
        p = torch.cat([ps, pe]).float()
        dl = p * (dp - (p * dp).sum(-1, keepdim=True))               # d logits [2 H, ., k]
        db = dc[4 * Dh:].view(2 * H, L, L)
        db[:H] = dl[:H]
        db[H:] = dl[H:].transpose(1, 2)                              # back to the bias plane's [k, i]
        dW += _mm_f32(dc, y[rows])
        dyln[rows] = _mm_f32(dc.t(), w)
    dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)   # + the residual's dy
    dgb = part.sum(0)
    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo


def _weights(wv, wb, wg, wo):
    bf = torch.bfloat16
    return torch.cat([wv.to(bf), wg.to(bf), wb.to(bf)]).contiguous(), wo.to(bf).contiguous()


class _BidirFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, mask, lnw, lnb, wv, wb, wg, wo, eps, H, Dh):
        w, wo16 = _weights(wv, wb, wg, wo)
        out, saved = _forward(pair, mask, lnw, lnb, eps, w, wo16, H, Dh, True)
        ctx.H, ctx.Dh, ctx.saved = H, Dh, saved
        ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wv, wb, wg, wo))
        ctx.save_for_backward(pair, lnw, w, wo16)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, w, wo16 = ctx.saved_tensors
        H, Dh = ctx.H, ctx.Dh
        dx, dlw, dlb, dW, dWo = _backward(dy, pair, lnw, w, wo16, ctx.saved, H, Dh)
        ctx.saved = None
        dWv, dWg, dWb = dW[:2 * Dh], dW[2 * Dh:4 * Dh], dW[4 * Dh:]
        grads = [t.to(d) for t, d in zip((dlw, dlb, dWv, dWb, dWg, dWo), ctx.dtypes, strict=True)]
        return (dx, None, *grads, None, None, None)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + bidirectional bias-only attention(pair)`` (the module's ``_forward_pytorch``), forward and, with grad, backward."""
    H, Dh = module.n_head, module.d_hidden
    params = (module.ln_pair.weight, module.ln_pair.bias, module.to_value.weight, module.to_bias.weight, module.to_gate.weight,
              module.to_out.weight)
    mask = None if mask is None else mask.contiguous()
    grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in params))
    if not grad:
        w, wo = _weights(*params[2:])
        return _forward(pair, mask, params[0], params[1], module.ln_pair.eps, w, wo, H, Dh, False)[0]
    return _BidirFn.apply(pair, mask, *params, module.ln_pair.eps, H, Dh)
