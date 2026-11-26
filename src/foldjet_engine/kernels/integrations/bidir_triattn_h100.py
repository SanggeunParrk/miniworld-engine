"""H100 (sm_90a) and A100 (sm_80: the same kernels built for sm_80, ``triangle_attention/cuda/sm80_bidir``) :class:`~foldjet_engine.modules.triangle.BidirectionalTriangleAttention`, bias-only (``use_self_attention=False``, the
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

import os

import torch

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)

#: pair widths the glue LayerNorm takes (C % 8 == 0, C <= 512)
MAX_C = 512


def refusal(module, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
    """Why the H100 path does not serve this call (None: it does). Shapes / dtypes / flags only."""
    if not pair.is_cuda:
        return "the kernels run on CUDA tensors only"
    if torch.cuda.get_device_capability(pair.device) not in ((9, 0), (8, 0), (12, 0), (10, 0)):
        return "the cuda backend of BidirectionalTriangleAttention is the H100 (sm_90a), A100 (sm_80) and RTX 5090 (sm_120a) path"
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


def _a100() -> bool:
    return torch.cuda.get_device_capability() in ((8, 0), (12, 0))  # A100, or the RTX 5090 (the same sources built for sm_120a)


def _b200() -> bool:
    return torch.cuda.get_device_capability() == (10, 0)  # B200: the same sources built for sm_100a (b200_bidir)


#: B200: the ending node's per-head contractions (forward O_e, backward dV_e and dP_e) and the softmax backward's row dot on the
#: sm_100a kernels of ``b200_bidir.bo`` -- one launch for every head instead of a cuBLAS batched GEMM (+ a torch sum) per head.
#: FOLDJET_ENGINE_BIDIR_B200_NATIVE=0 switches back to the torch ops (H100 / A100 / RTX 5090 never take this path).
B200_NATIVE = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_NATIVE", "1") != "0"

#: (S90 training, cuda:b200_sm100 61) the single-direction (use_self_attention=False) LayerNorm backward's input gradient dyln = dc^T W written as bf16 instead of fp32 (2 x 67 MB less traffic at L512): row 188 training step -2.4 .. -3.0 % (L256 199.7 -> 193.7 us, L512 568.7 -> 553.6, L768 1201.1 -> 1164.7, starting; ending -2.5 .. -2.7 %); the bf16 dyln rounds the cotangent once before the LayerNorm backward (accuracy: row 188 training, 8 cases, no RED, GREEN / YELLOW as before).
DYLN_BF16 = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_DYLN_BF16", "1") != "0"


#: the native ending GEMMs up to this length: they win at L128 (one launch for 4 heads instead of 4-8) and tie at L256; from L384 on
#: cuBLAS's batched GEMM per head is faster (B200 2026-11-17, measurer: ending eval L384 0.132 vs 0.151 ms, L768 0.422 vs 0.559)
BO_GEMM_MAX_L = 256


def _bo(L: int = 0):
    """The native B200 kernels (None: not a B200, switched off, or L > BO_GEMM_MAX_L when the caller passes the length)."""
    if not (B200_NATIVE and _b200()) or L > BO_GEMM_MAX_L:
        return None
    from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

    return b200_bidir.bo()


def _b200_wide():
    from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn

    return b200_triattn._wide()  # tri_scale_rows: the dropout scale (and residual) of the bias-only update on the B200


def _glue():
    if _b200():
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.glue()
    if _a100():  # the same sources built for sm_80 (sm_120a on the 5090)
        from foldjet_engine.kernels.triangle_attention.cuda import sm80_bidir

        return sm80_bidir.glue()
    from foldjet_engine.kernels.triangle_attention.cuda.h100_sm90_generic import _glue as glue

    return glue()


def _ext():
    if _b200():
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.ext()
    if _a100():
        from foldjet_engine.kernels.triangle_attention.cuda import sm80_bidir

        return sm80_bidir.ext()
    from foldjet_engine.kernels.triangle_attention.cuda import h100_bidir

    return h100_bidir.ext()


#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the bias-only paths' LayerNorm forward / backward and fused gate backward (C = 128, K = 128) on the B200 kernels of
#: ``b200_sources/lnbwd_sm100.cu``: x / dyln / dy of a row in flight together and a row step ahead, 2 CTAs an SM (fewer partial rows for the
#: column sum). forward y / mean / rstd and dx bitwise equal, d gamma / d beta fp32 sums regrouped. FOLDJET_ENGINE_BIDIR_B200_LNBWD=0 turns it off.
B200_LNBWD = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_LNBWD", "1") != "0"


#: (N1) the single-direction training backward's two weight-gradient GEMMs (dWo = dt^T u, dW = dc^T y: tall-K split-K GEMMs of 9-12 us at L128, 36 us at L512
#: with their reduce) off the main chain as branches of the captured graph (they read tensors the chain already has and nothing reads their results before the
#: end): the dWo GEMM beside the gate backward / core GEMMs, the dW GEMM beside the dyln GEMM and the LayerNorm backward. One sample (B = 1), B200, graph capture only.
#: FOLDJET_ENGINE_BIDIR_B200_SIDE=0 turns it off.
B200_SIDE = int(os.environ.get("FOLDJET_ENGINE_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch
B200_SIDE_MAX_L = int(os.environ.get("FOLDJET_ENGINE_BIDIR_B200_SIDE_MAX_L", "256"))   # the dWo branch only up to this length (beyond it the core GEMMs fill the SMs: it slows the chain); the dW branch at every length


def _ln_fwd(g, x, lnw, lnb, eps, L):
    """-> (y, mean, rstd) of the LayerNorm forward in the starting layout (the shared glue's ``ln_fwd``; C = 128 on the B200 kernel)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.lnb().ln_fwd128(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, 4)
    return g.ln_fwd(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, L, False)


def _ln_bwd(g, bo, dyln, x, dy, lnw, mean, rstd, L):
    """-> (dx, [d gamma; d beta] fp32 [2, C]) of the LayerNorm backward + the residual's dy (the shared glue's ``ln_bwd`` in the starting layout,
    its partial rows summed by ``bo_colsum``; C = 128 on the B200 kernel)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        dx, part = b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2, False)
        return dx, b200_bidir.lnb().colsum256(part)  # one launch (the shared colsum: two)
    else:
        dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)
    return dx, (bo.bo_colsum(part) if bo is not None else part.sum(0))  # B200: one deterministic launch (torch: a slow tall reduce)


#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the tall-K fp32-output contractions of the bias-only backward (dWo, dW, dP: K = 16384 .. L L) on tuned cublasLt
#: algorithms (``b200_sources/ltmm_sm100.cu``, chosen per shape by a deterministic / accurate timing sweep at the first eager call). FOLDJET_ENGINE_BIDIR_B200_LT=0 turns it off.
B200_LT = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_LT", "1") != "0"


def _lt():
    if not (B200_LT and _b200()):
        return None
    from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

    return b200_bidir.ltmm()


def _mm_f32(a, b):
    e = _lt()
    if e is not None and a.shape[1] >= 16384 and (a.stride(-1) == 1 or a.stride(-2) == 1) and (b.stride(-1) == 1 or b.stride(-2) == 1):
        r = e.lt_mm(a, b, a)
        if r.numel():  # an empty result: shape not calibrated yet (graph capture) or no accurate candidate
            return r
    return torch.mm(a, b, out_dtype=torch.float32)


def _bmm_f32(a, b):
    e = _lt()
    if e is not None and a.shape[2] >= 16384 and (a.stride(-1) == 1 or a.stride(-2) == 1) and (b.stride(-1) == 1 or b.stride(-2) == 1):
        r = e.lt_mm(a, b, a)
        if r.numel():  # an empty result: shape not calibrated yet (graph capture) or no accurate candidate
            return r
    return torch.bmm(a, b, out_dtype=torch.float32)


def _acc(acc, t):
    """acc += t; None (the B200's first sample) takes t itself: no zero fill and add launches, the same values."""
    if acc is None:
        return t
    acc += t
    return acc


#: the H100 path's host-side tails: False = the torch ops the engine 2.5.9 ran (cat / masked_fill / addmm / copies); both give the same
#: bits (tests/gpu/test_bidir_triattn_h100.py). The A100 / RTX 5090 builds always run the torch ops.
LEGACY = False
#: parts switched back to the torch ops one by one (tests): "logits", "gemm", "tail", "gbwd", "smbwd", "cast", "mm"
OFF: set = set()


#: d_pair 128, K = 2 d_hidden or d_hidden in (128, 256), L L % 128 == 0: the gate and the out-projection with its residual in one kernel
#: (``tail`` of triattn_bidir_sm90.cu). Inference: no u planes; training (u saved for the backward): the kernel also writes u. The same
#: arithmetic as gate_u + torch.addmm (u rounded to bf16, k16 steps in order, the product rounded to bf16 and then x added): the outputs and
#: u equal the unfused tail's bits (tests/gpu/test_bidir_triattn_h100.py). BIDIR_FUSED_TAIL=0 switches it off; BIDIR_FUSED_TAIL_SAVE=0 only the training use.
FUSED_TAIL = os.environ.get("BIDIR_FUSED_TAIL", "1") != "0"
FUSED_TAIL_SAVE = os.environ.get("BIDIR_FUSED_TAIL_SAVE", "1") != "0"


def _fused_tail(save: bool, C: int, K: int, LL: int) -> bool:
    return (
        FUSED_TAIL and (FUSED_TAIL_SAVE or not save) and C == 128 and K in (128, 256) and LL % 128 == 0 and _planes_ext("tail") is not None
    )


def _planes_ext(part: str = ""):
    """The H100 extension with the logits pass and the residual out-projection, or None (legacy switch, or the sm_80 / sm_120 build)."""
    if LEGACY or part in OFF or _a100():
        return None
    return _ext()


def _planes_gemm(w, y):
    """C = [Wv; Wg; Wb] y^T, the channel planes [N, rows of y]: ``torch.mm(w, y.t())``'s bits (the fastest bit-equal cublasLt algorithm)."""
    e = _planes_ext("gemm")
    if e is None:
        return torch.mm(w, y.t())
    return e.cgemm(w, y, torch.empty(w.shape[0], y.shape[0], device=y.device, dtype=y.dtype))


#: Training backward, d_pair 128, K in (128, 256), L L % 64 == 0: du = Wo^T dy^T and the gate's backward in one kernel (``gate_bwd_fused`` of
#: triattn_bidir_sm90.cu: no du planes). The same arithmetic as torch.mm + gate_bwd (du rounded to bf16, k16 steps in order): equal bits
#: (tests/gpu/test_bidir_triattn_h100.py). BIDIR_FUSED_GBWD=0 switches it off.
FUSED_GBWD = os.environ.get("BIDIR_FUSED_GBWD", "1") != "0"


def _gate_bwd(dy, wo, g, o, dg):
    """-> dO [K, L L] (dG written into ``dg``): ``gate_bwd(torch.mm(wo.t(), dy.t()), g, o, dg)``; dy [L L, C] rows of one sample, wo [C, K]."""
    K = wo.shape[1]
    if FUSED_GBWD and wo.shape[0] == 128 and K in (128, 256) and dy.shape[0] % 64 == 0 and _planes_ext("gbwd") is not None:
        if B200_LNBWD and _b200() and K == 128:  # (N1) the same kernel with the gate / o loads of a tile issued before its GEMM: bitwise equal
            from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

            return b200_bidir.lnb().gate_bwd128(dy.contiguous(), wo, g, o, dg, 2)
        return _ext().gate_bwd_fused(dy.contiguous(), wo, g, o, dg)
    return _ext().gate_bwd(torch.mm(wo.t(), dy.t()), g, o, dg)


#: The softmax gradient's last step in one kernel (``softmax_bwd_planes`` of triattn_bidir_sm90.cu): the same fp32 ops as the torch
#: expression, one pass in place of the cat / float / sub / mul / transpose-copy passes. BIDIR_FUSED_SMBWD=0 switches it off.
FUSED_SMBWD = os.environ.get("BIDIR_FUSED_SMBWD", "1") != "0"


def _softmax_bwd(p, dp, db, nstart):
    """db [P, L L] (rows of the gradient planes) = d logits of the softmax P (p [P, L, L] bf16, dp [P, L, L] fp32): planes < nstart as they are, the
    others transposed ([i, k] -> [k, i]); d logits = p (dp - (p dp).sum(-1))."""
    bo = _bo()
    s = bo.bo_rowdot(p, dp) if bo is not None else torch.mul(p, dp).sum(-1, keepdim=True)  # B200: one launch (torch: a multiply + a reduce)
    e = _planes_ext("smbwd") if FUSED_SMBWD else None
    if e is not None:
        e.softmax_bwd_planes(p, dp, s, db, nstart)
        return
    pf = p.float()
    dl = pf * (dp - s)
    out = db.view(p.shape)
    out[:nstart] = dl[:nstart]
    out[nstart:] = dl[nstart:].transpose(1, 2)


def _mm_f32_into(out, a, b):
    """out (a [rows, N] fp32 slice) = a @ b, the same GEMM as ``_mm_f32`` without its result tensor and the copy into ``out``."""
    if LEGACY or "mm" in OFF or _a100():
        out.copy_(_mm_f32(a, b))
    else:
        torch.mm(a, b, out_dtype=torch.float32, out=out)


def _probs(bias: torch.Tensor, keep: torch.Tensor | None, H: int, L: int) -> torch.Tensor:
    """The bias planes [2 H, L L] of one sample -> [2 H, L, L] bf16: P_s (rows :H) and P_e (rows H:), each a softmax over its last axis k (masked keys -inf)."""
    e = _planes_ext("logits")
    if e is not None:  # one pass: widen, transpose the ending half, mask
        logits = e.logits(bias, keep, H, L)
    else:
        b = bias.view(2 * H, L, L).float()
        logits = torch.cat([b[:H], b[H:].transpose(1, 2)])  # starting [h, j, k]; ending [h, i, k] = bias_e[h, k, i]
        if keep is not None:
            logits = logits.masked_fill(~keep[None, None, :], float("-inf"))
    return torch.softmax(logits, -1).to(torch.bfloat16)


#: The RTX 5090 (sm_120a): the plane contractions on CUTLASS tensor-op tiles (``pair_weighted_averaging.cuda.gemm_sm120``) instead of cuBLAS, whose heuristic picks 32 x 32 WMMA
#: tiles there for these shapes (L 256, 4 x 32-channel heads: the P v contraction 51 -> 41 us, the out-projection with its residual 63 -> 43 us; L 512: 348 -> 209, 248 -> 137 us)
#: and copies the residual into the output first (addmm). The same accumulation (bf16 products, fp32 sums rounded once): within bf16 rounding of the cuBLAS bits.
#: From L 256 (at L 128 the call is the same speed). FOLDJET_ENGINE_BIDIR_SM120_GEMM=0 switches it off (the A100 and H100 never take it).
SM120_GEMM = os.environ.get("FOLDJET_ENGINE_BIDIR_SM120_GEMM", "1") != "0"
_PV_CFG, _OUT_CFG = 0, 2  # gemm_sm120.CFGS: 128 x 128 x 32 (P v); 256 x 128 x 32 (the out-projection, N = d_pair 128)


def _sm120_gemm() -> bool:
    return SM120_GEMM and torch.cuda.get_device_capability() == (12, 0)


def _pv(a, p, out):
    """out[h] = a[h] p[h]^T (a [H, dh L, L] the value planes of the heads, p [H, L, L] the softmax rows, out [H, dh L, L])."""
    if _sm120_gemm() and p.shape[-1] >= 256 and a.is_contiguous() and p.is_contiguous() and out.is_contiguous():
        from foldjet_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

        e = a.new_empty((0,))
        gemm_sm120.ext().mm(a, p, e, out, False, True, _PV_CFG, e.view(torch.uint8), 1)
        return
    torch.bmm(a, p.transpose(1, 2), out=out)


def _out_proj(x, u, wo, out):
    """out = x + u^T wo^T (x, out: [L L, C] rows of one sample; u [K, L L] planes; wo [C, K])."""
    if _sm120_gemm() and u.shape[1] >= 256 * 256 and wo.shape[0] == 128 and wo.shape[1] % 8 == 0 and u.is_contiguous() and wo.is_contiguous() and x.is_contiguous() and out.is_contiguous():
        from foldjet_engine.kernels.pair_weighted_averaging.cuda import gemm_sm120

        e = x.new_empty((0,))
        gemm_sm120.ext().mm(u, wo, x, out, True, True, _OUT_CFG, e.view(torch.uint8), 1)
        return
    torch.addmm(x, u.t(), wo.t(), out=out)


def _forward(pair, mask, lnw, lnb, eps, w, wo, H, Dh, save):
    """-> (out [B, L, L, C], saved or None). w = [Wv; Wg; Wb] [4 Dh + 2 H, C], wo [C, 2 Dh] (bf16)."""
    g = _glue()
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    y, mean, rstd = _ln_fwd(g, x, lnw, lnb, eps, L)
    out = torch.empty_like(x)
    planes, probs, os_, us = [], [], [], []
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c = _planes_gemm(w, y[rows])  # [4 Dh + 2 H, L L]
        p16 = _probs(c[4 * Dh :], None if mask is None else mask[b], H, L)
        ps, pe = p16[:H], p16[H:]
        o = torch.empty(2 * Dh, LL, device=x.device, dtype=torch.bfloat16)
        _pv(c[:Dh].view(H, dh * L, L), ps, o[:Dh].view(H, dh * L, L))
        bo = _bo(L)
        if bo is not None:  # B200: every ending head in one launch
            bo.bo_pv(pe, c[Dh : 2 * Dh], dh, False, o[Dh:])
        else:
            for h in range(H):
                ch = slice(Dh + h * dh, Dh + (h + 1) * dh)
                torch.bmm(pe[h].expand(dh, L, L), c[ch].view(dh, L, L), out=o[ch].view(dh, L, L))
        if _fused_tail(save, C, 2 * Dh, LL):
            u = torch.empty(2 * Dh, LL, device=x.device, dtype=torch.bfloat16) if save else None
            _ext().tail(c[2 * Dh : 4 * Dh], o, wo, x[rows], out[rows], u)
        else:
            u = g.gate_u(c[2 * Dh : 4 * Dh], o, 0)
            _out_proj(x[rows], u, wo, out[rows])
        if save:
            planes.append(c)
            probs.append(p16)
            os_.append(o)
            us.append(u)
    saved = (y, mean, rstd, planes, probs, os_, us) if save else None
    return out.view(B, L, L, C), saved


def _backward(dy, pair, lnw, w, wo, saved, H, Dh):
    g = _glue()
    y, mean, rstd, planes, probs, os_, us = saved
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    dy = dy.contiguous().view(-1, C)
    bo = _bo()
    dW = dWo = None  # the first sample's products start the sums (the torch-ops path: zeros + add, the same values)
    if bo is None:
        dW = torch.zeros(w.shape, device=x.device, dtype=torch.float32)
        dWo = torch.zeros(wo.shape, device=x.device, dtype=torch.float32)
    dyln = torch.empty(B * LL, C, device=x.device, dtype=torch.float32)
    dc = torch.empty(w.shape[0], LL, device=x.device, dtype=torch.bfloat16)
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c, p16, o, u = planes[b], probs[b], os_[b], us[b]
        ps, pe = p16[:H], p16[H:]
        s_wo = None
        if (B200_SIDE & 1) and B == 1 and L <= B200_SIDE_MAX_L and dWo is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_wo = side.fork(x.device)
        if s_wo is not None:  # branch: dWo beside the gate backward and the core GEMMs
            with side.on(s_wo):
                dWo = _mm_f32(dy[rows].t(), u.t())
        else:
            dWo = _acc(dWo, _mm_f32(dy[rows].t(), u.t()))
        do = _gate_bwd(dy[rows], wo, c[2 * Dh : 4 * Dh], o, dc[2 * Dh : 4 * Dh])  # dO; dG into dc
        # starting: O_s[h] = V_s[h] P_s[h]^T
        dos = do[:Dh].view(H, dh * L, L)
        torch.bmm(dos, ps, out=dc[:Dh].view(H, dh * L, L))
        dp = torch.empty(2 * H, L, L, device=x.device, dtype=torch.float32)  # d P: starting [H, j, k], ending [H, i, k]
        if bo is not None:  # B200: straight into dp (no copy)
            torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32, out=dp[:H])  # [H, j, k]
        else:
            dp[:H] = torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32)  # [H, j, k]
        # ending: O_e[c] = P_e[h] V_e[c]
        if (
            bo is not None and L <= BO_GEMM_MAX_L
        ):  # B200: every ending head in one launch each (dP: a deterministic cluster sum over the head's channels)
            bo.bo_pv(pe, do[Dh : 2 * Dh], dh, True, dc[Dh : 2 * Dh])
            bo.bo_dp(do[Dh : 2 * Dh], c[Dh : 2 * Dh], dh, dp[H:])
        else:
            for h in range(H):
                ch = slice(Dh + h * dh, Dh + (h + 1) * dh)
                doe = do[ch].view(dh, L, L)
                torch.bmm(pe[h].t().expand(dh, L, L), doe, out=dc[ch].view(dh, L, L))
                torch.sum(torch.bmm(doe, c[ch].view(dh, L, L).transpose(1, 2), out_dtype=torch.float32), 0, out=dp[H + h])  # [i, k]
        _softmax_bwd(p16, dp, dc[4 * Dh :], H)  # d logits -> the bias planes' gradient (the ending half back to [k, i])
        s_w = None
        if (B200_SIDE & 2) and B == 1 and dW is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_w = side.fork(x.device)
        if s_w is not None:  # branch: dW beside the dyln GEMM and the LayerNorm backward
            with side.on(s_w):
                dW = _mm_f32(dc, y[rows])
        else:
            dW = _acc(dW, _mm_f32(dc, y[rows]))
        _mm_f32_into(dyln[rows], dc.t(), w)
    dx, dgb = _ln_bwd(g, bo, dyln, x, dy, lnw, mean, rstd, L)  # + the residual's dy
    if B == 1 and (s_wo is not None or s_w is not None):
        from foldjet_engine.kernels.bias_only_dit.cuda import side

        side.join(x.device, s_wo)
        side.join(x.device, s_w)
    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo


def _weights(wv, wb, wg, wo):
    """[Wv; Wg; Wb] and Wo as bf16 (one launch from fp32 masters on the H100 path)."""
    e = _planes_ext("cast")
    if e is not None and all(t.dtype == torch.float32 and t.is_contiguous() and t.is_cuda for t in (wv, wb, wg, wo)):
        w, wo16 = e.cast_weights(wv, wg, wb, wo)
        return w, wo16
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
        assert dlw is not None and dlb is not None and dW is not None and dWo is not None
        ctx.saved = None
        dWv, dWg, dWb = dW[: 2 * Dh], dW[2 * Dh : 4 * Dh], dW[4 * Dh :]
        grads = [t.to(d) for t, d in zip((dlw, dlb, dWv, dWb, dWg, dWo), ctx.dtypes, strict=True)]
        return (dx, None, *grads, None, None, None)


def forward(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + bidirectional bias-only attention(pair)`` (the module's ``_forward_pytorch``), forward and, with grad, backward."""
    H, Dh = module.n_head, module.d_hidden
    params = (
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.to_value.weight,
        module.to_bias.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    mask = None if mask is None else mask.contiguous()
    grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in params))
    if not grad:
        w, wo = _weights(*params[2:])
        return _forward(pair, mask, params[0], params[1], module.ln_pair.eps, w, wo, H, Dh, False)[0]
    return _BidirFn.apply(pair, mask, *params, module.ln_pair.eps, H, Dh)


# ================================================================================================= one direction (TriangleAttention)
# The bias-only single-direction TriangleAttention (``use_self_attention=False``) on the same kernels: one direction's planes
# [Wv; Wg; Wb] y^T -> [Dh | Dh | H, L L], the starting node's or the ending node's contraction above, the gate, the out-projection. The
# ending node works in the pair's own frame (P_e[h][i, k] = softmax_k bias[h, k, i], O_e = P_e V_e): no transposed copy of the pair. The
# residual and the training dropout (drop_row / drop_col, a [B, L, 1, 1] / [B, 1, L, 1] scale on the update) are applied here, so the
# LayerNorm backward adds the residual's gradient in its pass.
def single_refusal(module, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
    """Why the single-direction bias-only path does not serve this call (None: it does)."""
    if not pair.is_cuda:
        return "the kernels run on CUDA tensors only"
    if torch.cuda.get_device_capability(pair.device) not in (
        (9, 0),
        (8, 0),
        (12, 0),
        (10, 0),
    ):  # 12.0: the sm_80 glue built for sm_120a (sm80_bidir); 10.0: the glue built for sm_100a (b200_bidir)
        return "the bias-only TriangleAttention cuda path is the H100 (sm_90a), A100 (sm_80) and RTX 5090 (sm_120a) one"
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: bf16 activations only (the bias-only path's glue kernels)"
    if pair.dim() != 4 or pair.shape[1] != pair.shape[2]:
        return f"pair must be [B, L, L, d_pair], got {tuple(pair.shape)}"
    if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(pair.shape[:2])):
        return "mask must be a bool [B, L] key mask"
    C, L = pair.shape[-1], pair.shape[1]
    Dh = module.to_value.weight.shape[0]
    if C % 8 or C > MAX_C:
        return f"d_pair {C}: the glue LayerNorm takes d_pair % 8 == 0, <= {MAX_C}"
    if L % 8:
        return f"L {L}: the planes' GEMMs and the gate kernels take L % 8 == 0"
    if (Dh // module.n_head) % 8 or Dh % 8:
        return f"hidden width {Dh} / {module.n_head} heads: the planes' GEMMs take heads of a multiple of 8 channels"
    if module.ln_pair.weight.dtype != torch.float32 or module.ln_pair.bias is None or module.ln_pair.bias.dtype != torch.float32:
        return "the LayerNorm affine must be fp32 with a bias (the module's pin)"
    if module.to_gate.bias is not None or module.to_out.bias is not None or module.dual_output or module.untransposed_output:
        return "gate / out-projection biases or a Chai-1 output layout (not on the bias-only path)"
    if not module.starting and not module.transposed_bias:
        return "AlphaFold 3's ending-node bias orientation (the bias-only path reads the transposed pair's bias)"
    return None


def _forward1(pair, mask, drop, lnw, lnb, eps, w, wo, H, Dh, starting, save):
    """One direction -> (out = pair + drop * update [B, L, L, C], saved or None). w = [Wv; Wg; Wb] [2 Dh + H, C], wo [C, Dh] (bf16)."""
    g = _glue()
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    y, mean, rstd = _ln_fwd(g, x, lnw, lnb, eps, L)
    out = torch.empty_like(x)
    planes, probs, os_, us = [], [], [], []
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c = _planes_gemm(w, y[rows])  # [2 Dh + H, L L]
        e = _planes_ext("logits")
        if e is not None:
            logits = e.logits(c[2 * Dh :], None if mask is None else mask[b], H if starting else 0, L)
        else:
            logits = c[2 * Dh :].view(H, L, L).float()
            if not starting:
                logits = logits.transpose(1, 2)  # [h, i, k] = bias[h, k, i]
            if mask is not None:
                logits = logits.masked_fill(~mask[b][None, None, :], float("-inf"))
        p = torch.softmax(logits, -1).to(torch.bfloat16)
        o = torch.empty(Dh, LL, device=x.device, dtype=torch.bfloat16)
        if starting:
            _pv(c[:Dh].view(H, dh * L, L), p, o.view(H, dh * L, L))
        else:
            bo = _bo(L)
            if bo is not None:  # B200: every head in one launch
                bo.bo_pv(p, c[:Dh], dh, False, o)
            else:
                for h in range(H):
                    ch = slice(h * dh, (h + 1) * dh)
                    torch.bmm(p[h].expand(dh, L, L), c[ch].view(dh, L, L), out=o[ch].view(dh, L, L))
        if drop is None and _fused_tail(save, C, Dh, LL):
            u = torch.empty(Dh, LL, device=x.device, dtype=torch.bfloat16) if save else None
            _ext().tail(c[Dh : 2 * Dh], o, wo, x[rows], out[rows], u)
        else:
            u = g.gate_u(c[Dh : 2 * Dh], o, 0)
            if drop is None:
                _out_proj(x[rows], u, wo, out[rows])
            elif _b200():  # B200: scale and residual in one pass (the broadcast torch ops took ~3x the bandwidth time)
                t = _b200_wide().tri_scale_rows(torch.mm(u.t(), wo.t()), drop[b].reshape(L, C), L, not starting, x[rows])
                if B == 1:
                    out = t
                else:
                    out[rows] = t
            else:
                t = torch.mm(u.t(), wo.t()).view(L, L, C)
                out[rows] = (x[rows].view(L, L, C) + t * drop[b]).view(LL, C)
        if save:
            planes.append(c)
            probs.append(p)
            os_.append(o)
            us.append(u)
    saved = (y, mean, rstd, planes, probs, os_, us) if save else None
    return out.view(B, L, L, C), saved


def _backward1(dy, pair, drop, lnw, w, wo, saved, H, Dh, starting):
    g = _glue()
    y, mean, rstd, planes, probs, os_, us = saved
    B, L, _, C = pair.shape
    LL, dh = L * L, Dh // H
    x = pair.contiguous().view(-1, C)
    dy = dy.contiguous().view(-1, C)
    bo = _bo()
    dW = dWo = None  # the first sample's products start the sums (the torch-ops path: zeros + add, the same values)
    if bo is None:
        dW = torch.zeros(w.shape, device=x.device, dtype=torch.float32)
        dWo = torch.zeros(wo.shape, device=x.device, dtype=torch.float32)
    dyln = torch.empty(B * LL, C, device=x.device, dtype=torch.bfloat16 if (DYLN_BF16 and _b200()) else torch.float32)
    dc = torch.empty(w.shape[0], LL, device=x.device, dtype=torch.bfloat16)
    for b in range(B):
        rows = slice(b * LL, (b + 1) * LL)
        c, p, o, u = planes[b], probs[b], os_[b], us[b]
        if drop is None:
            dt = dy[rows]
        elif _b200():  # B200: one bandwidth pass (the broadcast torch multiply took ~3x)
            dt = _b200_wide().tri_scale_rows(dy[rows], drop[b].reshape(L, C), L, not starting)
        else:
            dt = (dy[rows].view(L, L, C) * drop[b]).view(LL, C)  # the update's gradient
        s_wo = None
        if (B200_SIDE & 1) and B == 1 and L <= B200_SIDE_MAX_L and dWo is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_wo = side.fork(x.device)
        if s_wo is not None:  # branch: dWo beside the gate backward and the core GEMMs
            with side.on(s_wo):
                dWo = _mm_f32(dt.t(), u.t())
        else:
            dWo = _acc(dWo, _mm_f32(dt.t(), u.t()))
        do = _gate_bwd(dt, wo, c[Dh : 2 * Dh], o, dc[Dh : 2 * Dh])  # dO; dG into dc
        if starting:  # O[h] = V[h] P[h]^T
            dos = do.view(H, dh * L, L)
            torch.bmm(dos, p, out=dc[:Dh].view(H, dh * L, L))
            dp = _bmm_f32(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L))  # [H, j, k]
        else:  # O[c] = P[h] V[c]
            dp = torch.empty(H, L, L, device=x.device, dtype=torch.float32)
            if bo is not None and L <= BO_GEMM_MAX_L:  # B200: every head in one launch each
                bo.bo_pv(p, do, dh, True, dc[:Dh])
                bo.bo_dp(do, c[:Dh], dh, dp)
            else:
                for h in range(H):
                    ch = slice(h * dh, (h + 1) * dh)
                    doe = do[ch].view(dh, L, L)
                    torch.bmm(p[h].t().expand(dh, L, L), doe, out=dc[ch].view(dh, L, L))
                    torch.sum(torch.bmm(doe, c[ch].view(dh, L, L).transpose(1, 2), out_dtype=torch.float32), 0, out=dp[h])  # [i, k]
        _softmax_bwd(p, dp, dc[2 * Dh :], H if starting else 0)  # d logits -> the bias plane's layout
        s_w = None
        if (B200_SIDE & 2) and B == 1 and dW is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_w = side.fork(x.device)
        if s_w is not None:  # branch: dW beside the dyln GEMM and the LayerNorm backward
            with side.on(s_w):
                dW = _mm_f32(dc, y[rows])
        else:
            dW = _acc(dW, _mm_f32(dc, y[rows]))
        if dyln.dtype == torch.bfloat16:
            torch.mm(dc.t(), w, out=dyln[rows])
        else:
            _mm_f32_into(dyln[rows], dc.t(), w)
    dx, dgb = _ln_bwd(g, bo, dyln, x, dy, lnw, mean, rstd, L)  # + the residual's dy
    if B == 1 and (s_wo is not None or s_w is not None):
        from foldjet_engine.kernels.bias_only_dit.cuda import side

        side.join(x.device, s_wo)
        side.join(x.device, s_w)
    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo


def _weights1(wv, wb, wg, wo):
    return _weights(wv, wb, wg, wo)


class _SingleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, mask, drop, lnw, lnb, wv, wb, wg, wo, eps, H, Dh, starting):
        w, wo16 = _weights1(wv, wb, wg, wo)
        out, saved = _forward1(pair, mask, drop, lnw, lnb, eps, w, wo16, H, Dh, starting, True)
        ctx.H, ctx.Dh, ctx.starting, ctx.saved, ctx.drop = H, Dh, starting, saved, drop
        ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wv, wb, wg, wo))
        ctx.save_for_backward(pair, lnw, w, wo16)
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, w, wo16 = ctx.saved_tensors
        H, Dh = ctx.H, ctx.Dh
        dx, dlw, dlb, dW, dWo = _backward1(dy, pair, ctx.drop, lnw, w, wo16, ctx.saved, H, Dh, ctx.starting)
        assert dlw is not None and dlb is not None and dW is not None and dWo is not None
        ctx.saved = ctx.drop = None
        dWv, dWg, dWb = dW[:Dh], dW[Dh : 2 * Dh], dW[2 * Dh :]
        grads = [t.to(d) for t, d in zip((dlw, dlb, dWv, dWb, dWg, dWo), ctx.dtypes, strict=True)]
        return (dx, None, None, *grads, None, None, None, None)


#: B200 (S90), inference at one 128-token tile (L <= 128), d_pair = d_hidden = 128 with 4 heads: the bias-only update is the self-attention
#: update with q = k = 0 (the logits are the bias alone, softmax(b_jk) is the same for every row i), so it runs on the fused one-kernel
#: L128 path of ``b200_triattn.fused_forward`` with zero Wq / Wk (one launch instead of the eight-launch plane chain; the redundant
#: q.k exponentials are 1.8 us of MUFU at L128). FOLDJET_ENGINE_TRIATTN_B200_BO_L128=0 switches it off. Above one tile the L^3 exponentials
#: and the extra q / k stores are not free: the chain stays.
B200_BO_L128 = os.environ.get("FOLDJET_ENGINE_TRIATTN_B200_BO_L128", "1") != "0"
_ZERO_QK: dict = {}


def _b200_one_tile(module, pair: torch.Tensor, mask: torch.Tensor | None, params) -> torch.Tensor | None:
    from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn as k

    ln_w, ln_b, wv, wb, wg, wo = params
    one_tile = k.L128_FUSED and pair.shape[1] <= k.TOK
    planes = k.BO_PLANES and not one_tile and pair.shape[0] == 1 and pair.shape[1] % k.TOK == 0
    if not ((one_tile or planes) and pair.shape[-1] == k.C and module.n_head == k.H):
        return None
    if tuple(wv.shape) != (k.C, k.C) or tuple(wg.shape) != (k.C, k.C) or tuple(wo.shape) != (k.C, k.C) or tuple(wb.shape) != (k.H, k.C):
        return None
    if module.to_gate.bias is not None or module.to_out.bias is not None or module.dual_output or module.untransposed_output:
        return None
    key = (wv.device, wv.dtype)
    z = _ZERO_QK.get(key)
    if z is None:
        z = _ZERO_QK[key] = torch.zeros(k.C, k.C, device=wv.device, dtype=wv.dtype)
    starting = module.starting
    if planes:
        return k.bias_only_planes(pair, ln_w, ln_b, wv, wg, wb, wo, mask, module.ln_pair.eps, starting, z)
    y, _ = k.fused_forward(pair, ln_w, ln_b, z, z, wv, wg, wb, wo, mask, module.ln_pair.eps, None, False, None, not starting, False)
    return y


def forward_single(module, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """``pair + drop(bias-only triangle attention(pair))`` of a single-direction ``TriangleAttention(use_self_attention=False)`` (the
    module's ``_forward_pytorch``; the dropout in training mode), forward and, with grad, backward."""
    from foldjet_engine.modules.functional import dropout_scale

    H, Dh = module.n_head, module.to_value.weight.shape[0]
    params = (
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.to_value.weight,
        module.to_bias.weight,
        module.to_gate.weight,
        module.to_out.weight,
    )
    mask = None if mask is None else mask.contiguous()
    drop = None
    if module.p_drop > 0.0 and module.training:
        if _b200():  # B200: the H100 Philox kernel built for sm_100a, one launch (bitwise the torch draw; None: not a BF16 square pair)
            from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn

            drop = b200_triattn.drop_scale(pair, module.p_drop)
            drop = None if drop is None else drop.unsqueeze(module._drop_dim)
        if drop is None:
            drop = dropout_scale(pair, module.p_drop, module._drop_dim).to(pair.dtype)  # [B, 1, L, C] / [B, L, 1, C]
    grad = torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in params))
    if not grad and drop is None and B200_BO_L128 and _b200():
        y = _b200_one_tile(module, pair, mask, params)
        if y is not None:
            return y
    if not grad:
        w, wo = _weights1(*params[2:])
        return _forward1(pair, mask, drop, params[0], params[1], module.ln_pair.eps, w, wo, H, Dh, module.starting, False)[0]
    return _SingleFn.apply(pair, mask, drop, *params, module.ln_pair.eps, H, Dh, module.starting)
