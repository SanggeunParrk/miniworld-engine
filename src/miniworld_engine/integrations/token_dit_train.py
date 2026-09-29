"""Fused token DiT TRAINING block on B200 (sm_100), wired to DiTBlock's parameter contract.

One autograd Function per ``DiTBlock`` call, forward and backward, with no Triton and no quack:

  * GEMMs: cuBLAS, bf16 operands, fp32 accumulation (activations bf16, weight gradients fp32);
  * the six conditioning projections as two GEMMs (every block-internal LayerNorm of cond shares its statistics, so its
    weight folds into the projection; the two output gates read raw cond), q|k|v|g as one GEMM;
  * the pair bias: a CUDA LayerNorm of the pair rows and one GEMM, head-major, the shared key mask folded in as -inf;
  * the attention core: ``kernels/augmented_attention/cuda/sm100`` (attn_fwd2, attn_dqb + attn_dkv);
  * every elementwise / row step: ``kernels/conditioned_transition/cuda/token_dit_train_rows.cu``;
  * the residual stream fp32 inside the block, the output in the input's dtype.

``serves()`` is the whole gate: autograd on, the engine's kernel backend (implementation TRITON or MINIWORLD), B200, bf16 (the input's dtype or ``compute_dtype``), the token widths
(768 / 16 x 48 / cond 384 / pair 128 / transition n = 2), B == 1, an even A, L a multiple of 128, a key mask of [B, L]
or none, LayerNorm eps 1e-5. Anything else keeps the module path. MINIWORLD_TOKEN_DIT_TRAIN=0 turns it off.

The forward and the backward are each one opaque op (``kernels/_compile.opaque``) with a fake implementation, inside an
autograd Function, so torch.compile keeps them as single nodes (as the H100 TriMul training path does).

Unlike the research stack runner (branch ``b200/token-dit``), a DiTBlock call sees one block, so the pair bias and the cond
LayerNorm are computed per block rather than once per stack.
"""

from __future__ import annotations

import os

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

D, DC, DP, H, DH = 768, 384, 128, 16, 48
EPS = 1e-5
BF = torch.bfloat16

ATT = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "to_query.weight", "to_query.bias", "to_key.weight", "to_value.weight",
       "to_gate.weight", "to_out.weight", "ln_pair.weight", "to_bias.weight")
QKN = ("norm_query.weight", "norm_key.weight")
TRN = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "expand_a.weight", "expand_b.weight", "squeeze.weight")


def serves(module, single, cond, pair, mask, compute_dtype=None) -> bool:
    if os.environ.get("MINIWORLD_TOKEN_DIT_TRAIN", "1") == "0" or not torch.is_grad_enabled():
        return False
    if settings.current().engine_backend == "triton":
        return False
    a = module.attention
    # the engine's kernels (TRITON or MINIWORLD resolve to them), as the sm_90 / sm_100 attention cores gate on
    from miniworld_engine.modules.dispatch import KernelBackend
    if a._backend != KernelBackend.TRITON:
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if not (single.dtype is BF or compute_dtype is BF) or single.dtype not in (BF, torch.float32):
        return False
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(p.requires_grad for p in module.parameters()):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != D or single.shape[0] % 2 or single.shape[2] % 128:
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (1, single.shape[2], single.shape[2], DP):
        return False
    if (a.n_head, module.transition.expand_a.weight.shape[0]) != (H, 2 * D):
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])):
        return False
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, module.transition.ada_ln_in.ln_in,
             module.transition.ada_ln_in.ln_cond)
    return all(n.eps == EPS for n in norms)


def _f32(t):
    return t.detach().float().contiguous()


_PACKS: dict = {}


def _pack(P, qk, dev):
    """The block's weights in the kernels' layouts (bf16 GEMM operands, folded cond / pair LayerNorm weights, fp32 bias
    vectors). Rebuilt only when a parameter changes -- an optimizer step bumps ``_version`` -- so a training step pays
    it once per block, not once per forward."""
    key = tuple((t.data_ptr(), t._version) for t in P.values())
    slot = next(iter(P.values())).data_ptr()
    hit = _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    g = lambda n: P[n]  # noqa: E731
    w1, w2 = _f32(g("attention.ada_ln_in.ln_cond.weight")), _f32(g("transition.ada_ln_in.ln_cond.weight"))
    Wraw = torch.cat([_f32(g("attention.ada_ln_in.to_scale.weight")), _f32(g("attention.ada_ln_in.to_bias.weight")),
                      _f32(g("transition.ada_ln_in.to_scale.weight")), _f32(g("transition.ada_ln_in.to_bias.weight"))])
    wp, Wb = _f32(g("attention.ln_pair.weight")), _f32(g("attention.to_bias.weight"))
    Wf = Wb * wp
    W = dict(
        w1=w1, w2=w2, Wraw=Wraw, Wn=torch.cat([Wraw[:2 * D] * w1, Wraw[2 * D:] * w2]).to(BF),   # cond-LN weights folded
        Wg=torch.cat([g("attention.to_scale.weight"), g("transition.to_scale.weight")]).detach().to(BF),
        bs1=_f32(g("attention.ada_ln_in.to_scale.bias")), bs2=_f32(g("transition.ada_ln_in.to_scale.bias")),
        bg1=_f32(g("attention.to_scale.bias")), bg2=_f32(g("transition.to_scale.bias")),
        Wqkvg=torch.cat([g("attention.to_query.weight"), g("attention.to_key.weight"), g("attention.to_value.weight"),
                         g("attention.to_gate.weight")]).detach().to(BF),
        bqkvg=torch.cat([g("attention.to_query.bias").detach().float(), torch.zeros(3 * D, device=dev)]).to(BF),
        nq=_f32(g("attention.norm_query.weight")) if qk else torch.ones(DH, device=dev),
        nk=_f32(g("attention.norm_key.weight")) if qk else torch.ones(DH, device=dev),
        wp=wp, Wb=Wb, Wf=Wf, Wf_bf=Wf.to(BF),
        Wo=g("attention.to_out.weight").detach().to(BF).contiguous(),
        Wab=torch.cat([g("transition.expand_a.weight"), g("transition.expand_b.weight")]).detach().to(BF),
        Wsq=g("transition.squeeze.weight").detach().to(BF).contiguous(),
    )
    _PACKS[slot] = (key, W)
    return W


def _names_for(qk):
    return ["attention." + n for n in ATT + (QKN if qk else ())] + ["transition." + n for n in TRN]


def _saved_like(single, pair):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd)."""
    A, _, L, _ = single.shape
    M, R, e = A * L, L * L, single.new_empty
    f32 = torch.float32
    return [e((M, D), dtype=f32), e((M, DC), dtype=BF), e((M, DC), dtype=BF), e((M, 2), dtype=f32), e((M, 4 * D), dtype=BF),
            e((M, 2 * D), dtype=BF), e((M, 2), dtype=f32), e((M, D), dtype=BF), e((M, 4 * D), dtype=BF), e((M, 32), dtype=f32),
            e((M, D), dtype=BF), e((M, D), dtype=BF), e((M, D), dtype=BF), e((H, L, L), dtype=BF), e((M, D), dtype=f32),
            e((A, H, L), dtype=f32), e((M, D), dtype=BF), e((M, D), dtype=BF), e((M, D), dtype=f32), e((M, 2), dtype=f32),
            e((M, D), dtype=BF), e((M, 4 * D), dtype=BF), e((M, D), dtype=BF), e((R, DP), dtype=BF), e((R, 2), dtype=f32)]


def _fwd_fake(single, cond, pair, mask, params, qk, eq, ek):
    return [torch.empty_like(single), *_saved_like(single, pair)]


@opaque(fake=_fwd_fake, name="token_dit_train_sm100_fwd")
def _fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         qk: bool, eq: float, ek: float) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    T = ext()
    names = _names_for(qk)
    P = dict(zip(names, params))
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    x = single.reshape(M, D).to(torch.float32, copy=True)                    # the fp32 residual (never aliases the input)
    c2 = cond.reshape(M, DC).contiguous()
    chat = torch.empty(M, DC, device=dev, dtype=BF); cbf = torch.empty_like(chat)
    cst = torch.empty(M, 2, device=dev)
    T.cond_prep(c2, chat, cbf, cst, EPS)
    W = _pack(P, qk, dev)
    Wn, Wraw, Wg, w1, w2 = W["Wn"], W["Wraw"], W["Wg"], W["w1"], W["w2"]
    G = torch.mm(chat, Wn.t())                                                  # [M, 4D] s1 | sh1 | s2 | sh2
    Gg = torch.mm(cbf, Wg.t())                                                  # [M, 2D] gate1 | gate2
    bs1, bs2, bg1, bg2 = W["bs1"], W["bs2"], W["bg1"], W["bg2"]
    xa = torch.empty(M, D, device=dev, dtype=BF); xst = torch.empty(M, 2, device=dev)
    T.adaln_a(x, G, bs1, xa, xst, EPS)
    Wqkvg, bqkvg = W["Wqkvg"], W["bqkvg"]
    qkvg = torch.addmm(bqkvg, xa, Wqkvg.t())
    qn = torch.empty(M, D, device=dev, dtype=BF); kn = torch.empty_like(qn); vc = torch.empty_like(qn)
    rqk = torch.empty(M, 32, device=dev)
    nq, nk = W["nq"], W["nk"]
    T.qknorm(qkvg, nq, nk, qn, kn, vc, rqk, eq, ek, qk)
    p2 = pair.reshape(R, DP)
    ph = torch.empty(R, DP, device=dev, dtype=BF); pst = torch.empty(R, 2, device=dev)
    T.pair_ln(p2, ph, pst, EPS)
    wp, Wb, Wf = W["wp"], W["Wb"], W["Wf"]                                     # Wf = Wb diag(wp): ln_pair weight folded
    bias = torch.mm(W["Wf_bf"], ph.t()).view(H, L, L)                           # head-major, natural units
    if mask is not None:
        bias.masked_fill_(~mask.reshape(1, 1, L), float("-inf"))
    O, LSE = sm100.forward(qn, kn, vc, bias, A, L)
    og = torch.empty(M, D, device=dev, dtype=BF)
    T.gate_o(O, qkvg, og)
    Wo = W["Wo"]
    y = torch.mm(og, Wo.t())
    x1 = torch.empty_like(x); xt = torch.empty(M, D, device=dev, dtype=BF); x1st = torch.empty(M, 2, device=dev)
    T.res_adaln_b(x, y, Gg, bg1, G, bs2, x1, xt, x1st, EPS)
    Wab = W["Wab"]
    ab = torch.mm(xt, Wab.t())                                                  # [M, 2 * 2D] a | b
    h = torch.empty(M, 2 * D, device=dev, dtype=BF)
    rows.swiglu_rows(ab, h)
    Wsq = W["Wsq"]
    z = torch.mm(h, Wsq.t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    T.res_c(x1, z, Gg, bg2, out)
    return [out.view(single.shape), x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias, O, LSE, og, y, x1, x1st, xt,
            ab, z, ph, pst]


def _bwd_fake(single, cond, pair, mask, params, saved, dout, qk, eq, ek):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_bwd_fake, name="token_dit_train_sm100_bwd")
def _bwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         saved: list[torch.Tensor], dout: torch.Tensor, qk: bool, eq: float, ek: float) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    T = ext()
    (x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias, O, LSE, og, y, x1, x1st, xt, ab, z, ph, pst) = saved
    names = _names_for(qk)
    W = _pack(dict(zip(names, params)), qk, single.device)
    Wn, Wraw, Wg, Wqkvg, Wo, Wab, Wsq, Wf = W["Wn"], W["Wraw"], W["Wg"], W["Wqkvg"], W["Wo"], W["Wab"], W["Wsq"], W["Wf"]
    nq, nk, bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (W[k] for k in ("nq", "nk", "bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb"))
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype, f32 = single.dtype, cond.dtype, pair.dtype, torch.float32
    c2, p2 = cond.reshape(M, DC).contiguous(), pair.reshape(R, DP)
    dout = dout.reshape(M, D).to(sdt).contiguous()
    from miniworld_engine.kernels.conditioned_transition.cuda.train import partials
    allp = partials(M, 7 * D, dev)                     # per-block column sums: bg2 bs2 bg1 bs1 bq | dwq dwk (per head)
    part = [allp[:, i * D:(i + 1) * D] for i in range(5)]
    pwqk = allp[:, 5 * D:]
    dG = torch.empty(M, 4 * D, device=dev, dtype=BF); dGg = torch.empty(M, 2 * D, device=dev, dtype=BF)
    dz = torch.empty(M, D, device=dev, dtype=BF)
    T.res_c_bwd(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = torch.mm(dz, Wsq)                                                      # [M, 2D]
    dab = torch.empty_like(ab); h = torch.empty_like(dh)
    T.swiglu_bwd(dh, ab, dab, h)
    dWsq = torch.mm(dz.t(), h, out_dtype=f32)
    dxt = torch.mm(dab, Wab)
    dWab = torch.mm(dab.t(), xt, out_dtype=f32)
    dx1 = torch.empty(M, D, device=dev); dy = torch.empty(M, D, device=dev, dtype=BF)
    T.res_adaln_b_bwd(dout, dxt, x1, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2])
    dog = torch.mm(dy, Wo)
    dWo = torch.mm(dy.t(), og, out_dtype=f32)
    dob = torch.empty(M, D, device=dev, dtype=BF); dd = torch.empty(A, H, L, device=dev)
    dqkvg = torch.empty(M, 4 * D, device=dev, dtype=BF)
    T.gate_o_bwd(dog, O, qkvg, dob, dd, dqkvg, L)
    DQ, DK, DV, DB = sm100.backward(qn, kn, vc, dob, bias, LSE, dd, A, L)
    T.qknorm_bwd(DQ, DK, DV, qkvg, rqk, nq, nk, dqkvg, part[4], pwqk, qk)
    dxa = torch.mm(dqkvg, Wqkvg)
    dWqkvg = torch.mm(dqkvg.t(), xa, out_dtype=f32)
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    T.adaln_a_bwd(dxa, x, xst, G, bs1, dx1, dx, dG, part[3])
    sums = allp.sum(0) if qk else allp[:, :5 * D].sum(0)                        # one reduction for every column sum
    pb = sums[:5 * D].view(5, D)
    dwqk = sums[5 * D:].view(2, H, DH).sum(1).reshape(-1) if qk else torch.zeros(2 * DH, device=dev)
    dchat = torch.mm(dG, Wn)
    dWn = torch.mm(dG.t(), chat, out_dtype=f32)
    dcg = torch.mm(dGg, Wg)
    dWg = torch.mm(dGg.t(), cbf, out_dtype=f32)
    dc = torch.empty(M, DC, device=dev, dtype=cdt)
    T.cond_bwd(dchat, dcg, c2, cst, dc)
    dWu = torch.empty_like(dWn); dw12 = torch.zeros(2, DC, device=dev)
    Ws1, Wb1, Ws2, Wb2 = (Wraw[i * D:(i + 1) * D] for i in range(4))
    T.unfold_lnw(dWn, Ws1, Wb1, Ws2, Wb2, w1, w2, dWu, dw12)
    dbv = DB.view(H, R).to(BF)                                                  # masked key columns are 0 (P = 0)
    dWf = torch.mm(dbv, ph, out_dtype=f32)                                      # [H, DP]
    dph = torch.mm(dbv.t(), Wf.to(BF), out_dtype=f32)                           # [R, DP]
    dpair = torch.empty(R, DP, device=dev, dtype=pdtype)
    T.pair_ln_bwd(dph, p2, pst, dpair)
    dWs1, dWb1, dWs2, dWb2 = dWu.split(D)
    grads = {
        "attention.ada_ln_in.ln_cond.weight": dw12[0], "attention.ada_ln_in.to_scale.weight": dWs1,
        "attention.ada_ln_in.to_scale.bias": pb[3], "attention.ada_ln_in.to_bias.weight": dWb1,
        "attention.to_scale.weight": dWg[:D], "attention.to_scale.bias": pb[2],
        "attention.to_query.weight": dWqkvg[:D], "attention.to_query.bias": pb[4], "attention.to_key.weight": dWqkvg[D:2 * D],
        "attention.to_value.weight": dWqkvg[2 * D:3 * D], "attention.to_gate.weight": dWqkvg[3 * D:],
        "attention.norm_query.weight": dwqk[:DH], "attention.norm_key.weight": dwqk[DH:],
        "attention.to_out.weight": dWo, "attention.ln_pair.weight": (dWf * Wb).sum(0), "attention.to_bias.weight": dWf * wp,
        "transition.ada_ln_in.ln_cond.weight": dw12[1], "transition.ada_ln_in.to_scale.weight": dWs2,
        "transition.ada_ln_in.to_scale.bias": pb[1], "transition.ada_ln_in.to_bias.weight": dWb2,
        "transition.to_scale.weight": dWg[D:], "transition.to_scale.bias": pb[0],
        "transition.expand_a.weight": dWab[:2 * D], "transition.expand_b.weight": dWab[2 * D:], "transition.squeeze.weight": dWsq,
    }
    # custom-op outputs may not alias each other: the gradients above are views of a few shared buffers, so each leaves as
    # its own copy (in the parameter's dtype)
    return [dx.view(A, 1, L, D), dc.view(A, 1, L, DC), dpair.view(1, L, L, DP),
            *(grads[n].to(p.dtype, copy=True).reshape(p.shape) for n, p in zip(names, params))]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, qk, eq, ek, *params):  # noqa: D102
        out, *saved = _fwd(single, cond, pair, mask, list(params), qk, eq, ek)
        ctx.save_for_backward(single, cond, pair, *params, *saved)
        ctx.meta = (mask, qk, eq, ek, len(params))
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        mask, qk, eq, ek, npar = ctx.meta
        vals = ctx.saved_tensors
        single, cond, pair = vals[:3]
        params, saved = list(vals[3:3 + npar]), list(vals[3 + npar:])
        dx, dc, dpair, *pg = _bwd(single, cond, pair, mask, params, saved, dout.contiguous(), qk, eq, ek)
        return (dx, dc, dpair, None, None, None, None, *(g if ctx.needs_input_grad[7 + i] else None for i, g in enumerate(pg)))


def block(module, single, cond, pair, mask=None):
    """One DiTBlock (attention + transition, both residuals) through the fused training path. Call ``serves`` first."""
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    return _Block.apply(single, cond, pair.contiguous(), mask, qk, eq, ek, *[module.get_parameter(n) for n in _names_for(qk)])


__all__ = ["block", "serves"]
