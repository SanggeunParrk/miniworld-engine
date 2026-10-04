"""Fused bias-only token DiT TRAINING block on B200 (sm_100), wired to ``BiasOnlyDiTBlock``'s parameter contract.

One autograd Function per block call, its forward and backward each one opaque op, CUDA and cuBLAS only: cuBLAS GEMMs (bf16
operands, fp32 accumulation, weight gradients in the parameter's dtype), the expand GEMM with the SwiGLU in its epilogue
(``gemm_swiglu2_sm100`` keeping a | b), and the row kernels of ``kernels/bias_only_dit/cuda/bias_only_dit_train_rows.cu``
(conditioning LN, AdaLN, residual + gate, their backwards, the pair bias on mma.sync). The attention is the bias-only one:

  forward   bias = LN(pair) Wf^T (pair_bias: LN(pair) stays on chip); P = softmax(bias) once per call for every sample
            (softmax_t, the key mask folded in, P^T written beside it); a = sigmoid(g) (P v) per head and sample (pv_gate_inf)
  backward  do = da sigmoid(g), dg = da a (1 - sigmoid(g)), D = sum da a per row and head (gate_bwd_rows: a = sigmoid(g) o, so
            the pre-gate o is never stored); dv = P^T do (pv_gate_inf without the gate, on the transposed P);
            dbias = P o (sum_a do v^T - D) (dpb_sm100); d pair and dWf = dbias LN(pair) (pair_bias_bwd) -> to_bias / ln_pair

``serves()`` is the whole gate: autograd on, the engine's kernels (implementation MINIWORLD or TRITON), B200, bf16 inputs, the
token widths (768; the attention as 16 heads x 48, 24 x 32, 12 x 64 or 16 x 64 / cond 384 / pair 128 / transition n = 2), B == 1, L a multiple of 128 up to 768, a per-sample
conditioning [A, 1, L, 384], a key mask [1, L] or none, LayerNorm eps 1e-5. MINIWORLD_BIAS_ONLY_DIT_TRAIN=0 turns it off.
"""

from __future__ import annotations

import math
import os

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

D, DC, DP = 768, 384, 128
#: (n_head, head width) served: 768 attention channels as 16 x 48, 24 x 32, 12 x 64, 1024 as 16 x 64 (read off the weights:
#: to_bias.weight rows = heads, to_value.weight rows = channels)
LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))
EPS = 1e-5
BF = torch.bfloat16

ATT = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "to_value.weight", "to_gate.weight", "to_out.weight", "ln_pair.weight",
       "to_bias.weight")
TRN = ("ada_ln_in.ln_cond.weight", "ada_ln_in.to_scale.weight", "ada_ln_in.to_scale.bias", "ada_ln_in.to_bias.weight",
       "to_scale.weight", "to_scale.bias", "expand_a.weight", "expand_b.weight", "squeeze.weight")
NAMES = ["attention." + n for n in ATT] + ["transition." + n for n in TRN]


def serves(module, single, cond, pair, mask=None) -> bool:
    from miniworld_engine.modules.exceptions import ImplementationType

    if os.environ.get("MINIWORLD_BIAS_ONLY_DIT_TRAIN", "1") == "0" or not torch.is_grad_enabled():
        return False
    if module.implementation not in (ImplementationType.MINIWORLD, ImplementationType.TRITON):
        return False
    if settings.current().engine_backend == "triton":
        return False
    if not (single.is_cuda and torch.cuda.get_device_capability(single.device) == (10, 0)):
        return False
    if not all(t.dtype is BF for t in (single, cond, pair)):
        return False
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(p.requires_grad for p in module.parameters()):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != D or single.shape[2] % 128 or single.shape[2] > 768:
        return False
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (1, single.shape[2], single.shape[2], DP):
        return False
    at = module.attention
    da = at.to_value.weight.shape[0]
    if (at.n_head, da // at.n_head) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * D:
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])):
        return False
    norms = (at.ada_ln_in.ln_in, at.ada_ln_in.ln_cond, at.ln_pair, module.transition.ada_ln_in.ln_in,
             module.transition.ada_ln_in.ln_cond)
    return all(n.eps == EPS for n in norms)


def _f32(t):
    return t.detach().float().contiguous()


_PACKS: dict = {}


def _pack(P, dev):
    """The block's weights in the kernels' layouts; rebuilt only when a parameter changes (an optimizer step bumps
    ``_version``), so a training step pays it once per block. Scoped to the CUDA-graph capture (``kernels._capture``): an
    eager pack is never reused inside a capture, whose replays would otherwise run on the weights of capture time."""
    key = tuple((t.data_ptr(), t._version) for t in P.values())
    slot = _capture.scoped(next(iter(P.values())).data_ptr())
    hit = None if slot is None else _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    g = lambda n: P[n]
    w1, w2 = _f32(g("attention.ada_ln_in.ln_cond.weight")), _f32(g("transition.ada_ln_in.ln_cond.weight"))
    Wraw = torch.cat([_f32(g("attention.ada_ln_in.to_scale.weight")), _f32(g("attention.ada_ln_in.to_bias.weight")),
                      _f32(g("transition.ada_ln_in.to_scale.weight")), _f32(g("transition.ada_ln_in.to_bias.weight"))])
    wp, Wb = _f32(g("attention.ln_pair.weight")), _f32(g("attention.to_bias.weight"))
    Wf = Wb * wp
    W = {
        "w1": w1, "w2": w2, "Wraw": Wraw, "Wn": torch.cat([Wraw[:2 * D] * w1, Wraw[2 * D:] * w2]).to(BF),   # cond-LN weights folded
        "Wg": torch.cat([g("attention.to_scale.weight"), g("transition.to_scale.weight")]).detach().to(BF),
        "bs1": _f32(g("attention.ada_ln_in.to_scale.bias")), "bs2": _f32(g("transition.ada_ln_in.to_scale.bias")),
        "bg1": _f32(g("attention.to_scale.bias")), "bg2": _f32(g("transition.to_scale.bias")),
        "Wvg": torch.cat([g("attention.to_value.weight"), g("attention.to_gate.weight")]).detach().to(BF).contiguous(),
        "wp": wp, "Wb": Wb, "Wf_bf": Wf.to(BF),
        "Wo": g("attention.to_out.weight").detach().to(BF).contiguous(),
        "Wab": torch.cat([g("transition.expand_a.weight"), g("transition.expand_b.weight")]).detach().to(BF),
        "Wsq": g("transition.squeeze.weight").detach().to(BF).contiguous(),
    }
    if slot is not None:
        _capture.prune(_PACKS)
        _PACKS[slot] = (key, W)
    return W


_OPS: dict = {}


def _op(kind, dev, nh=16, dh=48):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if (kind, idx, nh, dh) not in _OPS:
        from miniworld_engine.kernels.bias_only_dit import cuda as C
        _OPS[(kind, idx, nh, dh)] = (C.GemmSwigluAB(idx) if kind == "swiglu" else
                                     {"pv": C.PvGateCore, "dpb": C.DpbKernel}[kind](idx, nh=nh, dh=dh))
    return _OPS[(kind, idx, nh, dh)]


def _heads(params):
    return params[NAMES.index("attention.to_bias.weight")].shape[0]


def _width(params):
    """The attention's channels (n_head x head width: 768 or 1024)."""
    return params[NAMES.index("attention.to_value.weight")].shape[0]


def _saved_like(single, pair, H, DA):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd); H heads, DA
    attention channels."""
    A, _, L, _ = single.shape
    M, R, e = A * L, L * L, single.new_empty
    f32 = torch.float32
    return [e((M, DC), dtype=BF), e((M, 2), dtype=f32), e((M, 4 * D), dtype=BF),
            e((M, 2 * D), dtype=BF), e((M, 2), dtype=f32), e((M, D), dtype=BF), e((M, 2 * DA), dtype=BF), e((H, L, L), dtype=BF),
            e((H, L, L), dtype=BF),
            e((M, DA), dtype=BF), e((M, D), dtype=BF), e((M, 2), dtype=f32), e((M, D), dtype=BF),
            e((M, 4 * D), dtype=BF), e((M, 2 * D), dtype=BF), e((M, D), dtype=BF), e((R, 2), dtype=f32)]


def _fwd_fake(single, cond, pair, mask, params):
    return [torch.empty_like(single), *_saved_like(single, pair, _heads(params), _width(params))]


@opaque(fake=_fwd_fake, name="bias_only_dit_train_sm100_fwd")
def _fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None,
         params: list[torch.Tensor]) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated)."""
    from miniworld_engine.kernels.bias_only_dit import cuda as C
    from miniworld_engine.kernels.bias_only_dit.cuda.train import ext as bo_ext
    BT = bo_ext()
    W = _pack(dict(zip(NAMES, params, strict=True)), single.device)
    H, DA = W["Wf_bf"].shape[0], W["Wo"].shape[1]                           # heads, attention channels (LAYOUTS)
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    x = single.reshape(M, D).contiguous()                                    # the input as it is: the first residual rows
    c2 = cond.reshape(M, DC).contiguous()
    chat = torch.empty(M, DC, device=dev, dtype=BF)
    cst = torch.empty(M, 2, device=dev)
    BT.cond_ln_cuda(c2, chat, cst, EPS)
    G = torch.mm(chat, W["Wn"].t())                                          # [M, 4D] s1 | sh1 | s2 | sh2
    Gg = torch.mm(c2, W["Wg"].t())                                           # [M, 2D] gate1 | gate2
    xa = torch.empty(M, D, device=dev, dtype=BF); xst = torch.empty(M, 2, device=dev)
    BT.adaln_a_cuda(x, G, W["bs1"], xa, xst, EPS)
    vg = torch.mm(xa, W["Wvg"].t())                                          # [M, 2 DA] v | g
    p2 = pair.reshape(R, DP)
    P = torch.empty(H, L, L, device=dev, dtype=BF); pst = torch.empty(R, 2, device=dev)
    BT.pair_bias_cuda(p2, W["Wf_bf"], P, pst, EPS)                          # the bias, head-major; Wf = Wb diag(wp); LN(pair) stays on chip
    Pt = torch.empty(H, L, L, device=dev, dtype=BF)                         # P^T for the backward's dV, from the softmax kernel
    C.softmax_t(P.view(H * L, L), P.view(H * L, L), Pt.view(H * L, L), None if mask is None else mask.reshape(L).contiguous())
    og = torch.empty(M, DA, device=dev, dtype=BF)
    _op("pv", dev, H, DA // H)(vg[:, :DA], P.view(H * L, L), og, A, g=vg[:, DA:])   # a = sigmoid(g) (P v)
    y = torch.mm(og, W["Wo"].t())
    xt = torch.empty(M, D, device=dev, dtype=BF); x1st = torch.empty(M, 2, device=dev)
    BT.res_adaln_b_cuda(x, y, Gg, W["bg1"], G, W["bs2"], xt, x1st, EPS)            # x1 = x + sigmoid(g1) y is never stored
    ab = torch.empty(M, 4 * D, device=dev, dtype=BF)                         # [M, 2 * 2D] a | b
    h = torch.empty(M, 2 * D, device=dev, dtype=BF)
    _op("swiglu", dev)(xt, W["Wab"], ab, h)                                  # the SwiGLU in the expand GEMM's epilogue
    z = torch.mm(h, W["Wsq"].t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    BT.res_c_cuda(x, y, z, Gg, W["bg1"], W["bg2"], out)
    return [out.view(single.shape), chat, cst, G, Gg, xst, xa, vg, P, Pt, og, y, x1st, xt, ab, h, z, pst]


# the weight gradients leave _bwd as a few buffers (custom-op outputs may not alias each other); _Block.backward hands out views:
#   value | gate [2D, D], expand a | b [4D, D], attention | transition to_scale [2D, DC], squeeze [D, 2D], to_out [D, D]
#   (cuBLAS, in the dtype of the group's first parameter); one flat buffer in the AdaLN projections' dtype (the four projections
#   s1 | b1 | s2 | b2 [4D, DC], the four gate / scale biases, to_bias); one in the LayerNorm weights' dtype (the two cond-LN
#   weights, ln_pair: the engine's LayerNorms keep fp32 weights in a bf16 block)
_BIG = (("attention.to_value.weight", "attention.to_gate.weight"), ("transition.expand_a.weight", "transition.expand_b.weight"),
        ("attention.to_scale.weight", "transition.to_scale.weight"), ("transition.squeeze.weight",), ("attention.to_out.weight",))
_U, _NORMS = 4 * D * DC, 2 * DC + DP


def _flat(H):
    """Size of the small-gradient buffer for H heads (to_bias is [H, DP])."""
    return _U + 4 * D + H * DP
_SMALL = {   # name -> (buffer: 0 flat / 1 norms, offset, shape)
    "attention.ada_ln_in.to_scale.weight": (0, 0, (D, DC)), "attention.ada_ln_in.to_bias.weight": (0, D * DC, (D, DC)),
    "transition.ada_ln_in.to_scale.weight": (0, 2 * D * DC, (D, DC)), "transition.ada_ln_in.to_bias.weight": (0, 3 * D * DC, (D, DC)),
    "transition.to_scale.bias": (0, _U, (D,)), "transition.ada_ln_in.to_scale.bias": (0, _U + D, (D,)),
    "attention.to_scale.bias": (0, _U + 2 * D, (D,)), "attention.ada_ln_in.to_scale.bias": (0, _U + 3 * D, (D,)),
    "attention.to_bias.weight": (0, _U + 4 * D, None),          # [n_head, DP]
    "attention.ada_ln_in.ln_cond.weight": (1, 0, (DC,)), "transition.ada_ln_in.ln_cond.weight": (1, DC, (DC,)),
    "attention.ln_pair.weight": (1, 2 * DC, (DP,)),
}


def _grad_bufs_like(params):
    P = dict(zip(NAMES, params, strict=True))
    big = [P[g[0]].new_empty((sum(P[n].shape[0] for n in g), P[g[0]].shape[1])) for g in _BIG]
    flat = P["attention.ada_ln_in.to_scale.weight"].new_empty(_flat(_heads(params)))
    return [*big, flat, P["attention.ada_ln_in.ln_cond.weight"].new_empty(_NORMS)]


_PLANS: dict = {}


def _split_grads(bufs, params):
    """Per-parameter views of _bwd's gradient buffers, in NAMES order (the (buffer, rows / offset, shape) plan built once per
    parameter shape set)."""
    key = tuple(p.shape for p in params)
    plan = _PLANS.get(key)
    if plan is None:
        where = {}
        for b, g in enumerate(_BIG):
            r = 0
            for n in g:
                k = params[NAMES.index(n)].shape[0]
                where[n] = (b, r, r + k, None)
                r += k
        for n, (b, o, shape) in _SMALL.items():
            shape = shape or tuple(params[NAMES.index(n)].shape)
            where[n] = (len(_BIG) + b, o, o + math.prod(shape), shape)
        plan = _PLANS[key] = [where[n] for n in NAMES]
    out = []
    for (b, lo, hi, shape), p in zip(plan, params, strict=True):
        g = bufs[b][lo:hi] if shape is None else bufs[b][lo:hi].view(shape)
        out.append(g if g.dtype == p.dtype else g.to(p.dtype))
    return out


def _bwd_fake(single, cond, pair, mask, params, saved, dout):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *_grad_bufs_like(params)]


@opaque(fake=_bwd_fake, name="bias_only_dit_train_sm100_bwd")
def _bwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         saved: list[torch.Tensor], dout: torch.Tensor) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes."""
    from miniworld_engine.kernels.bias_only_dit import cuda as C
    from miniworld_engine.kernels.bias_only_dit.cuda.train import ext as bo_ext
    from miniworld_engine.kernels.bias_only_dit.cuda.train import partials
    BT = bo_ext()
    (chat, cst, G, Gg, xst, xa, vg, P, Pt, og, y, x1st, xt, ab, h, z, pst) = saved
    W = _pack(dict(zip(NAMES, params, strict=True)), single.device)
    Wn, Wraw, Wg, Wvg, Wo, Wab, Wsq = W["Wn"], W["Wraw"], W["Wg"], W["Wvg"], W["Wo"], W["Wab"], W["Wsq"]
    bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (W[k] for k in ("bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb"))
    H, DA = Wb.shape[0], Wo.shape[1]
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype, f32 = single.dtype, cond.dtype, pair.dtype, torch.float32
    x, c2, p2 = single.reshape(M, D).contiguous(), cond.reshape(M, DC).contiguous(), pair.reshape(R, DP)
    dout = dout.reshape(M, D).to(sdt).contiguous()
    # the big weight gradients leave cuBLAS in their parameter's dtype (fp32 accumulation either way), one GEMM per group of
    # parameters that share an input (_BIG): no fp32 -> bf16 copy afterwards; _Block.backward splits them into views
    pdt = {n: p.dtype for n, p in zip(NAMES, params, strict=True)}
    wgrad = lambda n, a, b: torch.mm(a.t(), b, out_dtype=torch.float32) if pdt[n] is torch.float32 else torch.mm(a.t(), b)
    part = partials(M, 4, dev)                          # per-block column sums: bg2 bs2 bg1 bs1
    flat = torch.empty(_flat(H), device=dev, dtype=pdt["attention.ada_ln_in.to_scale.weight"])    # the small gradients (_SMALL)
    norms = torch.empty(_NORMS, device=dev, dtype=pdt["attention.ada_ln_in.ln_cond.weight"])
    dG = torch.empty(M, 4 * D, device=dev, dtype=BF); dGg = torch.empty(M, 2 * D, device=dev, dtype=BF)
    dz = torch.empty(M, D, device=dev, dtype=BF)
    n0 = BT.res_c_bwd_cuda(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = torch.mm(dz, Wsq)                                                      # [M, 2D]
    dab = torch.empty_like(ab)
    BT.swiglu_bwd_cuda(dh, ab, dab)
    dWsq = wgrad("transition.squeeze.weight", dz, h)
    dxt = torch.mm(dab, Wab)
    dWab = wgrad("transition.expand_a.weight", dab, xt)
    dx1 = torch.empty(M, D, device=dev); dy = torch.empty(M, D, device=dev, dtype=BF)
    n12 = BT.res_adaln_b_bwd_cuda(dout, dxt, x, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2])
    dog = torch.mm(dy, Wo)
    dWo = wgrad("attention.to_out.weight", dy, og)
    # the bias-only attention's backward
    dvg = torch.empty(M, 2 * DA, device=dev, dtype=BF)
    do = torch.empty(M, DA, device=dev, dtype=BF); dd = torch.empty(A, H, L, device=dev)
    C.gate_bwd_rows(dog, og, vg[:, DA:], do, dvg[:, DA:], dd, L)
    _op("pv", dev, H, DA // H)(do, Pt.view(H * L, L), dvg[:, :DA], A)              # dv = P^T do
    dbias = torch.empty(H * L, L, device=dev, dtype=BF)
    _op("dpb", dev, H, DA // H)(do, vg[:, :DA], P.view(H * L, L), dd, dbias, A)   # P o (dP - D); masked keys get P = 0
    dxa = torch.mm(dvg, Wvg)
    dWvg = wgrad("attention.to_value.weight", dvg, xa)
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    n3 = BT.adaln_a_bwd_cuda(dxa, x, xst, G, bs1, dx1, dx, dG, part[3])
    dchat = torch.mm(dG, Wn)
    dWn = torch.mm(dG.t(), chat, out_dtype=f32)
    dcg = torch.mm(dGg, Wg)
    dWgg = wgrad("attention.to_scale.weight", dGg, c2)
    dc = torch.empty(M, DC, device=dev, dtype=cdt)
    BT.cond_bwd_cuda(dchat, dcg, c2, cst, dc)
    pw = torch.empty(192, DC, device=dev)                                       # cond-LN weight partials, one row per block
    BT.unfold_cuda(dWn, Wraw, w1, w2, flat[:_U].view(4 * D, DC), pw)
    # the pair LayerNorm backward with d LN(pair) = dbias^T Wf, and dWf = dbias LN(pair), in one kernel (LN(pair) rebuilt on chip)
    dpair = torch.empty(R, DP, device=dev, dtype=pdtype)
    pwf = torch.empty(BT.partial_rows(M), H, DP, device=dev)                     # dWf partials, one per block
    nwf = BT.pair_bias_bwd_cuda(dbias, p2, pst, W["Wf_bf"], dpair, pwf)
    # the step's last kernel: bias sums, cond-LN weights, ln_pair / to_bias from their partials, in the parameters' dtype
    BT.finalize_cuda(part, [n0, n12, n12, n3], pw, pwf, nwf, Wb, wp, flat[_U:], norms)
    return [dx.view(A, 1, L, D), dc.view(A, 1, L, DC), dpair.view(1, L, L, DP), dWvg, dWab, dWgg, dWsq, dWo, flat, norms]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, *params):
        out, *saved = _fwd(single, cond, pair, mask, list(params))
        ctx.save_for_backward(single, cond, pair, *params, *saved)
        ctx.meta = (mask, len(params))
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, npar = ctx.meta
        vals = ctx.saved_tensors
        single, cond, pair = vals[:3]
        params, saved = list(vals[3:3 + npar]), list(vals[3 + npar:])
        dx, dc, dpair, *bufs = _bwd(single, cond, pair, mask, params, saved, dout.contiguous())
        pg = _split_grads(bufs, params)
        return (dx, dc, dpair, None, *(g if ctx.needs_input_grad[4 + i] else None for i, g in enumerate(pg)))


def block(module, single, cond, pair, mask=None):
    """One BiasOnlyDiTBlock (attention + transition, both residuals) through the fused training path. Call ``serves`` first."""
    return _Block.apply(single, cond, pair.contiguous(), mask, *[module.get_parameter(n) for n in NAMES])


__all__ = ["NAMES", "block", "serves"]
