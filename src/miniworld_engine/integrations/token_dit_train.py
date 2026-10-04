"""Fused token DiT TRAINING block on B200 (sm_100), wired to DiTBlock's parameter contract.

One autograd Function per ``DiTBlock`` call, forward and backward, with no Triton and no quack, in one of two precisions --
bf16 (bf16 inputs, or ``compute_dtype=bf16``) or fp32 (fp32 inputs, no ``compute_dtype``; TF32 tensor cores):

  * GEMMs: cuBLAS, operands in the path's dtype (bf16, or fp32 on TF32 tensor cores whatever the caller's allow_tf32), fp32
    accumulation, fp32 weight gradients;
  * the six conditioning projections as two GEMMs (every block-internal LayerNorm of cond shares its statistics, so its
    weight folds into the projection; the two output gates read raw cond), q|k|v|g as one GEMM;
  * the pair bias: a CUDA LayerNorm of the pair rows and one GEMM, head-major, the shared key mask folded in as -inf;
  * the attention core: ``kernels/augmented_attention/cuda/sm100`` -- bf16: attn_fwd2, attn_dkv + attn_dqb; fp32: attn_fwd_tf32,
    attn_dkv_tf32 + attn_dqb_tf32 (kind::tf32 MMAs, fp32 softmax);
  * every elementwise / row step: ``kernels/conditioned_transition/cuda/token_dit_train_rows.cu``; in bf16 the expand GEMM
    carries the SwiGLU (``gemm_swiglu2_sm100.cu -DSAVE_AB``, which also writes [a | b] for the backward);
  * the residual stream fp32 inside the block, the output in the input's dtype.

``serves()`` is the whole gate: autograd on, the engine's kernel backend (implementation TRITON or MINIWORLD), B200, a bf16 or
fp32 input (bf16 operands when the input or ``compute_dtype`` is bf16, else fp32), the token widths
(768 / 16 x 48 / cond 384 / pair 128 / transition n = 2), B == 1, an even A, L a multiple of 128, a key mask of [B, L]
or none, LayerNorm eps 1e-5. Anything else keeps the module path. MINIWORLD_TOKEN_DIT_TRAIN=0 turns it off.

The forward and the backward are each one opaque op (``kernels/_compile.opaque``) with a fake implementation, inside an
autograd Function, so torch.compile keeps them as single nodes (as the H100 TriMul training path does).

Unlike the research stack runner (branch ``b200/token-dit``), a DiTBlock call sees one block, so the pair bias and the cond
LayerNorm are computed per block rather than once per stack.
"""

from __future__ import annotations

import contextlib
import os

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

D, DC, DP, H, DH = 768, 384, 128, 16, 48          # the default layout; serves() takes LAYOUTS
#: (heads, d_single): 16 x 48 (bf16 and fp32); 24 x 32, 12 x 64 and 16 x 64 (d 1024) in bf16
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
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
    if single.dtype not in (BF, torch.float32) or compute_dtype not in (None, BF, torch.float32):
        return False
    if not any(t.requires_grad for t in (single, cond, pair)) and not any(p.requires_grad for p in module.parameters()):
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[0] % 2 or single.shape[2] % 128:
        return False
    d = single.shape[-1]
    if tuple(cond.shape) != (*single.shape[:3], DC) or tuple(pair.shape) != (1, single.shape[2], single.shape[2], DP):
        return False
    if (a.n_head, d) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * d:
        return False
    if (a.n_head, d) != (H, D) and operand_dtype(single, compute_dtype) is not BF:
        return False                         # fp32 (TF32 kernels) at 16 x 48 only
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, single.shape[2])):
        return False
    norms = (a.ada_ln_in.ln_in, a.ada_ln_in.ln_cond, a.ln_pair, module.transition.ada_ln_in.ln_in,
             module.transition.ada_ln_in.ln_cond)
    return all(n.eps == EPS for n in norms)


def _f32(t):
    return t.detach().float().contiguous()


def operand_dtype(single, compute_dtype=None) -> torch.dtype:
    """The dtype of every GEMM operand / saved activation: bf16 when the input or ``compute_dtype`` is bf16, else fp32."""
    return BF if (single.dtype is BF or compute_dtype is BF) else torch.float32


@contextlib.contextmanager
def _tf32(on: bool):
    """cuBLAS on TF32 tensor cores for the fp32 path's GEMMs (restored after): the fp32 path is the TF32 recipe, and IEEE fp32
    GEMMs ran the block 5-6x slower."""
    if not on:
        yield
        return
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _mm32(a, b):
    """a @ b with an fp32 result (weight gradients): bf16 operands accumulate into fp32 output, fp32 ones already are."""
    return torch.mm(a, b, out_dtype=torch.float32) if a.dtype is BF else torch.mm(a, b)


_PACKS: dict = {}


def _pack(P, qk, dev, at):
    """The block's weights in the kernels' layouts (GEMM operands in ``at``, folded cond / pair LayerNorm weights, fp32 bias
    vectors). Rebuilt only when a parameter changes -- an optimizer step bumps ``_version`` -- so a training step pays
    it once per block, not once per forward. Scoped to the CUDA-graph capture (``kernels._capture``): an eager pack is never
    reused inside a capture, whose replays would otherwise run on the weights of capture time after every optimizer step."""
    key = tuple((t.data_ptr(), t._version) for t in P.values())
    slot = _capture.scoped((next(iter(P.values())).data_ptr(), at))
    hit = None if slot is None else _PACKS.get(slot)
    if hit is not None and hit[0] == key:
        return hit[1]
    g = lambda n: P[n]
    D = g("attention.to_out.weight").shape[0]
    DH = D // g("attention.to_bias.weight").shape[0]
    w1, w2 = _f32(g("attention.ada_ln_in.ln_cond.weight")), _f32(g("transition.ada_ln_in.ln_cond.weight"))
    Wraw = torch.cat([_f32(g("attention.ada_ln_in.to_scale.weight")), _f32(g("attention.ada_ln_in.to_bias.weight")),
                      _f32(g("transition.ada_ln_in.to_scale.weight")), _f32(g("transition.ada_ln_in.to_bias.weight"))])
    wp, Wb = _f32(g("attention.ln_pair.weight")), _f32(g("attention.to_bias.weight"))
    Wf = Wb * wp
    W = {
        "w1": w1, "w2": w2, "Wraw": Wraw, "Wn": torch.cat([Wraw[:2 * D] * w1, Wraw[2 * D:] * w2]).to(at),   # cond-LN weights folded
        "Wg": torch.cat([g("attention.to_scale.weight"), g("transition.to_scale.weight")]).detach().to(at),
        "bs1": _f32(g("attention.ada_ln_in.to_scale.bias")), "bs2": _f32(g("transition.ada_ln_in.to_scale.bias")),
        "bg1": _f32(g("attention.to_scale.bias")), "bg2": _f32(g("transition.to_scale.bias")),
        "Wqkvg": torch.cat([g("attention.to_query.weight"), g("attention.to_key.weight"), g("attention.to_value.weight"),
                         g("attention.to_gate.weight")]).detach().to(at),
        "bqkvg": torch.cat([g("attention.to_query.bias").detach().float(), torch.zeros(3 * D, device=dev)]).to(at),
        "nq": _f32(g("attention.norm_query.weight")) if qk else torch.ones(DH, device=dev),
        "nk": _f32(g("attention.norm_key.weight")) if qk else torch.ones(DH, device=dev),
        "wp": wp, "Wb": Wb, "Wf": Wf, "Wf_at": Wf.to(at),
        "Wo": g("attention.to_out.weight").detach().to(at).contiguous(),
        "Wab": torch.cat([g("transition.expand_a.weight"), g("transition.expand_b.weight")]).detach().to(at),
        "Wsq": g("transition.squeeze.weight").detach().to(at).contiguous(),
    }
    if slot is not None:
        _capture.prune(_PACKS)
        _PACKS[slot] = (key, W)
    return W


_GSW: dict = {}


def _swiglu_gemm(xt, wab):
    """The sm_100a expand GEMM with the SwiGLU epilogue that also writes [a | b] (``gemm_swiglu2_sm100.cu -DSAVE_AB``) where
    it applies (bf16, B200, M >= its row threshold, MINIWORLD_TOKEN_DIT_GEMM_SWIGLU); None keeps cuBLAS + the SwiGLU rows."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu
    if not gemm_swiglu.supported(xt, wab):
        return None
    idx = xt.device.index if xt.device.index is not None else torch.cuda.current_device()
    if idx not in _GSW:
        try:
            _GSW[idx] = gemm_swiglu.GemmSwiglu(idx, K=wab.shape[1], H=wab.shape[0] // 2, save_ab=True)
        except Exception as exc:         # a failed build keeps cuBLAS + the row pass
            import warnings
            warnings.warn(f"sm100 SwiGLU GEMM unavailable, keeping cuBLAS: {exc!r}", RuntimeWarning, stacklevel=2)
            _GSW[idx] = None
    return _GSW[idx]


def _names_for(qk):
    return ["attention." + n for n in ATT + (QKN if qk else ())] + ["transition." + n for n in TRN]


def _saved_like(single, pair, fp32, qk=True, H=H):
    """Shapes / dtypes of the forward's saved activations (the fake implementation and the contract of _fwd)."""
    A, _, L, D = single.shape
    M, R, e = A * L, L * L, single.new_empty
    f32, at = torch.float32, (torch.float32 if fp32 else BF)
    return [e((M, D), dtype=f32), e((M, DC), dtype=at), e((M, DC), dtype=at), e((M, 2), dtype=f32), e((M, 4 * D), dtype=at),
            e((M, 2 * D), dtype=at), e((M, 2), dtype=f32), e((M, D), dtype=at), e((M, 4 * D), dtype=at), e((M, 2 * H), dtype=f32),
            *(e((M, D), dtype=at) for _ in range(3 if qk else 0)), e((H, L, L), dtype=at), e((M, D), dtype=f32),
            e((A, H, L), dtype=f32), e((M, D), dtype=at), e((M, D), dtype=at), e((M, D), dtype=f32), e((M, 2), dtype=f32),
            e((M, D), dtype=at), e((M, 4 * D), dtype=at), e((M, D), dtype=at), e((R, DP), dtype=at), e((R, 2), dtype=f32)]


def _fwd_fake(single, cond, pair, mask, params, qk, eq, ek, fp32, heads):
    return [torch.empty_like(single), *_saved_like(single, pair, fp32, qk, heads)]


@opaque(fake=_fwd_fake, name="token_dit_train_sm100_fwd")
def _fwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         qk: bool, eq: float, ek: float, fp32: bool, heads: int) -> list[torch.Tensor]:
    """The block's forward: [out, *saved activations] (every output freshly allocated); ``fp32``: fp32 GEMM operands (on TF32
    tensor cores, whatever the caller's allow_tf32) and the TF32 attention kernels, else bf16."""
    with _tf32(fp32):
        return _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, heads)


def _fwd_body(single, cond, pair, mask, params, qk, eq, ek, fp32, H):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition import cuda as rows
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    A, _, L, D = single.shape
    DH = D // H
    T = ext(D, DH)
    names = _names_for(qk)
    P = dict(zip(names, params, strict=False))
    M, R, dev = A * L, L * L, single.device
    at = torch.float32 if fp32 else BF
    x = torch.empty(M, D, device=dev)                                          # the fp32 residual, written by adaln_a
    c2 = cond.reshape(M, DC).contiguous()
    chat = torch.empty(M, DC, device=dev, dtype=at); cbf = torch.empty_like(chat)
    cst = torch.empty(M, 2, device=dev)
    T.cond_prep(c2, chat, cbf, cst, EPS)
    W = _pack(P, qk, dev, at)
    Wn, Wg = W["Wn"], W["Wg"]
    G = torch.mm(chat, Wn.t())                                                  # [M, 4D] s1 | sh1 | s2 | sh2
    Gg = torch.mm(cbf, Wg.t())                                                  # [M, 2D] gate1 | gate2
    bs1, bs2, bg1, bg2 = W["bs1"], W["bs2"], W["bg1"], W["bg2"]
    xa = torch.empty(M, D, device=dev, dtype=at); xst = torch.empty(M, 2, device=dev)
    T.adaln_a(single.reshape(M, D).contiguous(), G, bs1, xa, xst, EPS, x)  # reads the input, writes x on the way
    Wqkvg, bqkvg = W["Wqkvg"], W["bqkvg"]
    qkvg = torch.addmm(bqkvg, xa, Wqkvg.t())
    rqk = torch.empty(M, 2 * H, device=dev)
    nq, nk = W["nq"], W["nk"]
    if qk:
        qn = torch.empty(M, D, device=dev, dtype=at); kn = torch.empty_like(qn); vc = torch.empty_like(qn)
        T.qknorm(qkvg, nq, nk, qn, kn, vc, rqk, eq, ek, qk)
    else:                                      # the core reads q / k / v as column views of qkvg: no copies
        qn, kn, vc = (qkvg[:, i * D:(i + 1) * D] for i in range(3))
    p2 = pair.reshape(R, DP)
    ph = torch.empty(R, DP, device=dev, dtype=at); pst = torch.empty(R, 2, device=dev)
    T.pair_ln(p2, ph, pst, EPS)
    bias = torch.mm(W["Wf_at"], ph.t()).view(H, L, L)                           # head-major, natural units; Wf = Wb diag(wp)
    if mask is not None:
        bias.masked_fill_(~mask.reshape(1, 1, L), float("-inf"))
    O, LSE = sm100.forward_tf32(qn, kn, vc, bias, A, L) if fp32 else sm100.forward(qn, kn, vc, bias, A, L, H, DH)
    og = torch.empty(M, D, device=dev, dtype=at)
    T.gate_o(O, qkvg, og)
    Wo = W["Wo"]
    y = torch.mm(og, Wo.t())
    x1 = torch.empty_like(x); xt = torch.empty(M, D, device=dev, dtype=at); x1st = torch.empty(M, 2, device=dev)
    T.res_adaln_b(x, y, Gg, bg1, G, bs2, x1, xt, x1st, EPS)
    Wab = W["Wab"]
    h = torch.empty(M, 2 * D, device=dev, dtype=at)
    gsw = _swiglu_gemm(xt, Wab)
    if gsw is not None:                                                         # expand GEMM + SwiGLU, a | b saved on the way
        ab = torch.empty(M, 4 * D, device=dev, dtype=at)
        gsw(xt, Wab, h, ab)
    else:
        ab = torch.mm(xt, Wab.t())                                              # [M, 2 * 2D] a | b
        rows.swiglu_rows(ab, h)
    Wsq = W["Wsq"]
    z = torch.mm(h, Wsq.t())
    out = torch.empty(M, D, device=dev, dtype=single.dtype)
    T.res_c(x1, z, Gg, bg2, out)
    # custom-op outputs may not alias each other: without QK-norm q / k / v are views of qkvg -- not outputs, recovered in _bwd
    qkv_out = [qn, kn, vc] if qk else []
    return [out.view(single.shape), x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, *qkv_out, bias, O, LSE, og, y, x1, x1st, xt,
            ab, z, ph, pst]


def _bwd_fake(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads):
    return [torch.empty_like(single), torch.empty_like(cond), torch.empty_like(pair), *(torch.empty_like(p) for p in params)]


@opaque(fake=_bwd_fake, name="token_dit_train_sm100_bwd")
def _bwd(single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, params: list[torch.Tensor],
         saved: list[torch.Tensor], dout: torch.Tensor, qk: bool, eq: float, ek: float, fp32: bool, heads: int) -> list[torch.Tensor]:
    """The block's backward: [d single, d cond, d pair, *d params] in the inputs' dtypes."""
    with _tf32(fp32):
        return _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, heads)


def _bwd_body(single, cond, pair, mask, params, saved, dout, qk, eq, ek, fp32, H):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100
    from miniworld_engine.kernels.conditioned_transition.cuda.train import ext
    D = single.shape[-1]
    DH = D // H
    T = ext(D, DH)
    if not qk:                                 # the forward's q / k / v: column views of qkvg (see _fwd_body)
        saved = [*saved[:10], *(saved[8][:, i * D:(i + 1) * D] for i in range(3)), *saved[10:]]   # after qkvg (8), rqk (9)
    (x, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias, O, LSE, og, y, x1, x1st, xt, ab, z, ph, pst) = saved
    names = _names_for(qk)
    at = torch.float32 if fp32 else BF
    W = _pack(dict(zip(names, params, strict=False)), qk, single.device, at)
    Wn, Wraw, Wg, Wqkvg, Wo, Wab, Wsq, Wf = W["Wn"], W["Wraw"], W["Wg"], W["Wqkvg"], W["Wo"], W["Wab"], W["Wsq"], W["Wf"]
    nq, nk, bs1, bs2, bg1, bg2, w1, w2, wp, Wb = (W[k] for k in ("nq", "nk", "bs1", "bs2", "bg1", "bg2", "w1", "w2", "wp", "Wb"))
    A, _, L, _ = single.shape
    M, R, dev = A * L, L * L, single.device
    sdt, cdt, pdtype = single.dtype, cond.dtype, pair.dtype
    c2, p2 = cond.reshape(M, DC).contiguous(), pair.reshape(R, DP)
    dout = dout.reshape(M, D).to(sdt).contiguous()
    from miniworld_engine.kernels.conditioned_transition.cuda.train import partials
    allp = partials(M, 7 * D, dev)                     # per-block column sums: bg2 bs2 bg1 bs1 bq | dwq dwk (per head)
    part = [allp[:, i * D:(i + 1) * D] for i in range(5)]
    pwqk = allp[:, 5 * D:]
    dG = torch.empty(M, 4 * D, device=dev, dtype=at); dGg = torch.empty(M, 2 * D, device=dev, dtype=at)
    dz = torch.empty(M, D, device=dev, dtype=at)
    T.res_c_bwd(dout, z, Gg, bg2, dz, dGg, part[0])
    dh = torch.mm(dz, Wsq)                                                      # [M, 2D]
    dab = torch.empty_like(ab); h = torch.empty_like(dh)
    T.swiglu_bwd(dh, ab, dab, h)
    dWsq = _mm32(dz.t(), h)
    dxt = torch.mm(dab, Wab)
    dWab = _mm32(dab.t(), xt)
    dx1 = torch.empty(M, D, device=dev); dy = torch.empty(M, D, device=dev, dtype=at)
    T.res_adaln_b_bwd(dout, dxt, x1, x1st, G, bs2, Gg, bg1, y, dx1, dy, dG, dGg, part[1], part[2])
    dog = torch.mm(dy, Wo)
    dWo = _mm32(dy.t(), og)
    dob = torch.empty(M, D, device=dev, dtype=at); dd = torch.empty(A, H, L, device=dev)
    dqkvg = torch.empty(M, 4 * D, device=dev, dtype=at)
    T.gate_o_bwd(dog, O, qkvg, dob, dd, dqkvg, L)
    DQ, DK, DV, DB = (sm100.backward_tf32(qn, kn, vc, dob, bias, LSE, dd, A, L) if fp32
                      else sm100.backward(qn, kn, vc, dob, bias, LSE, dd, A, L, H, DH))
    T.qknorm_bwd(DQ, DK, DV, qkvg, rqk, nq, nk, dqkvg, part[4], pwqk, qk)
    dxa = torch.mm(dqkvg, Wqkvg)
    dWqkvg = _mm32(dqkvg.t(), xa)
    dx = torch.empty(M, D, device=dev, dtype=sdt)
    T.adaln_a_bwd(dxa, x, xst, G, bs1, dx1, dx, dG, part[3])
    sums = allp.sum(0) if qk else allp[:, :5 * D].sum(0)                        # one reduction for every column sum
    pb = sums[:5 * D].view(5, D)
    dwqk = sums[5 * D:].view(2, H, DH).sum(1).reshape(-1) if qk else torch.zeros(2 * DH, device=dev)
    dchat = torch.mm(dG, Wn)
    dWn = _mm32(dG.t(), chat)
    dcg = torch.mm(dGg, Wg)
    dWg = _mm32(dGg.t(), cbf)
    dc = torch.empty(M, DC, device=dev, dtype=cdt)
    T.cond_bwd(dchat, dcg, c2, cst, dc)
    dWu = torch.empty_like(dWn); dw12 = torch.zeros(2, DC, device=dev)
    Ws1, Wb1, Ws2, Wb2 = (Wraw[i * D:(i + 1) * D] for i in range(4))
    T.unfold_lnw(dWn, Ws1, Wb1, Ws2, Wb2, w1, w2, dWu, dw12)
    dbv = DB.view(H, R).to(at)                                                  # masked key columns are 0 (P = 0)
    dWf = _mm32(dbv, ph)                                                        # [H, DP]
    dph = _mm32(dbv.t(), Wf.to(at))                                             # [R, DP]
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
    # its own tensor in the parameter's dtype -- one multi-tensor copy kernel for all of them (23 copies were ~45 us)
    outs = [torch.empty(p.shape, device=dev, dtype=p.dtype) for p in params]
    torch._foreach_copy_(outs, [grads[n].reshape(p.shape) for n, p in zip(names, params, strict=False)])
    return [dx.view(A, 1, L, D), dc.view(A, 1, L, DC), dpair.view(1, L, L, DP), *outs]


class _Block(torch.autograd.Function):
    @staticmethod
    def forward(ctx, single, cond, pair, mask, qk, eq, ek, fp32, heads, *params):
        out, *saved = _fwd(single, cond, pair, mask, list(params), qk, eq, ek, fp32, heads)
        ctx.save_for_backward(single, cond, pair, *params, *saved)
        ctx.meta = (mask, qk, eq, ek, fp32, len(params), heads)
        return out

    @staticmethod
    def backward(ctx, dout):
        mask, qk, eq, ek, fp32, npar, heads = ctx.meta
        vals = ctx.saved_tensors
        single, cond, pair = vals[:3]
        params, saved = list(vals[3:3 + npar]), list(vals[3 + npar:])
        dx, dc, dpair, *pg = _bwd(single, cond, pair, mask, params, saved, dout.contiguous(), qk, eq, ek, fp32, heads)
        return (dx, dc, dpair, None, None, None, None, None, None, *(g if ctx.needs_input_grad[9 + i] else None for i, g in enumerate(pg)))


def block(module, single, cond, pair, mask=None, compute_dtype=None):
    """One DiTBlock (attention + transition, both residuals) through the fused training path, in the precision
    ``operand_dtype`` picks. Call ``serves`` first."""
    a = module.attention
    qk = bool(a.use_qk_norm)
    eq = float(a.norm_query.effective_eps(torch.float32)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(torch.float32)) if qk else 0.0
    fp32 = operand_dtype(single, compute_dtype) is torch.float32
    return _Block.apply(single, cond, pair.contiguous(), mask, qk, eq, ek, fp32, int(a.n_head),
                        *[module.get_parameter(n) for n in _names_for(qk)])


__all__ = ["block", "serves"]
