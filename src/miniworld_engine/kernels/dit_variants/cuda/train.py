"""H100 TRAINING of the DiT block variants (Boltz-2, RoseTTAFold3, Chai-1; ``modules.dit_variant``): hand-written CUDA row kernels
(``train_rows.cu``, forward and backward), cuBLAS for every GEMM and the engine's sm_90a attention cores, in an ``autograd.Function``
per block whose forward and backward are each one opaque op.

What the block Function owns is the work over the A x L sample rows; everything on the L token rows (the conditioning tables, the
pair bias) is made by ordinary torch ops outside it, so autograd chains the gradients of the tables / bias / parameters through them:

    tables   [L, K D] of the AdaLN scales / shifts and the residual gates' logits, from the conditioning with the modules' own layers
    bias     [H, L, L] log2-unit attention bias, one stacked GEMM over the LayerNorm'd pair for every block of the stack (key mask: a
             separate additive row)
    block    ``_TokenBlock``: x [A L, D] and the tables / bias / weights in, x' out; d x, d tables (summed over the samples: one partial
             row per (token, split of the samples), fixed order), d bias, d weights out.

The residual stream is fp32 inside a sequential block (between the attention and the transition half), every reduction fp32, the saved
activations and GEMM operands bf16 (or fp32 with TF32 GEMMs). Token DiT layout only here (D = 768, 16 heads x 48, L a multiple of 128).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
HEADS, HEAD_DIM, WIDTH = 16, 48, 768
#: sample splits of the token-major backward kernels: blocks = L x S
SPLITS = 4
#: rows per block of the dq / dk / dv backward kernel (its column sums leave as one partial row per block)
QKV_RPB = 16
# table column blocks (units of D) of a token block: attention scale | shift | gate, transition scale | shift | gate
SC_A, SH_A, G_A, SC_T, SH_T, G_T = range(6)
_NAMES = ("out", "a", "xt", "qkvg", "qs", "kb", "vb", "stqk", "og", "lse", "ya", "y2", "xp", "st1", "st2", "ab", "h",
          "w_qkvg", "w_o", "w_exp", "w_sq")
_SAVED = _NAMES[1:]


def _ext():
    from miniworld_engine.kernels.dit_variants import cuda

    return cuda.train_ext()


def _mm_t(a, b):
    """a^T b (a, b [M, *]) in fp32: bf16 products with fp32 accumulation (never rounded to bf16), TF32 under the caller's allow_tf32."""
    if a.dtype == torch.bfloat16:
        return torch.mm(a.t(), b, out_dtype=torch.float32)
    return torch.mm(a.t(), b)


def _empty_like_spec(ref, shape, dtype=None):
    return ref.new_empty(shape, dtype=dtype or ref.dtype)


def _fwd_specs(M, L, at, xdt, par, gating, kq, up):
    D = WIDTH
    nq = 4 if gating else 3
    f32 = torch.float32
    hid = 2 * D
    spec = {
        "out": ((M, D), xdt), "a": ((M, D), at), "xt": ((M, D), at), "qkvg": ((M, nq * D), at), "qs": ((M, D), at), "kb": ((M, D), at),
        "vb": ((M, D), at), "stqk": ((M, 4) if kq else (0,), f32), "og": ((M, D), at), "lse": ((M // L, HEADS, L), f32), "ya": ((M, D), at),
        "y2": ((M, D), at), "xp": ((0,) if par else (M, D), f32), "st1": ((M, 2), f32), "st2": ((0,) if par else (M, 2), f32),
        "ab": ((M, (3 if up else 2) * hid), at), "h": ((M, hid), at),
        "w_qkvg": ((nq * D, D), at), "w_o": ((D, D), at), "w_exp": (((3 if up else 2) * hid, D), at), "w_sq": ((D, hid), at),
    }
    return spec


def _fwd_fake(x, tab, bias, km, wqkvg, bq, wo, wexp, wsq, qkw, par, sig, gating, kq, up, length, eps, eps_qk):
    spec = _fwd_specs(x.shape[0], length, tab.dtype, x.dtype, par, gating, kq, up)
    return [x.new_empty(spec[n][0], dtype=spec[n][1]) for n in _NAMES]


@opaque(fake=_fwd_fake, name="dit_variants_token_train_fwd")
def token_fwd(x: torch.Tensor, tab: torch.Tensor, bias: torch.Tensor, km: torch.Tensor | None, wqkvg: torch.Tensor, bq: torch.Tensor,
              wo: torch.Tensor, wexp: torch.Tensor, wsq: torch.Tensor, qkw: list[torch.Tensor], par: bool, sig: bool, gating: bool,
              kq: bool, up: bool, length: int, eps: float, eps_qk: float) -> list[torch.Tensor]:
    """One block's forward: [out, saved ...] (``_NAMES``). x [A L, D] (bf16 / fp32), tab [L, 6 D], bias [H, L, L] log2 units (bf16 /
    fp32 as the activations), km [A, L] additive fp32 key mask or None; fp32 master weights (cast here)."""
    from miniworld_engine.kernels.augmented_attention import cuda as attn

    ext = _ext()
    M, D = x.shape
    L, A = length, M // length
    at = tab.dtype
    w_qkvg, w_o, w_exp, w_sq = (w.to(at, copy=True) for w in (wqkvg, wo, wexp, wsq))
    spec = _fwd_specs(M, L, at, x.dtype, par, gating, kq, up)
    buf = {n: torch.empty(spec[n][0], device=x.device, dtype=spec[n][1]) for n in _NAMES if n not in ("w_qkvg", "w_o", "w_exp", "w_sq")}
    a, xt, qkvg, qs, kb, vb, og, ya, y2, st1, ab, h = (buf[n] for n in ("a", "xt", "qkvg", "qs", "kb", "vb", "og", "ya", "y2", "st1", "ab", "h"))
    cols = lambda c: c * D  # noqa: E731
    # AdaLN of the attention (and, parallel, of the transition: one pass over x)
    if par:
        ext.ln_mod_fwd(x, tab, [cols(SC_A), cols(SC_T)], [cols(SH_A), cols(SH_T)], [a, xt], st1, L, eps, sig)
    else:
        ext.ln_mod_fwd(x, tab, [cols(SC_A)], [cols(SH_A)], [a], st1, L, eps, sig)
    bias_vec = torch.zeros(qkvg.shape[1], device=x.device, dtype=at)
    bias_vec[:D] = bq.to(at)
    torch.addmm(bias_vec, a, w_qkvg.t(), out=qkvg)
    wq, bql, wk, bkl = (qkw if kq else (None, None, None, None))
    scale = LOG2E / math.sqrt(HEAD_DIM)
    ext.qkv_prep(qkvg[:, :D], qkvg[:, D:2 * D], qkvg[:, 2 * D:3 * D], qs, kb, vb, buf["stqk"] if kq else None, wq, bql, wk, bkl, scale, eps_qk, kq, at == torch.float32)
    if at == torch.float32:                       # the TF32 core: key-permuted bias / mask, v^T
        perm = _key_perm(L, x.device)
        bias_p = bias[..., perm].contiguous()
        km_p = None if km is None else km[:, perm].contiguous()
        vt = _transpose(vb)
        if gating:
            og, lse = attn.tf32_forward(qs, kb, vt, bias_p, km_p, A, L, qkvg=qkvg)
        else:
            og, lse = attn.tf32_forward(qs, kb, vt, bias_p, km_p, A, L)
        del vt
    elif gating:
        og, lse = attn.forward(qs, kb, vb, bias, km, A, L, qkvg=qkvg)
    else:
        lse = buf["lse"]
        og = torch.empty(M, D, device=x.device, dtype=at)
        attn._ext("attn_fwd", attn.fwd_defs() + ("FWD_BF16=1",)).attn_fwd(qs, kb, vb, bias, km, og, lse)
    torch.mm(og, w_o.t(), out=ya)
    out = buf["out"]
    if par:
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(x, ya, tab, cols(G_A), y2, cols(G_T), out, L)
        xp = buf["xp"]
        st2 = buf["st2"]
    else:
        xp, st2 = buf["xp"], buf["st2"]
        ext.res_ln_fwd(x, ya, tab, cols(G_A), cols(SC_T), cols(SH_T), xp, xt, st2, L, eps, sig)
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(xp, y2, tab, cols(G_T), None, 0, out, L)
    res = dict(buf, og=og, lse=lse, xp=xp, st2=st2, w_qkvg=w_qkvg, w_o=w_o, w_exp=w_exp, w_sq=w_sq)
    return [res[n] for n in _NAMES]




def _transpose(t):
    out = t.new_empty(t.shape[1], t.shape[0])
    _ext().transpose2d(t, out)
    return out

def _key_perm(length, device):
    from miniworld_engine.kernels.conditioned_transition import cuda as rows

    return rows.key_perm(length, device)


def _tf32_attn_bwd(qs, kb, vb, dob, bias, km, lse, dd, A, L):
    """The TF32 attention backward (``attn_tf32.cu``): dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias [16, L, L] fp32 (natural key
    order; the bias passed in is natural order too). The core takes q, k, dO transposed, and its bias / LSE / D / key mask in permuted orders."""
    from miniworld_engine.kernels.augmented_attention import cuda as attn

    perm = _key_perm(L, qs.device)
    inv = torch.argsort(perm)
    bias_p = bias[..., perm].contiguous()
    bias_tp = bias_p[:, perm][..., inv].transpose(1, 2).contiguous()      # [h, key, q position]
    km_p = None if km is None else km[:, perm].contiguous()
    dq, dk, dv, db_p = attn.tf32_backward(qs, kb, vb, dob, _transpose(qs), _transpose(kb), _transpose(dob), bias_p, bias_tp, lse, dd,
                                          lse[..., perm].contiguous(), dd[..., perm].contiguous(), km, km_p, A, L)
    return dq, dk, dv, db_p[..., inv]

def _bwd_fake(dout, x, tab, bias, km, saved, qkw, par, sig, gating, kq, up, length, eps, eps_qk):
    D = WIDTH
    return [torch.empty_like(x), tab.new_empty(tab.shape, dtype=torch.float32), bias.new_empty(bias.shape, dtype=torch.float32),
            tab.new_empty(saved[_SAVED.index("w_qkvg")].shape, dtype=torch.float32), tab.new_empty((D,), dtype=torch.float32),
            tab.new_empty((D, D), dtype=torch.float32), tab.new_empty(saved[_SAVED.index("w_exp")].shape, dtype=torch.float32),
            tab.new_empty(saved[_SAVED.index("w_sq")].shape, dtype=torch.float32),
            *[tab.new_empty((D,), dtype=torch.float32) for _ in range(4)]]


@opaque(fake=_bwd_fake, name="dit_variants_token_train_bwd")
def token_bwd(dout: torch.Tensor, x: torch.Tensor, tab: torch.Tensor, bias: torch.Tensor, km: torch.Tensor | None, saved: list[torch.Tensor],
              qkw: list[torch.Tensor], par: bool, sig: bool, gating: bool, kq: bool, up: bool, length: int, eps: float,
              eps_qk: float) -> list[torch.Tensor]:
    """The block's backward: [dx, dtab (fp32), dbias (fp32, bias' units), dwqkvg, dbq, dwo, dwexp, dwsq, dwq, dbq_ln, dwk, dbk_ln] (the last
    four zeros without kq)."""
    from miniworld_engine.kernels.augmented_attention import cuda as attn

    ext = _ext()
    s = dict(zip(_SAVED, saved, strict=True))
    M, D = x.shape
    L, A = length, M // length
    at = tab.dtype
    S = min(A, SPLITS)
    dev = x.device
    dout = dout.to(x.dtype).contiguous()
    part = torch.empty(S, L, 6 * D, device=dev, dtype=torch.float32)
    cols = lambda c: c * D  # noqa: E731
    w_qkvg, w_o, w_exp, w_sq = s["w_qkvg"], s["w_o"], s["w_exp"], s["w_sq"]
    qkvg, a, xt, og, ya, y2, ab, h = (s[n] for n in ("qkvg", "a", "xt", "og", "ya", "y2", "ab", "h"))
    dya = torch.empty(M, D, device=dev, dtype=at)
    dy2 = torch.empty(M, D, device=dev, dtype=at)
    if par:
        ext.res_bwd(dout, ya, tab, cols(G_A), dya, y2, cols(G_T), dy2, part, cols(G_A), cols(G_T), L, S)
    else:
        ext.res_bwd(dout, y2, tab, cols(G_T), dy2, None, 0, None, part, cols(G_T), 0, L, S)
    dh = torch.mm(dy2, w_sq)
    dwsq = _mm_t(dy2, h)
    dab = torch.empty_like(ab)
    ext.glu_bwd(ab, dh, dab, up)
    dt = torch.mm(dab, w_exp)
    dwexp = _mm_t(dab, xt)
    if not par:
        dxp = torch.empty(M, D, device=dev, dtype=torch.float32)
        ext.res_adaln_bwd(dout, dt, s["xp"], s["st2"], tab, cols(SC_T), cols(G_A), ya, dxp, dya, part, cols(SC_T), cols(SH_T), cols(G_A), L, S, sig)
    dog = torch.mm(dya, w_o)
    dwo = _mm_t(dya, og)
    nq = qkvg.shape[1] // D
    dO = torch.empty(M, D, device=dev, dtype=at)
    dqkvg = torch.empty(M, nq * D, device=dev, dtype=at)
    dd = torch.empty(A, HEADS, L, device=dev, dtype=torch.float32)
    ext.gate_o_bwd(dog, og, qkvg[:, 3 * D:] if gating else None, dO, dqkvg[:, 3 * D:] if gating else None, dd, L, HEAD_DIM, at == torch.float32)
    if at == torch.float32:
        dq, dk, dv, db = _tf32_attn_bwd(s["qs"], s["kb"], s["vb"], dO, bias, km, s["lse"], dd, A, L)
    else:
        dq, dk, dv, db = attn.backward(s["qs"], s["kb"], s["vb"], dO, bias, km, s["lse"], dd, A, L)
    nblk = (M + QKV_RPB - 1) // QKV_RPB
    qpart = torch.empty(nblk, (5 if kq else 1) * D, device=dev, dtype=torch.float32)
    wq, _, wk, _ = (qkw if kq else (None, None, None, None))
    ext.qkv_bwd(dq, dk, dv, qkvg[:, :D] if kq else None, qkvg[:, D:2 * D] if kq else None, s["stqk"] if kq else None, wq, wk,
                dqkvg[:, :D], dqkvg[:, D:2 * D], dqkvg[:, 2 * D:3 * D], qpart, QKV_RPB, kq)
    da = torch.mm(dqkvg, w_qkvg)
    dwqkvg = _mm_t(dqkvg, a)
    dx = torch.empty_like(x)
    if par:
        ext.adaln_bwd(x, s["st1"], dout, [da, dt], tab, [cols(SC_A), cols(SC_T)], [cols(SC_A), cols(SC_T)], [cols(SH_A), cols(SH_T)], dx, part, L, S, sig)
    else:
        ext.adaln_bwd(x, s["st1"], dxp, [da], tab, [cols(SC_A)], [cols(SC_A)], [cols(SH_A)], dx, part, L, S, sig)
    dtab = part.sum(0)
    qsum = qpart.sum(0)
    if kq:
        dwq, dbql, dwk, dbkl, dbq = (qsum[i * D:(i + 1) * D].clone() for i in range(5))
    else:
        dbq = qsum.clone()
        dwq, dbql, dwk, dbkl = (torch.zeros(D, device=dev, dtype=torch.float32) for _ in range(4))
    dbias = (db * LN2).contiguous() if db.dtype != torch.float32 else (db * LN2).contiguous()
    return [dx, dtab, dbias, dwqkvg, dbq, dwo, dwexp, dwsq, dwq, dbql, dwk, dbkl]


class _TokenBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tab, bias, km, wqkvg, bq, wo, wexp, wsq, wq, bql, wk, bkl, cfg):  # noqa: D102
        par, sig, gating, kq, up, length, eps, eps_qk = cfg
        qkw = [wq, bql, wk, bkl] if kq else []
        out, *saved = token_fwd(x, tab, bias, km, wqkvg, bq, wo, wexp, wsq, qkw, par, sig, gating, kq, up, length, eps, eps_qk)
        ctx.save_for_backward(x, tab, bias, km if km is not None else x.new_empty(0), *saved, wq, bql, wk, bkl)
        ctx.cfg, ctx.has_km, ctx.n_saved = cfg, km is not None, len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # noqa: D102
        par, sig, gating, kq, up, length, eps, eps_qk = ctx.cfg
        x, tab, bias, km, *rest = ctx.saved_tensors
        saved, qkw4 = rest[:ctx.n_saved], rest[ctx.n_saved:]
        res = token_bwd(dout, x, tab, bias, km if ctx.has_km else None, list(saved), list(qkw4) if kq else [], par, sig, gating, kq, up, length,
                        eps, eps_qk)
        dx, dtab, dbias, dwqkvg, dbq, dwo, dwexp, dwsq, dwq, dbql, dwk, dbkl = res
        return (dx, dtab.to(tab.dtype), dbias.to(bias.dtype), None, dwqkvg, dbq, dwo, dwexp, dwsq,
                *((dwq, dbql, dwk, dbkl) if kq else (None, None, None, None)), None)



def _pair_ln_fake(z, eps, out_bf16):
    return z.new_empty(z.shape, dtype=torch.bfloat16 if out_bf16 else z.dtype)


@opaque(fake=_pair_ln_fake, name="dit_variants_pair_ln_fwd")
def pair_ln_fwd(z: torch.Tensor, eps: float, out_bf16: bool) -> torch.Tensor:
    """LayerNorm (no affine) of the pair rows z [R, C] (C 128 | 256), bf16 or in z's dtype."""
    out = torch.empty(z.shape, device=z.device, dtype=torch.bfloat16 if out_bf16 else z.dtype)
    _ext().pair_ln_fwd(z, out, eps)
    return out


def _pair_ln_bwd_fake(z, g, eps):
    return torch.empty_like(z)


@opaque(fake=_pair_ln_bwd_fake, name="dit_variants_pair_ln_bwd")
def pair_ln_bwd(z: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """The LayerNorm backward of the pair rows: dz (z's dtype) from the gradient g of the normalised rows."""
    dz = torch.empty_like(z)
    _ext().pair_ln_bwd(z, g, dz, eps)
    return dz


class _PairLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, eps, out_bf16):  # noqa: D102
        ctx.save_for_backward(z)
        ctx.eps = eps
        return pair_ln_fwd(z, eps, out_bf16)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):  # noqa: D102
        (z,) = ctx.saved_tensors
        return pair_ln_bwd(z, g.contiguous(), ctx.eps), None, None

# ------------------------------------------------------------------------------------------------------------------ glue
def serves(stack, single, cond, pair, mask) -> bool:
    """A token stack call with a gradient on H100: bf16, D = 768 with 16 heads, L a multiple of 128, one conditioning row set shared by
    the samples, the CUDA training switch on."""
    from miniworld_engine.modules.exceptions import ImplementationType

    if stack.implementation == ImplementationType.PYTORCH or not single.is_cuda or single.ndim != 3:
        return False
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *stack.parameters()))):
        return False
    cfg = settings.current()
    if cfg.force_portable_kernels or not getattr(cfg, "dit_variant_train_cuda", True):
        return False
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return False
    a, length, d = single.shape
    blk = stack.blocks[0]
    if single.dtype not in (torch.bfloat16, torch.float32) or (d, blk.n_head) != (WIDTH, HEADS) or length % 128 or cond.ndim != 2 or cond.shape[0] != length:
        return False
    return (tuple(pair.shape[:2]) == (length, length) and pair.shape[-1] in (128, 256) and (mask is None or tuple(mask.shape) == (length,))
            and blk.expand.weight.shape[0] == 4 * d and a > 0)


def _tables(blk, s):
    """[L, 6 D] table of one block from the conditioning rows s [L, c]: the modules' own layers (autocast, autograd)."""
    def mod(norm):
        if norm.identity_scale:
            return norm.to_scale(s) + 1, norm.to_shift(s)
        sn = F.layer_norm(s.float(), s.shape[-1:], weight=norm.cond_norm_weight.float(), eps=1e-5).to(s.dtype)
        return norm.to_scale(sn), norm.to_shift(sn)

    sca, sha = mod(blk.attn_adaln)
    sct, sht = mod(blk.trans_adaln)
    parts = [sca, sha, blk.attn_zero(s), sct, sht, blk.trans_zero(s)]
    dt = parts[0].dtype
    return torch.cat([p.to(dt) for p in parts], dim=-1).contiguous()


def _pair_biases(stack, pair):
    """The blocks' biases [H, L, L] in log2 units (activation dtype), one LayerNorm of the pair and one stacked GEMM; the LN bias (Chai-1)
    only shifts a head's logits by a constant, which the softmax cancels (its gradient is exactly zero), so it is left out."""
    blocks = list(stack.blocks)
    length, c = pair.shape[0], pair.shape[-1]
    h = blocks[0].n_head
    eps = blocks[0].pair_norm.eps
    dt = torch.bfloat16 if pair.dtype == torch.bfloat16 or torch.is_autocast_enabled("cuda") else pair.dtype
    zhat = _PairLN.apply(pair.reshape(length * length, c).contiguous(), eps, dt == torch.bfloat16)
    w = torch.cat([b.to_pair_bias.weight.float() * b.pair_norm.weight.float()[None, :] for b in blocks]) * LOG2E
    out = torch.mm(w.to(dt), zhat.t())                          # [nb H, L L]
    return [out[h * i:h * (i + 1)].view(h, length, length) for i in range(len(blocks))]


def stack_train(stack, single, cond, pair, mask):
    """The token stack's forward under autograd on the H100 training kernels: x [A, L, D] -> [A, L, D]."""
    blocks = list(stack.blocks)
    a, length, d = single.shape
    biases = _pair_biases(stack, pair)
    km = None if mask is None else torch.where(mask.bool(), 0.0, -1e30).to(torch.float32).expand(a, length).contiguous()
    x = single.reshape(a * length, d).contiguous()
    empty = single.new_empty(0, dtype=torch.float32)
    for blk, bias in zip(blocks, biases, strict=True):
        tab = _tables(blk, cond)
        wqkvg = torch.cat([blk.to_q.weight, blk.to_k.weight, blk.to_v.weight, *((blk.to_g.weight,) if blk.gating else ())])
        wexp = torch.cat([blk.expand.weight, *((blk.up.weight,) if blk.up_gate else ())])
        qk = (blk.q_norm.weight, blk.q_norm.bias, blk.k_norm.weight, blk.k_norm.bias) if blk.kq_norm else (empty,) * 4
        cfg = (stack.parallel, not blk.attn_adaln.identity_scale, blk.gating, blk.kq_norm, blk.up_gate, length, blk.adaln_eps,
               blk.q_norm.eps if blk.kq_norm else 1e-5)
        x = _TokenBlock.apply(x, tab, bias, km, wqkvg, blk.to_q.bias, blk.to_out.weight, wexp, blk.squeeze.weight, *qk, cfg)
    return x.view(a, length, d)


# ====================================================================================================================== atom stack
# The atom DiT blocks (128 wide, 4 heads x 32, 32 queries x 128 keys per window): the same row kernels at D = 128, the local DiT's mma.sync
# window attention (``attn_fwd_mma`` / ``attn_bwd_mma``), cuBLAS. Per block the tables are [N, 8 D]: q-AdaLN scale | shift, k-AdaLN scale |
# shift, attention gate, transition scale | shift | gate. The slid key window's five edge windows (the only ones whose keys differ from the
# clipped window the kernel reads) are redone in torch over their own keys, forward and backward.
ATOM_D, ATOM_H, ATOM_DH, ATOM_WQ, ATOM_WK = 128, 4, 32, 32, 128
A_SC_Q, A_SH_Q, A_SC_K, A_SH_K, A_G_A, A_SC_T, A_SH_T, A_G_T = range(8)
ATOM_SPLITS = 2
_ANAMES = ("out", "xq", "xk", "xt", "q", "k", "v", "qn", "kn", "stqk", "g", "og", "lse", "ya", "y2", "xp", "st1", "st2", "ab", "h",
           "w_gq", "w_kv", "w_o", "w_exp", "w_sq")
_ASAVED = _ANAMES[1:]


def _local():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    return local





def _edge():
    from miniworld_engine.kernels.dit_variants import cuda

    return cuda.edge_ext()


class _EdgeBias(torch.autograd.Function):
    """The slid edge windows' bias [5, 4, 32, 128] (natural units, -inf off the valid keys) from the trunked pair z: LN(z[widx]) gamma . Wb."""

    @staticmethod
    def forward(ctx, z, gamma, wb, widx, valid, eps):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb, widx)
        ctx.eps = eps
        return _edge().edge_bias_fwd(z.index_select(0, widx).contiguous(), gamma.float().contiguous(), wb.float().contiguous(), valid, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, deb):  # noqa: D102
        z, gamma, wb, widx = ctx.saved_tensors
        dz_e, part = _edge().edge_bias_bwd(z.index_select(0, widx).contiguous(), gamma.float().contiguous(), wb.float().contiguous(), deb.contiguous(), ctx.eps)
        dz = torch.zeros_like(z)
        dz.index_add_(0, widx, dz_e)
        tot = part.sum(0)
        return dz, tot[:16].to(gamma.dtype), tot[16:].view(4, 16).to(wb.dtype), None, None, None

def _afwd_specs(M, N, at, xdt, par, gating, proj, kq, up):
    D = ATOM_D
    f32 = torch.float32
    hid = 2 * D
    nab = 3 if up else 2
    return {
        "out": ((M, D), xdt), "xq": ((M, D), at), "xk": ((M, D), at), "xt": ((M, D), at), "q": ((M, D), at), "k": ((M, D), at), "v": ((M, D), at),
        "qn": ((M, D) if kq else (0,), at), "kn": ((M, D) if kq else (0,), at), "stqk": ((M, 4) if kq else (0,), f32),
        "g": ((M, D) if gating else (0,), at), "og": ((M, D), at), "lse": ((M // N, ATOM_H, N), f32), "ya": ((M, D) if proj else (0,), at),
        "y2": ((M, D), at), "xp": ((0,) if par else (M, D), f32), "st1": ((M, 2), f32), "st2": ((0,) if par else (M, 2), f32),
        "ab": ((M, nab * hid), at), "h": ((M, hid), at),
        "w_gq": (((2 if gating else 1) * D, D), at), "w_kv": ((2 * D, D), at), "w_o": ((D, D) if proj else (0,), at), "w_exp": ((nab * hid, D), at),
        "w_sq": ((D, hid), at),
    }


def _afwd_fake(x, tab, bias2, widx, keys, ebias, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, par, sig, gating, proj, kq, up, slide, length, eps, eps_qk):
    spec = _afwd_specs(x.shape[0], length, tab.dtype, x.dtype, par, gating, proj, kq, up)
    return [x.new_empty(spec[n][0], dtype=spec[n][1]) for n in _ANAMES]


@opaque(fake=_afwd_fake, name="dit_variants_atom_train_fwd")
def atom_fwd(x: torch.Tensor, tab: torch.Tensor, bias2: torch.Tensor, widx: torch.Tensor, keys: torch.Tensor, ebias: torch.Tensor, wq: torch.Tensor,
             bq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wexp: torch.Tensor, wsq: torch.Tensor,
             qkw: list[torch.Tensor], par: bool, sig: bool, gating: bool, proj: bool, kq: bool, up: bool, slide: bool, length: int, eps: float,
             eps_qk: float) -> list[torch.Tensor]:
    """One atom block's forward: [out, saved ...] (``_ANAMES``). x [A N, 128], tab [N, 8 D], bias2 [4, nwin, 32, 128] fp32 (log2, -inf off the
    keys); the slid edge windows (widx [5], keys [5, 128], ebias [5, 4, 32, 128] natural units) when ``slide``."""
    ext = _ext()
    local = _local()
    M, D = x.shape
    N, A = length, M // length
    at = tab.dtype
    w_q, w_k, w_v, w_exp, w_sq = (w.to(at, copy=True) for w in (wq, wk, wv, wexp, wsq))
    w_gq = torch.cat([wg, wq]).to(at) if gating else w_q.clone()
    w_kv = torch.cat([wk, wv]).to(at)
    w_o = wo.to(at, copy=True) if proj else wo.new_empty(0)
    spec = _afwd_specs(M, N, at, x.dtype, par, gating, proj, kq, up)
    skip = ("w_gq", "w_kv", "w_o", "w_exp", "w_sq", "q", "og", "lse", "ya")
    buf = {n: torch.empty(spec[n][0], device=x.device, dtype=spec[n][1]) for n in _ANAMES if n not in skip}
    cols = lambda c: c * D  # noqa: E731
    xq, xk, xt, st1, k, v = buf["xq"], buf["xk"], buf["xt"], buf["st1"], buf["k"], buf["v"]
    if par:
        ext.ln_mod_fwd(x, tab, [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)], [cols(A_SH_Q), cols(A_SH_K), cols(A_SH_T)], [xq, xk, xt], st1, N, eps, sig)
    else:
        ext.ln_mod_fwd(x, tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], [xq, xk], st1, N, eps, sig)
    q = torch.addmm(bq.to(at), xq, w_q.t())
    torch.mm(xk, w_kv[:D].t(), out=k)
    torch.mm(xk, w_kv[D:].t(), out=v)
    g = buf["g"]
    if gating:
        torch.mm(xq, w_gq[:D].t(), out=g)
    f32 = at == torch.float32
    if kq:
        qn, kn = buf["qn"], buf["kn"]
        ext.qkv_prep(q, k, v, qn, kn, v if f32 else None, buf["stqk"], qkw[0], qkw[1], qkw[2], qkw[3], 1.0, eps_qk, True, f32)
    else:
        qn, kn = q, k
        if f32:       # the TF32 core's operands, rounded to nearest in place
            ext.qkv_prep(q, k, v, q, k, v, None, None, None, None, None, 1.0, eps_qk, False, True)
    o, lse = local.attn_fwd_mma(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2) if at == torch.bfloat16 else local.attn_fwd_mma32(
        qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), bias2)
    if slide:
        _edge().edge_attn_fwd(qn.view(A, N, D), kn.view(A, N, D), v.view(A, N, D), keys, widx, ebias, o)
    o = o.view(M, D)
    if gating:
        og = torch.empty_like(o)
        ext.gate_fwd(o, g, og)
    else:
        og = o
    ya = torch.mm(og, w_o.t()) if proj else og
    out, y2 = buf["out"], buf["y2"]
    ab, h = buf["ab"], buf["h"]
    if par:
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(x, ya, tab, cols(A_G_A), y2, cols(A_G_T), out, N)
    else:
        ext.res_ln_fwd(x, ya, tab, cols(A_G_A), cols(A_SC_T), cols(A_SH_T), buf["xp"], xt, buf["st2"], N, eps, sig)
        torch.mm(xt, w_exp.t(), out=ab)
        ext.glu_fwd(ab, h, up)
        torch.mm(h, w_sq.t(), out=y2)
        ext.res_out_fwd(buf["xp"], y2, tab, cols(A_G_T), None, 0, out, N)
    res = dict(buf, q=q, og=og, lse=lse, ya=ya if proj else ya.new_empty(0), w_gq=w_gq, w_kv=w_kv, w_o=w_o, w_exp=w_exp, w_sq=w_sq)
    return [res[n] for n in _ANAMES]


def _abwd_fake(dout, x, tab, bias2, widx, keys, ebias, saved, wq, wk, wv, wg, wo, wexp, wsq, qkw, par, sig, gating, proj, kq, up, slide, length, eps,
               eps_qk):
    D = ATOM_D
    f32 = torch.float32
    return [torch.empty_like(x), tab.new_empty(tab.shape, dtype=f32), bias2.new_empty(bias2.shape, dtype=f32), ebias.new_empty(ebias.shape, dtype=f32),
            wq.new_empty(wq.shape, dtype=f32), wq.new_empty((D,), dtype=f32), wk.new_empty(wk.shape, dtype=f32), wv.new_empty(wv.shape, dtype=f32),
            wg.new_empty(wg.shape, dtype=f32), wo.new_empty(wo.shape, dtype=f32), wexp.new_empty(wexp.shape, dtype=f32),
            wsq.new_empty(wsq.shape, dtype=f32), *[wq.new_empty((D,), dtype=f32) for _ in range(4)]]


@opaque(fake=_abwd_fake, name="dit_variants_atom_train_bwd")
def atom_bwd(dout: torch.Tensor, x: torch.Tensor, tab: torch.Tensor, bias2: torch.Tensor, widx: torch.Tensor, keys: torch.Tensor, ebias: torch.Tensor,
             saved: list[torch.Tensor], wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wexp: torch.Tensor,
             wsq: torch.Tensor, qkw: list[torch.Tensor], par: bool, sig: bool, gating: bool, proj: bool, kq: bool, up: bool, slide: bool,
             length: int, eps: float, eps_qk: float) -> list[torch.Tensor]:
    """The atom block's backward: [dx, dtab, dbias2, debias, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwq_ln, dbq_ln, dwk_ln, dbk_ln]."""
    ext = _ext()
    local = _local()
    s = dict(zip(_ASAVED, saved, strict=True))
    M, D = x.shape
    N, A = length, M // length
    at = tab.dtype
    S = min(A, ATOM_SPLITS)
    dev = x.device
    f32 = torch.float32
    dout = dout.to(x.dtype).contiguous()
    part = torch.empty(S, N, 8 * D, device=dev, dtype=f32)
    cols = lambda c: c * D  # noqa: E731
    w_gq, w_kv, w_o, w_exp, w_sq = s["w_gq"], s["w_kv"], s["w_o"], s["w_exp"], s["w_sq"]
    xq, xk, xt, og, ya, y2, ab, h = (s[n] for n in ("xq", "xk", "xt", "og", "ya", "y2", "ab", "h"))
    if not proj:
        ya = og
    dya = torch.empty(M, D, device=dev, dtype=at)
    dy2 = torch.empty(M, D, device=dev, dtype=at)
    if par:
        ext.res_bwd(dout, ya, tab, cols(A_G_A), dya, y2, cols(A_G_T), dy2, part, cols(A_G_A), cols(A_G_T), N, S)
    else:
        ext.res_bwd(dout, y2, tab, cols(A_G_T), dy2, None, 0, None, part, cols(A_G_T), 0, N, S)
    dh = torch.mm(dy2, w_sq)
    dwsq = _mm_t(dy2, h)
    dab = torch.empty_like(ab)
    ext.glu_bwd(ab, dh, dab, up)
    dt = torch.mm(dab, w_exp)
    dwexp = _mm_t(dab, xt)
    if not par:
        dxp = torch.empty(M, D, device=dev, dtype=f32)
        ext.res_adaln_bwd(dout, dt, s["xp"], s["st2"], tab, cols(A_SC_T), cols(A_G_A), ya, dxp, dya, part, cols(A_SC_T), cols(A_SH_T), cols(A_G_A), N, S, sig)
    dog = torch.mm(dya, w_o) if proj else dya
    dwo = _mm_t(dya, og) if proj else wo.new_empty(wo.shape, dtype=f32)
    dO = torch.empty(M, D, device=dev, dtype=at)
    dp = torch.empty(M, 4 * D, device=dev, dtype=at)               # [dg | dq | dk | dv]
    dd = torch.empty(A, ATOM_H, N, device=dev, dtype=f32)
    ext.gate_o_bwd(dog, og, s["g"] if gating else None, dO, dp[:, :D] if gating else None, dd, N, ATOM_DH, at == torch.float32)
    qa, ka = (s["qn"], s["kn"]) if kq else (s["q"], s["k"])
    if slide:
        # the five edge windows are the edge kernels' (their dO / D rows become zeros for the main kernel, which then adds nothing for them)
        dqe = torch.empty(A, 5, ATOM_WQ, D, device=dev, dtype=f32)
        dke = torch.empty(A, 5, ATOM_WK, D, device=dev, dtype=f32)
        dve = torch.empty(A, 5, ATOM_WK, D, device=dev, dtype=f32)
        dbp = torch.empty(A, 5, ATOM_H, ATOM_WQ, ATOM_WK, device=dev, dtype=f32)
        _edge().edge_bwd_compute(qa.view(A, N, D), ka.view(A, N, D), s["v"].view(A, N, D), dO.view(A, N, D), dd, keys, widx, ebias, dqe, dke, dve, dbp)
    dbias2 = local.attn_bwd_mma(qa.view(A, N, D), ka.view(A, N, D), s["v"].view(A, N, D), dO.view(A, N, D), bias2, s["lse"], dd, dp[:, D:],
                                at == torch.float32)
    if slide:
        _edge().edge_bwd_apply(dp[:, D:], keys, widx, dqe, dke, dve, A, N)
        debias = dbp.sum(0)
    else:
        debias = ebias.new_empty(ebias.shape, dtype=f32)
    if kq:
        nblk = (M + QKV_RPB - 1) // QKV_RPB
        qpart = torch.empty(nblk, 5 * D, device=dev, dtype=f32)
        ext.qkv_bwd(dp[:, D:2 * D], dp[:, 2 * D:3 * D], dp[:, 3 * D:], s["q"], s["k"], s["stqk"], qkw[0], qkw[2], dp[:, D:2 * D], dp[:, 2 * D:3 * D],
                    dp[:, 3 * D:], qpart, QKV_RPB, True)
        qsum = qpart.sum(0)
        dwq_ln, dbq_ln, dwk_ln, dbk_ln, dbq = (qsum[i * D:(i + 1) * D].clone() for i in range(5))
    else:
        nblk = (M + QKV_RPB - 1) // QKV_RPB
        qpart = torch.empty(nblk, D, device=dev, dtype=f32)
        ext.qkv_bwd(dp[:, D:2 * D], None, None, None, None, None, None, None, None, None, None, qpart, QKV_RPB, False)
        dbq = qpart.sum(0)
        dwq_ln, dbq_ln, dwk_ln, dbk_ln = (torch.zeros(D, device=dev, dtype=f32) for _ in range(4))
    if gating:
        dxq = torch.mm(dp[:, :2 * D], w_gq)
        dwgq = _mm_t(dp[:, :2 * D], xq)
        dwg, dwq = dwgq[:D], dwgq[D:]
    else:
        dxq = torch.mm(dp[:, D:2 * D], w_gq)
        dwq = _mm_t(dp[:, D:2 * D], xq)
        dwg = wg.new_empty(wg.shape, dtype=f32)
    dxk = torch.mm(dp[:, 2 * D:], w_kv)
    dwkv = _mm_t(dp[:, 2 * D:], xk)
    dx = torch.empty_like(x)
    if par:
        ext.adaln_bwd(x, s["st1"], dout, [dxq, dxk, dt], tab, [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)], [cols(A_SC_Q), cols(A_SC_K), cols(A_SC_T)],
                      [cols(A_SH_Q), cols(A_SH_K), cols(A_SH_T)], dx, part, N, S, sig)
    else:
        ext.adaln_bwd(x, s["st1"], dxp, [dxq, dxk], tab, [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SC_Q), cols(A_SC_K)], [cols(A_SH_Q), cols(A_SH_K)], dx,
                      part, N, S, sig)
    dtab = part.sum(0)
    return [dx, dtab, dbias2, debias, dwq.clone(), dbq, dwkv[:D].clone(), dwkv[D:].clone(), dwg.clone(), dwo, dwexp, dwsq,
            dwq_ln, dbq_ln, dwk_ln, dbk_ln]


class _AtomBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tab, bias2, widx, keys, ebias, wq, bq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl, cfg):  # noqa: D102
        par, sig, gating, proj, kq, up, slide, length, eps, eps_qk = cfg
        qkw = [wql, bql, wkl, bkl] if kq else []
        out, *saved = atom_fwd(x, tab, bias2, widx, keys, ebias, wq, bq, wk, wv, wg, wo, wexp, wsq, qkw, par, sig, gating, proj, kq, up, slide, length,
                               eps, eps_qk)
        ctx.save_for_backward(x, tab, bias2, widx, keys, ebias, *saved, wq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl)
        ctx.cfg, ctx.n_saved = cfg, len(saved)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):  # noqa: D102
        par, sig, gating, proj, kq, up, slide, length, eps, eps_qk = ctx.cfg
        x, tab, bias2, widx, keys, ebias, *rest = ctx.saved_tensors
        saved, (wq, wk, wv, wg, wo, wexp, wsq, wql, bql, wkl, bkl) = rest[:ctx.n_saved], rest[ctx.n_saved:]
        qkw = [wql, bql, wkl, bkl] if kq else []
        res = atom_bwd(dout, x, tab, bias2, widx, keys, ebias, list(saved), wq, wk, wv, wg, wo, wexp, wsq, qkw, par, sig, gating, proj, kq, up, slide,
                       length, eps, eps_qk)
        dx, dtab, dbias2, debias, dwq, dbq, dwk, dwv, dwg, dwo, dwexp, dwsq, dwql, dbql, dwkl, dbkl = res
        return (dx, dtab.to(tab.dtype), dbias2, None, None, debias if slide else None, dwq, dbq, dwk, dwv, dwg if gating else None,
                dwo if proj else None, dwexp, dwsq, *((dwql, dbql, dwkl, dbkl) if kq else (None, None, None, None)), None)


class _AtomPairBias(torch.autograd.Function):
    """bias2 [4, nwin, 32, 128] fp32 (log2 units, -inf off the keys) of one block from the trunked pair z: ``pair_bias_mma`` / ``pair_bias_bwd_mma``."""

    @staticmethod
    def forward(ctx, z, gamma, wb, kmask, n, eps):  # noqa: D102
        ctx.save_for_backward(z, gamma, wb)
        ctx.eps = eps
        return _local().pair_bias_mma(z, gamma, wb, kmask, n, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, dbias):  # noqa: D102
        z, gamma, wb = ctx.saved_tensors
        dz, dgamma, dwb = _local().pair_bias_bwd_mma(z, gamma, wb, dbias.contiguous(), ctx.eps)
        return dz, dgamma.to(gamma.dtype), dwb.to(wb.dtype), None, None, None


def atom_serves(stack, single, cond, pair, mask, uid) -> bool:
    """An atom stack call with a gradient on H100: bf16, 128 wide, N a multiple of 64, a conditioning row set shared by the samples, pair
    channels 16, the CUDA training switch on."""
    from miniworld_engine.modules.exceptions import ImplementationType

    if stack.implementation == ImplementationType.PYTORCH or not single.is_cuda or single.ndim != 3:
        return False
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (single, cond, pair, *stack.parameters()))):
        return False
    cfg = settings.current()
    if cfg.force_portable_kernels or not getattr(cfg, "dit_variant_train_cuda", True):
        return False
    if torch.cuda.get_device_capability(single.device) != (9, 0):
        return False
    a, n, d = single.shape
    return (single.dtype in (torch.bfloat16, torch.float32) and d == ATOM_D and n % 64 == 0 and cond.ndim == 2 and tuple(cond.shape) == (n, d)
            and tuple(pair.shape) == (n // 32, 32, 128, 16) and (mask is None or tuple(mask.shape) == (n,))
            and (uid is None or tuple(uid.shape) == (n,)) and stack.blocks[0].expand.weight.shape[0] == 4 * d and a > 0)


def _atom_tables(blk, c):
    def mod(norm):
        if norm.identity_scale:
            return norm.to_scale(c) + 1, norm.to_shift(c)
        cn = F.layer_norm(c.float(), c.shape[-1:], weight=norm.cond_norm_weight.float(), eps=1e-5).to(c.dtype)
        return norm.to_scale(cn), norm.to_shift(cn)

    scq, shq = mod(blk.q_adaln)
    sck, shk = mod(blk.k_adaln)
    sct, sht = mod(blk.trans_adaln)
    parts = [scq, shq, sck, shk, blk.attn_zero(c), sct, sht, blk.trans_zero(c)]
    dt = parts[0].dtype
    return torch.cat([p.to(dt) for p in parts], dim=-1).contiguous()


def atom_stack_train(stack, single, cond, pair, mask, uid):
    """The atom stack's forward under autograd on the H100 training kernels: x [A, N, 128] -> [A, N, 128]."""
    blocks = list(stack.blocks)
    a, n, d = single.shape
    nwin = n // ATOM_WQ
    dev = single.device
    kmask = torch.ones(n, dtype=torch.bool, device=dev) if mask is None else mask.bool()
    slide = stack.key_window == "slide"
    j = torch.arange(ATOM_WK, device=dev)
    # the clipped window's key of slot j: 32 w - 48 + j (outside [0, N): -inf already); a padding query keeps its keys
    same = None
    if uid is not None:
        idx = (ATOM_WQ * torch.arange(nwin, device=dev) - 48)[:, None] + j
        same = (uid.view(nwin, ATOM_WQ, 1) == uid[idx.clamp(0, n - 1)][:, None, :]) | ~kmask.view(nwin, ATOM_WQ, 1)
    widx = keys = valid = None
    if slide:
        n_real = kmask.sum()
        last = (n_real - 1) // 32
        widx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)
        starts = torch.minimum((ATOM_WQ * widx - 48).clamp(min=0), (n_real - ATOM_WK).clamp(min=0))
        keys = starts[:, None] + j
        valid = (keys < n_real)[:, None, :]
        keys = keys.clamp(max=n - 1)
        if uid is not None:
            rows = ATOM_WQ * widx[:, None] + torch.arange(ATOM_WQ, device=dev)
            valid = valid & ((uid[rows][:, :, None] == uid[keys][:, None, :]) | ~kmask[rows][:, :, None])
        valid = valid.expand(5, ATOM_WQ, ATOM_WK).to(torch.uint8).contiguous()
    z = pair.contiguous()
    x = single.reshape(a * n, d).contiguous()
    empty = single.new_empty(0, dtype=torch.float32)
    for i, blk in enumerate(blocks):
        norm = blk.pair_norm if stack.per_block_pair_norm else stack.pair_norm
        wb = blk.to_pair_bias.weight if stack.per_block_pair_norm else stack.to_pair_bias.weight[ATOM_H * i:ATOM_H * (i + 1)]
        bias2 = _AtomPairBias.apply(z, norm.weight, wb, kmask, n, norm.eps)
        if same is not None:
            bias2 = bias2.masked_fill(~same[None], float("-inf"))
        if slide:
            ebias = _EdgeBias.apply(z, norm.weight, wb, widx, valid, norm.eps)
        else:
            ebias = empty
        tab = _atom_tables(blk, cond)
        qk = (blk.q_norm.weight, blk.q_norm.bias, blk.k_norm.weight, blk.k_norm.bias) if blk.kq_norm else (empty,) * 4
        cfg = (stack.parallel, not blk.q_adaln.identity_scale, blk.gating, blk.project_output, blk.kq_norm, blk.up_gate, slide, n, blk.adaln_eps,
               blk.q_norm.eps if blk.kq_norm else 1e-5)
        wg = blk.to_g.weight if blk.gating else empty
        wo = blk.to_out.weight if blk.project_output else empty
        wexp = torch.cat([blk.expand.weight, *((blk.up.weight,) if blk.up_gate else ())])
        x = _AtomBlock.apply(x, tab, bias2, widx if slide else empty, keys if slide else empty, ebias, blk.to_q.weight, blk.to_q.bias, blk.to_k.weight,
                             blk.to_v.weight, wg, wo, wexp, blk.squeeze.weight, *qk, cfg)
    return x.view(a, n, d)
