"""H100 (sm_90a) TriangleAttention, starting node, d_pair = 128, 4 heads x 32, bf16.

Kernel bodies are raw-PTX wgmma / TMA hand-CUDA in ``h100_sources/`` (no CUTLASS / CuTe), on the kernel boundaries of the B200
path (``b200_triattn.py``): front (LN + projections + bias) | attention core | tail (gate + out-projection + residual), and the
matching backward.
"""

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")

C = 128  # d_pair
H = 4  # heads
D = C // H  # head dim (the kernels take 32 only)
TOK = 128  # query / key tile; L must be a multiple of it


def _flags():
    return [
        *host_flags(),
        "-std=c++17",
        "-O3",
        *gencodes("90a"),
        "--use_fast_math",
        "-DNDEBUG",
        f"-I{_SRC}",
        "-Xptxas=-v",
    ]


@functools.lru_cache(maxsize=None)
def _core(rows: int = 2, defs: tuple[str, ...] = ()):
    """The attention core; `defs` = extra -D schedule knobs (experiments), part of the build name."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_core_r{rows}{tag}_v12",
        sources=[str(_SRC / "triattn_sm90.cu")],
        extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}", *(f"-D{d}" for d in defs)],
        verbose=True,
    )


@functools.lru_cache(maxsize=None)
def _core_h16(defs: tuple[str, ...] = ()):
    """The inference core on fp16 operands (h100_sources/triattn_h16_sm90.cu): q / k / v / sbias fp16, out bf16."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_core_h16{tag}_v1",
        sources=[str(_SRC / "triattn_h16_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


# inference(): the front writes q / k / v and the bias in fp16 for the fp16 core (faster, and closer to fp32: see triattn_h16_sm90.cu)
INFERENCE_FP16 = True


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


def attention(q, k, v, bias, scale=D**-0.5, want_lse=False, rows=2, defs=()):
    """softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S] -> (out, lse, flags)."""
    return _core(rows, tuple(defs)).triattn_fwd(
        q, k, v, scaled_bias(bias, scale), scale, want_lse
    )


def scaled_bias(bias, scale=D**-0.5):
    """The core's bias operand: bf16(bias / scale) [B, H, S, S] (the module's front writes it directly)."""
    return (bias.float() / scale).to(torch.bfloat16).contiguous()


@functools.lru_cache(maxsize=None)
def _mod(defs: tuple[str, ...] = ()):
    """The module kernels around the core (front, tail, backward); `defs` = extra -D knobs (experiments)."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_mod{tag}_v10",
        sources=[str(_SRC / "triattn_mod_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


def pack_front(wq, wk, wv, wg, wb):
    """[Wq; Wk; Wv; Wg] -> bf16 [512, 128]; Wb -> bf16 [16, 128] (zero rows below the 4 heads)."""
    w4 = torch.cat([wq, wk, wv, wg]).to(torch.bfloat16).contiguous()
    wb16 = wb.new_zeros(16, wb.shape[1], dtype=torch.bfloat16)
    wb16[: wb.shape[0]] = wb.to(torch.bfloat16)
    return w4, wb16


def flip_bias(sbias, mask):
    """The ending node on AlphaFold 3's bias orientation, from the front's output: the transpose of the bias the front projected
    from the transposed pair, with the key mask applied afterwards (to the transposed axis).  ``sbias`` is [B, H, L, L] bf16 made
    with no mask; masked keys are -inf, as the front writes them."""
    out = sbias.transpose(-1, -2)
    if mask is not None:
        out = out.masked_fill(~mask[:, None, None, :], float("-inf"))
    return out.contiguous()


def inference(pair, lnw, lnb, w4, wb16, wo16, mask=None, eps=1e-5, trans=False, out=None, flip=False, gate_bias=None, out_bias=None):
    """pair + TriangleAttention(pair) for the starting node, no grad: front (LN + projections + bias) -> core -> tail (gate +
    out-projection + residual).  w4, wb16 = pack_front(...); wo16 = Wo in bf16; `out`: a contiguous tensor like pair to write into.  ``flip``: the ending node (``trans``) takes its
    bias from the pair as it is (:func:`flip_bias`). ``gate_bias`` [128] / ``out_bias`` [C]: biases of the gate / out-projection
    (RoseTTAFold3), added to the front's gate and to the result."""
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    empty = torch.empty(0, device=pair.device)
    q, k, v, g, sbias = m.tri_front(
        x,
        lnw,
        lnb,
        eps,
        w4,
        wb16,
        mask if mask is not None and not flip else empty.bool(),
        B,
        L,
        D**-0.5,
        trans,
        INFERENCE_FP16,
    )
    if flip:
        sbias = flip_bias(sbias, mask)
    shp = (B, L, L, H, D)
    core = _core_h16().triattn_fwd_h16 if INFERENCE_FP16 else _core().triattn_fwd
    o, _, _, _ = core(q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False)
    # the biases in fp32 inside the tail, before its one rounding: added to the bf16 gate / sum afterwards they were a second
    # rounding each, the out bias at the residual's scale (RoseTTAFold3's trunk drifted from its reference by it)
    fp32 = lambda t: None if t is None else t.float().contiguous()  # noqa: E731
    return m.tri_tail(
        g, o.view(-1, C), x, wo16, empty.bfloat16(), L, trans, out, bo=fp32(out_bias), gb=fp32(gate_bias)
    ).view(B, L, L, C)


@functools.lru_cache(maxsize=None)
def _bwd(defs: tuple[str, ...] = ()):
    """The core backward kernels (delta, key side, query side); `defs` = extra -D knobs (experiments)."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_bwd{tag}_v12",
        sources=[str(_SRC / "triattn_bwd_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


def _pack_params(wq, wk, wv, wg, wb, wo):
    """(w4, wb16, wo16) in one launch when the six parameters share a dtype (fp32 master or bf16); the torch pack otherwise."""
    if (
        len({t.dtype for t in (wq, wk, wv, wg, wb, wo)}) == 1
        and wq.dtype in (torch.float32, torch.bfloat16)
        and all(t.is_contiguous() for t in (wq, wk, wv, wg, wb, wo))
    ):
        return _mod().tri_pack(wq, wk, wv, wg, wb, wo)
    w4, wb16 = pack_front(wq, wk, wv, wg, wb)
    return w4, wb16, wo.to(torch.bfloat16).contiguous()


def train_forward(
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need=True, trans=False
):
    """The module forward with the tensors its backward reads: -> (out [B, L, L, 128], (q, k, v, g, sbias, o, lse)); o / lse are empty
    unless `need`.  Every returned tensor is a fresh allocation (none aliases an argument), so it can be an output of a custom op."""
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    w4, wb16, wo16 = _pack_params(wq, wk, wv, wg, wb, wo)
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    empty = torch.empty(0, device=pair.device)
    q, k, v, g, sbias = m.tri_front(
        x,
        lw,
        lb,
        eps,
        w4,
        wb16,
        mask if mask is not None else empty.bool(),
        B,
        L,
        D**-0.5,
        trans,
    )
    shp = (B, L, L, H, D)
    o, lse, _, _ = _core().triattn_fwd(
        q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, need
    )
    drop = drop if drop is not None else empty.bfloat16()
    out = m.tri_tail(g, o.view(-1, C), x, wo16, drop, L, trans)
    return out.view(B, L, L, C), (q, k, v, g, sbias, o, lse)


def train_backward(
    dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, eps, drop, saved, trans=False
):
    """Gradients (pair, lnw, lnb, wq, wk, wv, wg, wb, wo) of `train_forward`, each in its input's dtype."""
    q, k, v, g, sbias, o, lse = saved
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m, bw = _mod(), _bwd()
    w4, _, wo16 = _pack_params(wq, wk, wv, wg, wb, wo)
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    empty = torch.empty(0, device=pair.device)
    drop = drop if drop is not None else empty.bfloat16()
    dbias = torch.zeros(
        B, H, L, L, device=x.device, dtype=torch.float32
    )  # the query-side kernel adds its partials in
    dy = dy.contiguous().view(-1, C)
    do, dg, delta, dpair, dwo_part = m.tri_gate_bwd(
        dy, g, o.view(-1, C), wo16, drop, B, L, trans
    )
    shp = (B, L, L, H, D)
    q5, k5, v5, do5 = q.view(shp), k.view(shp), v.view(shp), do.view(shp)
    dk, dv = bw.triattn_bwd_kv(q5, k5, v5, sbias, do5, lse, delta, D**-0.5)
    dq = bwd_q(bw, q5, k5, v5, sbias, do5, lse, delta, D**-0.5, dbias)
    dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
    m.tri_head_bwd(dq, dk, dv, dg, dbias, x, w4, wb, lw, eps, dpair, B, L, trans)
    # fp32 outputs whenever any parameter is fp32 (master weights, or the bf16 model's fp32 LayerNorm affine): the wrapper's `.to` then
    # rounds the bf16 ones exactly as the kernel's own bf16 store would, and gamma / beta keep their fp32 value
    fp32 = wq.dtype == torch.float32 or lnw.dtype == torch.float32
    dwq, dwk, dwv, dwg, dgamma, dbeta, dwb, dwo = m.tri_wgrad(
        dq, dk, dv, dg, dbias, x, eps, B, L, w4, wb, lw, lb, dwo_part, fp32, trans
    )
    # separate buffers from the kernel (no slices: the outputs of a custom op may not alias each other); .to is a no-op in the common case
    return (
        dpair.view(B, L, L, C),
        dgamma.to(lnw.dtype),
        dbeta.to(lnb.dtype),
        dwq.to(wq.dtype),
        dwk.to(wk.dtype),
        dwv.to(wv.dtype),
        dwg.to(wg.dtype),
        dwb.to(wb.dtype),
        dwo.to(wo.dtype),
    )


class TriAttnFused(torch.autograd.Function):
    """The whole module (residual included): pair [B, L, L, 128] -> pair + drop o TriangleAttention(pair), starting node.
    Parameters may be bf16 or fp32 (master weights); the gradients come back in their dtype.
    drop: the module's broadcast dropout scale as [B, L, 128] bf16 (index = the row's last position), or None.
    (The module path goes through the custom ops of ``integrations.triattn_h100_sm90``, which call `train_forward` / `train_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop):
        need = any(ctx.needs_input_grad)
        out, saved = train_forward(
            pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need
        )
        if need:
            ctx.save_for_backward(
                pair,
                lnw,
                lnb,
                wq,
                wk,
                wv,
                wg,
                wb,
                wo,
                drop if drop is not None else pair.new_empty(0),
                *saved,
            )
            ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, wq, wk, wv, wg, wb, wo, drop, *saved = ctx.saved_tensors
        grads = train_backward(
            dy,
            pair,
            lnw,
            lnb,
            wq,
            wk,
            wv,
            wg,
            wb,
            wo,
            ctx.eps,
            drop if drop.numel() else None,
            saved,
        )
        return (*grads, None, None, None)


def triangle_attention(
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None
):
    """pair + drop o TriangleAttention(pair) for the starting node; the caller transposes for the ending node."""
    return TriAttnFused.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop)


#: Query-side backward launches: tasks = (b, h, 128-query tile) x slabs, each a whole tile's pair rows (its dbias partials
#: reduce-added in order).  A launch of many samples was slower per sample than launches of a few (B48: 1.07-1.55x over H 4-16 and
#: L128-768, 2026-10-20): the tasks after the first wave start out of step with the other query tiles of their (b, h), which then
#: read the same K / V rows from DRAM at different times.  So B samples run in launches of ``g`` samples whose tasks make one wave of
#: ~96 CTAs (one sample, any g >= 1: the single launch as before).
BWD_Q_WAVE = 96


def bwd_q(bw, q, k, v, sbias, do, lse, delta, scale, dbias, dq=None):
    """dq of the core's query side (dbias partials added into ``dbias``), the B samples in launches of ``g`` samples (see BWD_Q_WAVE)."""
    B, _, S, H, _ = q.shape
    g = max(1, BWD_Q_WAVE // (H * (S // TOK)))
    if g >= B:
        return bw.triattn_bwd_q(q, k, v, sbias, do, lse, delta, scale, dbias, dq)
    if dq is None:
        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    for i in range(0, B, g):
        s = slice(i, min(B, i + g))
        bw.triattn_bwd_q(
            q[s], k[s], v[s], sbias[s], do[s], lse[s], delta[s], scale, dbias[s], dq[s]
        )
    return dq
