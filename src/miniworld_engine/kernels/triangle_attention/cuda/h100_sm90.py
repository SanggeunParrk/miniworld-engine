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
        name=f"triattn_h100_core_r{rows}{tag}_v10",
        sources=[str(_SRC / "triattn_sm90.cu")],
        extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}", *(f"-D{d}" for d in defs)],
        verbose=True,
    )


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
        name=f"triattn_h100_mod{tag}_v8",
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


def inference(pair, lnw, lnb, w4, wb16, wo16, mask=None, eps=1e-5):
    """pair + TriangleAttention(pair) for the starting node, no grad: front (LN + projections + bias) -> core -> tail (gate +
    out-projection + residual).  w4, wb16 = pack_front(...); wo16 = Wo in bf16."""
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
        mask if mask is not None else empty.bool(),
        B,
        L,
        D**-0.5,
    )
    shp = (B, L, L, H, D)
    o, _, _, _ = _core().triattn_fwd(
        q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, False
    )
    return m.tri_tail(g, o.view(-1, C), x, wo16, empty.bfloat16(), L).view(B, L, L, C)


@functools.lru_cache(maxsize=None)
def _bwd(defs: tuple[str, ...] = ()):
    """The core backward kernels (delta, key side, query side); `defs` = extra -D knobs (experiments)."""
    ensure_cuda_home()
    tag = "".join(f"_{d.replace('=', '').lower()}" for d in defs)
    return load_extension(
        name=f"triattn_h100_bwd{tag}_v10",
        sources=[str(_SRC / "triattn_bwd_sm90.cu")],
        extra_cuda_cflags=[*_flags(), *(f"-D{d}" for d in defs)],
        verbose=True,
    )


def train_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need=True):
    """The module forward with the tensors its backward reads: -> (out [B, L, L, 128], (q, k, v, g, sbias, o, lse)); o / lse are empty
    unless `need`.  Every returned tensor is a fresh allocation (none aliases an argument), so it can be an output of a custom op."""
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    w4, wb16 = pack_front(wq, wk, wv, wg, wb)
    wo16 = wo.to(torch.bfloat16).contiguous()
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
    )
    shp = (B, L, L, H, D)
    o, lse, _, _ = _core().triattn_fwd(
        q.view(shp), k.view(shp), v.view(shp), sbias, D**-0.5, need
    )
    drop = drop if drop is not None else empty.bfloat16()
    out = m.tri_tail(g, o.view(-1, C), x, wo16, drop, L)
    return out.view(B, L, L, C), (q, k, v, g, sbias, o, lse)


def train_backward(dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, eps, drop, saved):
    """Gradients (pair, lnw, lnb, wq, wk, wv, wg, wb, wo) of `train_forward`, each in its input's dtype."""
    q, k, v, g, sbias, o, lse = saved
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m, bw = _mod(), _bwd()
    w4, _ = pack_front(wq, wk, wv, wg, wb)
    wo16 = wo.to(torch.bfloat16).contiguous()
    lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
    empty = torch.empty(0, device=pair.device)
    drop = drop if drop is not None else empty.bfloat16()
    dbias = torch.zeros(
        B, H, L, L, device=x.device, dtype=torch.float32
    )  # the query-side kernel adds its partials in
    dy = dy.contiguous().view(-1, C)
    do, dg, delta, dpair, dwo_part = m.tri_gate_bwd(
        dy, g, o.view(-1, C), wo16, drop, B, L
    )
    shp = (B, L, L, H, D)
    q5, k5, v5, do5 = q.view(shp), k.view(shp), v.view(shp), do.view(shp)
    dk, dv = bw.triattn_bwd_kv(q5, k5, v5, sbias, do5, lse, delta, D**-0.5)
    dq = bw.triattn_bwd_q(q5, k5, v5, sbias, do5, lse, delta, D**-0.5, dbias)
    dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
    m.tri_head_bwd(dq, dk, dv, dg, dbias, x, w4, wb, lw, eps, dpair, B, L)
    fp32 = wq.dtype == torch.float32
    dw, dgb, dwo = m.tri_wgrad(
        dq, dk, dv, dg, dbias, x, eps, B, L, w4, wb, lw, lb, dwo_part, fp32
    )
    # (clones: the outputs of a custom op may not alias each other, and these are slices of a few buffers)
    return (
        dpair.view(B, L, L, C),
        dgb[0].to(lnw.dtype).clone(),
        dgb[1].to(lnb.dtype).clone(),
        *(dw[t].to(w.dtype).clone() for t, w in enumerate((wq, wk, wv, wg))),
        dgb[2:].to(wb.dtype).clone(),
        dwo.to(wo.dtype).clone(),
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
