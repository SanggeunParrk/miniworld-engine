"""B200 (sm_100a) TriangleAttention, starting node, d_pair = 128, 4 heads x 32, bf16, optional broadcast dropout.

Forward:  tri_front (LN + q|k|v|g projections + head-major masked bias) -> triattn_fwd (attention + base-2 LSE)
          -> tri_tail (sigmoid gate + to_out + dropout scale + residual).
Backward: tri_gate_bwd (do, dg, delta, dWo) -> triattn_bwd (KV-owned dK/dV + query-owned dQ/dbias)
          -> tri_head_bwd (projection dgrad + LN backward + residual) -> tri_wgrad (parameter gradients on tcgen05).
Kernel bodies are tcgen05 / TMEM / TMA hand-CUDA in ``b200_sources/``, from the B200 research branch ``b200/msa``
(2026-09-27). Every call owns its buffers; nothing is cached across calls except the compiled extensions.
"""

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_SRC = Path(__file__).with_name("b200_sources")

C = 128          # d_pair
H = 4            # heads
D = C // H       # head dim (the kernels take 32 only)
TOK = 128        # query / key / token tile of every kernel; L must be a multiple of it


def _flags():
    return [*host_flags(), "-std=c++17", "-O3", *gencodes("100a"), "--use_fast_math", "-DNDEBUG", f"-I{_SRC}"]


# Per-length schedule of the attention core, from a sweep of every registered L (2026-09-29, triattn.md "Schedule per
# length"): two pair rows per forward task up to L = 384 (more tasks for the 148 SMs; -15 % at L256), four above.
# The backward kernels are the same in both builds (query-side QR = 4 was best at every L).
def _rows_per_task(length: int) -> int:
    return 2 if length <= 384 else 4


@functools.lru_cache(maxsize=None)
def _core(rows: int = 4):
    ensure_cuda_home()
    return load_extension(name=f"triattn_b200_core_r{rows}_v5", sources=[str(_SRC / "triattn_sm100.cu")],
                          extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}"], verbose=False)


@functools.lru_cache(maxsize=1)
def _mod():
    ensure_cuda_home()
    return load_extension(name="triattn_b200_mod_v3", sources=[str(_SRC / "triattn_mod_sm100.cu")],
                          extra_cuda_cflags=_flags(), verbose=False)


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


class TriAttnCore(torch.autograd.Function):
    """out = softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S]."""

    @staticmethod
    def forward(ctx, q, k, v, bias, scale):
        out, lse, _ = _core(_rows_per_task(q.shape[2])).triattn_fwd(q, k, v, bias, scale, any(ctx.needs_input_grad))
        ctx.save_for_backward(q, k, v, bias, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, bias, out, lse = ctx.saved_tensors
        e = _core(_rows_per_task(q.shape[2]))
        dout = dout.contiguous()
        delta = e.triattn_delta(dout, out)
        E = q.new_empty(0)
        dq, dk, dv, db = e.triattn_bwd(q, k, v, bias, dout, lse, delta, ctx.scale, q.new_empty(0, dtype=torch.float32), E, E, E)
        return dq, dk, dv, db.to(bias.dtype), None


# One zeroed fp32 workspace for every gradient accumulator of the backward (one memset instead of four):
# [dWo C*C | wgrad M 4*C*C | wgrad vectors 8*C + 4, padded | dbias B*4*L*L]; each piece starts on a 128-byte boundary.
_WS_DWO, _WS_MT, _WS_VEC, _WS_DB = 0, C * C, 5 * C * C, 5 * C * C + 1056


class TriAttnFused(torch.autograd.Function):
    """The whole module (residual included): pair [B, L, L, 128] -> pair + TriangleAttention(pair).
    Parameters may be bf16 or fp32 (master weights): one pack kernel stages them, and the gradients come back in their dtype.
    drop: the module's broadcast dropout scale as [B, L, 128] bf16 (index = the row's last position), or None."""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop):
        B, L, _, _ = pair.shape
        x = pair.contiguous().view(-1, C)
        m = _mod()
        w4, wo16, prm = m.tri_pack_params(wq.contiguous(), wk.contiguous(), wv.contiguous(), wg.contiguous(), wo.contiguous(),
                                          lnw.contiguous(), lnb.contiguous(), wb.contiguous())
        lw, lb, wbf = prm[0], prm[1], prm[2:]
        q, k, v, g, bias = m.tri_front(x, lw, lb, eps, w4, wbf,
                                       mask if mask is not None else torch.empty(0, device=pair.device, dtype=torch.bool), B, L)
        shp = (B, L, L, H, D)
        need = any(ctx.needs_input_grad)
        o, lse, _ = _core(_rows_per_task(L)).triattn_fwd(q.view(shp), k.view(shp), v.view(shp), bias, D ** -0.5, need)
        drop = drop if drop is not None else x.new_empty(0)
        out = m.tri_tail(g, o.view(-1, C), x, wo16, drop, L)
        if need:
            ctx.save_for_backward(x, prm, w4, wo16, q, k, v, g, bias, o, lse, drop)
            ctx.dims = (B, L, eps)
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out.view(B, L, L, C)

    @staticmethod
    def backward(ctx, dy):
        x, prm, w4, wo16, q, k, v, g, bias, o, lse, drop = ctx.saved_tensors
        B, L, eps = ctx.dims
        dt = ctx.dtypes
        lw, lb, wbf = prm[0], prm[1], prm[2:]
        m, e = _mod(), _core(_rows_per_task(L))
        ws = torch.zeros(_WS_DB + B * H * L * L, device=x.device, dtype=torch.float32)
        dy = dy.contiguous().view(-1, C)
        do, dg, delta, dwo, dpair = m.tri_gate_bwd(dy, g, o.view(-1, C), wo16, B, L, ws[_WS_DWO:_WS_MT], drop)
        shp = (B, L, L, H, D)
        E = q.new_empty(0)
        dq, dk, dv, db = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, D ** -0.5, ws[_WS_DB:], E, E, E)
        dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
        m.tri_head_bwd(dq, dk, dv, dg, db, x, w4, wbf, lw, lb, eps, dpair, B, L)
        fp32 = dt[2] == torch.float32
        dw, dgb, dwo = m.tri_wgrad(dq, dk, dv, dg, db, x, eps, B, L, w4, wbf, lw, lb, True, dwo, ws[_WS_MT:_WS_DB], fp32)
        return (dpair.view(B, L, L, C), dgb[0].to(dt[0]), dgb[1].to(dt[1]), *(dw[t].to(dt[2 + t]) for t in range(4)),
                dgb[2:].to(dt[6]), dwo.to(dt[7]), None, None, None)


def triangle_attention(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node; the caller transposes for the ending node."""
    return TriAttnFused.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop)


# ------------------------------------------------------------------------------------------------------------------------------
# Other widths: d_pair C in 64 .. 512 (a multiple of 64), heads of 16 or 32 channels. Forward: tri_wfront (LN + the five
# projections as 2-CTA products, weights streamed in 64-column chunks; row statistics kept when training) -> the head-dim-32
# core above (a head dim of 16 is zero-padded to 32) -> tri_wtail (sigmoid gate + out-projection + dropout + residual).
# Backward (WideTrain): tri_scale_rows (dropout) -> tri_wgbwd (du = dy_s . Wo on tcgen05, the gate math and delta in its
# drain) -> the core's backward -> tri_whbwd ([dq | dk | dv | dg | db] . W on tcgen05, the LN backward and the residual in its
# drain, xhat | 1 written) -> the weight gradients from G = D^T [xhat | 1] (one cuBLAS GEMM, K = L^2) by the algebra of the
# d_pair 128 path, finished by tri_wfinish. All hand kernels in triattn_wide_sm100.cu.
WIDE_HEAD_DIMS = (16, 32)
WIDE_MAX_HEADS = 16                   # the packed bias block is 16 rows


@functools.lru_cache(maxsize=1)
def _wide():
    ensure_cuda_home()
    return load_extension(name="triattn_b200_wide_v31", sources=[str(_SRC / "triattn_wide_sm100.cu")],
                          extra_cuda_cflags=_flags(), verbose=False)


def wide_supports(d_pair: int, d_hidden: int, n_head: int, length: int) -> bool:
    """Shapes the wide path takes (inference and training)."""
    if d_pair % 64 or not 64 <= d_pair <= 512 or d_hidden % n_head or not supports(length):
        return False
    return d_hidden // n_head in WIDE_HEAD_DIMS and n_head <= WIDE_MAX_HEADS


def pack_wide_weights(wq, wk, wv, wg, wb, wo):
    """-> (wcat, wo_p, wo_t, wall_t), HDp = heads x 32 (a 16-channel head is zero-padded to 32: its padded q / k / v / g
    channels come out 0 and meet zero columns of Wo):
      wcat   [Wq; Wk; Wv; Wg; Wb (16 rows)]  [4 HDp + 16, C] bf16   the front's B
      wo_p   Wo                               [C, HDp] bf16          the tail's B
      wo_t   Wo^T                             [HDp, C] bf16          the gate backward's B
      wall_t [Wq; Wk; Wv; Wg; Wb (64 rows)]^T [C, 4 HDp + 64] bf16   the head backward's B"""
    n_head = wb.shape[0]
    hd = wq.shape[0] // n_head
    C = wq.shape[1]
    bf = torch.bfloat16

    def pad_rows(w):
        w = w.to(bf)
        if hd == D:
            return w
        out = w.new_zeros(n_head, D, C)
        out[:, :hd] = w.view(n_head, hd, C)
        return out.view(n_head * D, C)

    wb16 = wb.new_zeros(16, C, dtype=bf)
    wb16[:n_head] = wb.to(bf)
    rows = [pad_rows(w) for w in (wq, wk, wv, wg)]
    wcat = torch.cat(rows + [wb16]).contiguous()
    wb64 = wb.new_zeros(64, C, dtype=bf)
    wb64[:n_head] = wb.to(bf)
    wall_t = torch.cat(rows + [wb64]).t().contiguous()
    if hd == D:
        wo_p = wo.to(bf).contiguous()
    else:
        wo_p = wo.new_zeros(C, n_head, D, dtype=bf)
        wo_p[:, :, :hd] = wo.to(bf).view(C, n_head, hd)
        wo_p = wo_p.view(C, n_head * D)
    return wcat, wo_p, wo_p.t().contiguous(), wall_t


def wide_inference(pair, lnw, lnb, packed, n_head, head_dim, mask=None, eps=1e-5):
    """pair + TriangleAttention(pair) for the starting node, no grad; packed = pack_wide_weights(...).
    Three launches: tri_wfront (LN + the five projections, head-major masked bias) -> triattn_fwd -> tri_wtail
    (sigmoid gate + out-projection + residual)."""
    B, L, _, C = pair.shape
    wcat, wo_p = packed[:2]
    HDp = n_head * D
    w = _wide()
    x = pair.contiguous().view(-1, C)
    q, k, v, g, bias = w.tri_wfront(x, lnw, lnb, eps, wcat, n_head,
                                    mask if mask is not None else torch.empty(0, device=pair.device, dtype=torch.bool), B, L,
                                    x.new_empty(0, dtype=torch.float32))
    shp = (B, L, L, n_head, D)
    o, _, _ = _core(_rows_per_task(L)).triattn_fwd(q.view(shp), k.view(shp), v.view(shp), bias, head_dim ** -0.5, False)
    return w.tri_wtail(g, o.view(-1, HDp), x, wo_p, x.new_empty(0), L).view(B, L, L, C)


def _mm32(a, b):
    """a @ b on cuBLAS with fp32 output (bf16 operands, fp32 accumulation)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:                                            # a torch without mm(out_dtype=)
        return torch.mm(a, b).float()


class WideTrain(torch.autograd.Function):
    """The wide path with a backward: pair [B, L, L, C] -> pair + drop o TriangleAttention(pair) (starting node).
    packed = pack_wide_weights(...) of the same parameters (bf16 staging; the gradients come back in each parameter's dtype)."""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed):
        B, L, _, C = pair.shape
        wcat, wo_p = packed[:2]
        HDp = n_head * D
        w = _wide()
        x = pair.contiguous().view(-1, C)
        need = any(ctx.needs_input_grad)
        stats = torch.empty(x.shape[0], 2, device=x.device, dtype=torch.float32) if need else x.new_empty(0, dtype=torch.float32)
        q, k, v, g, bias = w.tri_wfront(x, lnw, lnb, eps, wcat, n_head,
                                        mask if mask is not None else torch.empty(0, device=pair.device, dtype=torch.bool), B, L, stats)
        shp = (B, L, L, n_head, D)
        o, lse, _ = _core(_rows_per_task(L)).triattn_fwd(q.view(shp), k.view(shp), v.view(shp), bias, head_dim ** -0.5, need)
        drop = drop if drop is not None else x.new_empty(0)
        out = w.tri_wtail(g, o.view(-1, HDp), x, wo_p, drop, L)
        if need:
            ctx.save_for_backward(x, q, k, v, g, bias, o, lse, drop, lnw, lnb, stats)
            ctx.packed = packed
            ctx.dims = (B, L, C, n_head, head_dim, eps)
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out.view(B, L, L, C)

    @staticmethod
    def backward(ctx, dy):
        x, q, k, v, g, bias, o, lse, drop, lnw, lnb, stats = ctx.saved_tensors
        B, L, C, n_head, head_dim, eps = ctx.dims
        wcat, _, wo_t, wall_t = ctx.packed
        HDp = n_head * D
        R = B * L * L
        w, e = _wide(), _core(_rows_per_task(L))
        dy = dy.contiguous().view(-1, C)
        dys = w.tri_scale_rows(dy, drop, L) if drop.numel() else dy
        Dall = x.new_empty(R, 4 * HDp + 64)                       # [dq | dk | dv | dg | db rows], written in place by their producers
        cut = lambda i, n=HDp: Dall[:, i * HDp:i * HDp + n]
        do, _, u, delta = w.tri_wgbwd(dys, g, o.view(-1, HDp), wo_t, cut(3), B, L)
        shp = (B, L, L, n_head, D)
        _, _, _, db = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim ** -0.5,
                                    q.new_empty(0, dtype=torch.float32), cut(0), cut(1), cut(2))
        w.tri_db_rows(db, cut(4, 64), B, L)
        dpair, xh = w.tri_whbwd(Dall, x, dy, lnw, stats, wall_t)
        # parameter gradients: G = D^T [xhat | 1] (one GEMM, K = L^2), then dW / dgamma / dbeta / dWb by the d_pair 128 algebra
        # (y = gamma o xhat + beta feeds every projection) and dWo without the padded channels, in tri_wfinish
        G = _mm32(Dall.t(), xh)
        dwo = _mm32(dys.t(), u)
        dt = ctx.dtypes
        fp32 = dt[2] == torch.float32
        dw, dwb, dwo, dgb = w.tri_wfinish(G, wcat, lnw, lnb, dwo, n_head, head_dim, fp32)
        return (dpair.view(B, L, L, C), dgb[0].to(dt[0]), dgb[1].to(dt[1]), *(dw[i].to(dt[2 + i]) for i in range(4)), dwb.to(dt[6]),
                dwo.to(dt[7]), None, None, None, None, None, None)


def wide_train(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, packed, n_head, head_dim, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node with a backward (drop: [B, L, C] bf16 or None)."""
    return WideTrain.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed)

