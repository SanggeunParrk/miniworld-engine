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

from miniworld_engine import settings
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
    return load_extension(name=f"triattn_b200_core_r{rows}_v8", sources=[str(_SRC / "triattn_sm100.cu")],
                          extra_cuda_cflags=[*_flags(), f"-DTA_R={rows}"], verbose=False)


@functools.lru_cache(maxsize=1)
def _mod():
    ensure_cuda_home()
    return load_extension(name="triattn_b200_mod_v7", sources=[str(_SRC / "triattn_mod_sm100.cu")],
                          extra_cuda_cflags=_flags(), verbose=False)


def supports(length: int) -> bool:
    """Lengths the kernels take: whole 128-token tiles."""
    return length > 0 and length % TOK == 0


class TriAttnCore(torch.autograd.Function):
    """out = softmax(scale q.k + bias) v on [B, N, S, H, 32] tensors, bias [B, H, S, S]."""

    @staticmethod
    def forward(ctx, q, k, v, bias, scale):
        out, lse = _core_forward(q, k, v, bias, scale, any(ctx.needs_input_grad))
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


#: The attention core's forward in launches of a few samples (B200, 2026-10-20).  Its persistent kernel walks its tasks in a static
#: round robin, and in a long launch the CTAs drift apart and stop sharing K / V rows in L2: with four pair rows a task (L > 384),
#: module inference of 48 samples at L768 took 43.9 ms in one launch and 41.6 ms in launches of ~36 864 tasks (8 samples; L640 B48
#: 26.7 -> 25.8 ms, d_pair 256 L768 B16 29.5 -> 29.0 ms; training +0.5-1.7 %).  The forward with two rows a task (L <= 384) does not
#: drift.  Measured and not kept: a dynamic task counter in the forward (each CTA claiming its next task from a global counter, the
#: H100 fix) was 4-5 % slower per sample at every launch size on B200, and the same grouping of the two backward kernels changed the
#: module's training step by -0.5 to +0.1 % (isolated, with idle gaps between launches, they had looked 6-10 % faster in groups).
#: Each launch is the same kernel on a contiguous range of samples (the same bits).
GROUP_TASKS = 36864


def fwd_group(batch: int, heads: int, length: int) -> int:
    """Samples per launch of the core's forward: ~GROUP_TASKS tasks ((sample, head, 128-query tile) x 4 pair rows) at L > 384, all
    samples at L <= 384 (two rows a task: no drift)."""
    rows = _rows_per_task(length)
    if rows != 4:
        return batch
    per = heads * (length // TOK) * ((length + rows - 1) // rows)
    return max(1, min(batch, GROUP_TASKS // per))


def _core_forward(q, k, v, bias, scale, need):
    """The core's forward over [B, N, S, H, 32] tensors in launches of `fwd_group` samples -> (out [B, N, S, H, 32], lse or empty)."""
    B, N, S, H, _ = q.shape
    e = _core(_rows_per_task(S))
    g = fwd_group(B, H, S)
    if g >= B:
        E = q.new_empty(0)                                     # (pybind: the optional outputs are passed explicitly)
        o, lse, _ = e.triattn_fwd(q, k, v, bias, scale, need, E, E)
        return o, lse
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty((B, N, H, S) if need else (0,), device=q.device, dtype=torch.float32)
    for i in range(0, B, g):
        j = min(B, i + g)
        e.triattn_fwd(q[i:j], k[i:j], v[i:j], bias[i:j], scale, need, o[i:j], lse[i:j] if need else lse)
    return o, lse


_WS_DWO, _WS_MT, _WS_VEC, _WS_DB = 0, C * C, 5 * C * C, 5 * C * C + 1056


def _empty_mask(device):
    return torch.empty(0, device=device, dtype=torch.bool)


def fused_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None, need=False, out=None, trans=False):
    """The d_pair 128 module forward (starting node, residual included) on B samples: -> (y [B, L, L, 128], saved).
    saved = (prm, w4, wo16, q, k, v, g, bias, o, lse): what `fused_backward` reads (lse empty unless `need`); every one a fresh
    allocation (none aliases an argument: they can be outputs of a custom op).  `out`: a contiguous bf16 tensor of the pair's size
    to write y into (a chunk's rows of a batch's output), or None.  Kernels: tri_pack_params -> tri_front -> triattn_fwd -> tri_tail.
    trans: the ending node -- pair, y (and the backward's dy, dx) in the caller's [B, I, J, C] layout, read and written through 4-D
    tensor maps by the kernels that touch them (the saved intermediates stay in the starting-node layout: no transposed copies)."""
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    m = _mod()
    w4, wo16, prm = m.tri_pack_params(wq.contiguous(), wk.contiguous(), wv.contiguous(), wg.contiguous(), wo.contiguous(),
                                      lnw.contiguous(), lnb.contiguous(), wb.contiguous())
    lw, lb, wbf = prm[0], prm[1], prm[2:]
    q, k, v, g, bias = m.tri_front(x, lw, lb, eps, w4, wbf, mask if mask is not None else _empty_mask(pair.device), B, L, trans)
    shp = (B, L, L, H, D)
    o, lse = _core_forward(q.view(shp), k.view(shp), v.view(shp), bias, D ** -0.5, need)
    y = m.tri_tail(g, o.view(-1, C), x, wo16, drop if drop is not None else x.new_empty(0), L,
                   out if out is not None else x.new_empty(0), trans)
    return (out if out is not None else y.view(B, L, L, C)), (prm, w4, wo16, q, k, v, g, bias, o, lse)


def split_grads(fused, raw, dtypes):
    """(pair, lnw, lnb, wq, wk, wv, wg, wb, wo) gradients from a backward's stacked outputs (views of them, cast to ``dtypes``, the
    eight parameters' dtypes, where they differ)."""
    if fused:
        dpair, dgb, dw, dwo = raw          # dgb = [dgamma; dbeta; dWb (4 rows)], dw = [dWq; dWk; dWv; dWg]
        wb = dgb[2:]
    else:
        dpair, dgb, dw, wb, dwo = raw      # dgb = [dgamma; dbeta] fp32
    return (dpair, dgb[0].to(dtypes[0]), dgb[1].to(dtypes[1]), *(dw[i].to(dtypes[2 + i]) for i in range(4)), wb.to(dtypes[6]),
            dwo.to(dtypes[7]))


def fused_backward(dy, pair, eps, drop, saved, dtypes, trans=False):
    """Stacked gradients (dpair, [dgamma; dbeta; dWb], [dWq; dWk; dWv; dWg], dWo) of `fused_forward` (`split_grads` takes them apart);
    dtypes = the eight parameters' dtypes (the weight gradients come back in fp32 when Wq is fp32, else bf16).
    Kernels: tri_gate_bwd -> triattn_bwd (biasT, key side, query side) -> tri_head_bwd -> tri_wgrad (+ slot sum, finish)."""
    prm, w4, wo16, q, k, v, g, bias, o, lse = saved
    B, L, _, _ = pair.shape
    x = pair.contiguous().view(-1, C)
    lw, lb, wbf = prm[0], prm[1], prm[2:]
    m, e = _mod(), _core(_rows_per_task(L))
    det = bool(settings.current().triattn_b200_deterministic_wgrad)
    if det:
        # dbias: the query-side kernel reduce-adds into zeros; the parameter gradients' partials go to per-CTA slots (no zeroing),
        # summed in CTA order (tri_slot_sum)
        dbws = torch.zeros(B * H * L * L, device=x.device, dtype=torch.float32)
        E32 = x.new_empty(0, dtype=torch.float32)
        wdwo = wmt = E32
    else:
        # one zeroed fp32 workspace for every gradient accumulator (one memset): [dWo C*C | wgrad M 4*C*C | wgrad vectors 8*C + 4,
        # padded | dbias B*4*L*L]; each piece starts on a 128-byte boundary
        ws = torch.zeros(_WS_DB + B * H * L * L, device=x.device, dtype=torch.float32)
        dbws, wdwo, wmt = ws[_WS_DB:], ws[_WS_DWO:_WS_MT], ws[_WS_MT:_WS_DB]
    dy = dy.contiguous().view(-1, C)
    drop = drop if drop is not None else x.new_empty(0)
    do, dg, delta, dwo, dpair = m.tri_gate_bwd(dy, g, o.view(-1, C), wo16, B, L, wdwo, drop, det, trans)
    shp = (B, L, L, H, D)
    E = q.new_empty(0)
    dq, dk, dv, db = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, D ** -0.5, dbws, E, E, E)
    dq, dk, dv = dq.view(-1, C), dk.view(-1, C), dv.view(-1, C)
    m.tri_head_bwd(dq, dk, dv, dg, db, x, w4, wbf, lw, lb, eps, dpair, B, L, trans)
    dt = dtypes
    fp32 = dt[2] == torch.float32
    dw, dgb, dwo = m.tri_wgrad(dq, dk, dv, dg, db, x, eps, B, L, w4, wbf, lw, lb, True, dwo, wmt, fp32, det, trans)
    return dpair.view(B, L, L, C), dgb, dw, dwo


class TriAttnFused(torch.autograd.Function):
    """The whole module (residual included): pair [B, L, L, 128] -> pair + TriangleAttention(pair).
    Parameters may be bf16 or fp32 (master weights): one pack kernel stages them, and the gradients come back in their dtype.
    drop: the module's broadcast dropout scale as [B, L, 128] bf16 (index = the row's last position), or None.
    (The module path goes through the custom ops of ``integrations.triattn_b200``, which call `fused_forward` / `fused_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop):
        need = any(ctx.needs_input_grad)
        out, saved = fused_forward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, need)
        if need:
            ctx.save_for_backward(pair, drop if drop is not None else pair.new_empty(0), *saved)
            ctx.eps = eps
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, drop, *saved = ctx.saved_tensors
        raw = fused_backward(dy, pair, ctx.eps, drop if drop.numel() else None, saved, ctx.dtypes)
        return (*split_grads(True, raw, ctx.dtypes), None, None, None)


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
    return load_extension(name="triattn_b200_wide_v33", sources=[str(_SRC / "triattn_wide_sm100.cu")],
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


def wide_inference(pair, lnw, lnb, packed, n_head, head_dim, mask=None, eps=1e-5, out=None, trans=False):
    """pair + TriangleAttention(pair) for the starting node, no grad; packed = pack_wide_weights(...).
    Three launches: tri_wfront (LN + the five projections, head-major masked bias) -> triattn_fwd -> tri_wtail
    (sigmoid gate + out-projection + residual).  `out`: a contiguous bf16 tensor like the pair to write into, or None."""
    y, _ = wide_forward(pair, lnw, lnb, packed, n_head, head_dim, mask, eps, None, False, out, trans)
    return y


def wide_forward(pair, lnw, lnb, packed, n_head, head_dim, mask=None, eps=1e-5, drop=None, need=False, out=None, trans=False):
    """-> (y [B, L, L, C], saved = (q, k, v, g, bias, o, lse, stats)): lse / stats empty unless `need`; every saved tensor is a fresh
    allocation (none aliases an argument)."""
    B, L, _, C = pair.shape
    wcat, wo_p = packed[:2]
    HDp = n_head * D
    w = _wide()
    x = pair.contiguous().view(-1, C)
    stats = torch.empty(x.shape[0], 2, device=x.device, dtype=torch.float32) if need else x.new_empty(0, dtype=torch.float32)
    q, k, v, g, bias = w.tri_wfront(x, lnw, lnb, eps, wcat, n_head, mask if mask is not None else _empty_mask(pair.device), B, L, stats, trans)
    shp = (B, L, L, n_head, D)
    o, lse = _core_forward(q.view(shp), k.view(shp), v.view(shp), bias, head_dim ** -0.5, need)
    y = w.tri_wtail(g, o.view(-1, HDp), x, wo_p, drop if drop is not None else x.new_empty(0), L,
                    out if out is not None else x.new_empty(0), trans)
    return (out if out is not None else y.view(B, L, L, C)), (q, k, v, g, bias, o, lse, stats)


def _mm32(a, b):
    """a @ b on cuBLAS with fp32 output (bf16 operands, fp32 accumulation)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:                                            # a torch without mm(out_dtype=)
        return torch.mm(a, b).float()


def wide_backward(dy, pair, lnw, lnb, packed, n_head, head_dim, eps, drop, saved, dtypes, trans=False):
    """Stacked gradients (dpair, [dgamma; dbeta] fp32, [dWq; dWk; dWv; dWg], dWb, dWo) of `wide_forward` (`split_grads` takes them
    apart; the weight gradients in fp32 when Wq is fp32, else bf16)."""
    q, k, v, g, bias, o, lse, stats = saved
    B, L, _, C = pair.shape
    x = pair.contiguous().view(-1, C)
    wcat, _, wo_t, wall_t = packed
    HDp = n_head * D
    R = B * L * L
    w, e = _wide(), _core(_rows_per_task(L))
    dy = dy.contiguous().view(-1, C)
    dys = w.tri_scale_rows(dy, drop, L, trans) if drop is not None and drop.numel() else dy   # (the caller's layout, as u below)
    Dall = x.new_empty(R, 4 * HDp + 64)                       # [dq | dk | dv | dg | db rows], written in place by their producers
    cut = lambda i, n=HDp: Dall[:, i * HDp:i * HDp + n]
    do, _, u, delta = w.tri_wgbwd(dys, g, o.view(-1, HDp), wo_t, cut(3), B, L, trans)
    shp = (B, L, L, n_head, D)
    _, _, _, db = e.triattn_bwd(q.view(shp), k.view(shp), v.view(shp), bias, do.view(shp), lse, delta, head_dim ** -0.5,
                                q.new_empty(0, dtype=torch.float32), cut(0), cut(1), cut(2))
    w.tri_db_rows(db, cut(4, 64), B, L)
    dpair, xh = w.tri_whbwd(Dall, x, dy, lnw, stats, wall_t, L, trans)
    # parameter gradients: G = D^T [xhat | 1] (one GEMM, K = L^2), then dW / dgamma / dbeta / dWb by the d_pair 128 algebra
    # (y = gamma o xhat + beta feeds every projection) and dWo without the padded channels, in tri_wfinish
    G = _mm32(Dall.t(), xh)
    dwo = _mm32(dys.t(), u)
    dt = dtypes
    fp32 = dt[2] == torch.float32
    dw, dwb, dwo, dgb = w.tri_wfinish(G, wcat, lnw, lnb, dwo, n_head, head_dim, fp32)
    return dpair.view(B, L, L, C), dgb, dw, dwb, dwo


class WideTrain(torch.autograd.Function):
    """The wide path with a backward: pair [B, L, L, C] -> pair + drop o TriangleAttention(pair) (starting node).
    packed = pack_wide_weights(...) of the same parameters (bf16 staging; the gradients come back in each parameter's dtype).
    (The module path goes through the custom ops of ``integrations.triattn_b200``, which call `wide_forward` / `wide_backward`.)"""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed):
        need = any(ctx.needs_input_grad)
        out, saved = wide_forward(pair, lnw, lnb, packed, n_head, head_dim, mask, eps, drop, need)
        if need:
            ctx.save_for_backward(pair, lnw, lnb, drop if drop is not None else pair.new_empty(0), *saved)
            ctx.packed = packed
            ctx.dims = (n_head, head_dim, eps)
            ctx.dtypes = tuple(t.dtype for t in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return out

    @staticmethod
    def backward(ctx, dy):
        pair, lnw, lnb, drop, *saved = ctx.saved_tensors
        n_head, head_dim, eps = ctx.dims
        raw = wide_backward(dy, pair, lnw, lnb, ctx.packed, n_head, head_dim, eps, drop if drop.numel() else None, saved, ctx.dtypes)
        return (*split_grads(False, raw, ctx.dtypes), None, None, None, None, None, None)


def wide_train(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, packed, n_head, head_dim, mask=None, eps=1e-5, drop=None):
    """pair + drop o TriangleAttention(pair) for the starting node with a backward (drop: [B, L, C] bf16 or None)."""
    return WideTrain.apply(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, n_head, head_dim, packed)
