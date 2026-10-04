"""B200 (sm_100a) TriMul training for every width D64-D512, either direction, bf16, B=1.

D64 / D128 (``forward_small`` / ``backward_small``; L a multiple of 128): forward = the inference front (k1w,
cuBLAS contractions) and k3g's saving epilogue (x_n, LN_out statistics); backward = two fused kernels around the contraction
gradients:
  b1s / b1g   output side: gate / projection gradients, LN_out backward -> dG, dt; dWg, dWp, dg_o, db_o (front CTAs hand dP
              to LN-backward CTAs through an L2 ring). b1s: D64 (H = 64 / 128); b1g: D128 (H = 256 bidirectional / 128).
  cuBLAS      contraction gradients -> plane gradient da [P, L, L]
  b7m         D64 input side, one CTA per 128-token tile: pre-activations recomputed, gate gradients,
              dxn = [dpre | dG] [W1 ; Wg] (packed W1 resident in shared memory, dW1 in TMEM), LN_in backward + residual -> dx;
              dW1, dg_i, db_i
  b7g         D128 input side, over P / 64 plane chunks (8 bidirectional, 4 one direction) per source group (the packed
              W1, 128 / 256 KB, does not fit next to b7m's buffers)

D256 / 384 / 512:

Forward = the inference path (``b200_infer``: k1w, cuBLAS contractions, k3w) with k3w also storing p (output projection) and g
(gate logit), and the planes, the contraction output t and both LayerNorms' statistics kept for the backward.

Backward:
  gate_bwd    dp = dy ds sigmoid(g), dg = dp p (1 - sigmoid(g)) -> dpr = dp rs_o, dg; per token the LN_out backward's row sums
              S1 = dp . sp, S2 = dp . (p - ep) (fold identities: no pass over the H channels); per column r0, r1 for dWp
  cuBLAS      dor^T = Wp^T dpr^T = rs_o do^T              [H, M], channel-major like t
  lnout_bwd   dt = g_o dor - rs_o (S1 + xhat S2) / H, elementwise over [H, M]; dg_o, db_o
  cuBLAS      dWp = g_o (dpr^T t^T - r1) + b_o r0          (t read as stored: the LN_out fold again)
  cuBLAS      contraction gradients -> plane gradient da [P, L, L]
  k1wb        recomputes k1w's pre-activations and writes dpre = (dg_c, dp_c) token-major in the packed-row order
  cuBLAS      [dW1 ; dWg] = [dpre | dg]^T xn,  dxn = [dpre | dg] [W1 ; Wg]
  lnin_bwd    dx = dy + LN_in backward of dxn; dg_i, db_i
Sources: ``b200_sources/{k1w,k3g,b1s,b7m,b1g,b7g}.cu`` (D64 / D128), ``b200_sources/{k1w,k3w,wide_aux,k1wb,wide_bwd}.cu``.
Every call owns its buffers.
"""

import torch
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque

WIDTHS = (64, 128, 256, 384, 512)
EPS = 1e-5
TOK = 128
B1_RSF = 4       # b1s / b1g ring slots per front CTA
B7_RD = 12       # b7g ring slots per source group


def _ext():
    """The one B200 TriMul extension (shared with inference; see ``b200_sources/bind_b200.cpp``)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import _ext as ext

    return ext()


def supports(width: int, length: int, direction: int, hidden: int | None = None) -> bool:
    """direction 0 = bidirectional, 1 / 2 = one direction. k1w / k1wb / k3w tile 128 tokens and lnout_bwd 256, so L is a
    multiple of 16; the D64 / D128 path's k3g save and b1s / b1g group tiles by column (L a multiple of 128, at most 80 columns
    of tiles for the front CTAs). hidden: the contraction width (``b200_infer.hidden_ok``)."""
    if not hidden_ok(width, width if hidden is None else hidden, direction):
        return False
    if width <= 128:
        return width in WIDTHS and 0 < length <= 80 * TOK and length % TOK == 0
    return width in WIDTHS and length > 0 and length % 16 == 0


def _front_ctas(length: int, width: int, direction: int, planes: int | None = None) -> int:
    """b1s / b1g front CTAs: the largest multiple of L/128 <= 80 (D64, 4 D planes) / 96 (D64, 2 D planes; D128, 4 D planes) /
    112 (D128, 2 D planes), measured on L128-L768 -- per plane count, so a one-direction hidden-2 D block takes the
    bidirectional value (the kernels' per-token work is the same)."""
    rows = length // TOK
    planes = (4 if direction == 0 else 2) * width if planes is None else planes
    cap = {(64, True): 80, (64, False): 96, (128, True): 96, (128, False): 112}[width, planes == 4 * width]
    return (cap // rows) * rows


def _b7_source_groups(length: int, nch: int) -> int:
    """b7g source groups of nch source CTAs each (the rest of the SMs consume): 18 x 4 one direction; 11 / 12 x 8
    bidirectional (L <= 384 / above), measured on L128-L768."""
    if nch == 4:
        return 18
    return 11 if length <= 384 else 12


def _pack_w1(wl, wlg, wr, wrg):
    p = wl.shape[0] + wr.shape[0]
    g = torch.cat((wlg, wrg))
    v = torch.cat((wl, wr))
    return torch.stack((g.reshape(p // 64, 64, -1), v.reshape(p // 64, 64, -1)), 1).reshape(2 * p, -1)


def _unpack_w1(d, p):
    c = d.reshape(p // 64, 2, 64, -1)
    g = c[:, 0].reshape(p, -1)
    v = c[:, 1].reshape(p, -1)
    h = p // 2
    return v[:h], g[:h], v[h:], g[h:]          # dWl, dWlg, dWr, dWrg


from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import _contract, _fold, contracted, hidden_ok, planes_of


def _contract_bwd(planes, dt, d, direction):
    """Outgoing t = a b^T: da = dt b, db = dt^T a; incoming t = a^T b: da = b dt^T, db = a dt. Planes [P, (B,) n, n]; the samples fold
    into cuBLAS's batch dim as in ``_contract`` (whose note on the ``out=`` buffer applies here too)."""
    n = planes.shape[-1]
    samples = planes.numel() // (planes.shape[0] * n * n)
    dpl = torch.empty((planes.shape[0] * samples, n, n), dtype=planes.dtype, device=planes.device)
    f = _fold
    h, h2, h3 = d * samples, 2 * d * samples, 3 * d * samples
    if direction == 0:
        a_o, a_i, b_o, b_i = f(planes[:d]), f(planes[d:2 * d]), f(planes[2 * d:3 * d]), f(planes[3 * d:])
        dt_o, dt_i = f(dt[:d]), f(dt[d:])
        torch.bmm(dt_o, b_o, out=dpl[:h])
        torch.bmm(dt_o.transpose(1, 2), a_o, out=dpl[h2:h3])
        torch.bmm(b_i, dt_i.transpose(1, 2), out=dpl[h:h2])
        torch.bmm(a_i, dt_i, out=dpl[h3:])
    elif direction == 1:
        torch.bmm(f(dt), f(planes[d:]), out=dpl[:h])
        torch.bmm(f(dt).transpose(1, 2), f(planes[:d]), out=dpl[h:])
    else:
        torch.bmm(f(planes[d:]), f(dt).transpose(1, 2), out=dpl[:h])
        torch.bmm(f(planes[:d]), f(dt), out=dpl[h:])
    return dpl.view(planes.shape)


def _forward_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, mean / rstd (LN_in), mean / rstd (LN_out), k3w vectors, p, g."""
    x = leaves[0]
    n, d = x.shape[1], x.shape[-1]
    m, p = n * n, (4 if direction == 0 else 2) * d
    f32 = torch.float32
    return [torch.empty_like(x), x.new_empty((p, n, n)), x.new_empty((p // 2, n, n)), x.new_empty((m,), dtype=f32),
            x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32),
            x.new_empty((4, d), dtype=f32), x.new_empty((m, d)), x.new_empty((m, d))]


@opaque(fake=_forward_fake, name="trimul_b200_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int) -> list[torch.Tensor]:
    """leaves = x, wl, wlg, wr, wrg, wg, wp (bf16, [out, in]), gi, bi, go, bo (fp32); mask [L] bool token mask or None;
    ds [L, D] bf16 row-dropout scale or None; direction 0 = bidirectional, 1 = outgoing, 2 = incoming."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    n, d = x.shape[1], x.shape[-1]
    m = n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        mean = x.new_empty((m,), dtype=torch.float32)
        rstd = torch.empty_like(mean)
        E.k1w_stats(x2, mean, rstd, EPS)
        planes = x.new_empty(((4 if direction == 0 else 2) * d, n, n))
        E.k1w_forward(x2, wl, wlg, wr, wrg, mask, mean, rstd, gi, bi, planes, 0, EPS)
        t = _contract(planes, d, direction)
        mo = torch.empty_like(mean)
        ro = torch.empty_like(mean)
        E.wide_ln_stats(t, mo, ro, EPS)
        wpq = torch.empty_like(wp)
        wgq = torch.empty_like(wg)
        vec = x.new_empty((4, d), dtype=torch.float32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        y = torch.empty_like(x)
        pv = torch.empty_like(x2)
        gv = torch.empty_like(x2)
        E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, ds, y.view(m, d), n, 0, pv, gv)
    return [y, planes, t, mean, rstd, mo, ro, vec, pv, gv]


def _backward_fake(leaves, mask, ds, direction, saved, dy):
    """One gradient per leaf, contiguous: dx in x's dtype, the parameter gradients fp32 (the kernels' accumulators)."""
    return [torch.empty_like(leaves[0]), *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves[1:])]


@opaque(fake=_backward_fake, name="trimul_b200_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int,
             saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """gate_bwd -> do GEMM -> lnout_bwd -> dWp -> contraction gradients -> k1wb -> weight / input GEMMs -> lnin_bwd."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    planes, t, mean, rstd, mo, ro, vec, pv, gv = saved
    n, d = x.shape[1], x.shape[-1]
    m = n * n
    p = planes.shape[0]
    h = p // 2
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        dpr = torch.empty_like(x2)                                  # dp rs_o (dp itself is never needed)
        dcat = x.new_empty((m, 2 * p + d))                          # [dpre | dg]
        s1 = torch.empty_like(mean)
        s2 = torch.empty_like(mean)
        acc = x.new_empty((4 * d + 2 * h,), dtype=torch.float32)    # r0, r1, dg_i, db_i, dg_o, db_o
        r0, r1, dgi, dbi = acc[:d], acc[d:2 * d], acc[2 * d:3 * d], acc[3 * d:4 * d]
        dgo, dbo = acc[4 * d:4 * d + h], acc[4 * d + h:]
        E.wide_gate_bwd(dy2, pv, gv, ds, vec, mo, ro, dpr, dcat[:, 2 * p:], s1, s2, acc[:2 * d], n)
        dt = torch.empty_like(t)
        # (dpr Wp)^T = rs_o do^T, channel-major like t and dt, so lnout_bwd transposes nothing
        E.wide_lnout_bwd(torch.mm(wp.t(), dpr.t()), t, mo, ro, s1, s2, go, dt, acc[4 * d:])
        dwp = go[None, :] * (torch.mm(dpr.t(), t.view(h, m).t(), out_dtype=torch.float32) - r1[:, None]) + r0[:, None] * bo[None, :]
        dpl = _contract_bwd(planes, dt, d, direction)
        del dt
        wl_, wlg_, wr_, wrg_ = (w if w.stride(1) == 1 else w.contiguous() for w in (wl, wlg, wr, wrg))
        E.k1wb_forward(x2, wl_, wlg_, wr_, wrg_, mask, mean, rstd, gi, bi, dpl, dcat, 0)
        del dpl
        xn = torch.empty_like(x2)
        E.wide_ln_apply(x2, mean, rstd, gi, bi, xn)
        dw = torch.mm(dcat.t(), xn, out_dtype=torch.float32)        # [2P + D, D]
        dwl, dwlg, dwr, dwrg = _unpack_w1(dw[:2 * p], p)
        dxn = dcat @ torch.cat((_pack_w1(wl, wlg, wr, wrg), wg))
        dx = torch.empty_like(x)
        E.wide_lnin_bwd(dxn, x2, dy2, mean, rstd, gi, dx.view(m, d), acc[2 * d:4 * d])
    grads = [dwl, dwlg, dwr, dwrg, dw[2 * p:], dwp, dgi, dbi, dgo, dbo]
    # the small gradients are views of shared buffers; a custom op may not return aliasing outputs. fp32: the parameters may be
    # an fp32 master (the autograd function casts them to bf16 for the kernels and hands these back in the parameters' dtype)
    return [dx, *(g.to(torch.float32, copy=True) for g in grads)]


def _forward_small_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, x_n, LN_out mean / rstd."""
    x = leaves[0]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m, p = bsz * n * n, planes_of(d, leaves[1].shape[0], direction)
    f32 = torch.float32
    return [torch.empty_like(x), x.new_empty((p, bsz, n, n)), x.new_empty((p // 2, bsz, n, n)), x.new_empty((m, d)),
            x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32)]


@opaque(fake=_forward_small_fake, name="trimul_b200_train_small_fwd")
def forward_small(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int) -> list[torch.Tensor]:
    """D64 / D128 training forward; arguments as ``forward`` except ds, which is required ([B * L, D] bf16, ones without dropout), and
    x [B, L, L, D] with the samples b-major (B > 1 for D64 only: the D128 backward kernels take one sample); mask [B * L]. One
    direction, the hidden width (wl's rows) may be 2 D (``b200_infer.hidden_ok``)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import _front

    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        hid = wl.shape[0]
        p = planes_of(d, hid, direction)
        planes = x.new_empty((p, bsz, n, n))
        wpp = x.new_empty((d, p // 2))     # k3g's output projection, columns in its TMEM K order
        E.k1w_forward(x2, *_front(E, wl, wlg, wr, wrg, wp, wpp), mask, None, None, gi, bi, planes, 0, EPS)
        t = _contract(planes, contracted(d, hid, direction), direction)
        y = torch.empty_like(x)
        xn = torch.empty_like(x2)
        mo = x.new_empty((m,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 1, ds, xn, mo, ro, 0)
    return [y, planes, t, xn, mo, ro]


def _backward_small_fake(leaves, mask, ds, direction, saved, dy):
    """One gradient per leaf, contiguous: dx in x's dtype, the parameter gradients fp32 (the kernels' accumulators)."""
    return [torch.empty_like(leaves[0]), *(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves[1:])]


@opaque(fake=_backward_small_fake, name="trimul_b200_train_small_bwd")
def backward_small(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int,
               saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """b1s / b1g -> contraction gradients -> b7m / b7g. Weight gradients accumulate atomically in one fp32 buffer (one memset),
    which also holds the ring flags and the kernels' per-CTA LayerNorm-gradient rows: the fp32 LayerNorm gradients are those
    rows summed in a fixed order by ``lnpart_sum`` (atomics would add ~1e-4 relative run-to-run noise to them)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, _bi, go, bo = leaves
    planes, t, xn, mo, ro = saved
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    p = planes.shape[0]
    h = p // 2
    nf = _front_ctas(n, d, direction, p)
    nch = p // 64
    sg = _b7_source_groups(n, nch)
    nflag7 = sg * B7_RD * nch if d == 128 else 0
    nsm = torch.cuda.get_device_properties(x.device).multi_processor_count
    shapes = [(d, d), (d, h), (nsm, 2 * h), (2 * p, d), (nsm, 2 * d)]
    sizes = [int(torch.Size(s).numel()) for s in shapes]
    E = _ext()
    with torch.cuda.device(x.device):
        acc = torch.zeros(sum(sizes) + nf * B1_RSF + nflag7, device=x.device, dtype=torch.float32)
        views, o = [], 0
        for s, k in zip(shapes, sizes, strict=True):
            views.append(acc[o:o + k].view(s))
            o += k
        dwg, dwp, lnp_o, dw1, lnp_i = views
        flags = acc[o:o + nf * B1_RSF].view(torch.int32)
        flags7 = acc[o + nf * B1_RSF:].view(torch.int32)
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        dg = torch.empty_like(x2)
        dt = torch.empty_like(t)
        ring = x.new_empty((nf * B1_RSF * TOK, d))
        (E.b1g_backward if d == 128 else E.b1s_backward)(dy2, xn, t, ds, mo, ro, wg, wp, go, bo, dg, dt, dwg, dwp, lnp_o, ring, flags,
                                                         n, nf)
        del ring
        dpl = _contract_bwd(planes, dt, contracted(d, wl.shape[0], direction), direction)
        del dt
        if mask is None:
            pair_mask = x.new_ones((m,), dtype=torch.float32)
        else:
            tm = mask.view(bsz, n)
            pair_mask = (tm[:, :, None] & tm[:, None, :]).float().view(m)
        dx = torch.empty_like(x)
        w1 = _pack_w1(wl, wlg, wr, wrg)
        if d == 128:
            ring7 = x.new_empty((sg * B7_RD * nch * TOK, 128))
            E.b7g_backward(x2, xn, dy2, dpl.view(p, m), dg, pair_mask, w1, wg, gi, dx.view(m, d), dw1, lnp_i, ring7, flags7, EPS,
                           sg)
        else:
            E.b7m_backward(x2, xn, dy2, dpl.view(p, m), dg, pair_mask, w1, wg, gi, dx.view(m, d), dw1, lnp_i, EPS)
        dwl, dwlg, dwr, dwrg = _unpack_w1(dw1, p)
        lns = x.new_empty((2 * h + 2 * d,), dtype=torch.float32)
        E.lnpart_sum(lnp_o, lnp_i, lns)
        dgo, dbo, dgi, dbi = lns.split((h, h, d, d))
    grads = [dwl, dwlg, dwr, dwrg, dwg, dwp, dgi, dbi, dgo, dbo]
    return [dx, *(g.to(torch.float32, copy=True) for g in grads)]


def _kernel_leaves(leaves):
    """The kernels' operands: x, the six projection weights in x's dtype (bf16) and contiguous, the four LayerNorm vectors fp32.
    The casts happen here, outside autograd, so the parameters can be an fp32 master and still receive unrounded fp32 gradients."""
    x = leaves[0]
    return [x, *(w.to(x.dtype).contiguous() for w in leaves[1:7]), *(v.float().contiguous() for v in leaves[7:])]


class _TrainingSmall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        leaves = _kernel_leaves(leaves)
        y, *saved = forward_small(leaves, mask, ds, direction)
        ctx.direction = direction
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:13]), list(vals[13:])
        dx, *grads = backward_small(leaves, mask, ds, ctx.direction, saved, dy)
        return (None, None, None, dx, *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)))


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        leaves = _kernel_leaves(leaves)
        y, *saved = forward(leaves, mask, ds, direction)
        ctx.direction = direction
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:13]), list(vals[13:])
        dx, *grads = backward(leaves, mask, ds, ctx.direction, saved, dy)
        return (None, None, None, dx, *(g.to(dt) for g, dt in zip(grads, ctx.param_dtypes, strict=True)))


def trimul_train(leaves, mask, ds, direction):
    """Training step's forward with autograd (x + ds * trimul(x)); see ``forward`` for the arguments. The parameter leaves may be
    in any float dtype (bf16, or an fp32 master): the kernels run bf16 and each parameter gets its gradient in its own dtype, fp32
    ones unrounded."""
    x = leaves[0]
    d = x.shape[-1]
    if d <= 128:
        if ds is None:
            ds = x.new_ones((x.shape[0] * x.shape[1], d))
        return _TrainingSmall.apply(direction, mask, ds, *leaves)
    return _Training.apply(direction, mask, ds, *leaves)
