"""B200 (sm_100a) TriMul training for every width D64-D512, either direction, bf16, B samples per launch (tokens b-major, M = B L L:
the kernels are token-flat but for the per-sample token mask and row-dropout scale; the contractions batch over channels x B).

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
D64 / D128 weights: every kernel form of the parameters (bf16 row-major front matrices, the packed W1, k3g's permuted Wp, Wg, Wp)
comes from one ``trimul_forms`` launch, cached while the parameters are unchanged (``kernel_forms``: one launch per block per step,
shared by the no-grad recycles and the training forward). Parameter gradients: every kernel writes per-CTA partials (no atomics) and
``grad_reduce`` sums them in a fixed order straight into one fp32 buffer in parameter order, whose views are the gradients:
deterministic, and no unpack / copy kernels.

Sources: ``b200_sources/{k1w,k3g,b1s,b7m,b1g,b7g,wide_aux}.cu`` (D64 / D128), ``b200_sources/{k1w,k3w,wide_aux,k1wb,wide_bwd}.cu``.
Every call owns its buffers. The ``*_into`` ops are the same launches writing y (forward) / dx (backward) into a caller's buffer:
``trimul_train_chunked`` runs a batch in chunks of samples as one autograd node, each chunk writing its rows of y and dx in place.
"""

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._device import num_sms
from torch.autograd.function import once_differentiable

from miniworld_engine.kernels._compile import opaque

WIDTHS = (64, 128, 256, 384, 512)
EPS = 1e-5
TOK = 128
B1_RSF = 4       # b1s / b1g ring slots per front CTA
B7_RD = 12       # b7g ring slots per source group


#: Fixed-arity return types of the ``*_into`` ops (the saved sets of the wide / D64-D128 forwards, the ten parameter gradients).
_SAVED9 = tuple[(torch.Tensor,) * 9]
_SAVED6 = tuple[(torch.Tensor,) * 6]
_GRADS10 = tuple[(torch.Tensor,) * 10]


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
    """The packed W1 of b7m / b7g (chunk c = 64 gate rows, then 64 value rows, of the planes [left | right]); the kernels get it from
    ``trimul_forms`` -- this statement is the reference the tests hold it to."""
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


def _forms_op_fake(params):
    """The one bf16 buffer of forms_op: 8 R D + 2 D R + D D elements (see wide_aux.cu forms_kernel)."""
    r, d = params[0].shape
    return params[0].new_empty((8 * r * d + 2 * d * r + d * d,), dtype=torch.bfloat16)


@opaque(fake=_forms_op_fake, name="trimul_b200_forms")
def forms_op(params: list[torch.Tensor]) -> torch.Tensor:
    """wl, wlg, wr, wrg [R, D], wg [D, D], wp [D, R] (fp32 or bf16, any strides) -> one bf16 buffer of the kernels' weight forms
    (``wide_aux.cu`` ``forms_kernel``)."""
    out = _forms_op_fake(params)
    with torch.cuda.device(out.device):
        _ext().trimul_forms(*params[:6], out)
    return out


def kernel_forms(params):
    """[wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, w1, wpp]: the D64 / D128 kernels' operands from the module's ten parameters (fp32
    master or bf16, the front matrices in any layout) -- the four front matrices row-major bf16, Wg / Wp bf16, the LayerNorm vectors fp32,
    the packed W1 of b7m / b7g and k3g's permuted Wp -- from one ``trimul_forms`` launch, cached while the parameters are unchanged
    (``kernels._capture.param_forms``: inside a CUDA-graph capture once per capture, so a training step's no-grad recycles and its
    training forward share one launch per block)."""
    params = tuple(params)

    def build():
        wl = params[0]
        r, d = wl.shape
        rd = r * d
        buf = forms_op(list(params[:6]))
        o = 8 * rd
        vec = [v.detach().float().contiguous() for v in params[6:10]]
        return (*buf[:4 * rd].view(4, r, d).unbind(0), buf[o + d * r:o + d * r + d * d].view(d, d), buf[o + d * r + d * d:].view(d, r),
                *vec, buf[4 * rd:o].view(4 * r, d), buf[o:o + d * r].view(d, r))

    if torch.compiler.is_compiling():
        return list(build())   # traced: graph ops (the cache's capture-id query is a graph break)
    return list(_capture.param_forms("trimul_b200_forms", params, build))


def _grad_total(leaves):
    """Elements of ``grad_reduce``'s output: the ten parameter gradients in parameter order."""
    return sum(int(t.numel()) for t in leaves[1:11])


def _split_grads(buf, leaves):
    """Views of ``grad_reduce``'s buffer shaped like the parameters (wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo)."""
    out, o = [], 0
    for t in leaves[1:11]:
        k = int(t.numel())
        out.append(buf[o:o + k].view(t.shape))
        o += k
    return out


#: int32 ring flags of one D64 / D128 backward (b1s / b1g, then b7g), zeroed by the training forward's k3g (its zero_buf): a fixed
#: size, the most any shape needs (front CTAs <= 112 x B1_RSF, b7g source groups <= 18 x B7_RD x 8 chunks), so that the saved set's
#: shape does not depend on L (torch.compile with dynamic shapes).
FLAG_WORDS = 112 * B1_RSF + 18 * B7_RD * 8


def _flag_words(n, d, direction, p):
    """The flags one backward uses (<= FLAG_WORDS)."""
    nf = _front_ctas(n, d, direction, p)
    nch = p // 64
    return nf * B1_RSF + (_b7_source_groups(n, nch) * B7_RD * nch if d == 128 else 0)


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


def _sample_major(width: int, hidden: int, direction: int) -> bool:
    """D64 training keeps its channel-major activations (planes, t, dt, the plane gradient) sample-major: ``[P / 64, B, 64, L, L]``,
    each 64-channel chunk (one contraction operand) holding the B samples' planes one after another, instead of ``[P, B, L, L]``.
    k1w's plane stores then step one sample's L^2 tokens between channels whatever B is; with the channel stride B L^2, k1w ran at
    half its one-sample rate at B4 L768 (580 vs 4 x 72 us). The contractions fold (sample, channel) of a chunk into cuBLAS's batch
    exactly as before (the same per-sample products), and k3g / b1s / b7m read the same tiles through 4-D maps
    (``tmap::make_chmajor``). Every chunk must be one operand: the contracted width is 64."""
    return width == 64 and contracted(width, hidden, direction) == 64


def _contract_sm(planes, direction):
    """``_contract`` for sample-major planes [P / 64, B, 64, n, n] -> t [H / 64, B, 64, n, n] (chunk g of planes is operand g). As in
    ``_contract`` the ``out=`` buffers are slices of one 3-D tensor and the 5-D result is a view of it."""
    n, k = planes.shape[-1], planes.shape[1] * 64
    f = lambda a: a.view(-1, n, n)
    g = 2 if direction == 0 else 1
    t = planes.new_empty((g * k, n, n))
    if direction == 0:
        torch.bmm(f(planes[0]), f(planes[2]).transpose(1, 2), out=t[:k])
        torch.bmm(f(planes[1]).transpose(1, 2), f(planes[3]), out=t[k:])
    elif direction == 1:
        torch.bmm(f(planes[0]), f(planes[1]).transpose(1, 2), out=t)
    else:
        torch.bmm(f(planes[0]).transpose(1, 2), f(planes[1]), out=t)
    return t.view(g, *planes.shape[1:])


def _contract_bwd_sm(planes, dt, direction):
    """``_contract_bwd`` for sample-major planes / dt; returns the plane gradient in the planes' layout."""
    n, k = planes.shape[-1], planes.shape[1] * 64
    f = lambda a: a.view(-1, n, n)
    dpl = planes.new_empty((planes.shape[0] * k, n, n))
    if direction == 0:
        torch.bmm(f(dt[0]), f(planes[2]), out=dpl[:k])
        torch.bmm(f(planes[3]), f(dt[1]).transpose(1, 2), out=dpl[k:2 * k])
        torch.bmm(f(dt[0]).transpose(1, 2), f(planes[0]), out=dpl[2 * k:3 * k])
        torch.bmm(f(planes[1]), f(dt[1]), out=dpl[3 * k:])
    elif direction == 1:
        torch.bmm(f(dt[0]), f(planes[1]), out=dpl[:k])
        torch.bmm(f(dt[0]).transpose(1, 2), f(planes[0]), out=dpl[k:])
    else:
        torch.bmm(f(planes[1]), f(dt[0]).transpose(1, 2), out=dpl[:k])
        torch.bmm(f(planes[0]), f(dt[0]), out=dpl[k:])
    return dpl.view(planes.shape)


def _forward_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, mean / rstd (LN_in), mean / rstd (LN_out), k3w vectors, p, g."""
    return [torch.empty_like(leaves[0]), *_forward_into_fake(leaves, mask, ds, direction, leaves[0])]


def _forward_into_fake(leaves, mask, ds, direction, y):
    """The saved set of ``forward`` (y is written into ``y``)."""
    x = leaves[0]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m, p = bsz * n * n, (4 if direction == 0 else 2) * d
    f32 = torch.float32
    return (x.new_empty((p, bsz, n, n)), x.new_empty((p // 2, bsz, n, n)), x.new_empty((m,), dtype=f32),
            x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32),
            x.new_empty((4, d), dtype=f32), x.new_empty((m, d)), x.new_empty((m, d)))


@opaque(fake=_forward_fake, name="trimul_b200_train_fwd")
def forward(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int) -> list[torch.Tensor]:
    """leaves = x, wl, wlg, wr, wrg, wg, wp (bf16, [out, in]), gi, bi, go, bo (fp32); x [B, L, L, D] (samples b-major); mask [B * L]
    bool token mask or None; ds [B * L, D] bf16 row-dropout scale or None; direction 0 = bidirectional, 1 = outgoing, 2 = incoming."""
    return list(_forward_body(leaves, mask, ds, direction, None))


@opaque(fake=_forward_into_fake, name="trimul_b200_train_fwd_into", mutates_args=("y",))
def forward_into(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int,
                 y: torch.Tensor) -> _SAVED9:
    """``forward`` writing y into the caller's ``y``; returns the saved set (a fixed-arity tuple: a mutating custom op returning
    ``Tensor[]`` cannot be functionalized)."""
    return tuple(_forward_body(leaves, mask, ds, direction, y)[1:])


def _forward_body(leaves, mask, ds, direction, y):
    """-> [y, *saved]; y None: allocated here (where the one-launch path always allocated it)."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        mean = x.new_empty((m,), dtype=torch.float32)
        rstd = torch.empty_like(mean)
        E.k1w_stats(x2, mean, rstd, EPS)
        planes = x.new_empty(((4 if direction == 0 else 2) * d, bsz, n, n))
        E.k1w_forward(x2, wl, wlg, wr, wrg, mask, mean, rstd, gi, bi, planes, 0, EPS)
        t = _contract(planes, d, direction)
        mo = torch.empty_like(mean)
        ro = torch.empty_like(mean)
        E.wide_ln_stats(t, mo, ro, EPS)
        wpq = torch.empty_like(wp)
        wgq = torch.empty_like(wg)
        vec = x.new_empty((4, d), dtype=torch.float32)
        E.wide_fold_prep(wp, go, bo, wg, gi, bi, wpq, wgq, vec)
        if y is None:
            y = torch.empty_like(x)
        pv = torch.empty_like(x2)
        gv = torch.empty_like(x2)
        E.k3w_forward(x2, t.view(t.shape[0], m), wpq, wgq, vec, mo, ro, mean, rstd, ds, y.view(m, d), n, 0, pv, gv)
    return [y, planes, t, mean, rstd, mo, ro, vec, pv, gv]


def _grads_fake(leaves):
    """The parameter gradients: fp32 (the kernels' accumulators), contiguous."""
    return tuple(torch.empty(t.shape, dtype=torch.float32, device=t.device) for t in leaves[1:])


def _backward_fake(leaves, mask, ds, direction, saved, dy):
    """One gradient per leaf, contiguous: dx in x's dtype, the parameter gradients fp32 (the kernels' accumulators)."""
    return [torch.empty_like(leaves[0]), *_grads_fake(leaves)]


def _backward_into_fake(leaves, mask, ds, direction, saved, dy, dx):
    """The parameter gradients (dx is written into ``dx``)."""
    return _grads_fake(leaves)


@opaque(fake=_backward_fake, name="trimul_b200_train_bwd")
def backward(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int,
             saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """gate_bwd -> do GEMM -> lnout_bwd -> dWp -> contraction gradients -> k1wb -> weight / input GEMMs -> lnin_bwd."""
    return _backward_body(leaves, mask, ds, direction, saved, dy, None)


@opaque(fake=_backward_into_fake, name="trimul_b200_train_bwd_into", mutates_args=("dx",))
def backward_into(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor | None, direction: int,
                  saved: list[torch.Tensor], dy: torch.Tensor, dx: torch.Tensor) -> _GRADS10:
    """``backward`` writing dx into the caller's ``dx``; returns the parameter gradients."""
    return tuple(_backward_body(leaves, mask, ds, direction, saved, dy, dx)[1:])


def _backward_body(leaves, mask, ds, direction, saved, dy, dx):
    """-> [dx, *parameter gradients]; dx None: allocated here."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = leaves
    planes, t, mean, rstd, mo, ro, vec, pv, gv = saved
    n, d = x.shape[1], x.shape[-1]
    m = x.shape[0] * n * n
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
        if dx is None:
            dx = torch.empty_like(x)
        E.wide_lnin_bwd(dxn, x2, dy2, mean, rstd, gi, dx.view(m, d), acc[2 * d:4 * d])
    grads = [dwl, dwlg, dwr, dwrg, dw[2 * p:], dwp, dgi, dbi, dgo, dbo]
    # the small gradients are views of shared buffers; a custom op may not return aliasing outputs. fp32: the parameters may be
    # an fp32 master (the autograd function casts them to bf16 for the kernels and hands these back in the parameters' dtype)
    return [dx, *(g.to(torch.float32, copy=True) for g in grads)]


# The D64 / D128 ops carry an _sm suffix since the D64 saved set became sample-major (5-D planes / t): inductor's FX graph cache
# keys a graph on its code and input metadata, not on a custom op's fake output, so a graph compiled when these ops returned 4-D
# planes would be reused under the old names (and fail its size asserts). New names give new keys. Renamed again (_smf) when the leaves became the
# kernel forms and the saved set / gradient outputs changed (cached weight forms, flags buffer, one gradient buffer).
def _forward_small_fake(leaves, mask, ds, direction):
    """y and the saved set: planes, t, x_n, LN_out mean / rstd."""
    return [torch.empty_like(leaves[0]), *_forward_small_into_fake(leaves, mask, ds, direction, leaves[0])]


def _forward_small_into_fake(leaves, mask, ds, direction, y):
    """The saved set of ``forward_small`` (y is written into ``y``): planes, t, x_n, LN_out mean / rstd, the backward's zeroed flags."""
    x = leaves[0]
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m, p = bsz * n * n, planes_of(d, leaves[1].shape[0], direction)
    f32 = torch.float32
    if _sample_major(d, leaves[1].shape[0], direction):
        pl, tt = x.new_empty((p // 64, bsz, 64, n, n)), x.new_empty((p // 128, bsz, 64, n, n))
    else:
        pl, tt = x.new_empty((p, bsz, n, n)), x.new_empty((p // 2, bsz, n, n))
    return (pl, tt, x.new_empty((m, d)),
            x.new_empty((m,), dtype=f32), x.new_empty((m,), dtype=f32), x.new_empty((FLAG_WORDS,), dtype=f32))


@opaque(fake=_forward_small_fake, name="trimul_b200_train_small_fwd_smf")
def forward_small(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int) -> list[torch.Tensor]:
    """D64 / D128 training forward; arguments as ``forward`` except ds, which is required ([B * L, D] bf16, ones without dropout), and
    leaves, which are x and ``kernel_forms`` (the front matrices row-major, then w1 and wpp). One direction, the hidden width (wl's
    rows) may be 2 D (``b200_infer.hidden_ok``)."""
    return list(_forward_small_body(leaves, mask, ds, direction, None))


@opaque(fake=_forward_small_into_fake, name="trimul_b200_train_small_fwd_into_smf", mutates_args=("y",))
def forward_small_into(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int,
                       y: torch.Tensor) -> _SAVED6:
    """``forward_small`` writing y into the caller's ``y``; returns the saved set."""
    return tuple(_forward_small_body(leaves, mask, ds, direction, y)[1:])


def _forward_small_body(leaves, mask, ds, direction, y):
    """-> [y, *saved]; y None: allocated here."""
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo, _w1, wpp = leaves
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    E = _ext()
    with torch.cuda.device(x.device):
        x2 = x.reshape(m, d)
        hid = wl.shape[0]
        p = planes_of(d, hid, direction)
        sm = _sample_major(d, hid, direction)
        planes = x.new_empty((p // 64, bsz, 64, n, n) if sm else (p, bsz, n, n))
        E.k1w_forward(x2, wl, wlg, wr, wrg, mask, None, None, gi, bi, planes, 0, EPS)
        t = _contract_sm(planes, direction) if sm else _contract(planes, contracted(d, hid, direction), direction)
        if y is None:
            y = torch.empty_like(x)
        xn = torch.empty_like(x2)
        mo = x.new_empty((m,), dtype=torch.float32)
        ro = torch.empty_like(mo)
        zb = x.new_empty((FLAG_WORDS,), dtype=torch.float32)   # the backward's ring flags, zeroed by k3g
        E.k3g_forward(x2, t, wpp, wg, gi, bi, go, bo, y.view(m, d), n, EPS, 1, ds, xn, mo, ro, 0, zb)
    return [y, planes, t, xn, mo, ro, zb]


def _backward_small_fake(leaves, mask, ds, direction, saved, dy):
    """dx in x's dtype and the ten parameter gradients as one fp32 buffer in parameter order (``_split_grads``)."""
    return [torch.empty_like(leaves[0]), _backward_small_into_fake(leaves, mask, ds, direction, saved, dy, None)]


def _backward_small_into_fake(leaves, mask, ds, direction, saved, dy, dx):
    """The parameter gradients as one fp32 buffer (dx is written into ``dx``)."""
    return leaves[0].new_empty((_grad_total(leaves),), dtype=torch.float32)


@opaque(fake=_backward_small_fake, name="trimul_b200_train_small_bwd_smf")
def backward_small(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int,
                   saved: list[torch.Tensor], dy: torch.Tensor) -> list[torch.Tensor]:
    """b1s / b1g -> contraction gradients -> b7m / b7g -> grad_reduce. Every kernel writes its parameter gradients as per-CTA partials
    (no atomics); ``grad_reduce`` sums them in a fixed order into one fp32 buffer in parameter order -> [dx, buffer]: deterministic. The
    ring flags come zeroed from the forward (saved[5])."""
    return _backward_small_body(leaves, mask, ds, direction, saved, dy, None)


@opaque(fake=_backward_small_into_fake, name="trimul_b200_train_small_bwd_into_smf", mutates_args=("dx",))
def backward_small_into(leaves: list[torch.Tensor], mask: torch.Tensor | None, ds: torch.Tensor, direction: int,
                        saved: list[torch.Tensor], dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """``backward_small`` writing dx into the caller's ``dx``; returns the parameter-gradient buffer (one tensor: a mutating custom op
    returning ``Tensor[]`` cannot be functionalized)."""
    return _backward_small_body(leaves, mask, ds, direction, saved, dy, dx)[1]


def _backward_small_body(leaves, mask, ds, direction, saved, dy, dx):
    """-> [dx, parameter-gradient buffer]; dx None: allocated here."""
    x, wl, wlg, wr, wrg, wg, wp, gi, _bi, go, bo, w1, _wpp = leaves
    planes, t, xn, mo, ro, zb = saved
    bsz, n, d = x.shape[0], x.shape[1], x.shape[-1]
    m = bsz * n * n
    sm = planes.dim() == 5                  # sample-major (``_sample_major``)
    p = planes.shape[0] * (64 if sm else 1)
    h = p // 2
    nf = _front_ctas(n, d, direction, p)
    nch = p // 64
    sg = _b7_source_groups(n, nch)
    nsm = num_sms(x.device)
    f32 = torch.float32
    E = _ext()
    assert _flag_words(n, d, direction, p) <= FLAG_WORDS
    with torch.cuda.device(x.device):
        flags = zb.view(torch.int32)
        x2 = x.reshape(m, d)
        dy2 = dy.contiguous().view(m, d)
        dg = torch.empty_like(x2)
        dt = torch.empty_like(t)
        ring = x.new_empty((nf * B1_RSF * TOK, d))
        part_o = x.new_empty((nf, d * (d + h)), dtype=f32)           # front CTAs' (dWg | dWp) partials
        lnp_o = x.new_empty((nsm - nf, 2 * h), dtype=f32)            # LN CTAs' (dg_o | db_o) rows
        (E.b1g_backward if d == 128 else E.b1s_backward)(dy2, xn, t, ds, mo, ro, wg, wp, go, bo, dg, dt, part_o, lnp_o, ring,
                                                         flags[:nf * B1_RSF], n, nf)
        del ring
        dpl = (_contract_bwd_sm(planes, dt, direction) if sm
               else _contract_bwd(planes, dt, contracted(d, wl.shape[0], direction), direction))
        del dt
        if dx is None:
            dx = torch.empty_like(x)
        if d == 128:
            ns, nc = sg, nsm - nch * sg
            part_i = x.new_empty((ns, 2 * p * d), dtype=f32)          # source groups' dW1 partials
            lnp_i = x.new_empty((nc, 2 * d), dtype=f32)
            ring7 = x.new_empty((sg * B7_RD * nch * TOK, 128))
            E.b7g_backward(x2, xn, dy2, dpl.view(p, m), dg, mask, w1, wg, gi, dx.view(m, d), part_i, lnp_i, ring7,
                           flags[nf * B1_RSF:], EPS, sg, n)
        else:
            ns = nc = min(m // TOK, nsm)
            part_i = x.new_empty((ns, 2 * p * d), dtype=f32)
            lnp_i = x.new_empty((nc, 2 * d), dtype=f32)
            E.b7m_backward(x2, xn, dy2, dpl if sm else dpl.view(p, m), dg, mask, w1, wg, gi, dx.view(m, d), part_i, lnp_i, EPS, n)
        grads = x.new_empty((_grad_total(leaves),), dtype=f32)
        E.grad_reduce(part_o, nf, lnp_o, nsm - nf, part_i, ns, lnp_i, nc, grads)
    return [dx, grads]


def _kernel_leaves(leaves):
    """The kernels' operands: x, the six projection weights in x's dtype (bf16) and contiguous, the four LayerNorm vectors fp32.
    The casts happen here, outside autograd, so the parameters can be an fp32 master and still receive unrounded fp32 gradients."""
    x = leaves[0]
    return [x, *(w.to(x.dtype).contiguous() for w in leaves[1:7]), *(v.float().contiguous() for v in leaves[7:])]


class _TrainingSmall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, direction, mask, ds, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        leaves = [leaves[0], *kernel_forms(leaves[1:])]
        y, *saved = forward_small(leaves, mask, ds, direction)
        ctx.direction = direction
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:15]), list(vals[15:])
        dx, grads = backward_small(leaves, mask, ds, ctx.direction, saved, dy)
        return (None, None, None, dx, *(g.to(dt) for g, dt in zip(_split_grads(grads, leaves), ctx.param_dtypes, strict=True)))


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


def _chunk_ops(width):
    return ((forward_small_into, backward_small_into) if width <= 128 else (forward_into, backward_into))


class _Chunked(torch.autograd.Function):
    """A batch in chunks of samples (``sizes``, b-major, in order) as ONE autograd node. y and dx are allocated once and each chunk's
    launch writes its rows of them in place (``*_into``), so nothing is copied or concatenated. Plain chunks keep their saved tensors
    from the forward to the backward; checkpointed chunks (``checkpoint``) save nothing and run their forward again in the backward,
    so only the chunk being differentiated holds saved tensors. The parameter gradients are the chunks' fp32 gradients summed in chunk
    order (deterministic). Each chunk runs exactly the launches of a one-launch call of its samples."""

    @staticmethod
    def forward(ctx, direction, sizes, checkpoint, mask, ds, *leaves):
        ctx.param_dtypes = [t.dtype for t in leaves[1:]]
        small = leaves[0].shape[-1] <= 128
        leaves = [leaves[0], *kernel_forms(leaves[1:])] if small else _kernel_leaves(leaves)
        ctx.nleaves = len(leaves)
        x = leaves[0]
        n = x.shape[1]
        fwd, _ = _chunk_ops(x.shape[-1])
        y = torch.empty_like(x)
        saved, counts, start = [], [], 0
        for size in sizes:
            stop = start + size
            part = fwd([x[start:stop], *leaves[1:]], None if mask is None else mask[start * n:stop * n],
                       None if ds is None else ds[start * n:stop * n], direction, y[start:stop])
            if checkpoint:
                del part
                counts.append(0)
            else:
                saved.extend(part)
                counts.append(len(part))
            start = stop
        ctx.direction, ctx.sizes, ctx.checkpoint, ctx.counts = direction, sizes, checkpoint, counts
        ctx.save_for_backward(mask, ds, *leaves, *saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        vals = ctx.saved_tensors
        nl = 2 + ctx.nleaves
        mask, ds, leaves, saved = vals[0], vals[1], list(vals[2:nl]), list(vals[nl:])
        x = leaves[0]
        n = x.shape[1]
        small = x.shape[-1] <= 128
        fwd, bwd = _chunk_ops(x.shape[-1])
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        total = None
        start, at = 0, 0
        for size, count in zip(ctx.sizes, ctx.counts, strict=True):
            stop = start + size
            sub = [x[start:stop], *leaves[1:]]
            m = None if mask is None else mask[start * n:stop * n]
            d = None if ds is None else ds[start * n:stop * n]
            if ctx.checkpoint:
                part = fwd(sub, m, d, ctx.direction, torch.empty_like(sub[0]))
            else:
                part = saved[at:at + count]
                at += count
            grads = bwd(sub, m, d, ctx.direction, part, dy[start:stop], dx[start:stop])
            del part
            if small:
                total = grads if total is None else total.add_(grads)
            else:
                total = grads if total is None else [a.add_(b) for a, b in zip(total, grads, strict=True)]
            start = stop
        if small:
            total = _split_grads(total, leaves)
        return (None, None, None, None, None, dx, *(g.to(dt) for g, dt in zip(total, ctx.param_dtypes, strict=True)))


def trimul_train_chunked(leaves, mask, ds, direction, sizes, checkpoint):
    """``trimul_train`` for a batch run as chunks of ``sizes`` samples (``integrations.trimul_b200._plan``), one autograd node. mask
    [B * L] or None, ds [B * L, D] or None."""
    x = leaves[0]
    d = x.shape[-1]
    if d <= 128 and ds is None:
        ds = x.new_ones((x.shape[0] * x.shape[1], d))
    return _Chunked.apply(direction, tuple(sizes), bool(checkpoint), mask, ds, *leaves)
