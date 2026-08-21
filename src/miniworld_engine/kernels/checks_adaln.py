"""Reference checks for the ``adaln`` and ``conditioned_transition`` kernels.

One argument-free function per registry kernel, returning ``(actual, expected)`` -- or a dict of
those pairs when the kernel writes more than one buffer. ``autotune/run_all.py::check_one`` calls
it and reports ``max|a-e| / max|e|`` per pair. A driver only proves the kernel *runs*; these say
whether the number it wrote is the number the op is defined to produce.

Shapes and launcher calls are lifted verbatim from ``drivers_adaln.py`` (aligned mode: M=512,
D=128, DC=128, ND=512, eps=1e-5) -- the checker must hit the same compiled kernel the driver does,
so the shape constants are imported rather than re-chosen here, and that is also what carries
``MINIWORLD_SHAPE_MODE=ragged``'s partial tail tiles into these references without restating a
single extent. Nothing below writes a shape literal: the two derived widths are ``2 * _D``
([scale|bias]) and ``2 * _ND`` ([a|b] / [da|db]), which are packings of an imported extent, not
extents of their own. dtypes follow the drivers too: adaLN is bf16, the conditioned_transition
tail is fp32 ("fp32 io with TF32 tensor cores" in every file of it).

The references:

    adaLN         y = sigmoid(cond_aff @ Wsᵀ + sb) * LN(x) + cond_aff @ Wbᵀ,
                  cond_aff = LN(cond) * lnw, both LNs affine-free with the biased variance
                  (``modules/adaptive_layernorm/module.py``'s PYTORCH path, restated at the top of
                  ``adaln/triton/inference.py``). ``_adaln_epilogue`` gets [scale|bias] from an
                  outside GEMM, so its reference is the epilogue half only.
    tail fwd      a = x@Waᵀ ; b = x@Wbᵀ ; h = silu(a)*b ; out = h@Wsᵀ ;
                  scale = cond@Wscᵀ + bsc ; y = sigmoid(scale)*out
    tail bwd      the same expression differentiated by fp32 autograd (``_swiglu_bwd`` /
                  ``_gate_bwd`` below), never by re-typing the kernel's own closed form -- a
                  transcription of ``silu_p = sa*(1 + a*(1-sa))`` would agree with a sign error in
                  the kernel. The GEMM that consumes each grad is then applied outside autograd,
                  which is what the kernels fuse.

Every reference GEMM runs with TF32 off, so it is a true-fp32 reference against the kernels'
``input_precision="tf32"`` dots; that gap (~1e-3 relative here) is the bulk of what these checks
report and it sits well inside ``check_one``'s 5e-2 band.

Two things the drivers do that a checker must NOT copy. ``drivers_adaln.adaln_bwd_pre_dx`` and
``adaln_bwd_dx_dlnw`` fill mean/rstd with ``1.0``, and ``adaln_bwd_pre``/``adaln_bwd_pre_dx`` pass
a ``randn`` gate. Both are fine for a driver (it only has to reach the kernel) and both put the
kernel outside the regime where "matching" means anything: a saved rstd that is not
``1/sqrt(var(x)+eps)`` makes the LN-backward algebra the derivative of nothing, and a gate outside
(0,1) is not any ``sigmoid(scale)``, so the only reference left would be a transcription of the
kernel's own closed form. The checkers below keep every shape, dtype and stride the driver's and
change only those VALUES -- real row statistics of the x/cond they pass, and gate = sigmoid(randn)
-- so the comparison is against a defined function. The drivers are not edited.

Saved values, two opposite rules, one per kernel:

  * saved ACTIVATIONS (x_hat, cond_norm, gate) are consumed as-is, so the reference must consume
    the SAME numbers -- hence ``_main_train``'s reconstruction of a leaf whose LayerNorm reproduces
    the saved x_hat/cond_norm bit-for-bit, and the ``logit(gate)`` pinning that makes the
    reference's ``sigmoid`` land exactly on the saved gate.
  * saved STATISTICS (mean, rstd) are consumed by a LayerNorm BACKWARD, which is the derivative of
    a normalization whose mean/rstd are functions of x. A reference that froze them as constants
    would drop the two centering terms and report a large, plausible-looking error that is not the
    kernel's. So every LN-backward reference here differentiates through ``F.layer_norm`` and is
    handed the true statistics of the same x.

Gradients are taken by fp32 autograd, never by hand: writing ``dscale = dy*x_norm*g*(1-g)`` into
the reference is a chance to repeat the exact algebra slip the kernel may have made and then agree
with it. ``scale`` is reached through ``logit`` so the sigmoid derivative also comes from autograd.
Where a saved gate has saturated to exactly 1.0 in bf16, ``logit`` is +inf, ``sigmoid`` is 1.0 and
autograd's ``g*(1-g)`` is 0 -- which is what the kernel computes there too.

The packed backward buffers need no special handling: ``dab = [da|db]`` and the partial-looking
``dout``/``dscale``/``ab`` emissions are all final values for their own tensor (the kernels write
them once, from the ``pid == 0`` program of the redundant axis), so each is compared directly
instead of being reconstructed at the launcher level.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .drivers_adaln import (
    FP32,
    _D,
    _DC,
    _EPS,
    _M,
    _ND,
    _SHAPE_KEY,
    _adaln_args,
    _ct_args,
    _rand,
)


def _fixed() -> None:
    """Fixed RNG stream per checker: a "WRONG NUMBERS" line reproduces on the next run."""
    torch.manual_seed(0)


@contextlib.contextmanager
def _no_tf32():
    """True-fp32 reference GEMMs (the kernels use tl.dot(input_precision="tf32"))."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _ln(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """LayerNorm, no affine, biased variance -- computed in fp32 as the kernels do in-register."""
    return F.layer_norm(x.float(), (x.shape[-1],), eps=eps)


def _ln_stats(xf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(x_hat, rstd) for that same LayerNorm; ``xf`` already fp32. rstd is the saved-stat form."""
    return (F.layer_norm(xf, (xf.shape[-1],), eps=_EPS),
            torch.rsqrt(xf.var(dim=-1, correction=0) + _EPS))


def _true_stats(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, rstd) fp32 (M,) -- what a forward would have SAVED for this exact input.

    The drivers hand these two buffers to the backward kernels as ``ones``; see the module
    docstring for why a checker cannot.
    """
    tf = t.float()
    return tf.mean(dim=-1), torch.rsqrt(tf.var(dim=-1, correction=0) + _EPS)


# ── conditioned_transition reference stages ───────────────────────────────────────────────────


def _expand(x, wa, wb):
    """(a, b, h) with h = silu(a)*b -- stages 1+2 of the tail."""
    with _no_tf32():
        a = x.float() @ wa.float().t()
        b = x.float() @ wb.float().t()
    return a, b, F.silu(a) * b


def _squeeze_gate_ref(h, cond, ws, wsc, bsc):
    """(out, scale, y) -- stages 3+4+5 of the tail."""
    with _no_tf32():
        out = h.float() @ ws.float().t()
        scale = torch.addmm(bsc.float(), cond.float(), wsc.float().t())
    return out, scale, torch.sigmoid(scale) * out


def _swiglu_bwd(a, b, dh):
    """(da, db) by fp32 autograd through h = silu(a)*b."""
    ar = a.float().detach().requires_grad_(True)
    br = b.float().detach().requires_grad_(True)
    (F.silu(ar) * br).backward(dh.float())
    return ar.grad, br.grad


def _gate_bwd(out, scale, dy):
    """(dout, dscale) by fp32 autograd through y = sigmoid(scale)*out."""
    o = out.float().detach().requires_grad_(True)
    s = scale.float().detach().requires_grad_(True)
    (torch.sigmoid(s) * o).backward(dy.float())
    return o.grad, s.grad


# ── adaLN ─────────────────────────────────────────────────────────────────────────────────────


def _adaln_ref(x, cond, lnw, ws, sb, wb):
    """(y, x_hat, cond_norm, gate, rstd_x, rstd_c) in fp32 -- the whole adaLN forward.

    Written from ``modules/adaptive_layernorm/module.py``'s PYTORCH branch: two affine-free
    LayerNorms (the cond one weighted by lnw), a biased Linear to scale, an unbiased one to bias,
    and a sigmoid gate. Both GEMMs run with TF32 off.
    """
    x_hat, rstd_x = _ln_stats(x.detach().float())
    cond_norm, rstd_c = _ln_stats(cond.detach().float())
    cond_aff = cond_norm * lnw.detach().float()
    with _no_tf32():
        scale = torch.addmm(sb.detach().float(), cond_aff, ws.detach().float().t())
        bias = cond_aff @ wb.detach().float().t()
    gate = torch.sigmoid(scale)
    return gate * x_hat + bias, x_hat, cond_norm, gate, rstd_x, rstd_c


def _gate_bwd_saved(gate, x_norm, dy):
    """(dscale, dxn) by fp32 autograd through y = sigmoid(scale)*x_norm, at the SAVED gate.

    ``scale`` is pinned at ``logit(gate)`` so ``sigmoid(scale)`` reproduces the saved gate the
    kernel actually read; autograd then supplies the sigmoid derivative and the product rule.
    """
    s = torch.logit(gate.float()).detach().requires_grad_(True)
    xn = x_norm.float().detach().requires_grad_(True)
    (torch.sigmoid(s) * xn).backward(dy.float())
    return s.grad, xn.grad


def _main_train():
    """main.py's autograd Function, forward+backward once: (kernel grads, reference grads).

    Returned in the Function's own order: (dx, dcond, dlnw, dWs, dsb, dWb). The three main.py
    backward kernels are launched together by one ``backward()`` -- exactly as the drivers reach
    them -- so the three checkers below each call this and compare their own kernel's buffers.

    Two deliberate differences from ``drivers_adaln._adaln_main(backward=True)``, neither of which
    changes a shape, a dtype, a stride or which compiled kernel runs:
      * all six inputs get requires_grad, not just x. The three kernels always compute all six
        gradients; without this autograd discards five of them and the checker could only see dx.
      * the saved activations are read back off ``y.grad_fn`` so the reference consumes the same
        x_hat / cond_norm / gate the backward kernels read (see the module docstring).
    """
    from .adaln.triton.main import triton_adaptive_layer_norm

    _fixed()
    # batched=True: the Function reshapes x/cond itself and keys the forward and (via
    # ctx.orig_x_shape) all three backward kernels off the PRE-flatten shape, so it takes the
    # (1, M, D) activation -- the driver's call and the only one ``length_of`` accepts.
    args = _adaln_args(batched=True)
    for t in args:
        t.requires_grad_(True)
    x, cond, lnw, ws, sb, wb = args
    y = triton_adaptive_layer_norm(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
    x_hat, cond_norm, gate, _, _, _, rstd_x, rstd_c = y.grad_fn.saved_tensors
    dy = torch.randn_like(y)
    got = torch.autograd.grad(y, args, dy)
    # dx/dcond come back at x/cond's (1, M, D) shape (the backward reshapes them to
    # ctx.orig_x_shape); the reference's leaves are the 2-D saved activations, so view the two
    # activation grads at (M, D). A view of the same numbers -- nothing is dropped or reduced.
    got = (got[0][0], got[1][0], *got[2:])

    # LN(x_hat/rstd) == x_hat and rstd(x_hat/rstd) == rstd exactly (LayerNorm is shift-invariant
    # and var(x_hat) = 1 - eps*rstd^2), so this leaf reproduces the SAVED activation while leaving
    # mean/rstd differentiable -- the two properties the backward reference needs at once.
    with _no_tf32():   # covers the reference's backward GEMMs (dW = grad^T @ aff) too
        xr = (x_hat / rstd_x[:, None]).detach().requires_grad_(True)
        cr = (cond_norm / rstd_c[:, None]).detach().requires_grad_(True)
        lw, w_s, b_s, w_b = (t.float().detach().requires_grad_(True) for t in (lnw, ws, sb, wb))
        aff = F.layer_norm(cr, (_DC,), eps=_EPS) * lw
        scale = torch.addmm(b_s, aff, w_s.t())
        # Pin scale's VALUE at logit(saved gate) -- a detached shift, so every gradient path into
        # aff / w_s / b_s is untouched while sigmoid(scale) lands on the gate the kernels read.
        scale = scale + (torch.logit(gate.float()) - scale).detach()
        y_ref = torch.sigmoid(scale) * F.layer_norm(xr, (_D,), eps=_EPS) + aff @ w_b.t()
        y_ref.backward(dy[0].float())   # dy at the reference's (M, D) view of the same seed
    return got, (xr.grad, cr.grad, lw.grad, w_s.grad, b_s.grad, w_b.grad)


def layernorm_fwd_strided():
    """fused3._ln_kernel (HAS_W=True) via inference._cond_affine: aff = LN(cond) * lnw.

    ``ln_cond`` of the module: affine-free biased-variance LayerNorm times a weight, no bias. The
    reduce axis is _DC, which ragged mode moves to a different partial tail than _D.
    """
    from .adaln.triton.inference import _cond_affine

    _fixed()
    cond, lnw = _rand(_M, _DC), _rand(_DC)
    aff = _cond_affine(cond, lnw, _EPS, shape_key=_SHAPE_KEY)
    return aff, _ln(cond) * lnw.float()


def adaln_fwd():
    """inference._adaln_fused_kernel: the whole adaLN forward in one kernel, y only."""
    from .adaln.triton.inference import adaln_inference_fused

    _fixed()
    # OUTER entry point: it reshapes x/cond itself and takes the key from the PRE-flatten
    # shape, so it needs the (1, M, D) activation the driver passes (``length_of`` refuses a
    # 2-D (M, D), where shape[-2] is M and not L). y comes back at that same shape; the
    # reference runs on x[0]/cond[0] -- views of the very rows the kernel read, no copy and
    # no second draw -- and y[0] is the whole of y at B=1.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    y = adaln_inference_fused(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
    return y[0], _adaln_ref(x[0], cond[0], lnw, ws, sb, wb)[0]


def adaln_fwd_saveact():
    """main.adaln_fwd_kernel: y AND the five buffers it saves for the backward.

    x_hat / cond_norm / gate / rstd_x / rstd_cond are read back off ``y.grad_fn`` (the forward
    stores them under its own x_offsets / c_offsets masks, which is a separate correctness question
    from y). ``requires_grad`` on x is the only change from the driver -- it makes grad_fn exist.
    """
    from .adaln.triton.main import triton_adaptive_layer_norm

    _fixed()
    # OUTER entry point (see ``adaln_fwd``): the (1, M, D) activation, as the driver passes it.
    # The five saved buffers are allocated off the forward's own x_2d and so are 2-D/1-D; only
    # y carries x's shape back, so y[0] is what lines up with the 2-D reference.
    x, cond, lnw, ws, sb, wb = _adaln_args(batched=True)
    x.requires_grad_(True)
    y = triton_adaptive_layer_norm(x, cond, lnw, ws, sb, wb, _EPS, _EPS)
    x_hat, cond_norm, gate, _, _, _, rstd_x, rstd_c = y.grad_fn.saved_tensors
    e_y, e_xh, e_cn, e_g, e_rx, e_rc = _adaln_ref(x[0], cond[0], lnw, ws, sb, wb)
    return {"Y": (y[0], e_y), "XHat": (x_hat, e_xh), "CondNorm": (cond_norm, e_cn),
            "Gate": (gate, e_g), "RstdX": (rstd_x, e_rx), "RstdC": (rstd_c, e_rc)}


def adaln_epilogue_saveact():
    """training._epilogue_train_kernel: y, mean_x, rstd_x and gate, with HAS_SB=True.

    sb is the raw (M, 2N) [scale|bias] the outside GEMM produced and ``scale_bias`` (N,) is folded
    into the SCALE half in the epilogue (kernel: ``scale = SB[:, :N] + ScaleBias``,
    ``bias = SB[:, N:]``). All four buffers are outputs the training forward saves, so all four are
    compared; the statistics are the kernel's own, not consumed, so they are recomputed here.
    """
    from .adaln.triton.training import _epilogue_train

    _fixed()
    x, sb, scale_b = _rand(_M, _D), _rand(_M, 2 * _D), _rand(_D)
    y, mean, rstd, gate = _epilogue_train(x, sb, _EPS, scale_b, shape_key=_SHAPE_KEY)
    x_hat, e_rstd = _ln_stats(x.float())
    e_gate = torch.sigmoid(sb[:, :_D].float() + scale_b.float())
    return {"Y": (y, e_gate * x_hat + sb[:, _D:].float()), "Mean": (mean, x.float().mean(dim=-1)),
            "Rstd": (rstd, e_rstd), "Gate": (gate, e_gate)}


def adaln_bwd_pre():
    """fused3._bwd_elem_kernel: (dscale, dxn) from (dy, x_norm, gate), elementwise.

    gate is sigmoid(randn) rather than the driver's randn -- a gate outside (0,1) is not any
    sigmoid(scale), so there would be no reference but the kernel's own formula (module docstring).
    x_norm is left as the driver's randn: the kernel is elementwise in it and the reference treats
    it as a leaf, so its value carries no assumption.
    """
    from .adaln.triton.fused3 import _bwd_elem

    _fixed()
    dy, x_norm = _rand(_M, _D), _rand(_M, _D)
    gate = torch.sigmoid(_rand(_M, _D).float()).to(dy.dtype)
    dscale, dxn = _bwd_elem(dy, x_norm, gate, shape_key=_SHAPE_KEY)
    e_dscale, e_dxn = _gate_bwd_saved(gate, x_norm, dy)
    return {"DScale": (dscale, e_dscale), "DXn": (dxn, e_dxn)}


def adaln_bwd_pre_dx():
    """training._bwd_x_kernel: D=(2N,M) [dscale;dy] stacked, and dx = LN-bwd(dy*gate) fused.

    mean_x/rstd_x are the TRUE statistics of the x passed (the driver fills them with 1.0) and gate
    is sigmoid(randn); see the module docstring. dx is an LN backward, so the reference
    differentiates through F.layer_norm -- freezing the statistics would drop the two centering
    terms. dscale comes from the same autograd tape, and dy is copied straight into D's upper half.
    """
    from .adaln.triton.training import _bwd_x

    _fixed()
    dy, x = _rand(_M, _D), _rand(_M, _D)
    gate = torch.sigmoid(_rand(_M, _D).float()).to(dy.dtype)
    mean_x, rstd_x = _true_stats(x)
    d_stack, dx = _bwd_x(dy, x, mean_x, rstd_x, gate, shape_key=_SHAPE_KEY)
    xr = x.float().detach().requires_grad_(True)
    s = torch.logit(gate.float()).detach().requires_grad_(True)
    (torch.sigmoid(s) * F.layer_norm(xr, (_D,), eps=_EPS)).backward(dy.float())
    return {"D": (d_stack, torch.cat([s.grad, dy.float()], dim=1).t()), "DX": (dx, xr.grad)}


def adaln_bwd_dx_dlnw():
    """training._dgrad_condln_kernel: dcond_aff = D^T@w_cat in-kernel, then the cond LN backward.

    mean_c/rstd_c are the true statistics of the cond passed (the driver fills them with 1.0).
    D and w_cat stay the driver's randn -- the kernel only contracts them, so their values carry no
    assumption. The GEMM is applied outside autograd (TF32 off) and its result is the vector-Jacobian
    seed for ``LN(cond)*lnw``, which gives dcond and dlnw from one tape; dlnw lands via fp32
    atomics, so its add order can differ run to run.
    """
    from .adaln.triton.training import _dgrad_condln

    _fixed()
    d_stack, w_cat = _rand(2 * _D, _M), _rand(2 * _D, _DC)
    cond, lnw = _rand(_M, _DC), _rand(_DC)
    mean_c, rstd_c = _true_stats(cond)
    dcond, dlnw = _dgrad_condln(d_stack, w_cat, cond, mean_c, rstd_c, lnw, shape_key=_SHAPE_KEY)
    with _no_tf32():
        dcond_aff = d_stack.float().t() @ w_cat.float()
    cr = cond.float().detach().requires_grad_(True)
    lw = lnw.float().detach().requires_grad_(True)
    (F.layer_norm(cr, (_DC,), eps=_EPS) * lw).backward(dcond_aff)
    return {"DCond": (dcond, cr.grad), "DLnW": (dlnw, lw.grad)}


def adaln_bwd_dx_dbias():
    """main.adaln_bwd_input_kernel: dx, dcond and dscale_b (the (N,) atomic accumulation)."""
    got, exp = _main_train()
    return {"DX": (got[0], exp[0]), "DCond": (got[1], exp[1]), "DScaleB": (got[4], exp[4])}


def adaln_bwd_dw():
    """main.adaln_bwd_weight_kernel: dWs = dscale^T@cond_aff and dWb = dy^T@cond_aff."""
    got, exp = _main_train()
    return {"DScaleW": (got[3], exp[3]), "DBiasW": (got[5], exp[5])}


def adaln_bwd_dlnw():
    """main.adaln_bwd_lnw_kernel: dlnw = sum_m (dscale@Ws + dy@Wb) * cond_norm."""
    got, exp = _main_train()
    return got[2], exp[2]


def adaln_epilogue():
    """inference._adaln_epilogue_kernel: y = sigmoid(SB[:, :N]) * LN(x) + SB[:, N:].

    SB is the (M, 2N) [scale|bias] the outside GEMM produced, so scale already carries to_scale's
    bias and the epilogue applies no weights of its own.
    """
    from .adaln.triton.inference import _adaln_epilogue

    _fixed()
    x, sb = _rand(_M, _D), _rand(_M, 2 * _D)
    y = _adaln_epilogue(x, sb, _EPS, shape_key=_SHAPE_KEY)
    scale, bias = sb[:, :_D].float(), sb[:, _D:].float()
    return y, torch.sigmoid(scale) * _ln(x) + bias


def adaln_gemm_gate():
    """fused3._gemm_gate_kernel: the dual in-kernel GEMM + gate, both SAVE_GATE settings.

    y = sigmoid(cond_norm@Wsᵀ + sb) * x_norm + cond_norm@Wbᵀ, and with SAVE_GATE=True the same
    kernel also stores gate = sigmoid(scale) for the backward -- so both outputs are compared.
    The kernel's inputs are the already-normalized x_norm/cond_norm (the two LN kernels are
    upstream), which is what the driver feeds it.
    """
    from .adaln.triton.fused3 import _gemm_gate, _gemm_gate_train

    _fixed()
    x_norm, cond_norm, _, ws, sb, wb = _adaln_args()
    y = _gemm_gate(x_norm, cond_norm, ws, wb, sb, shape_key=_SHAPE_KEY)
    y_save, gate = _gemm_gate_train(x_norm, cond_norm, ws, wb, sb, shape_key=_SHAPE_KEY)
    with _no_tf32():
        scale = torch.addmm(sb.float(), cond_norm.float(), ws.float().t())
        bias = cond_norm.float() @ wb.float().t()
    exp_gate = torch.sigmoid(scale)
    exp_y = exp_gate * x_norm.float() + bias
    return {"Y": (y, exp_y), "Ysave": (y_save, exp_y), "Gate": (gate, exp_gate)}


# ── conditioned_transition: forward ───────────────────────────────────────────────────────────


def cond_transition_fwd_b2b():
    """inference._cond_transition_inference_kernel: the whole tail in ONE kernel, y only.

    h (M, ND) never reaches HBM here -- the launcher returns just y -- so the reference is the
    full five-stage expression and nothing intermediate is observable to compare.
    """
    from .conditioned_transition.triton.inference import cond_transition_inference

    _fixed()
    x, cond, wa, wb, ws, wsc, bsc = _ct_args()
    # OUTER entry point: it takes the flat matrix but names L through ``length=``, exactly as
    # the driver does; without it the launcher falls to ``length_of(x.shape)`` on a 2-D x.
    y = cond_transition_inference(x, cond, wa, wb, ws, wsc, bsc, length=_M)
    return y, _squeeze_gate_ref(_expand(x, wa, wb)[2], cond, ws, wsc, bsc)[2]


def cond_transition_expand_swiglu():
    """composed._expand_swiglu_kernel: h = silu(x@Waᵀ) * (x@Wbᵀ)."""
    from .conditioned_transition.triton.composed import _expand_swiglu

    _fixed()
    x, _, wa, wb, *_ = _ct_args()
    h = _expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)
    return h, _expand(x, wa, wb)[2]


def cond_transition_expand_swiglu_saveact():
    """train_fused._fwd_expand_swiglu_kernel: same h, plus the packed pre-activations ab=[a|b]."""
    from .conditioned_transition.triton.train_fused import _fwd_expand_swiglu

    _fixed()
    x, _, wa, wb, *_ = _ct_args()
    h, ab = _fwd_expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)
    a, b, exp_h = _expand(x, wa, wb)
    return {"H": (h, exp_h), "AB": (ab, torch.cat([a, b], dim=1))}


def cond_transition_swiglu():
    """training._swiglu_fwd_kernel: h = silu(a)*b, elementwise over the (M, ND) expand halves."""
    from .conditioned_transition.triton.training import _swiglu

    _fixed()
    a, b = _rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32)
    return _swiglu(a, b, shape_key=_SHAPE_KEY), F.silu(a) * b


def cond_transition_squeeze_gate():
    """composed._squeeze_gate_kernel: y = sigmoid(cond@Wscᵀ + bsc) * (h@Wsᵀ)."""
    from .conditioned_transition.triton.composed import _squeeze_gate

    _fixed()
    _, cond, _, _, ws, wsc, bsc = _ct_args()
    h = _rand(_M, _ND, dtype=FP32)
    y = _squeeze_gate(h, cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)
    return y, _squeeze_gate_ref(h, cond, ws, wsc, bsc)[2]


def cond_transition_squeeze_gate_saveact():
    """train_fused._fwd_squeeze_gate_kernel: same y, plus the saved out and scale."""
    from .conditioned_transition.triton.train_fused import _fwd_squeeze_gate

    _fixed()
    _, cond, _, _, ws, wsc, bsc = _ct_args()
    h = _rand(_M, _ND, dtype=FP32)
    y, out, scale = _fwd_squeeze_gate(h, cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)
    exp_out, exp_scale, exp_y = _squeeze_gate_ref(h, cond, ws, wsc, bsc)
    return {"Y": (y, exp_y), "Out": (out, exp_out), "Scale": (scale, exp_scale)}


def cond_transition_fwd_b2b_saveact():
    """training._b2b_fwd_train_kernel: the whole tail in one kernel -> (y, ab, h, out, scale).

    All five buffers are compared: the b2b keeps a/b/h in registers between the two GEMMs, so a
    y-only check would not see the saved-for-backward stores (which the ``pid_d == 0`` store mask
    makes a separate correctness question from y).
    """
    from .conditioned_transition.triton.training import _b2b_fwd_train

    _fixed()
    x, cond, wa, wb, ws, wsc, bsc = _ct_args()
    y, ab, h, out, scale = _b2b_fwd_train(x, cond, wa, wb, ws, wsc, bsc,
                                          shape_key=_SHAPE_KEY)
    a, b, exp_h = _expand(x, wa, wb)
    exp_out, exp_scale, exp_y = _squeeze_gate_ref(exp_h, cond, ws, wsc, bsc)
    return {"Y": (y, exp_y), "AB": (ab, torch.cat([a, b], dim=1)), "H": (h, exp_h),
            "Out": (out, exp_out), "Scale": (scale, exp_scale)}


# ── conditioned_transition: backward ──────────────────────────────────────────────────────────


def cond_transition_bwd_swiglu_flat():
    """training._swiglu_bwd_kernel via _swiglu_bwd_packed(a, b, dh) -> dab = [da|db] (M, 2ND)."""
    from .conditioned_transition.triton.training import _swiglu_bwd_packed

    _fixed()
    a = _rand(_M, _ND, dtype=FP32)
    b = _rand(_M, _ND, dtype=FP32)
    dh = _rand(_M, _ND, dtype=FP32)
    dab = _swiglu_bwd_packed(a, b, dh, shape_key=_SHAPE_KEY)
    da, db = _swiglu_bwd(a, b, dh)
    return dab, torch.cat([da, db], dim=1)


def cond_transition_bwd_swiglu_packed():
    """train_fused._swiglu_bwd_pack_kernel: dab = [da|db] from (dh, ab=[a|b]).

    Same math as the flat kernel above, different contract (2-D tiling, a/b arrive packed), so it
    gets its own reference against the same autograd grads.
    """
    from .conditioned_transition.triton.train_fused import _swiglu_bwd_pack

    _fixed()
    dh = _rand(_M, _ND, dtype=FP32)
    ab = _rand(_M, 2 * _ND, dtype=FP32)
    dab = _swiglu_bwd_pack(dh, ab, shape_key=_SHAPE_KEY)
    da, db = _swiglu_bwd(ab[:, :_ND], ab[:, _ND:], dh)
    return dab, torch.cat([da, db], dim=1)


def cond_transition_bwd_swiglu_dx():
    """train_fused._dx_fused_kernel: dx = da@Wa + db@Wb, da/db recomputed per tile from (dh, ab).

    Two dots per ND tile (Wa and Wb separately), where the packed kernel below does one over the
    concatenated 2ND axis -- so the same dx is reachable two ways and both are checked.
    """
    from .conditioned_transition.triton.train_fused import _dx_fused

    _fixed()
    _, _, wa, wb, *_ = _ct_args()
    dh = _rand(_M, _ND, dtype=FP32)
    ab = _rand(_M, 2 * _ND, dtype=FP32)
    dx = _dx_fused(dh, ab, wa, wb, shape_key=_SHAPE_KEY)
    da, db = _swiglu_bwd(ab[:, :_ND], ab[:, _ND:], dh)
    with _no_tf32():
        exp_dx = da @ wa.float() + db @ wb.float()
    return dx, exp_dx


def cond_transition_bwd_swiglu_dx_packed():
    """train_fused._dx_swiglubwd_kernel: dx = dab @ Wcat (one GEMM) and the emitted dab.

    dab is a real output here, not a scratch partial (the backward hands it to the dWa/dWb wgrad),
    so it is compared alongside dx.
    """
    from .conditioned_transition.triton.train_fused import _dx_swiglubwd

    _fixed()
    _, _, wa, wb, *_ = _ct_args()
    dh = _rand(_M, _ND, dtype=FP32)
    ab = _rand(_M, 2 * _ND, dtype=FP32)
    wcat = torch.cat([wa, wb], dim=0)
    dx, dab = _dx_swiglubwd(dh, ab, wcat, shape_key=_SHAPE_KEY)
    da, db = _swiglu_bwd(ab[:, :_ND], ab[:, _ND:], dh)
    exp_dab = torch.cat([da, db], dim=1)
    with _no_tf32():
        exp_dx = exp_dab @ wcat.float()
    return {"DX": (dx, exp_dx), "DAB": (dab, exp_dab)}


def cond_transition_bwd_gate_squeeze_dx():
    """train_fused._dh_gatebwd_kernel: dh = (sigmoid(scale)*dy) @ Ws, plus dout and dscale.

    The gate backward is fused into the dh-GEMM prologue and the two elementwise grads are emitted
    from the pid_n == 0 column of programs, so all three buffers are final values and compared.
    """
    from .conditioned_transition.triton.train_fused import _dh_gatebwd

    _fixed()
    _, _, _, _, ws, *_ = _ct_args()
    out = _rand(_M, _D, dtype=FP32)
    scale = _rand(_M, _D, dtype=FP32)
    dy = _rand(_M, _D, dtype=FP32)
    dh, dout, dscale = _dh_gatebwd(out, scale, dy, ws, _ND, shape_key=_SHAPE_KEY)
    exp_dout, exp_dscale = _gate_bwd(out, scale, dy)
    with _no_tf32():
        exp_dh = exp_dout @ ws.float()
    return {"DH": (dh, exp_dh), "DOut": (dout, exp_dout), "DScale": (dscale, exp_dscale)}


def cond_transition_bwd_gemm():
    """train_fused._dgemm_kernel as the backward calls it: dcond = dscale(M,D) @ Wsc(D,DC).

    N=_DC and K=_D are passed separately (the launcher takes M/N/K as ints), so the GEMM's N and
    K axes are two different non-aligned widths under ragged mode.
    """
    from .conditioned_transition.triton.train_fused import _dgemm

    _fixed()
    _, _, _, _, _, wsc, _ = _ct_args()
    dscale = _rand(_M, _D, dtype=FP32)
    dcond = _dgemm(dscale, wsc, _M, _DC, _D, wsc.stride(0), wsc.stride(1),
                   shape_key=_SHAPE_KEY)
    with _no_tf32():
        exp = dscale @ wsc.float()
    return dcond, exp


def cond_transition_bwd_dw():
    """train_fused._wgrad_kernel: dW(N,K) = g(M,N)ᵀ @ x(M,K), the dWs shape (D, ND)."""
    from .conditioned_transition.triton.train_fused import _wgrad

    _fixed()
    g = _rand(_M, _D, dtype=FP32)
    x = _rand(_M, _ND, dtype=FP32)
    dw = _wgrad(g, x, _D, _ND, shape_key=_SHAPE_KEY)
    with _no_tf32():
        exp = g.t() @ x
    return dw, exp

