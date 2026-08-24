"""Reference checks for the ``adaln`` family.

adaLN and its ``conditioned_transition`` tail were one module (``checks_adaln.py``). The rules
these references follow -- same shapes as the driver, fp32 autograd rather than a hand-derived
formula, saved activations consumed as-is while saved statistics are recomputed -- are written
out in ``checks/conditioned_transition.py``. The helpers both families use are in
``checks/__init__.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.checks import _fixed, _no_tf32
from miniworld_engine.kernels.drivers import _rand
from miniworld_engine.kernels.drivers.adaln import _EPS, _adaln_args
from miniworld_engine.kernels.drivers.conditioned_transition import (
    _D,
    _DC,
    _M,
    _SHAPE_KEY,
)

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
    from miniworld_engine.kernels.adaln.triton.main import triton_adaptive_layer_norm

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
    from miniworld_engine.kernels.adaln.triton.inference import _cond_affine

    _fixed()
    cond, lnw = _rand(_M, _DC), _rand(_DC)
    aff = _cond_affine(cond, lnw, _EPS, shape_key=_SHAPE_KEY)
    return aff, _ln(cond) * lnw.float()


def adaln_fwd():
    """inference._adaln_fused_kernel: the whole adaLN forward in one kernel, y only."""
    from miniworld_engine.kernels.adaln.triton.inference import adaln_inference_fused

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
    from miniworld_engine.kernels.adaln.triton.main import triton_adaptive_layer_norm

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
    from miniworld_engine.kernels.adaln.triton.training import _epilogue_train

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
    from miniworld_engine.kernels.adaln.triton.fused3 import _bwd_elem

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
    from miniworld_engine.kernels.adaln.triton.training import _bwd_x

    _fixed()
    dy, x = _rand(_M, _D), _rand(_M, _D)
    gate = torch.sigmoid(_rand(_M, _D).float()).to(dy.dtype)
    mean_x, rstd_x = _true_stats(x)
    d_stack, dx = _bwd_x(dy, x, mean_x, rstd_x, gate, shape_key=_SHAPE_KEY)
    xr = x.float().detach().requires_grad_(True)
    s = torch.logit(gate.float()).detach().requires_grad_(True)
    (torch.sigmoid(s) * F.layer_norm(xr, (_D,), eps=_EPS)).backward(dy.float())
    assert s.grad is not None and xr.grad is not None
    return {"D": (d_stack, torch.cat([s.grad, dy.float()], dim=1).t()), "DX": (dx, xr.grad)}


def adaln_bwd_dx_dlnw():
    """training._dgrad_condln_kernel: dcond_aff = D^T@w_cat in-kernel, then the cond LN backward.

    mean_c/rstd_c are the true statistics of the cond passed (the driver fills them with 1.0).
    D and w_cat stay the driver's randn -- the kernel only contracts them, so their values carry no
    assumption. The GEMM is applied outside autograd (TF32 off) and its result is the vector-Jacobian
    seed for ``LN(cond)*lnw``, which gives dcond and dlnw from one tape; dlnw lands via fp32
    atomics, so its add order can differ run to run.
    """
    from miniworld_engine.kernels.adaln.triton.training import _dgrad_condln

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
    from miniworld_engine.kernels.adaln.triton.inference import _adaln_epilogue

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
    from miniworld_engine.kernels.adaln.triton.fused3 import _gemm_gate, _gemm_gate_train

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
