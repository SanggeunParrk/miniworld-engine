"""Reference checks for the ``conditioned_transition`` family.

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

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.checks import _fixed, _no_tf32
from miniworld_engine.kernels.drivers import FP32, _rand
from miniworld_engine.kernels.drivers.conditioned_transition import (
    _D,
    _DC,
    _M,
    _ND,
    _SHAPE_KEY,
    _ct_args,
)

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


# ── conditioned_transition: forward ───────────────────────────────────────────────────────────


def cond_transition_fwd_b2b():
    """inference._cond_transition_inference_kernel: the whole tail in ONE kernel, y only.

    h (M, ND) never reaches HBM here -- the launcher returns just y -- so the reference is the
    full five-stage expression and nothing intermediate is observable to compare.
    """
    from miniworld_engine.kernels.conditioned_transition.triton.inference import cond_transition_inference

    _fixed()
    x, cond, wa, wb, ws, wsc, bsc = _ct_args()
    # OUTER entry point: it takes the flat matrix but names L through ``length=``, exactly as
    # the driver does; without it the launcher falls to ``length_of(x.shape)`` on a 2-D x.
    y = cond_transition_inference(x, cond, wa, wb, ws, wsc, bsc, length=_M)
    return y, _squeeze_gate_ref(_expand(x, wa, wb)[2], cond, ws, wsc, bsc)[2]


def cond_transition_expand_swiglu():
    """composed._expand_swiglu_kernel: h = silu(x@Waᵀ) * (x@Wbᵀ)."""
    from miniworld_engine.kernels.conditioned_transition.triton.composed import _expand_swiglu

    _fixed()
    x, _, wa, wb, *_ = _ct_args()
    h = _expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)
    return h, _expand(x, wa, wb)[2]


def cond_transition_expand_swiglu_saveact():
    """train_fused._fwd_expand_swiglu_kernel: same h, plus the packed pre-activations ab=[a|b]."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _fwd_expand_swiglu

    _fixed()
    x, _, wa, wb, *_ = _ct_args()
    h, ab = _fwd_expand_swiglu(x, wa, wb, shape_key=_SHAPE_KEY)
    a, b, exp_h = _expand(x, wa, wb)
    return {"H": (h, exp_h), "AB": (ab, torch.cat([a, b], dim=1))}


def cond_transition_swiglu():
    """training._swiglu_fwd_kernel: h = silu(a)*b, elementwise over the (M, ND) expand halves."""
    from miniworld_engine.kernels.conditioned_transition.triton.training import _swiglu

    _fixed()
    a, b = _rand(_M, _ND, dtype=FP32), _rand(_M, _ND, dtype=FP32)
    return _swiglu(a, b, shape_key=_SHAPE_KEY), F.silu(a) * b


def cond_transition_squeeze_gate():
    """composed._squeeze_gate_kernel: y = sigmoid(cond@Wscᵀ + bsc) * (h@Wsᵀ)."""
    from miniworld_engine.kernels.conditioned_transition.triton.composed import _squeeze_gate

    _fixed()
    _, cond, _, _, ws, wsc, bsc = _ct_args()
    h = _rand(_M, _ND, dtype=FP32)
    y = _squeeze_gate(h, cond, ws, wsc, bsc, shape_key=_SHAPE_KEY)
    return y, _squeeze_gate_ref(h, cond, ws, wsc, bsc)[2]


def cond_transition_squeeze_gate_saveact():
    """train_fused._fwd_squeeze_gate_kernel: same y, plus the saved out and scale."""
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _fwd_squeeze_gate

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
    from miniworld_engine.kernels.conditioned_transition.triton.training import _b2b_fwd_train

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
    from miniworld_engine.kernels.conditioned_transition.triton.training import _swiglu_bwd_packed

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _swiglu_bwd_pack

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _dx_fused

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _dx_swiglubwd

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _dh_gatebwd

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _dgemm

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
    from miniworld_engine.kernels.conditioned_transition.triton.train_fused import _wgrad

    _fixed()
    g = _rand(_M, _D, dtype=FP32)
    x = _rand(_M, _ND, dtype=FP32)
    dw = _wgrad(g, x, _D, _ND, shape_key=_SHAPE_KEY)
    with _no_tf32():
        exp = g.t() @ x
    return dw, exp
