"""Torch references for the ``transition`` family.

``autotune.run_all.check_one`` calls one of these functions and compares what came back:
a checker returns ``(actual, expected)`` -- or a dict of named pairs -- and the runner
reports ``max|a-e| / max|e|`` per pair against its 5e-2 bf16 band. Launching proves a
kernel runs; only a reference proves the number.

Two rules hold everywhere in this file:

* **Same operands as the driver.** Every checker imports the shapes and calls the same
  launcher as its twin in ``drivers_trans.py`` (``_transition_operands``, ``ROWS``,
  ``TRIMUL_ROWS``/``TRIMUL_D``, and the ``settings.configure(transition_lnbwd_cuda=False)``
  bypass), so a passing check speaks about the launch the runner actually recorded.
* **Reference = what the source computes, not what the op is named.** Where a kernel takes
  precomputed state (LayerNorm ``rstd``/``c1``), the reference is fed the SAME state, and
  the intermediate that the kernel rounds to bf16 before a ``tl.dot`` is rounded in the
  reference too -- otherwise the comparison measures the reference's extra precision.

The maths the kernels here implement, once:

    LN from folded stats   xn = (x*rstd - c1)*gamma + beta          (c1 = mean*rstd, so
                                x*rstd - c1 == (x-mean)*rstd)
    SwiGLU expand          a = xn @ Wa^T ; b = xn @ Wb^T ; h = silu(a)*b
    squeeze                y = h @ Ws^T
    SwiGLU gate backward   dA = ge*b*silu'(a) ; dB = ge*silu(a)     (silu' = sig+silu*(1-sig))
    gated projection       out = sigmoid(x@wg^T) * (x@wp^T)         (triangle_multiplication)

References run in fp32: the kernels accumulate their GEMMs in fp32 from bf16 operands, so
an fp32 torch matmul of the same bf16 tensors is the tight reference, not a looser one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels.checks import _proj
from miniworld_engine.kernels.drivers import ACT_DTYPE, rows2d
from miniworld_engine.kernels.drivers.transition import (
    EPS,
    K_LARGE,
    K_SMALL,
    N_EXPAND,
    ROWS,
    _pair_x,
    _transition_operands,
)


def _stats(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rstd, c1=mean*rstd) from the same kernel the launchers use (stats.py:stats_triton)."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    return stats_triton(x2, EPS)


def _xn(x2, rstd, c1, gamma, beta) -> torch.Tensor:
    """LN from saved stats, fp32, still fp32 on return (cast at the call site if the kernel does).

    This is the kernels' contract verbatim: ``xn = (x*rstd - c1)*g + beta`` with
    ``c1 = mean*rstd`` -- NOT ``mean``. (transition/triton/fused.py:120.)
    """
    return (x2.float() * rstd[:, None] - c1[:, None]) * gamma.float() + beta.float()


def _swiglu_bwd(a, b, ge) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h, dA, dB) for the SwiGLU gate backward, all fp32."""
    sig = torch.sigmoid(a)
    silu = a * sig
    gf = ge.float()
    return silu * b, gf * b * (sig + silu * (1.0 - sig)), gf * silu


# --------------------------------------------------------------------------- transition fwd


def transition_expand_swiglu_triton():
    """transition_fwd_kernel emits ``expand`` in bf16; ``TritonTransitionFunction.forward``
    then does the squeeze with torch.matmul (main.py:138), so the reference covers both:
    round h to bf16 exactly where the kernel stores it, then squeeze."""
    from miniworld_engine.kernels.transition.triton.main import triton_transition

    # ``_pair_x()`` -- the driver's own helper for this one launcher. TritonTransitionFunction
    # reads both_key(rows_of(x.shape)) before its view(-1, d), and ``length_of`` refuses the
    # flat (M, K) that ``_transition_operands`` hands every other entry point here. Only x moves;
    # the weights still come from ``_transition_operands`` so K/ND are unchanged, and in aligned
    # mode _pair_x's M = L*L is exactly ROWS.
    _, _, _, wa, wb, ws = _transition_operands()
    x = _pair_x()
    y = triton_transition(x, wa, wb, ws, N_EXPAND)

    a, b = _proj(x, wa, wb)
    h = (a * torch.sigmoid(a) * b).to(x.dtype)      # kernel's bf16 `expand` store
    return y, h.float() @ ws.float().T


def transition_layernorm_expand_swiglu_triton():
    """LN(from stats) + SwiGLU expand -> (M, ND). The stats are passed IN so the reference
    normalizes with the identical rstd/c1 (the launcher would otherwise call stats_triton
    itself -- same kernel, same inputs, same values, but then unobserved)."""
    from miniworld_engine.kernels.transition.triton.fused import transition_expand_gate

    x2, g, beta, wa, wb, _ = _transition_operands()
    rstd, c1 = _stats(x2)
    expand = transition_expand_gate(x2, g, beta, wa, wb, EPS, stats=(rstd, c1))

    xn = _xn(x2, rstd, c1, g, beta).to(x2.dtype)    # kernel casts xn to bf16 before both dots
    a, b = _proj(xn, wa, wb)
    return expand, a * torch.sigmoid(a) * b


def transition_fwd_b2b_triton():
    """LN + expand + SwiGLU + squeeze in one kernel: h is rounded to bf16 in registers
    (fused.py:`h = (a*sigmoid(a)*b).to(x_ptr.dtype...)`) and then contracted with Ws, so the
    reference rounds h too. fuse_stats=False -> stats come from outside, as in the driver."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b

    x2, g, beta, wa, wb, ws = _transition_operands(k=K_SMALL)
    rstd, c1 = _stats(x2)
    out = transition_b2b(x2, g, beta, wa, wb, ws, EPS, stats=(rstd, c1), fuse_stats=False)

    xn = _xn(x2, rstd, c1, g, beta).to(x2.dtype)
    a, b = _proj(xn, wa, wb)
    h = (a * torch.sigmoid(a) * b).to(x2.dtype)
    return out, h.float() @ ws.float().T


def transition_fwd_b2b_ktiled_triton():
    """Same maths as ``transition_fwd_b2b_triton`` at K=256 (the kernel's K>128 reason to
    exist). This launcher computes the stats itself and does not return them, so the
    reference re-runs the same deterministic stats kernel on the same x2."""
    from miniworld_engine.kernels.transition.triton.fused import transition_b2b_ktiled

    x2, g, beta, wa, wb, ws = _transition_operands(k=K_LARGE)
    out = transition_b2b_ktiled(x2, g, beta, wa, wb, ws, EPS)

    rstd, c1 = _stats(x2)
    xn = _xn(x2, rstd, c1, g, beta).to(x2.dtype)
    a, b = _proj(xn, wa, wb)
    h = (a * torch.sigmoid(a) * b).to(x2.dtype)
    return out, h.float() @ ws.float().T


# --------------------------------------------------------------------------- transition bwd


def transition_bwd_swiglu_recompute_triton():
    """Version A stacked (NORMALIZE/STORE_H/STACK_DAB = True): normalize x from saved stats,
    recompute a/b once, emit h, dAB=[dA|dB] and the normalized xn. dAB is column-stacked,
    dA in [0:ND) and dB in [ND:2*ND) (fused.py:748-757)."""
    from miniworld_engine.kernels.transition.triton.fused import (
        _transition_expand_gatebwd_stacked,
    )

    x2, g, beta, wa, wb, _ = _transition_operands()
    rstd, c1 = _stats(x2)
    ge = rows2d(ROWS, wa.shape[0])
    h, dAB, xn = _transition_expand_gatebwd_stacked(x2, rstd, c1, g, beta, wa, wb, ge)

    xn_ref = _xn(x2, rstd, c1, g, beta).to(x2.dtype)   # cast to bf16 in-kernel before the dots
    a, b = _proj(xn_ref, wa, wb)
    h_ref, dA, dB = _swiglu_bwd(a, b, ge)
    return {
        "h": (h, h_ref),
        "dAB": (dAB, torch.cat((dA, dB), dim=1)),
        "xn": (xn, xn_ref),
    }


def layernorm_bwd_foldstats_triton():
    """LN backward from saved stats -> (dx, dgamma, dbeta), the dgamma/dbeta column partials
    scattered over NUM_REPLICAS fp32 buffers and summed by the launcher.

    ``transition_lnbwd_cuda`` defaults True and would route bf16/K<=512 to the hand-CUDA LN
    backward instead of this kernel, so it is switched off for the launch exactly as the
    driver does. Reference: fp32 autograd through ``F.layer_norm`` -- an independent
    derivation of dx/dgamma/dbeta rather than a restatement of the kernel's own algebra
    (which would hide a wrong mean/rstd fold)."""
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd

    x2, g, _, _, _, _ = _transition_operands()
    rstd, c1 = _stats(x2)
    dxn = torch.empty_like(x2).normal_()

    # Pinned only while the settings still declare it (v2.2.0 retires the legacy transition_*
    # switches); the driver applies the same guard.
    pinned = hasattr(settings.current(), "transition_lnbwd_cuda")
    if pinned:
        previous = settings.current().transition_lnbwd_cuda
        settings.configure(transition_lnbwd_cuda=False)
    try:
        dx, dgamma, dbeta = _transition_ln_bwd(dxn, x2, rstd, c1, g)
    finally:
        if pinned:
            settings.configure(transition_lnbwd_cuda=previous)

    xf = x2.float().requires_grad_(True)
    gf = g.float().requires_grad_(True)
    bf = torch.zeros_like(gf).requires_grad_(True)
    F.layer_norm(xf, (x2.shape[1],), gf, bf, EPS).backward(dxn.float())
    return {"dx": (dx, xf.grad), "dgamma": (dgamma, gf.grad), "dbeta": (dbeta, bf.grad)}


# ── the vendored transition_cuda extension ───────────────────────────────────────────────────
#
# These three kernels had no driver at all until now (nothing in the package imports the
# extension), so they had never produced a number. A driver alone would only prove they run, which
# is exactly the state that let three masking bugs sit in this repo, so they get references too.
#
# The math is the transition the module defines, written from torch ops rather than transcribed
# from the .cu: h = silu(x @ wa.T) * (x @ wb.T), y = h @ ws.T, with nn.Linear-style (out, in)
# weights. A reference transcribed from the kernel would agree with the kernel's sign errors.


def _transition_ref(x, wa, wb, ws):
    """fp32 transition reference. tf32 is forced off: it silently costs ~10 bits of mantissa on
    an A6000 and would put the reference inside the error it is supposed to measure."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        xf, af, bf, sf = (t.float() for t in (x, wa, wb, ws))
        a = xf @ af.t()
        b = xf @ bf.t()
        h = torch.nn.functional.silu(a) * b
        return h @ sf.t(), a, b, h
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _transition_cuda_fwd_pair(dtype):
    from miniworld_engine.kernels.drivers.transition import (
        _CUDA_N,
        _transition_cuda_ext,
        _transition_cuda_operands,
    )

    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(dtype)
    y = ext.forward(x, wa, wb, ws, _CUDA_N)
    ref, _, _, _ = _transition_ref(x, wa, wb, ws)
    return {"y": (y, ref)}


def transition_cast_cuda():
    """Same launch as ``transition_swiglu_cuda``; both kernels run inside this one forward, and
    the output is the only observable either of them has from Python. bf16 outright, matching
    its driver -- an all-fp32 forward launches no cast_kernel (see the driver's docstring)."""
    return _transition_cuda_fwd_pair(torch.bfloat16)


def transition_swiglu_cuda():
    return _transition_cuda_fwd_pair(ACT_DTYPE)


def transition_bwd_cuda():
    """``backward`` returns the grads in the order the .cpp assembles them. Which tensor is which
    is asserted by shape rather than assumed from position: dx matches x, dwa/dwb match the expand
    weights, dws matches the squeeze weight, and the four shapes are mutually distinct at these
    extents, so the mapping is unambiguous."""
    from miniworld_engine.kernels.drivers.transition import (
        _CUDA_N,
        _transition_cuda_ext,
        _transition_cuda_operands,
    )

    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(ACT_DTYPE)
    g = torch.randn_like(x).contiguous()
    got = ext.backward(g, x, wa, wb, ws, _CUDA_N)

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        xf = x.float().requires_grad_(True)
        af = wa.float().requires_grad_(True)
        bf = wb.float().requires_grad_(True)
        sf = ws.float().requires_grad_(True)
        h = torch.nn.functional.silu(xf @ af.t()) * (xf @ bf.t())
        (h @ sf.t()).backward(g.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    # Order is stated by the kernel, not guessed: transition_cuda_kernel.cu:480 returns
    # {dx, grad_a_weight, grad_b_weight, grad_squeeze_weight}. Shape cannot disambiguate it --
    # expand_a and expand_b are both (nN, N), so dwa and dwb are the same shape, which is what the
    # first version of this checker tripped over. Shape is still asserted as a cross-check, so a
    # future reordering in the .cu shows up as a shape mismatch rather than a silent swap.
    names = ("dx", "dwa", "dwb", "dws")
    refs = (xf.grad, af.grad, bf.grad, sf.grad)
    if len(got) != 4:
        raise AssertionError(f"backward returned {len(got)} tensors, expected 4")
    out = {}
    for name, actual, expected in zip(names, got, refs, strict=False):
        if tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(
                f"{name}: kernel returned {tuple(actual.shape)}, reference {tuple(expected.shape)} "
                "-- the return order in transition_cuda_kernel.cu:480 may have changed"
            )
        out[name] = (actual, expected)
    return out


def transition_squeeze_residual_triton():
    from miniworld_engine.kernels.drivers.transition import SHAPE_KEY
    from miniworld_engine.kernels.transition.triton.residual import squeeze_residual
    h = rows2d(ROWS, N_EXPAND * K_SMALL)
    w = rows2d(K_SMALL, N_EXPAND * K_SMALL)
    r = rows2d(ROWS, K_SMALL)
    expected = ((h.float() @ w.float().T).to(h.dtype).float() + r.float()).to(r.dtype)
    return squeeze_residual(h, w, r, SHAPE_KEY), expected
