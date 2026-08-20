"""Torch references for the ``transition`` / ``triangle_multiplication`` kernels.

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
import torch.nn.functional as F  # noqa: N812

from miniworld_engine.kernels.drivers import dev, rows2d
from miniworld_engine.kernels.drivers_trans import (
    EPS,
    K_LARGE,
    K_SMALL,
    N_EXPAND,
    ND_SMALL,
    ROWS,
    TRIMUL_D,
    TRIMUL_ROWS,
    _transition_operands,
)


def _stats(x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rstd, c1=mean*rstd) from the same kernel the launchers use (stats.py:stats_triton)."""
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton

    return stats_triton(x2, EPS)


def _xn(x2, rstd, c1, gamma, beta) -> torch.Tensor:
    """LN from saved stats, fp32, still fp32 on return (cast at the call site if the kernel does).

    This is the kernels' contract verbatim: ``xn = (x*rstd - c1)*g + beta`` with
    ``c1 = mean*rstd`` -- NOT ``mean``. (transition/cute/fused.py:65,
    transition/triton/fused.py:120.)
    """
    return (x2.float() * rstd[:, None] - c1[:, None]) * gamma.float() + beta.float()


def _proj(xn, wa, wb) -> tuple[torch.Tensor, torch.Tensor]:
    """(a, b) = (xn @ wa^T, xn @ wb^T) in fp32 -- the kernels' two fp32 accumulators."""
    xf = xn.float()
    return xf @ wa.float().T, xf @ wb.float().T


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

    x2, _, _, wa, wb, ws = _transition_operands()
    y = triton_transition(x2, wa, wb, ws, N_EXPAND)

    a, b = _proj(x2, wa, wb)
    h = (a * torch.sigmoid(a) * b).to(x2.dtype)     # kernel's bf16 `expand` store
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


def transition_fold_triton():
    """Weight prefold. The reference recomputes what the source computes, term for term
    (fold.py:64-81), including two details a "looks right" reference gets wrong:
      * S is the rowsum of the fp32 gamma-scaled weight, taken BEFORE B's bf16 cast;
      * B2 contracts the RAW W with beta (no gamma), unlike B and S.
    Both outputs are interleaved gate/up per j: row 2j from Wa, row 2j+1 from Wb."""
    from miniworld_engine.kernels.transition.triton.fold import fold_swiglu_triton

    _, g, beta, wa, wb, _ = _transition_operands()
    B, S, B2 = fold_swiglu_triton(wa, wb, g, beta)

    n, k = wa.shape
    gf, bef = g.float(), beta.float()
    ba, bb = wa.float() * gf, wb.float() * gf
    b_ref = torch.stack((ba.to(B.dtype), bb.to(B.dtype)), dim=1).reshape(2 * n, k)
    s_ref = torch.stack((ba.sum(dim=1), bb.sum(dim=1)), dim=1).reshape(2 * n)
    b2_ref = torch.stack((wa.float() @ bef, wb.float() @ bef), dim=1).reshape(2 * n)
    return {"B": (B, b_ref), "S": (S, s_ref), "B2": (B2, b2_ref)}


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


def swiglu_gate_bwd_sm100():
    """SM100 cute gate-backward. Same three outputs as the Triton recompute kernel above
    (h, dA, dB), from the SAVED xn -- no LN inside -- so the reference is the plain
    dual-projection + SwiGLU-backward epilogue (gatebwd_sm100.py:700-705)."""
    from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import (
        transition_expand_gatebwd_sm100,
    )

    xn, _, _, wa, wb, _ = _transition_operands()
    ge = rows2d(ROWS, wa.shape[0])
    h, dA, dB = transition_expand_gatebwd_sm100(xn, wa, wb, ge)

    a, b = _proj(xn, wa, wb)
    h_ref, dA_ref, dB_ref = _swiglu_bwd(a, b, ge)
    return {"h": (h, h_ref), "dA": (dA, dA_ref), "dB": (dB, dB_ref)}


def transition_bwd_epilogue_triton():
    """In-place ``dA *= ge; dB *= ge``. The kernel is IN-PLACE, so the reference has to hold
    a copy of the pre-launch dA/dB; ``restore_value`` on the autotuner means the multiply
    lands exactly once no matter how many configs it benchmarks."""
    from miniworld_engine.kernels.transition.cute.gatebwd_sm100 import _grad_mul_inplace

    dA, dB, ge = (rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL), rows2d(ROWS, ND_SMALL))
    dA0, dB0 = dA.clone(), dB.clone()
    _grad_mul_inplace(dA, dB, ge)

    gf = ge.float()
    return {"dA": (dA, dA0.float() * gf), "dB": (dB, dB0.float() * gf)}


def transition_bwd_transpose_packed_triton():
    """Pure layout: (M, ND) -> (M, 2*ND) with out[m, 2j] = out[m, 2j+1] = ge[m, j], i.e.
    ``repeat_interleave(2, dim=1)`` -- exact, not approximate. The second pair feeds a
    transposed VIEW (column stride != 1), which the launcher explicitly supports
    (backward_gatebwd.py:94) and which the contiguous case cannot exercise."""
    from miniworld_engine.kernels.transition.cute.backward_gatebwd import _cdup_interleave

    ge = rows2d(ROWS, ND_SMALL)
    view = rows2d(ND_SMALL, ROWS).T                 # (ROWS, ND_SMALL), stride (1, ROWS)
    return {
        "contiguous": (_cdup_interleave(ge), ge.repeat_interleave(2, dim=1)),
        "strided_view": (_cdup_interleave(view), view.repeat_interleave(2, dim=1)),
    }


def layernorm_bwd_privatized_triton():
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

    previous = settings.current().transition_lnbwd_cuda
    settings.configure(transition_lnbwd_cuda=False)
    try:
        dx, dgamma, dbeta = _transition_ln_bwd(dxn, x2, rstd, c1, g)
    finally:
        settings.configure(transition_lnbwd_cuda=previous)

    xf = x2.float().requires_grad_(True)
    gf = g.float().requires_grad_(True)
    bf = torch.zeros_like(gf).requires_grad_(True)
    F.layer_norm(xf, (x2.shape[1],), gf, bf, EPS).backward(dxn.float())
    return {"dx": (dx, xf.grad), "dgamma": (dgamma, gf.grad), "dbeta": (dbeta, bf.grad)}


def layernorm_fwd_recompute_foldstats_triton():
    """xn re-materialization from FOLDED stats: the kernel is handed ``c1 = mean*rstd``, not
    ``mean``, and computes ``x*rstd - c1`` (cute/fused.py:65). Two pairs:
      * ``xn``  -- that contract literally, with the same rstd/c1 the kernel got;
      * ``vs_layer_norm`` -- the same output against a real fp32 LayerNorm, which is what
        proves the folded form ``x*rstd - c1 == (x-mean)*rstd`` rather than assuming it."""
    from miniworld_engine.kernels.transition.cute.fused import _xn_recompute

    x2, g, beta, _, _, _ = _transition_operands()
    rstd, c1 = _stats(x2)
    xn = _xn_recompute(x2, rstd, c1, g, beta)

    contract = _xn(x2, rstd, c1, g, beta)
    ln = F.layer_norm(x2.float(), (x2.shape[1],), g.float(), beta.float(), EPS)
    return {"xn": (xn, contract), "vs_layer_norm": (xn, ln)}


# ------------------------------------------------------- triangle_multiplication (dt-v1)


def trimul_gemm_gate_saveact_triton():
    """Input gated GEMM: ``ab = sigmoid(xn@wg^T) * (xn@wp^T)`` plus the saved gate
    ``sig_m = sigmoid(xn@wg^T)`` (fp32, reused by the backward). The driver passes
    mask=None -> APPLY_MASK=False, so no row scale enters either output, and
    TRANSPOSE_OUT=True -> both outputs are written (N, M), hence the transposed reference."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _input_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    wg, wp = rows2d(2 * TRIMUL_D, TRIMUL_D), rows2d(2 * TRIMUL_D, TRIMUL_D)
    ab, sig_m = _input_gemm_fwd(xn, wg, wp, None, True)

    gate, proj = _proj(xn, wg, wp)
    sig = torch.sigmoid(gate)
    return {"ab": (ab, (sig * proj).T), "sig_m": (sig_m, sig.T)}


def trimul_outproj_gemm_gate_saveact_triton():
    """Output gated GEMM: the gate reads x_normed and the projection reads x_out -- two
    DIFFERENT activations through two weights (baseline_dtv1.py:405-406), unlike the input
    path where both dots share one operand. Outputs stay (M, N); sig is fp32."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import _output_gemm_fwd

    xn = rows2d(TRIMUL_ROWS, TRIMUL_D)
    x_out = rows2d(TRIMUL_ROWS, TRIMUL_D)
    wg, wp = rows2d(TRIMUL_D, TRIMUL_D), rows2d(TRIMUL_D, TRIMUL_D)
    ab, sig = _output_gemm_fwd(xn, x_out, wg, wp)

    gate = xn.float() @ wg.float().T
    proj = x_out.float() @ wp.float().T
    s = torch.sigmoid(gate)
    return {"ab": (ab, s * proj), "sig": (sig, s)}


def gated_projection_bwd_gate_recompute_flat_triton():
    """Elementwise backward of the gated projection, from the SAVED sigmoid (so nothing is
    recomputed here): ``d_gate = grad*ab*(1-sig_m)``, ``d_proj = grad*sig_m``, both in fp32
    then cast to grad's dtype. The launcher packs them into one (2N, M) buffer, d_gate in
    the first N rows and d_proj in the last N (baseline_dtv1.py:490-512), so the two halves
    are checked separately -- a swapped pack would otherwise pass on symmetry alone.
    sig_m is drawn from U[0,1) because it is a stored sigmoid, and fp32 as the launcher
    requires."""
    from miniworld_engine.modules.triangle_multiplication.baseline_dtv1 import (
        _elemwise_bwd_combined,
    )

    n = 2 * TRIMUL_D
    grad, ab = rows2d(n, TRIMUL_ROWS), rows2d(n, TRIMUL_ROWS)
    sig_m = torch.rand(n, TRIMUL_ROWS, device=dev(), dtype=torch.float32)
    d_combined = _elemwise_bwd_combined(grad, ab, sig_m)

    gf = grad.float()
    return {
        "d_gate": (d_combined[:n], gf * ab.float() * (1.0 - sig_m)),
        "d_proj": (d_combined[n:], gf * sig_m),
    }


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


def _transition_cuda_fwd_pair():
    from .drivers_trans import _CUDA_N, _transition_cuda_ext, _transition_cuda_operands

    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(torch.bfloat16)
    y = ext.forward(x, wa, wb, ws, _CUDA_N)
    ref, _, _, _ = _transition_ref(x, wa, wb, ws)
    return {"y": (y, ref)}


def transition_cast_cuda():
    """Same launch as ``transition_swiglu_cuda``; both kernels run inside this one forward, and
    the output is the only observable either of them has from Python."""
    return _transition_cuda_fwd_pair()


def transition_swiglu_cuda():
    return _transition_cuda_fwd_pair()


def transition_bwd_cuda():
    """``backward`` returns the grads in the order the .cpp assembles them. Which tensor is which
    is asserted by shape rather than assumed from position: dx matches x, dwa/dwb match the expand
    weights, dws matches the squeeze weight, and the four shapes are mutually distinct at these
    extents, so the mapping is unambiguous."""
    from .drivers_trans import _CUDA_N, _transition_cuda_ext, _transition_cuda_operands

    ext = _transition_cuda_ext()
    x, wa, wb, ws = _transition_cuda_operands(torch.bfloat16)
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
    for name, actual, expected in zip(names, got, refs):
        if tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(
                f"{name}: kernel returned {tuple(actual.shape)}, reference {tuple(expected.shape)} "
                "-- the return order in transition_cuda_kernel.cu:480 may have changed"
            )
        out[name] = (actual, expected)
    return out
