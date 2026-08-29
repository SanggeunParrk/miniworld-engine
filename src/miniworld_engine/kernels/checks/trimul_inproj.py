"""Accuracy checkers for the ``trimul_inproj`` family.

A checker takes no arguments and returns what ``autotune.run_all.check_one`` compares: either one
``(actual, expected)`` tensor pair, or a ``dict`` of named pairs. ``run_one`` already proved the
kernel *launches*; these prove the number it wrote is the number the op is defined to produce.

Every reference here is plain torch (fp32 arithmetic over the *same* bf16 input bits, so no input
quantization enters the comparison) and every buffer is built by the helpers in
``drivers_trimul`` -- ``D``/``L``/``M``, ``_x``/``_rows``/``_w``/``_bdll`` are imported rather than
respelled, so a checker can never drift from the shape its driver launches at. That is also what
makes the tile sweep work: ``drivers_trimul`` routes ``D`` and ``L`` through
``drivers.ragged()``, and because every extent below is either one of those two or derived from
them (``M = L*L``, ``2*D``/``4*D``/``5*H`` packed widths, the ``(D,)`` LN weights, the ``(L, D)``
drop scale), ``MINIWORLD_SHAPE_MODE=ragged`` moves the references with the kernels instead of
against them. No shape is written as a literal here.

Two rules the layouts below follow, because the packed/channel-major kernels in this group make
them easy to get wrong:

* **Return a dict, one entry per block.** ``check_one`` scores a pair by
  ``max|a-e| / max|e|``, so stacking blocks of different magnitude into one tensor lets a large
  block hide a wrong small one. ``_dconcat*`` (4-5 stacked gradient blocks) and the front kernels
  (left/right/preact) are split per block, which also makes a *block-order* error report as a
  named failure instead of a vague one.
* **Where the launcher wraps the kernel in cuBLAS GEMMs, check the kernel.** For the two
  "recompute" backward kernels the autograd path folds the kernel's four raw outputs into six
  GEMMs; those checkers launch the kernel at its own launch site (copied from the autograd
  ``backward`` verbatim) so a failure localizes to the kernel and not to a matmul around it.

The four cute kernels in this group (``trimul_gemm_gate_sm100_cute``, ``trimul_gemm_sm100_cute``,
``trimul_gemm_gate_packed_sm100_cute``, ``trimul_outproj_gemm_gate_sm90_cute``) have no checker:
their drivers do not reach a launch on this card, so there is no output to compare. See SKIPPED.

Imports stay LAZY inside each checker, for the reason ``drivers_trimul`` gives: some of these
modules import ``quack`` at module scope, and a top-level import here would make one missing
dependency fail every checker in the file instead of the one it belongs to.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels.checks import _f
from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers.trimul_inproj import D, L, M, _bdll, _rows, _w, _x

EPS = 1e-5      # trimul_back_triton's own default


# ── trimul_inproj: front / back (triton) ─────────────────────────────────────────────────────

def gated_projection_gate_dropres_triton():
    """gate_elem.py _gate_mul_kernel: y = res + ds * (sigmoid(glogit) * proj).

    ADD_RESIDUAL and USE_DROPOUT are constexpr AND autotune key entries, so the plain and the
    fused-residual+dropout forms are different compilations of different configs. Both are
    checked. ``ds`` is the row-broadcast drop scale [L, N] indexed by ``m % L``; it is filled with
    distinct positive values rather than a 0/1 mask so a wrong row index cannot pass.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton

    x_n, proj, Wg = _rows(), _rows(), _w()
    y, gate = gate_elem_triton(x_n, proj, Wg, return_gate=True)
    g = torch.sigmoid(_f(x_n) @ _f(Wg))
    p = _f(proj)

    res = _rows()
    ds = torch.rand(L, D, device=dev(), dtype=BF16)
    y_dr = gate_elem_triton(x_n, proj, Wg, residual=res, dropscale=ds, seq_len=L)
    rows = torch.arange(M, device=dev()) % L
    return {"y": (y, g * p),
            "gate": (gate, g),
            "y_dropres": (y_dr, _f(res) + _f(ds)[rows] * (g * p))}


def gated_projection_bwd_gate_dropres_triton():
    """gate_elem.py _gate_elem_bwd_ew_kernel: d_proj = dy*gate, d_glogit = dy*proj*gate*(1-gate).

    NOTHING HERE IS STOCHASTIC, despite the name. The kernel never draws a mask: dropout enters
    as ``ds_ptr``, a caller-supplied row-broadcast **drop scale** ``[L, N]`` (the pairformer's
    ``drop_row mask/(1-p)``), and for row ``m`` the kernel uses row ``ds[m % L]``. So the checker
    just passes a fixed tensor and the comparison is exactly reproducible. It is filled with
    distinct positive values (``rand``) rather than 0/1: a 0/1 mask cannot distinguish
    ``ds[m % L]`` from a wrong row index whenever both rows happen to be 1, and a uniform scale
    cannot distinguish it from no indexing at all.

    Three compilations are checked, because both switches are ``tl.constexpr``:
      * base -- the driver's own call (USE_DROPOUT=False, FROM_PREACT=False);
      * dropout -- ``USE_DROPOUT`` is in the autotune key, so it is separately tuned as well as
        separately compiled: ``dy_eff = dy * ds[m % L]``, the grad of ``y = ds*(proj*gate)``;
      * from_preact -- ``gate_ptr`` holds the pre-sigmoid logit and the kernel recomputes
        ``gate = sigmoid(preact)`` itself (the fused-forward save path). Not in the autotune key,
        but a distinct specialization the driver never reaches.

    ``gate`` is fed ``sigmoid(noise)``, not raw noise: the kernel's ``gate*(1-gate)`` factor is
    only meaningful for a gate in (0, 1), and this is a SAVED statistic -- so the reference
    consumes the same saved bf16 values (upcast exactly) instead of recomputing a sigmoid, which
    would compare a different function. The reference gets ``d_proj`` and ``dy*proj`` from torch
    autograd over ``y = ds * proj * gate`` and then applies the sigmoid derivative to the saved
    gate; the from_preact reference is a full autograd chain from the preact leaf, so there the
    derivative is torch's end to end.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_bwd_ew

    dy, proj = _rows(), _rows()
    preact = _rows()
    gate = torch.sigmoid(preact)                      # a real gate, in (0, 1)
    ds = torch.rand(L, D, device=dev(), dtype=BF16)   # drop scale rows, all distinct
    rows = torch.arange(M, device=dev()) % L

    d_proj, d_glogit = gate_elem_bwd_ew(dy, proj, gate)
    d_proj_dr, d_glogit_dr = gate_elem_bwd_ew(dy, proj, gate, dropscale=ds, seq_len=L)
    d_proj_pa, d_glogit_pa = gate_elem_bwd_ew(dy, proj, preact, from_preact=True)

    dyf, gf = _f(dy), _f(gate)
    sig_prime = gf * (1.0 - gf)                       # sigmoid' from the SAVED gate

    def _grads(dy_eff):
        """(d_proj, dy_eff*proj) from autograd over y = proj * gate."""
        p = _f(proj).requires_grad_()
        g = _f(gate).requires_grad_()
        (p * g).backward(dy_eff)
        return p.grad, g.grad

    dp_ref, dg_ref = _grads(dyf)
    dp_dr_ref, dg_dr_ref = _grads(dyf * _f(ds)[rows])

    pa = _f(preact).requires_grad_()
    pp = _f(proj).requires_grad_()
    (pp * torch.sigmoid(pa)).backward(dyf)

    return {"d_proj": (d_proj, dp_ref),
            "d_glogit": (d_glogit, dg_ref * sig_prime),
            "d_proj_drop": (d_proj_dr, dp_dr_ref),
            "d_glogit_drop": (d_glogit_dr, dg_dr_ref * sig_prime),
            "d_proj_preact": (d_proj_pa, pp.grad),
            "d_glogit_preact": (d_glogit_pa, pa.grad)}


def trimul_gemm_gate_mmajor_triton():
    """bidirectional.py _bidir_front_kernel: gated in-projection, M-MAJOR outputs.

    One packed B operand (K, 4*H2) holds all four weights with (gate, proj) columns INTERLEAVED
    per side, and the kernel writes three differently-laid-out results:
      * left/right  -> (B, H2, L, L) bdll, i.e. channel-major -- transpose of the (M, H2) result;
      * preact      -> (4*H2, M), left rows [0:2H2) then right, each side interleaved
                       (row 2c = gate logit of channel c, row 2c+1 = proj of channel c).
    The reference builds the interleave with the same ``stack(dim=2).reshape`` the launcher uses
    for the weights, so the two cannot disagree about the packing convention.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        bidir_front_triton,
    )

    h2 = 2 * D                                  # per-side hidden = 2*d_hidden
    x_n = _x()
    WL, WLg, WR, WRg = _w(h2), _w(h2), _w(h2), _w(h2)
    left, right, preact = bidir_front_triton(x_n, WL, WLg, WR, WRg)

    xf = _f(x_n).reshape(M, D)
    gL, pL = xf @ _f(WLg), xf @ _f(WL)          # (M, H2)
    gR, pR = xf @ _f(WRg), xf @ _f(WR)
    return {
        "left": (left.reshape(h2, M), (torch.sigmoid(gL) * pL).t()),
        "right": (right.reshape(h2, M), (torch.sigmoid(gR) * pR).t()),
        "preact_left": (preact[:2 * h2],
                        torch.stack([gL, pL], dim=2).reshape(M, 2 * h2).t()),
        "preact_right": (preact[2 * h2:],
                         torch.stack([gR, pR], dim=2).reshape(M, 2 * h2).t()),
    }


def trimul_outproj_layernorm_gemm_gate_triton():
    """back.py _back_kernel: LN_out + proj GEMM + gate GEMM + multiply, in one kernel.

        y = sigmoid(x_n @ Wg) * (LayerNorm_D(tri) @ Wp)  [+ residual]

    ``tri`` arrives CHANNEL-MAJOR as (D, M) -- the LN reduces over the channel axis, which is the
    strided one -- so the reference transposes it to (M, D) before layer_norm. LN's normed row is
    rounded to bf16 before the GEMM in the kernel; the reference does the same so the check is of
    the kernel's schedule, not of that deliberate cast.

    ADD_RESIDUAL is constexpr and in the autotune key: both compilations are checked.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton

    tri, x_n = _bdll(), _x()
    Wp, Wg = _w(), _w()
    ln_w = torch.randn(D, device=dev(), dtype=BF16)
    ln_b = torch.randn(D, device=dev(), dtype=BF16)
    res = _x()
    y = trimul_back_triton(tri, x_n, Wp, Wg, ln_w, ln_b, eps=EPS)
    y_res = trimul_back_triton(tri, x_n, Wp, Wg, ln_w, ln_b, eps=EPS, residual=res)

    norm = torch.nn.functional.layer_norm(_f(tri).reshape(D, M).t(), (D,),
                                          _f(ln_w), _f(ln_b), EPS)
    proj = _f(norm.to(BF16)) @ _f(Wp)
    gate = torch.sigmoid(_f(x_n).reshape(M, D) @ _f(Wg))
    ref = (gate * proj).reshape(1, L, L, D)
    return {"y": (y, ref), "y_residual": (y_res, ref + _f(res))}


def trimul_bwd_gate_packed_triton():
    """back_fused.py _dconcat_kernel: 4 gate-backward blocks STACKED into one (4D, M) buffer.

    Block order, read off the stores: [d_gLlog ; d_pL ; d_gRlog ; d_pR], each D rows, and each
    one is its own dict entry -- ``check_one`` scores ``max|a-e| / max|e|`` over whichever tensor
    it is handed, so the four stacked as one (4D, M) tensor would let the largest block's
    magnitude absorb a wrong (or block-offset) small one.

    The input the kernel reads is the INTERLEAVED preact: plane 2d is channel d's gate logit,
    plane 2d+1 its projection, and the right side is offset by 2D. The reference de-interleaves
    with strided slices and then gets its numbers from **torch autograd** on the forward
    expression ``sigmoid(glog) * p``, rather than from a hand-written product -- so the
    sigmoid-derivative algebra is torch's, not a restatement of the kernel's own
    ``dL * pL * gL * (1 - gL)``, and a sign or factor dropped in the kernel cannot be dropped
    identically in the reference.

    ``front_bwd_dW`` is used as the launch site (it is what the driver calls); only its first
    return, the kernel's own ``dconc``, is compared. Its other returns are the four cuBLAS weight
    grads, which would check each block through a GEMM instead of elementwise.
    """
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW

    d_left, d_right = _bdll(), _bdll()
    preact, x_n = _bdll(4 * D), _x()
    dconc = front_bwd_dW(d_left, d_right, preact, x_n, _w(), _w(), _w(), _w())[0]

    p = _f(preact).reshape(4 * D, M)
    gLlog = p[0:2 * D:2].clone().requires_grad_()
    pL = p[1:2 * D:2].clone().requires_grad_()
    gRlog = p[2 * D:4 * D:2].clone().requires_grad_()
    pR = p[2 * D + 1:4 * D:2].clone().requires_grad_()
    torch.autograd.backward([torch.sigmoid(gLlog) * pL, torch.sigmoid(gRlog) * pR],
                            [_f(d_left).reshape(D, M), _f(d_right).reshape(D, M)])
    return {"d_gLlog": (dconc[:D], gLlog.grad),
            "d_pL": (dconc[D:2 * D], pL.grad),
            "d_gRlog": (dconc[2 * D:3 * D], gRlog.grad),
            "d_pR": (dconc[3 * D:], pR.grad)}


def trimul_bwd_gate_packed_recompute_triton():
    """back_fused.py _dconcat_sig_kernel: 4 gate-backward blocks STACKED into one (4D, M) buffer.

    Block order, read off the stores: [d_gLlog ; d_pL ; d_gRlog ; d_pR], each D rows.

    "recompute" here means the grads are rebuilt from the FORWARD OUTPUTS instead of the raw
    logits: ``d_glog = dout * lr * (1 - sg)`` where ``lr = sg * proj``, which equals the textbook
    ``dout * proj * sg * (1-sg)``; ``proj = lr / sg`` is never formed. The reference is written in
    the ``lr``/``sg`` form the kernel uses, since ``lr`` and ``sg`` are its actual inputs.

    ``sg`` is a real sigmoid, not raw noise -- the kernel's ``(1 - sg)`` factor is only meaningful
    for sg in (0, 1). ``sg`` is (2D, M): left rows [0:D), right rows [D:2D).
    """
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import (
        front_bwd_dW_sig,
    )

    d_left, d_right = _bdll(), _bdll()
    left, right = _bdll(), _bdll()
    sg = torch.sigmoid(_bdll(2 * D))
    x_n = _x()
    dconc = front_bwd_dW_sig(d_left, d_right, left, right, sg, x_n,
                             _w(), _w(), _w(), _w())[0]

    dL, dR = _f(d_left).reshape(D, M), _f(d_right).reshape(D, M)
    lrL, lrR = _f(left).reshape(D, M), _f(right).reshape(D, M)
    sg2 = _f(sg).reshape(2 * D, M)
    sgL, sgR = sg2[:D], sg2[D:]
    return {"d_gLlog": (dconc[:D], dL * lrL * (1.0 - sgL)),
            "d_pL": (dconc[D:2 * D], dL * sgL),
            "d_gRlog": (dconc[2 * D:3 * D], dR * lrR * (1.0 - sgR)),
            "d_pR": (dconc[3 * D:], dR * sgR)}


# ── trimul_inproj/cute: the two @triton.jit kernels living under cute/ ───────────────────────

