"""Accuracy checkers for the trimul_inproj / tm1 / tm2 / gated_projection kernels.

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
import triton

from miniworld_engine.kernels.drivers import BF16, dev
from miniworld_engine.kernels.drivers_trimul import D, L, M, _bdll, _rows, _w, _x

EPS = 1e-5      # trimul_back_triton's own default


def _f(t: torch.Tensor) -> torch.Tensor:
    """bf16 -> fp32, exactly (no value changes). Reference math runs here."""
    return t.detach().float()


def _exact_fp32_matmul() -> None:
    """Pin the reference's fp32 matmuls to real fp32.

    Every GEMM reference below runs ``fp32 @ fp32``. If TF32 were enabled the reference would
    silently drop its operands to 10 mantissa bits -- coarser than the bf16 kernel's fp32
    accumulation -- and a genuine kernel error of a few 1e-3 would be indistinguishable from the
    reference's own rounding. Torch's default for matmul is already False; this makes the
    requirement explicit and survives a process that changed it (the checkers run under
    ``--isolate``, one subprocess per kernel, so the write cannot leak into another kernel's run,
    and none of these kernels goes through an fp32 cuBLAS call itself).
    """
    torch.backends.cuda.matmul.allow_tf32 = False


# ── gated_projection/triton/main.py ──────────────────────────────────────────────────────────

def gated_projection_gate_triton():
    """sigmoid_gate_fwd_kernel: out = sigmoid(gate) * rep, row-tiled (M, R) with a column loop.

    Launched at TritonGatedProjectionFunction.forward's launch site. The public
    ``triton_gated_projection`` would append its ``@ out_weight`` cuBLAS GEMM to the result; the
    kernel is the sigmoid-multiply, so it is checked without the GEMM's error on top.
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        get_seq_group,
        sigmoid_gate_fwd_kernel,
    )

    gate, x = _rows(), _rows()
    out = torch.empty_like(x)
    grid = lambda meta: [triton.cdiv(M, meta["BLOCK_M1"])]
    sigmoid_gate_fwd_kernel[grid](gate, x, gate.stride(0), x.stride(0), out, M, D,
                                  shape_key=get_seq_group(M))
    return out, torch.sigmoid(_f(gate)) * _f(x)


def gated_projection_bwd_gate_triton():
    """sigmoid_gate_bwd_kernel: d_rep = dy*s, d_gate = dy*rep*s*(1-s), s = sigmoid(gate).

    Same launch as ``drivers_trimul.gated_projection_bwd_gate_triton`` (the autograd backward
    returns ``.float()`` grads, so the kernel is launched directly there too).
    """
    from miniworld_engine.kernels.gated_projection.triton.main import (
        get_seq_group,
        sigmoid_gate_bwd_kernel,
    )

    gate, x, grad_out = _rows(), _rows(), _rows()
    dgate, dx = torch.empty_like(gate), torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)
    sigmoid_gate_bwd_kernel[grid](gate, x, grad_out, dgate, dx, gate.stride(0), x.stride(0),
                                  M, D, shape_key=get_seq_group(M))
    s = torch.sigmoid(_f(gate))
    dy = _f(grad_out)
    return {"dgate": (dgate, dy * _f(x) * s * (1.0 - s)),
            "drep": (dx, dy * s)}


def gated_projection_gate_flat_triton():
    """_sigmul_fwd: the flat (1-D, all-contiguous) form of the same y = sigmoid(g) * o."""
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    gate, out = _x(), _x()
    return sigmoid_gate_fused(gate, out), torch.sigmoid(_f(gate)) * _f(out)


def gated_projection_bwd_gate_flat_triton():
    """_sigmul_bwd, via _SigmoidGate.backward.

    ``da`` is random rather than the driver's ``.sum()`` (all-ones): a uniform upstream grad
    cannot distinguish ``da*o*s*(1-s)`` from an expression that drops the ``da`` factor.
    """
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import sigmoid_gate_fused

    gate, out = _x().requires_grad_(), _x().requires_grad_()
    da = _x()
    sigmoid_gate_fused(gate, out).backward(da)
    s = torch.sigmoid(_f(gate))
    return {"dg": (gate.grad, _f(da) * _f(out) * s * (1.0 - s)),
            "do": (out.grad, _f(da) * s)}


# ── tm1 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_gemm_gate_triton():
    """fused_sigmoid_gate_fwd_kernel: (sigmoid(x@WLg)*(x@WL), sigmoid(x@WRg)*(x@WR)).

    ``TritonTM1Function.forward`` is this kernel and nothing else -- no cuBLAS around it -- so the
    public ``triton_tm1`` is the kernel's launch site and the comparison is of the kernel alone.

    ``N`` is the contraction extent AND the output width here (square weights), so at a ragged D
    the ``for k_start in range(0, N, BLOCK_K)`` loop and the ``offs_n`` output tile go partial at
    once: a K column past ``N`` reaching either accumulator, or an N column past ``N`` reaching
    the store, both move the answer away from a reference that contracts exactly D columns.

    Reference: ``tm1.reference.tm1_pytorch``, the module's own definition, in fp32.
    """
    from miniworld_engine.kernels.tm1.reference import tm1_pytorch
    from miniworld_engine.kernels.tm1.triton.main import triton_tm1

    _exact_fp32_matmul()
    # ``_x()``, not ``_rows()``: TritonTM1Function reads token_key(length_of(x.shape)) BEFORE its
    # own rearrange to (M, d), and ``length_of`` refuses an already-flattened (M, d) -- shape[-2]
    # there is M, not L. It reshapes left/right back to x's shape, and the reference contracts
    # over the last axis either way, so the comparison is the same numbers at (1, L, L, D).
    x = _x()
    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    left, right = triton_tm1(x, WL, WLg, WR, WRg)
    ref_l, ref_r = tm1_pytorch(_f(x), _f(WL), _f(WLg), _f(WR), _f(WRg))
    return {"left": (left, ref_l), "right": (right, ref_r)}


def trimul_bwd_gate_recompute_triton():
    """fused_sigmoid_gate_bwd_kernel: recomputes the four in-projection GEMMs, emits 4 grads.

    ``dLA = dLB * LB * (1 - Lg)`` with ``dLB = dleft * Lg`` is the same expression as the textbook
    ``dleft * LB * s * (1-s)`` -- substituting dLB gives it term for term -- so the reference is
    written in the textbook form and the identity is what gets tested.

    Launched at TritonTM1Function.backward's launch site. dLA/dLB/dRA/dRB are the kernel's own
    outputs; the autograd path reduces them through six cuBLAS GEMMs into (dx, 4x dW), which would
    both blur which block is wrong and add matmul error to the comparison.
    """
    from miniworld_engine.kernels.tm1.triton.main import (
        fused_sigmoid_gate_bwd_kernel,
        get_seq_group,
    )

    x = _rows()
    WL, WLg, WR, WRg = _w(), _w(), _w(), _w()
    dleft, dright = _rows(), _rows()
    dLA, dLB, dRA, dRB = (torch.empty_like(x) for _ in range(4))
    grid = lambda meta: [triton.cdiv(M, meta["BLOCK_M1"]) * triton.cdiv(D, meta["BLOCK_N"])]
    fused_sigmoid_gate_bwd_kernel[grid](x, WLg, WL, WRg, WR, dleft, dright,
                                        dLA, dLB, dRA, dRB, M, D, shape_key=get_seq_group(M))
    xf = _f(x)
    LB, RB = xf @ _f(WL), xf @ _f(WR)
    Lg, Rg = torch.sigmoid(xf @ _f(WLg)), torch.sigmoid(xf @ _f(WRg))
    dl, dr = _f(dleft), _f(dright)
    return {"dLA": (dLA, dl * LB * Lg * (1.0 - Lg)),
            "dLB": (dLB, dl * Lg),
            "dRA": (dRA, dr * RB * Rg * (1.0 - Rg)),
            "dRB": (dRB, dr * Rg)}


def gated_projection_gate_inplace_flat_triton():
    """tm1/cute/launch.py _gate_mul_kernel: IN-PLACE proj *= sigmoid(gate) over a flat buffer.

    proj is both operand and destination, so the reference is taken from a copy made before the
    launch. (The kernel's @autotune carries ``restore_value=['proj_ptr']``, so the tuning sweep
    itself does not compound the multiply -- exactly one application survives.)
    """
    from miniworld_engine.kernels.tm1.cute.launch import _fused_gate_mul

    proj, gate = _bdll().contiguous(), _bdll().contiguous()
    ref = torch.sigmoid(_f(gate)) * _f(proj)
    _fused_gate_mul(proj, gate)
    return proj, ref


def gated_projection_gate_packed_flat_triton():
    """tm1/cute/launch.py _glu_wide_kernel: the two operands are HALVES OF ONE (1, 2D, L, L).

    Flattened, ``wide`` is the gate channels [0:D] then the proj channels [D:2D], each L*L long;
    the kernel pairs flat element e with e + D*L*L. Slicing the channel axis reproduces exactly
    that pairing on the reference side.
    """
    from miniworld_engine.kernels.tm1.cute.launch import _glu_wide

    wide = _bdll(2 * D)
    out = torch.empty(1, D, L, L, device=dev(), dtype=BF16)
    _glu_wide(out, wide, D, L)
    return out, torch.sigmoid(_f(wide[:, :D])) * _f(wide[:, D:])


# ── tm2 ──────────────────────────────────────────────────────────────────────────────────────

def trimul_outproj_gemm_gate_triton():
    """fused_sigmoid_gate2_fwd_kernel: out = sigmoid(x_gate @ W_gate) * (x_out @ W_out).

    Two independent GEMMs sharing one K loop and one output tile. ``TritonTM2Function.forward``
    is this kernel alone, so ``triton_tm2`` is its launch site.

    The two operands are DIFFERENT tensors (``x`` gates, ``y`` is projected), so the driver's two
    separate ``_rows()`` draws are what makes a crossed ``x_gate``/``x_out`` or ``W_gate``/``W_out``
    pointer visible at all -- with one shared input the swap is invisible. As in tm1, ``N`` is both
    the contraction and the output extent, so a ragged D makes both tails partial at once.

    Reference: ``tm2.reference.tm2_pytorch``, in fp32.
    """
    from miniworld_engine.kernels.tm2.reference import tm2_pytorch
    from miniworld_engine.kernels.tm2.triton.main import triton_tm2

    _exact_fp32_matmul()
    # ``_x()`` for both, for the same reason as tm1 above: TritonTM2Function keys on
    # length_of(x.shape) before its rearrange. They are still two INDEPENDENT draws, so a
    # crossed x_gate/x_out or W_gate/W_out pointer stays visible.
    x, y = _x(), _x()
    Wg, Wo = _w(), _w()
    out = triton_tm2(x, y, Wg, Wo)
    return out, tm2_pytorch(_f(x), _f(y), _f(Wg), _f(Wo))


def trimul_outproj_bwd_gate_recompute_triton():
    """fused_sigmoid_gate2_bwd_kernel: recomputes A = x@Wg and B = y@Wo, emits dA/dB.

    ``dA = dB * B * (1-g)`` with ``dB = dy * g`` is again the textbook ``dy * B * g * (1-g)``;
    the reference uses the textbook form. Launched at TritonTM2Function.backward's launch site,
    for the same reason as the tm1 backward above.
    """
    from miniworld_engine.kernels.tm2.triton.main import (
        fused_sigmoid_gate2_bwd_kernel,
        get_seq_group,
    )

    x, y = _rows(), _rows()
    Wg, Wo = _w(), _w()
    grad_out = _rows()
    dA, dB = torch.empty_like(x), torch.empty_like(x)
    grid = lambda meta: [triton.cdiv(M, meta["BLOCK_M1"]) * triton.cdiv(D, meta["BLOCK_N"])]
    fused_sigmoid_gate2_bwd_kernel[grid](x, y, Wg, Wo, grad_out, dA, dB, M, D,
                                        shape_key=get_seq_group(M))
    g = torch.sigmoid(_f(x) @ _f(Wg))
    B = _f(y) @ _f(Wo)
    dy = _f(grad_out)
    return {"dA": (dA, dy * B * g * (1.0 - g)),
            "dB": (dB, dy * g)}


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


def trimul_gemm_gate_packed_mmajor_triton():
    """front.py _lr_kernel: the gated in-projection halves, written CHANNEL-MAJOR.

        left  = sigmoid(x @ WLg) * (x @ WL)   -> (B, D, L, L)
        right = sigmoid(x @ WRg) * (x @ WR)   -> (B, D, L, L)

    Three things this kernel can get wrong, and how the comparison catches each:

    * **The K loop.** ``K = D`` is the contraction extent, and the loop is
      ``for k0 in range(0, K, BLOCK_K)`` with ``rk = k0 + arange(BLOCK_K)``. At a ragged D the
      last iteration is a PARTIAL tile, so any column of ``x``/``Wlr`` past ``K`` that reaches
      the accumulator adds a product of two out-of-range values. The reference contracts exactly
      ``D`` columns, so such a term shows up as a large ``rel`` -- this is the same failure mode
      that took the three fixed kernels from 2e-03 to 6e-01, and the reason
      ``_bidir_front_kernel`` (same loop shape) was found reading past its weight at K=125.
    * **The packed weight.** The launcher interleaves each side's (gate, proj) columns into ONE
      (D, 4D) operand and the kernel recovers them with ``reshape(...,2)`` + ``tl.split``. A
      swapped g/p or a mis-set right-half base offset (``+ 2*D``) changes which weight each
      output used; ``left`` and ``right`` are separate dict entries and the reference builds the
      pair from the *unpacked* WL/WLg/WR/WRg, so it cannot agree with the kernel about the
      packing by construction.
    * **The transposing store.** The result is (M, D) in registers and (D, M) in memory
      (``rd[:, None] * LL + rm[None, :]``). The reference is the reference module's ``bdll``
      output, so a missing/extra transpose is a shape or a large-rel failure, not a wash.

    ``gate`` (the other kernel this launcher runs) is deliberately NOT returned here; it is
    ``_gate_kernel``'s output and has its own checker. Same launch as the driver.
    """
    from miniworld_engine.kernels.trimul_inproj.reference import trimul_inproj_pytorch
    from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton

    _exact_fp32_matmul()
    x = _x()
    WL, WLg, WR, WRg, Wg = _w(), _w(), _w(), _w(), _w()
    left, right, _ = trimul_front_triton(x, WL, WLg, WR, WRg, Wg)
    ref_l, ref_r, _ = trimul_inproj_pytorch(_f(x), _f(WL), _f(WLg), _f(WR), _f(WRg), _f(Wg))
    return {"left": (left, ref_l), "right": (right, ref_r)}


def trimul_outproj_gemm_sigmoid_triton():
    """front.py _gate_kernel: gate = sigmoid(x @ Wg), (M, D) row-major, viewed (B, L, L, D).

    The second, lighter launch of the same ``trimul_front_triton`` front: one (BLOCK_M1, BLOCK_N)
    accumulator over the same partial-tile K loop as ``_lr_kernel``, a plain unary sigmoid, and a
    fully contiguous store. Checked against the reference module's ``gate`` (which keeps blld),
    so the K-loop tail and the (K, D) row-major weight read are both measured; ``left``/``right``
    belong to ``_lr_kernel`` and are dropped here. Same launch as the driver.
    """
    from miniworld_engine.kernels.trimul_inproj.reference import trimul_inproj_pytorch
    from miniworld_engine.kernels.trimul_inproj.triton.front import trimul_front_triton

    _exact_fp32_matmul()
    x = _x()
    WL, WLg, WR, WRg, Wg = _w(), _w(), _w(), _w(), _w()
    gate = trimul_front_triton(x, WL, WLg, WR, WRg, Wg)[2]
    ref_gate = trimul_inproj_pytorch(_f(x), _f(WL), _f(WLg), _f(WR), _f(WRg), _f(Wg))[2]
    return gate, ref_gate


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
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton

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
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW_sig

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


def trimul_bwd_gate_transpose_packed_triton():
    """back_fused.py _dconcat5_kernel: 5 blocks stacked into one (5D, M) buffer.

    Block order, read off the stores: [d_gLlog ; d_pL ; d_gRlog ; d_pR ; d_glogit], each D rows.
    The first four come from the interleaved ``preact`` (row 2d = gate logit, row 2d+1 = proj,
    with the right side offset by 2D) exactly as ``_dconcat_kernel``; the fifth is the TRANSPOSE
    that names this kernel -- d_glogit arrives (M, D) row-major and is relaid channel-major (D, M)
    via the strided read ``dglog_ptr + m*D + d``.

    The kernel is launched at ``front_bwd_dW_glogit``'s launch site (same argument prep as the
    driver) rather than through it: that launcher returns only its cuBLAS products (dxn and five
    dW), which would check each block through a GEMM instead of elementwise.
    """
    from miniworld_engine.kernels.trimul_inproj.triton._autotune import get_seq_group
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import _dconcat5_kernel

    d_left, d_right, preact, x_n = _bdll(), _bdll(), _bdll(4 * D), _x()
    d_glogit = _rows()
    H, DM = D, D * M                            # square single-dir: gate width == hidden width
    dconc5 = torch.empty(5 * H, M, device=dev(), dtype=x_n.dtype)
    _dconcat5_kernel[lambda meta: (triton.cdiv(DM, meta["BLOCK_E"]),)](
        d_left.reshape(H * M), d_right.reshape(H * M), preact.reshape(4 * H, M),
        d_glogit.reshape(M, H), dconc5, M, DM, D=H, shape_key=get_seq_group(M))

    dL, dR = _f(d_left).reshape(H, M), _f(d_right).reshape(H, M)
    p = _f(preact).reshape(4 * H, M)
    gL = torch.sigmoid(p[0:2 * H:2])
    pL = p[1:2 * H:2]
    gR = torch.sigmoid(p[2 * H:4 * H:2])
    pR = p[2 * H + 1:4 * H:2]
    return {"d_gLlog": (dconc5[:H], dL * pL * gL * (1.0 - gL)),
            "d_pL": (dconc5[H:2 * H], dL * gL),
            "d_gRlog": (dconc5[2 * H:3 * H], dR * pR * gR * (1.0 - gR)),
            "d_pR": (dconc5[3 * H:4 * H], dR * gR),
            "d_glogit": (dconc5[4 * H:], _f(d_glogit).reshape(M, H).t())}


# ── trimul_inproj/cute: the two @triton.jit kernels living under cute/ ───────────────────────

def trimul_transpose_triton():
    """front_sm100.py _transpose_kernel: a pure transpose, (M, 2D) row-major -> (2D, M)."""
    from miniworld_engine.kernels.trimul_inproj.cute.front_sm100 import _transpose_blld_to_bdll

    src = _rows(2 * D)
    out = torch.empty(2 * D, M, device=dev(), dtype=BF16)
    _transpose_blld_to_bdll(src, out)
    return out, _f(src).t()


def gated_projection_gate_packed_mmajor_triton():
    """front_train_sm100.py _glu_bdll_kernel: preact (4H, M) -> lr (2H, M), channel-major GLU.

    ONE buffer holds both sides and both operands: within a side, even plane 2d is the gate logit
    of channel d and odd plane 2d+1 is its proj; the right side starts at plane 2H. Output ``lr``
    is left planes [0:H) then right planes [H:2H). Strided slices reproduce that de-interleave.

    Same launch as ``drivers_trimul.gated_projection_gate_packed_mmajor_triton`` (the enclosing
    ``trimul_front_sm100_train`` fallback also runs a quack/sm100 front GEMM, which is a
    different kernel).
    """
    from miniworld_engine.kernels.trimul_inproj.cute.front_train_sm100 import (
        _glu_bdll_kernel,
        get_seq_group,
    )

    h = D
    preact = torch.randn(4 * h, M, device=dev(), dtype=BF16)
    lr = torch.empty(2 * h, M, device=dev(), dtype=BF16)
    grid = lambda meta: (triton.cdiv(h * M, meta["BLOCK_E"]),)
    _glu_bdll_kernel[grid](preact, lr, H=h, M=M, shape_key=get_seq_group(M))

    p = _f(preact)
    ref_l = torch.sigmoid(p[0:2 * h:2]) * p[1:2 * h:2]
    ref_r = torch.sigmoid(p[2 * h:4 * h:2]) * p[2 * h + 1:4 * h:2]
    return {"lr_left": (lr[:h], ref_l), "lr_right": (lr[h:], ref_r)}
