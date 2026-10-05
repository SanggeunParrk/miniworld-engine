"""Automatic dispatch to the A100 (sm_80) ConditionedTransition: hand-CUDA row passes and cuBLAS GEMMs, no Triton (AF3 Algorithm 25).

    xa = AdaLN(x, cond)                         # ``integrations/adaln_sm80.py``'s passes: LN(cond) w, one GEMM, LN(x) + sigmoid gate + sum
    [a | b] = xa [Wa; Wb]^T ; h = silu(a) b     # one GEMM, then the SwiGLU row pass
    z = h Ws^T ; g = cond Wsc^T + bsc           # two GEMMs (g reads the raw conditioning)
    y = x + sigmoid(g) z                        # one pass; ``delta`` (no residual) leaves the x out

Contract: the engine's kernel backend (implementation TRITON or MINIWORLD) with the engine backend not forced to Triton, sm_80, bf16 or fp32 operands (the module's compute dtype; fp32 runs
on TF32 tensor cores), ``d_hidden`` and ``d_cond`` each one of 128 / 384 / 768, any expansion ``n``, ``cond`` with the leading dims of ``x`` (one conditioning per row) or, with no
gradient, one conditioning shared by the samples of ``x`` (the first dim expanded: stride 0, or size 1: the AdaLN and gate tables then have the L rows of one sample).
``MINIWORLD_CONDTRANS_SM80=0`` turns it off (the module's Triton tail, with the AdaLN module's own switch ``MINIWORLD_ADALN_SM80``); a failed extension build warns once and keeps the
module path.

Inference is one opaque op.  Training is an autograd Function whose forward and backward are each one opaque op: the forward saves the AdaLN's tensors (LN(cond) w, the two statistics, [S | B]),
xa, [a | b], z and the gate logits; the backward runs the gate pass (dz, dg), the squeeze / SwiGLU / expand GEMMs with the SwiGLU backward between them (h is recomputed for the squeeze's
weight gradient), the gate's and the AdaLN's backward passes (the residual gradient and d cond of the gate join the AdaLN backward's own passes), the weight gradients as cuBLAS GEMMs with fp32
outputs.  Column sums (the biases, d w) come from per-block partial rows added in a fixed order.  Numerics and timings: ``docs/gpus/a100/conditioned_transition/conditioned_transition.md``.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.integrations.adaln_sm80 import (
    WIDTHS,
    Branch,
    _mm,
    _mm_op,
    _period,
    _tf32,
    _wgrad,
    cached,
    cast_params,
    like,
    master,
)
from miniworld_engine.kernels._compile import opaque

BF = torch.bfloat16
_FAILED = False
# Paired A100 training measurements: row/cuBLAS wins through 16384 rows;
# the fused atom kernels win from the next measured size, 24576 rows.
_ATOM_TRAIN_ROW_THRESHOLD = 16384


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the two sm_80 extensions (the AdaLN rows and the tail's); False, with one warning, when the toolchain fails (the module path then serves)."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _adaln().available()
        _tail().available()
    except Exception as exc:  # a toolchain problem keeps the module path
        _FAILED = True
        warnings.warn(f"sm_80 ConditionedTransition kernels unavailable, keeping the module path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def _adaln():
    from miniworld_engine.kernels.adaln.cuda import sm80
    return sm80


def _tail():
    from miniworld_engine.kernels.conditioned_transition.cuda import sm80
    return sm80


def serves(module, x: torch.Tensor, cond: torch.Tensor, compute_dtype: torch.dtype, grad: bool) -> bool:
    """Whether ``module`` (a ``ConditionedTransition`` whose backend resolved to the engine's kernels) runs this call on the sm_80 path: ``compute_dtype`` is the dtype the module
    casts x / cond / the weights to, ``grad`` its ``needs_backward``."""
    if os.environ.get("MINIWORLD_CONDTRANS_SM80", "1") == "0" or settings.current().engine_backend == "triton":
        return False
    if not x.is_cuda or not cond.is_cuda or compute_dtype not in (BF, torch.float32):
        return False
    if x.shape[-1] not in WIDTHS or cond.shape[-1] not in WIDTHS or x.numel() == 0:
        return False
    ada = module.ada_ln_in
    if ada.ln_cond.weight is None or ada.to_scale.bias is None or ada.to_bias.bias is not None or module.to_scale.bias is None:
        return False
    if any(w.bias is not None for w in (module.expand_a, module.expand_b, module.squeeze)):
        return False
    if torch.cuda.get_device_capability(x.device) != (8, 0):
        return False
    # update() expands shared conditioning in autograd before the training
    # launch, so its gradient is reduced back to the shared input shape.
    return _period(x, cond, False) is not None and _loads()


def _params(module) -> list[torch.Tensor]:
    """lnw (fp32), Ws1, sb1, Wb1 (the AdaLN), Wa, Wb, Ws (expand, squeeze), Wsc, bsc (the gate): the module's parameters themselves."""
    ada = module.ada_ln_in
    return [ada.ln_cond.weight, ada.to_scale.weight, ada.to_scale.bias, ada.to_bias.weight, module.expand_a.weight, module.expand_b.weight, module.squeeze.weight,
            module.to_scale.weight, module.to_scale.bias]


def _leaves(module, dt: torch.dtype, cache: bool) -> list[torch.Tensor]:
    """The parameters in ``dt`` (lnw stays fp32): casts made outside autograd, kept per parameter version with ``cache`` (inference)."""
    ps = _params(module)
    return [ps[0], *cast_params(tuple(ps[1:]), dt, cache)]


def _pack(ws1, wb1, wa, wb, cache: bool):
    """([Ws1; Wb1] [2 d, dc], [Wa; Wb] [2 n d, d]); with ``cache`` kept per parameter version."""
    def build():
        return torch.cat([ws1, wb1]), torch.cat([wa, wb])

    return cached("pack", (ws1, wb1, wa, wb), build) if cache else build()


def _pack_tf32(wa, wb, ws):
    """``(wab [512, 128], wsp [128, 256])``, the TF32-rounded and permuted weights of the fp32 fused tail (``kernels/conditioned_transition/cuda/sm80.py: pack_tf32``), kept per parameter version."""
    return cached("pack_tf32", (wa, wb, ws), lambda: _tail().pack_tf32(wa, wb, ws))



# ------------------------------------------------------------------------------------------------------------ inference
def _inference_fake(x, cond, lnw, ws1, sb1, wb1, wa, wb, ws, wsc, bsc, eps_x, eps_c, residual):
    return torch.empty_like(x)


@opaque(fake=_inference_fake, name="conditioned_transition_sm80_inference")
def inference(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws1: torch.Tensor, sb1: torch.Tensor, wb1: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
              ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor, eps_x: float, eps_c: float, residual: bool) -> torch.Tensor:
    """x [M, d], cond [P, dc] (P divides M: row r of x takes conditioning row r % P) -> x + tail(AdaLN(x, cond)) (residual) or the update alone."""
    ka, kt = _adaln(), _tail()
    with torch.cuda.device(x.device):
        wcat, wab = _pack(ws1, wb1, wa, wb, True)
        # Small inputs occupy too few SMs with the persistent atom kernels;
        # the row passes plus cuBLAS win through 4096 rows on A100.
        if x.shape[0] > 4096 and kt.atom_supported(x, cond, wa) and ka.atom_supported(x, cond):
            xa = ka.atom_fwd(x, cond, lnw, ws1, wb1, sb1, eps_x, eps_c)[0]
            return kt.atom_fwd(xa, x if residual else None, cond, wab, ws, wsc, bsc)[0]
        if kt.atom_tf32_supported(x, cond, wa) and ka.atom_tf32_supported(x, cond):    # d = dc = 128, fp32: the AdaLN in one kernel, the tail in two, all on TF32 tensor cores
            xa = ka.atom_fwd_tf32(x, cond, lnw, ws1, wb1, sb1, eps_x, eps_c)
            wab32, wsp = _pack_tf32(wa, wb, ws)
            return kt.atom_fwd_tf32(xa, x if residual else None, cond, wab32, wsp, wsc, bsc)
        br = Branch(x.device)
        g = br.run(lambda: _gate_logits(cond, wsc, bsc), cond, wsc, bsc)         # the gate GEMM beside the chain below: nothing needs it before the last pass
        aff, _ = ka.cond_ln(cond, lnw, x.dtype, eps_c)
        xa, _ = ka.epilogue(x, _mm_op(aff, wcat.t()), sb1, eps_x)
        h = kt.swiglu_fwd(_mm_op(xa, wab.t()))
        z = _mm_op(h, ws.t())
        br.join()
        return kt.gate_res_fwd(x if residual else None, z, g)


def _gate_logits(cond, wsc, bsc):
    """g = cond Wsc^T + bsc in the operand dtype (bf16: the bias joins the GEMM's accumulator: one rounding; fp32: TF32)."""
    if cond.dtype is BF:
        return torch.addmm(bsc, cond, wsc.t())
    with _tf32(True):                              # the bias joins the GEMM here too (one cuBLAS epilogue, no extra pass over g)
        return torch.addmm(bsc, cond, wsc.t())


# -------------------------------------------------------------------------------------------------------------- training
def _train_fwd_fake(x, cond, lnw, ws1, sb1, wb1, wa, wb, ws, wsc, bsc, eps_x, eps_c, residual):
    m, d = x.shape
    dc, nd = cond.shape[1], wa.shape[0]
    e = x.new_empty
    if x.shape[0] > _ATOM_TRAIN_ROW_THRESHOLD and _tail().atom_supported(x, cond, wa) and _adaln().atom_supported(x, cond):          # the fused atom kernels keep xa, rn(z) and the statistics only
        return [torch.empty_like(x), e((0,)), e((m, 2), dtype=torch.float32), e((0,)), e((m, 2), dtype=torch.float32), e((m, d)), e((0,)), e((m, d)), e((0,)), e((0,)), e((2 * nd, d))]
    return [torch.empty_like(x), e((m, dc)), e((m, 2), dtype=torch.float32), e((m, 2 * d)), e((m, 2), dtype=torch.float32), e((m, d)), e((m, 2 * nd)), e((m, d)), e((m, d)),
            e((2 * d, dc)), e((2 * nd, d))]


@opaque(fake=_train_fwd_fake, name="conditioned_transition_sm80_train_fwd")
def train_fwd(x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws1: torch.Tensor, sb1: torch.Tensor, wb1: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
              ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor, eps_x: float, eps_c: float, residual: bool) -> list[torch.Tensor]:
    """The forward saving what the backward reads: [y, LN(cond) w [M, dc], cond (mean, rstd) [M, 2], [S | B] [M, 2 d], x (mean, rstd) [M, 2], xa [M, d], [a | b] [M, 2 n d], z [M, d],
    g [M, d], [Ws1; Wb1] [2 d, dc], [Wa; Wb] [2 n d, d]]; cond is one row per row of x (the packed weights are read again by the backward: packed once per step)."""
    ka, kt = _adaln(), _tail()
    with torch.cuda.device(x.device):
        if x.shape[0] > _ATOM_TRAIN_ROW_THRESHOLD and kt.atom_supported(x, cond, wa) and ka.atom_supported(x, cond):          # large d = dc = 128 bf16 inputs: fused rows; small inputs use cuBLAS below
            xa, xst, cst = ka.atom_fwd(x, cond, lnw, ws1, wb1, sb1, eps_x, eps_c, stats=True)
            wab = torch.cat([wa, wb])
            y, z = kt.atom_fwd(xa, x if residual else None, cond, wab, ws, wsc, bsc, save_z=True)
            e = x.new_empty
            return [y, e((0,)), cst, e((0,)), xst, xa, e((0,)), z, e((0,)), e((0,)), wab]          # custom-op outputs may not alias: one empty tensor each
        wcat, wab = _pack(ws1, wb1, wa, wb, False)
        br = Branch(x.device)
        g = br.run(lambda: _gate_logits(cond, wsc, bsc), cond, wsc, bsc)         # the gate GEMM beside the chain below
        aff, cst = ka.cond_ln(cond, lnw, x.dtype, eps_c, stats=True)
        sbt = _mm_op(aff, wcat.t())
        xa, xst = ka.epilogue(x, sbt, sb1, eps_x, stats=True)
        ab = _mm_op(xa, wab.t())
        z = _mm_op(kt.swiglu_fwd(ab), ws.t())
        br.join()
        y = kt.gate_res_fwd(x if residual else None, z, g)
    return [y, aff, cst, sbt, xst, xa, ab, z, g, wcat, wab]


def _train_bwd_fake(dy, x, cond, lnw, ws1, sb1, wb1, wa, wb, ws, wsc, bsc, aff, cst, sbt, xst, xa, ab, z, g, wcat, wab, residual, fp32_grads=False):
    """Fake of ``train_bwd``: the gradient shapes and dtypes."""
    return [torch.empty_like(x), torch.empty_like(cond)] + [torch.empty(t.shape, dtype=torch.float32 if fp32_grads or t is lnw else t.dtype, device=t.device) for t in (lnw, ws1, sb1, wb1, wa, wb, ws, wsc, bsc)]


@opaque(fake=_train_bwd_fake, name="conditioned_transition_sm80_train_bwd")
def train_bwd(dy: torch.Tensor, x: torch.Tensor, cond: torch.Tensor, lnw: torch.Tensor, ws1: torch.Tensor, sb1: torch.Tensor, wb1: torch.Tensor, wa: torch.Tensor,
              wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor, aff: torch.Tensor, cst: torch.Tensor, sbt: torch.Tensor, xst: torch.Tensor,
              xa: torch.Tensor, ab: torch.Tensor, z: torch.Tensor, g: torch.Tensor, wcat: torch.Tensor, wab: torch.Tensor, residual: bool, fp32_grads: bool = False) -> list[torch.Tensor]:
    """[dx, dcond, d lnw, dWs1, d sb1, dWb1, dWa, dWb, dWs, dWsc, d bsc] (each in its input's dtype; the weights' and biases' fp32 when ``fp32_grads``: fp32 master parameters, unrounded) from the gradient dy [M, d] and the forward's saved tensors."""
    ka, kt = _adaln(), _tail()
    d, nd = x.shape[1], wa.shape[0]
    with torch.cuda.device(x.device):
        if x.shape[0] > _ATOM_TRAIN_ROW_THRESHOLD and kt.atom_supported(x, cond, wa) and ka.atom_supported(x, cond):          # same decomposition as train_fwd and its fake
            dz, dg, dcond2, hh, dab, pbsc = kt.atom_bwd_gate(dy, z, cond, xa, wab, ws, wsc, bsc)
            dxa = kt.atom_bwd_dxa(dab, wab)
            dws, dwab, dwsc = _wgrad(dz, hh), _wgrad(dab, xa), _wgrad(dg, cond)
            dx, dcond, dsc, aff, psb1, plw = ka.atom_bwd(dxa, x, cond, xst, cst, lnw, ws1, wb1, sb1, dy if residual else None, dcond2)
            dws1, dwb1 = _wgrad(dsc, aff), _wgrad(dxa, aff)
        else:
            dz, dg, pbsc = kt.gate_res_bwd(dy, z, g)
            br = Branch(x.device)                                          # the weight gradients (and the gate's data gradient) beside the data gradient's chain of row passes and GEMMs
            dwsc = br.run(lambda: _wgrad(dg, cond), dg, cond)
            dcond2 = br.run(lambda: _mm_op(dg, wsc), dg, wsc)
            dcond2_ready = br.mark()
            dab, h = kt.swiglu_bwd(_mm_op(dz, ws), ab)                     # dh = dz Ws [M, n d] stays in the operand dtype
            dws = br.run(lambda: _wgrad(dz, h), dz, h)
            dwab = br.run(lambda: _wgrad(dab, xa), dab, xa)
            dxa = _mm_op(dab, wab)                                         # the AdaLN's dy [M, d]
            dm, dx, psb1 = ka.bwd_x(dxa, x, xst, sbt, sb1, dy if residual else None, x.dtype)
            dw1 = br.run(lambda: _wgrad(dm, aff), dm, aff)
            dca = _mm(dm, wcat)
            br.wait(dcond2_ready, dcond2)
            dcond, plw = ka.cond_bwd(dca, cond, cst, lnw, dcond2)
            br.join()
            dws1, dwb1 = dw1[:d], dw1[d:]
        dbsc, dsb1, dlnw, dws1, dwb1, dwa, dwb, dws, dwsc = ka.finish(            # the column sums and the casts of the weight gradients: one launch
            [pbsc, psb1, plw], [like(bsc, fp32_grads), like(sb1, fp32_grads), lnw], [dws1, dwb1, dwab[:nd], dwab[nd:], dws, dwsc],
            [like(t, fp32_grads) for t in (ws1, wb1, wa, wb, ws, wsc)])
    return [dx, dcond, dlnw, dws1, dsb1, dwb1, dwa, dwb, dws, dwsc, dbsc]


class _Training(torch.autograd.Function):
    """Takes the parameters themselves: the casts to the kernels' dtype happen here, outside autograd, and fp32 masters' weight gradients come back unrounded."""

    @staticmethod
    def forward(ctx, residual, eps_x, eps_c, x, cond, *params):
        ctx.fp32_grads = master((params[1],), x.dtype)
        leaves = [params[0], *cast_params(tuple(params[1:]), x.dtype, False)]
        y, *saved = train_fwd(x, cond, *leaves, eps_x, eps_c, residual)
        ctx.save_for_backward(x, cond, *leaves, *saved)
        ctx.residual = residual
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        v = ctx.saved_tensors
        grads = train_bwd(dy.contiguous(), *v, ctx.residual, ctx.fp32_grads)
        return (None, None, None, *grads)


# ------------------------------------------------------------------------------------------------------------ module entry
def update(module, x: torch.Tensor, cond: torch.Tensor, residual: bool, compute_dtype: torch.dtype, grad: bool) -> torch.Tensor:
    """``ConditionedTransition`` on the sm_80 path (``serves`` must have accepted the call): ``x + delta`` (``residual``) or ``delta``; ``grad`` = ``needs_backward``."""
    d, dc = x.shape[-1], cond.shape[-1]
    xs = x.to(compute_dtype).reshape(-1, d).contiguous()
    if grad:                         # one conditioning per row (a shared one is expanded: autograd sums the per-row gradients)
        cs = cond.to(compute_dtype).expand(*x.shape[:-1], dc).reshape(-1, dc).contiguous()
    else:
        cs = cond if _period(x, cond, False) == xs.shape[0] else cond[0]          # shared: the first sample's rows are the table
        cs = cs.to(compute_dtype).reshape(-1, dc).contiguous()
    eps = (float(module.ada_ln_in.ln_in.eps), float(module.ada_ln_in.ln_cond.eps))
    if grad:
        y = _Training.apply(residual, *eps, xs, cs, *_params(module))
    else:
        y = inference(xs, cs, *_leaves(module, compute_dtype, True), *eps, residual)
    return y.reshape(x.shape)
