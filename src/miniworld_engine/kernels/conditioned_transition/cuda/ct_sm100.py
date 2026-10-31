"""B200 (sm_100a) ConditionedTransition tail, inference: two tcgen05 GEMMs of the generic sm_100a Transition
(``kernels/transition/cuda/sm100/generic/tg_gemm.cu``, launched through ``fused_generic_sm100a``)::

    h = silu(x Wa^T) * (x Wb^T)                              SWIGLU: both expand products in tensor memory, SwiGLU in the epilogue
    y = res + sigmoid(cond Wsc^T + bsc) * (h Ws^T)            CTGATE: the squeeze and the gate projection as the two phases of one
                                                              output tile (two accumulators), gate and residual in the epilogue

``x`` is the output of the module's AdaLN (the tail is post-AdaLN, as on every other path). bf16 on the bf16 tensor cores, fp32 on the
TF32 ones (where torch allows TF32); any row count. The weights are cast (fp32: rounded to the TF32 grid) once per parameter version
(``_capture.param_forms``). The H100 tail is ``ct_inference`` (sm_90a wgmma); the training tail on both cards is ``ct_train``.
"""

from __future__ import annotations

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int, d: int) -> bool:
    """sm_100, x / cond of one dtype (bf16, or fp32 with TF32 allowed), D a multiple of 64, the hidden width a multiple of 128, the
    conditioning width a multiple of one 128-byte K-block (64 bf16 / 32 fp32)."""
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as g

    if not (x.device.type == "cuda" and cond.device.type == "cuda" and x.dtype == cond.dtype and x.dtype in (torch.bfloat16, torch.float32)):
        return False
    if x.dtype is torch.float32 and not g._tf32_allowed():
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if not g._is_b200(idx):
        return False
    kq = 32 if x.dtype is torch.float32 else 64
    return d % 64 == 0 and nd % 128 == 0 and cond.shape[-1] % kq == 0 and x.shape[-1] == d and x.numel() > 0


def _weights(wa, wb, ws, wsc, bsc, f32: bool):
    """(wa, wb, ws, wsc) in the activation dtype (fp32: TF32 grid) and the gate bias in fp32 (rounded to bf16 first for bf16)."""
    from miniworld_engine.kernels._tf32 import tf32_grid
    from miniworld_engine.kernels.transition.cuda.fused_generic_sm100a import _prep

    dt = torch.float32 if f32 else torch.bfloat16
    src = wa.dtype
    oa, ob, os_, _, _ = _prep(wa, wb.to(src), ws.to(src), f32, False)
    wsc_ = wsc.to(dt).contiguous()
    if f32:
        wsc_ = tf32_grid(wsc_)
    return oa, ob, os_, wsc_, bsc.to(dt).float().contiguous()


def _tail_fake(xn, cond, wa, wb, ws, wsc, bsc, res):
    """Output structure of ``_tail``: y [M, D] in xn's dtype."""
    return torch.empty((xn.shape[0], ws.shape[0]), dtype=xn.dtype, device=xn.device)


@opaque(fake=_tail_fake, name="conditioned_transition_tail_sm100a")
def _tail(xn: torch.Tensor, cond: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
          res: torch.Tensor | None) -> torch.Tensor:
    """xn [M, D], cond [M, DC], the weights from ``_weights``; returns y [M, D] (res + the gated update, or the update alone)."""
    from miniworld_engine.kernels.transition.cuda.fused_generic_sm100a import _K

    k = _K(xn)
    m, d = xn.shape
    nd = wa.shape[0]
    h = torch.empty((m, nd), dtype=xn.dtype, device=xn.device)
    k.gemm("swiglu", m, nd, xn, wa, h, b1=wb, k0=d)
    y = torch.empty((m, d), dtype=xn.dtype, device=xn.device)
    k.gemm("ctgate", m, d, h, ws, y, a1=cond, b1=wsc, bias=bsc, res=res, k0=nd, k1=cond.shape[1], ldres=d if res is not None else 0)
    return y


#: fp32 (TF32) tail: "cublas" (the expand, squeeze and gate GEMMs on cuBLAS TF32 between the CUDA SwiGLU and gate row kernels of ct_rows.cu,
#: the weights packed once per parameter version) or "tcgen05" (the two generic sm_100a GEMMs). Measured (ladder, A = 5, us): d768 c384
#: L128 / 384 / 768 the tcgen05 tail's CT 61.2 / 90.0 / 114.5 against the Triton family's cuBLAS route 51.0 / 73.5 / 106.3 -- the TF32
#: tcgen05 GEMMs of the generic Transition kernel are below cuBLAS at these thin K.
F32_ROUTE = "cublas"


def _f32_weights(wa, wb, ws, wsc, bsc):
    """[Wa; Wb], Ws, Wsc, bsc as contiguous fp32 (cuBLAS rounds TF32 operands itself)."""
    f = lambda w: w.float().contiguous()  # noqa: E731
    return torch.cat([wa, wb]).float().contiguous(), f(ws), f(wsc), f(bsc)


def _tail_cublas_fake(xn, cond, wab, ws, wsc, bsc, res):
    """Output structure of ``_tail_cublas``: y [M, D] like xn."""
    return torch.empty((xn.shape[0], ws.shape[0]), dtype=xn.dtype, device=xn.device)


@opaque(fake=_tail_cublas_fake, name="conditioned_transition_tail_f32_cublas_sm100a")
def _tail_cublas(xn: torch.Tensor, cond: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                 res: torch.Tensor | None) -> torch.Tensor:
    """fp32: ab = xn [Wa; Wb]^T, h = silu(a) b (swiglu_fwd), sp = cond Wsc^T + bsc, y = res + sigmoid(sp) * (h Ws^T) (sigmul_fwd)."""
    from miniworld_engine.kernels._tf32 import tf32_matmul

    from .ct_train import ext as rows_ext

    e = rows_ext()
    with tf32_matmul():
        ab = torch.mm(xn, wab.t())
        h = e.swiglu_fwd(ab)
        out = torch.mm(h, ws.t())
        sp = torch.addmm(bsc, cond, wsc.t())
    return e.sigmul_fwd(out, sp, res)


def ct_inference_sm100(xn, cond, wa, wb, ws, wsc, bsc, residual=None):
    """``residual + sigmoid(cond Wsc^T + bsc) * (silu(xn Wa^T) * (xn Wb^T)) Ws^T`` without a backward: ``xn`` [M, D], ``cond`` [M, DC] of
    one dtype, ``residual`` [M, D] in that dtype or None (then the update alone); the weights in any dtype."""
    f32 = xn.dtype is torch.float32
    if f32 and F32_ROUTE == "cublas":
        ww = _capture.param_forms("ct_sm100a_f32_cublas", (wa, wb, ws, wsc, bsc), lambda: _f32_weights(wa, wb, ws, wsc, bsc))
        return _tail_cublas(xn.contiguous(), cond.contiguous(), *ww, residual.contiguous() if residual is not None else None)
    ww = _capture.param_forms(f"ct_sm100a_{xn.dtype}", (wa, wb, ws, wsc, bsc), lambda: _weights(wa, wb, ws, wsc, bsc, f32))
    res = residual.contiguous() if residual is not None else None
    return _tail(xn.contiguous(), cond.contiguous(), *ww, res)


__all__ = ["ct_inference_sm100", "supported"]
