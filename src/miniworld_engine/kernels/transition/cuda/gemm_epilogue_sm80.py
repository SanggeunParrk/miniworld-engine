"""A100 hand-CUDA versions of the GEMM-with-epilogue building blocks the Transition and the triangle-multiplication front share, for the kernel-level benches
(``gemm_epilogue``, ``gemm_epilogue_bwd``, ``dual_gemm_epilogue``, ``dual_gemm_epilogue_bwd``): the wide Transition's kernels (``sm80/tw_kernels_sm80.cuh``) with other
epilogues.

* LayerNorm + Linear (``y = LN(x) W^T``, N = K = D in 128 / 256 / 512): the LayerNorm row kernel (xn, mean / rstd in fp32) -> the tile GEMM (``gemm_res``, no residual);
  backward: dxn = dy W (cuBLAS, fp32), the LayerNorm backward (``ln_bwd`` without the residual branch, dgamma / dbeta partials reduced in a fixed order), dW = dy^T xn (cuBLAS).
* The gated dual-GEMM front of the triangle multiplication (``left = (x WL) sigmoid(x WLg)``, ``right = (x WR) sigmoid(x WRg)``, weights [D, H] used as ``x @ W``): the
  dual-B tile GEMM with the ``a sigmoid(b)`` epilogue (EPI 1) once per side; backward: the gate kernel's twin (``d_p = d sigmoid(g)``, ``d_g = d p sigmoid(g) (1 - sigmoid(g))``,
  p and g recomputed) then cuBLAS for dxn and the four weight gradients.
"""

import torch

from ..._compile import opaque
from . import fused_wide_sm80 as _wide
from .fused_sm80 import _is_fake


def supported(x: torch.Tensor, d_out: int) -> bool:
    """bf16 on an A100, width 128 / 256 / 512 in (the row kernels) and a multiple of 128 out (the tile GEMM's N), the wide switches on, the extension builds."""
    if _wide.switched_off() or not x.is_cuda or x.dtype is not torch.bfloat16 or x.shape[-1] not in (128, 256, 512) or d_out % 128:
        return False
    return _wide._is_ampere(x.device.index if x.device.index is not None else torch.cuda.current_device()) and _wide._loads()


# ---------------------------------------------------------------------------------------------------------------------------------- LayerNorm + Linear
def _lnl_fwd_fake(x, gamma, beta, w, eps, save):
    """(y [M, N] like x, xn like x and (mean, rstd) [M, 2] f32 when ``save`` -- empty otherwise)."""
    keep = save
    return (x.new_empty((x.shape[0], w.shape[0])), x.new_empty(x.shape) if keep else x.new_empty((0,)),
            torch.empty((x.shape[0] if keep else 0, 2), dtype=torch.float32, device=x.device))


@opaque(fake=_lnl_fwd_fake, name="transition_ln_linear_fwd_sm80")
def _lnl_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, w: torch.Tensor, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y = LN(x) W^T for x [M, D], w [N, D] bf16; gamma / beta in any float dtype.  ``save``: also xn and (mean, rstd)."""
    if _is_fake(x, w):
        return _lnl_fwd_fake(x, gamma, beta, w, eps, save)
    ext = _wide._ext()
    xn, stats = ext.ln_fwd(x, _wide._f32(gamma, not save), _wide._f32(beta, not save), eps, save)
    y = ext.gemm_res(xn, w, x.new_empty((0,)), 0)
    return y, xn if save else x.new_empty((0,)), stats


def _lnl_bwd_fake(dy, x, xn, stats, gamma, w, gdt):
    """(dx like x, dgamma [D] and dbeta [D] in ``gdt``, dW like w) -- fresh tensors, nothing aliases an input."""
    return (torch.empty_like(x), torch.empty((x.shape[1],), dtype=gdt, device=x.device), torch.empty((x.shape[1],), dtype=gdt, device=x.device), torch.empty_like(w))


@opaque(fake=_lnl_bwd_fake, name="transition_ln_linear_bwd_sm80")
def _lnl_bwd(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, stats: torch.Tensor, gamma: torch.Tensor, w: torch.Tensor,
             gdt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta, dW) of y = LN(x) W^T: dxn = dy W (fp32), the LayerNorm backward (no residual branch), dW = dy^T xn."""
    if _is_fake(dy, x):
        return _lnl_bwd_fake(dy, x, xn, stats, gamma, w, gdt)
    ext = _wide._ext()
    dxn = _wide._mm_f32(dy, w)
    dx, dgam, dbeta = ext.ln_bwd(dxn, x, stats, _wide._f32(gamma, False), x.new_empty((0,)), gdt)
    return dx, dgam, dbeta, torch.mm(dy.t(), xn)


class _LayerNormLinearSM80(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, w, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        y, xn, stats = _lnl_fwd(flat, gamma, beta, w.contiguous(), float(eps), True)
        ctx.save_for_backward(flat, xn, stats, gamma, w)
        ctx.shape, ctx.bdt = shape, beta.dtype
        return y.reshape(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        flat, xn, stats, gamma, w = ctx.saved_tensors
        dx, dgam, dbeta, dw = _lnl_bwd(dy.reshape(-1, dy.shape[-1]).contiguous(), flat, xn, stats, gamma, w.contiguous(), gamma.dtype)
        return dx.reshape(ctx.shape), dgam, dbeta.to(ctx.bdt), dw.to(w.dtype), None


def layernorm_linear_sm80(x, gamma, beta, w, eps=1e-5):
    """``F.linear(F.layer_norm(x, (D,), gamma, beta, eps), w)`` for bf16 x [..., D] and w [N, D].  Call ``supported(x, N)`` first."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, w))):
        shape = x.shape
        y, _, _ = _lnl_fwd(x.reshape(-1, shape[-1]).contiguous(), gamma, beta, w.contiguous(), float(eps), False)
        return y.reshape(*shape[:-1], w.shape[0])
    return _LayerNormLinearSM80.apply(x, gamma, beta, w, eps)


# ----------------------------------------------------------------------------------------------------------------------------- the gated dual-GEMM front
def _front_fwd_fake(x, wl, wlg, wr, wrg):
    """(left, right), each [M, H] like x (H = the weights' second dimension)."""
    return x.new_empty((x.shape[0], wl.shape[1])), x.new_empty((x.shape[0], wl.shape[1]))


@opaque(fake=_front_fwd_fake, name="transition_dual_gemm_gate_fwd_sm80")
def _front_fwd(x: torch.Tensor, wl: torch.Tensor, wlg: torch.Tensor, wr: torch.Tensor, wrg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(left, right) = ((x WL) sigmoid(x WLg), (x WR) sigmoid(x WRg)) for x [M, D] and weights [D, H] bf16: two launches of the dual-B GEMM with the gate epilogue."""
    if _is_fake(x, wl):
        return _front_fwd_fake(x, wl, wlg, wr, wrg)
    ext = _wide._ext()
    cfg = _wide._tiles(x.shape[0], x.shape[1], wl.shape[1])[0]
    return (ext.dual_swiglu(x, wl.t().contiguous(), wlg.t().contiguous(), cfg, 1), ext.dual_swiglu(x, wr.t().contiguous(), wrg.t().contiguous(), cfg, 1))


def dual_gemm_gate_sm80(x, wl, wlg, wr, wrg):
    """The triangle-multiplication front on an activation x [..., D]: (left, right) of shape [..., H].  bf16, D % 32 == 0, H % 64 == 0; call ``_wide._loads()`` first."""
    shape = x.shape
    left, right = _front_fwd(x.reshape(-1, shape[-1]).contiguous(), wl, wlg, wr, wrg)
    return left.reshape(*shape[:-1], wl.shape[1]), right.reshape(*shape[:-1], wl.shape[1])


def _front_bwd_fake(dl, dr, x, wl, wlg, wr, wrg):
    """(dx like x, [dWL | dWLg] and [dWR | dWRg], each [D, 2 H] like x) -- fresh tensors."""
    return torch.empty_like(x), x.new_empty((x.shape[1], 2 * wl.shape[1])), x.new_empty((x.shape[1], 2 * wl.shape[1]))


@opaque(fake=_front_bwd_fake, name="transition_dual_gemm_gate_bwd_sm80")
def _front_bwd(dl: torch.Tensor, dr: torch.Tensor, x: torch.Tensor, wl: torch.Tensor, wlg: torch.Tensor, wr: torch.Tensor,
               wrg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, [dWL | dWLg], [dWR | dWRg]) with dx [M, D], the weight gradients [D, 2 H] (p, g recomputed from x; d_p = d sigmoid(g), d_g = d p sigmoid(g) (1 - sigmoid(g)))."""
    if _is_fake(dl, x):
        return _front_bwd_fake(dl, dr, x, wl, wlg, wr, wrg)
    ext = _wide._ext()
    cfg = _wide._tiles(x.shape[0], x.shape[1], wl.shape[1])[0]
    dab_l, _ = ext.gate_bwd(x, wl.t().contiguous(), wlg.t().contiguous(), dl, cfg, 1)         # [d_p | d_g] of the left side [M, 2 H]
    dab_r, _ = ext.gate_bwd(x, wr.t().contiguous(), wrg.t().contiguous(), dr, cfg, 1)
    dx = torch.mm(dab_l, torch.cat((wl, wlg), 1).t())                                          # d_p WL^T + d_g WLg^T + (the right side)
    dx.addmm_(dab_r, torch.cat((wr, wrg), 1).t())
    return dx, torch.mm(x.t(), dab_l), torch.mm(x.t(), dab_r)


def dual_gemm_gate_bwd_sm80(d_left, d_right, x_n, wl, wlg, wr, wrg):
    """(dx_n, dWL, dWLg, dWR, dWRg) of the gated dual-GEMM front; d_left / d_right / x_n [..., H] / [..., D] bf16."""
    h = wl.shape[1]
    dx, dw_l, dw_r = _front_bwd(d_left.reshape(-1, h).contiguous(), d_right.reshape(-1, h).contiguous(), x_n.reshape(-1, x_n.shape[-1]).contiguous(), wl, wlg, wr, wrg)
    return dx.reshape(x_n.shape), dw_l[:, :h], dw_l[:, h:], dw_r[:, :h], dw_r[:, h:]
