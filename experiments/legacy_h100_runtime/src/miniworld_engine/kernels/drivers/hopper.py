"""Native Hopper build drivers, using the builder's actual row/width axes."""
import torch

from miniworld_engine.kernels.drivers import (
    BF16,
    both_level_is_pair,
    dev,
    driver_length,
    driver_width,
)


def _inputs():
    length, width = driver_length(128), driver_width(128)
    rows = length * length if both_level_is_pair(length) else length
    x = torch.randn(rows, width, device=dev(), dtype=BF16)
    gamma = torch.randn(width, device=dev(), dtype=BF16)
    beta = torch.randn_like(gamma)
    wa = torch.randn(4 * width, width, device=dev(), dtype=BF16) / width**0.5
    wb = torch.randn_like(wa) / width**0.5
    return x, gamma, beta, wa, wb


def layernorm_linear_m1():
    from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
        layernorm_linear_cute,
    )
    x, gamma, beta, wa, _ = _inputs()
    for view in (x, x.t().contiguous().t()):
        layernorm_linear_cute(view, gamma, beta, wa, None, eps=1e-5)


def transition_swiglu_fwd():
    from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import (
        transition_expand_swiglu_cute,
    )
    x, gamma, beta, wa, wb = _inputs()
    transition_expand_swiglu_cute(x, gamma, beta, wa, wb, 1e-5)


def transition_gate_bwd():
    from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
        transition_expand_gatebwd_cute,
    )
    x, _, _, wa, wb = _inputs()
    grad = torch.randn(x.shape[0], wa.shape[0], device=x.device, dtype=x.dtype)
    transition_expand_gatebwd_cute(x, grad, wa, wb)


def _lnbwd(dab):
    from miniworld_engine.kernels.layernorm_linear.cute.dgrad_lnbwd import (
        dgrad_lnbwd_cute,
    )
    from miniworld_engine.kernels.transition.cute.dab_lnbwd import (
        transition_dab_lnbwd_cute,
    )
    x, gamma, _, wa, _ = _inputs()
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
    grad = torch.randn(x.shape[0], wa.shape[0], device=x.device, dtype=x.dtype)
    if dab:
        transition_dab_lnbwd_cute(grad, wa, x, gamma, rstd, mean * rstd)
    else:
        xhat = ((x.float() - mean[:, None]) * rstd[:, None]).to(x.dtype)
        dgrad_lnbwd_cute(grad, wa, xhat, gamma, rstd)


def dab_lnbwd():
    _lnbwd(True)


def dgrad_lnbwd():
    _lnbwd(False)


def _cuda(kind):
    from miniworld_engine.kernels.transition import cuda
    x, gamma, beta, wa, wb = _inputs()
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
    args = (x, rstd, mean * rstd, gamma, beta, wa, wb)
    if kind == "b2b":
        ws = torch.randn(x.shape[-1], wa.shape[0], device=x.device, dtype=x.dtype) / wa.shape[0]**0.5
        cuda.transition_b2b_fwd(*args, ws)
        cuda.transition_b2b_fwd_saved(*args, ws)
    elif kind == "expand_gate":
        cuda.transition_expand_gate_fwd(*args)
    else:
        grad = torch.randn(x.shape[0], wa.shape[0], device=x.device, dtype=x.dtype)
        cuda.transition_expand_gatebwd_wgmma(*args, grad)


def transition_fwd_b2b_sm90_cuda():
    _cuda("b2b")


def transition_expand_gate_sm90_cuda():
    _cuda("expand_gate")


def transition_bwd_gate_sm90_cuda():
    _cuda("gatebwd")


def trimul_output_bwd_rows():
    from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import (
        dgrad_ln_rows,
    )
    m,n=driver_length(128)**2,driver_width(128)
    k=2*n
    dy=torch.randn(m,n,device=dev(),dtype=BF16)
    w=torch.randn(n,k,device=dev(),dtype=BF16)
    xhat=torch.randn(m,k,device=dev(),dtype=BF16)
    gamma=torch.randn(k,device=dev(),dtype=BF16)
    stats=[torch.randn(m,device=dev(),dtype=torch.float32) for _ in range(3)]
    dgrad_ln_rows(dy,w,xhat,gamma,*stats)
