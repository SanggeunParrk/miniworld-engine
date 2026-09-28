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


def _cuda(kind):
    from miniworld_engine.kernels.transition import cuda
    x, gamma, beta, wa, wb = _inputs()
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
    args = (x, rstd, mean * rstd, gamma, beta, wa, wb)
    if kind == "b2b":
        ws = torch.randn(x.shape[-1], wa.shape[0], device=x.device, dtype=x.dtype) / wa.shape[0]**0.5
        cuda.transition_b2b_fwd(*args, ws)
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
